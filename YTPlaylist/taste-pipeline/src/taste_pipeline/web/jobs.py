"""Asyncio-friendly job runner for taste-pipeline web UI operations.

Provides :class:`Job` (a per-run state record) and :class:`JobRunner`
(a thread-pool-backed scheduler that enforces one-at-a-time-per-kind
serialization, cooperative cancellation, and run-record persistence).

The runner is pure infrastructure — it does not import or call any
pipeline function (feed, metadata, download, embed, library). Callers
inject the work as ``run_fn(report)`` so the runner can be tested in
isolation and reused across pipeline stages. The dependency-injection
shape mirrors ``library.scan_library(config, embedder)``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, final

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "paused"]
_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({"succeeded", "failed", "cancelled", "paused"})
_MAX_HYDRATED_RUNS: Final[int] = 50

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class JobConflictError(RuntimeError):
    """Raised when a new job of the same kind is submitted while another is active."""


class ProgressReporter(Protocol):
    """Progress sink injected into ``run_fn`` by :class:`JobRunner`.

    Writes the current progress fraction, appends a human-readable log
    line, and -- when supplied -- replaces the job's structured ``detail``
    payload (the Index page reads ``detail`` for the current song, queue,
    elapsed time, and ETA). Also exposes :meth:`stop_requested` so a
    pipeline pass can honor a pause/cancel cooperatively.
    """

    def __call__(
        self,
        progress: float,
        message: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Record one progress update; ``detail`` replaces the structured payload when supplied."""
        ...

    def stop_requested(self) -> bool:
        """Return ``True`` once pause/cancel has been requested for this job."""
        ...


@dataclass(slots=True)
class _StopSignal:
    """Per-job cooperative stop signal: one event plus the reason it was set.

    ``reason`` is the terminal status the job takes when the signal fires:
    ``"cancelled"`` for :meth:`JobRunner.cancel` and ``"paused"`` for
    :meth:`JobRunner.pause`.
    """

    event: threading.Event = field(default_factory=threading.Event)
    reason: JobStatus = "cancelled"


@dataclass(slots=True)
class Job:
    """A single unit of work tracked by :class:`JobRunner`.

    Mutable while running; transitions ``queued`` -> ``running`` -> one of
    ``succeeded``/``failed``/``cancelled``. The full state is persisted
    as JSON at ``<data_dir>/runs/<job_id>.json`` once the job reaches a
    terminal state.

    The ``result`` field is populated by :class:`JobRunner` from the
    return value of ``run_fn`` when the job completes successfully. It is
    intended for pipeline passes that produce a payload the UI needs to
    surface after the job finishes -- e.g. the ``check_url`` job kind
    stashes a :class:`~taste_pipeline.check.CheckResult` here so the
    dashboard can render the verdict without re-querying. Passes that do
    not need to surface data leave it as ``None``.
    """

    id: uuid.UUID
    kind: str
    status: JobStatus = "queued"
    progress: float = 0.0
    log_lines: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, object] | None = None
    detail: dict[str, object] | None = None


def _record_sort_key(job: Job) -> datetime:
    """Sort key for hydration: ``started_at`` with ``finished_at`` as fallback."""
    return job.started_at or job.finished_at or datetime.min.replace(tzinfo=UTC)


def _load_run_record(record_path: Path) -> Job | None:
    """Parse one persisted run record into a terminal :class:`Job`; ``None`` when unusable."""
    try:
        raw = cast("dict[str, object]", json.loads(record_path.read_text(encoding="utf-8")))
        status = raw["status"]
        if status not in _TERMINAL_STATUSES:
            return None
        job_id = uuid.UUID(cast("str", raw["id"]))
        kind = cast("str", raw["kind"])
        started_raw = raw.get("started_at")
        finished_raw = raw.get("finished_at")
        started_at = datetime.fromisoformat(str(started_raw)) if started_raw else None
        finished_at = datetime.fromisoformat(str(finished_raw)) if finished_raw else None
        progress = float(cast("float", raw.get("progress", 0.0)))
        log_lines = [str(line) for line in cast("list[object]", raw.get("log_lines") or [])]
        error_raw = raw.get("error")
        detail_raw = raw.get("detail")
        detail = cast("dict[str, object]", detail_raw) if isinstance(detail_raw, dict) else None
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    return Job(
        id=job_id,
        kind=kind,
        status=status,
        progress=progress,
        log_lines=log_lines,
        started_at=started_at,
        finished_at=finished_at,
        error=str(error_raw) if error_raw is not None else None,
        detail=detail,
    )


@final
class _JobReport:
    """Callable progress sink bound to one job; also exposes its stop signal.

    The runner hands this object to ``run_fn`` as the ``report`` argument.
    Calling it records progress/log/detail; :meth:`stop_requested` lets a
    resumable pass (``library.scan_library``) check whether pause/cancel was
    requested at each file boundary. Once the stop event is set, calling the
    report is a no-op -- the same contract :meth:`JobRunner.cancel` always had.
    """

    __slots__ = ("_job", "_lock", "_stop")

    def __init__(self, job: Job, stop: _StopSignal, lock: threading.Lock) -> None:
        self._job = job
        self._stop = stop
        self._lock = lock

    def __call__(
        self,
        progress: float,
        message: str,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Record one progress update; no-op once pause/cancel has been requested."""
        with self._lock:
            if self._stop.event.is_set():
                return
            self._job.progress = progress
            self._job.log_lines.append(message)
            if detail is not None:
                self._job.detail = detail

    def stop_requested(self) -> bool:
        """Return ``True`` once pause/cancel has been requested for this job."""
        return self._stop.event.is_set()


class JobRunner:
    """Thread-pool-backed job scheduler with per-kind serialization and run-record persistence.

    Each ``submit(kind, run_fn)`` schedules ``run_fn`` on a thread pool.
    Concurrent jobs of the same ``kind`` are rejected with
    :class:`JobConflictError`; jobs of different kinds run in parallel up
    to ``max_workers``. Cancellation is cooperative -- :meth:`cancel` sets
    an event, marks the job ``"cancelled"`` immediately, and stamps
    ``finished_at``; :meth:`pause` does the same but marks the job
    ``"paused"`` so a resumable pass can continue from its last checkpoint
    later. Subsequent ``report()`` calls become no-ops, and the run record
    is persisted as soon as ``run_fn`` returns.
    """

    def __init__(self, data_dir: Path, *, max_workers: int = 2) -> None:
        """Create the runner, ensure ``<data_dir>/runs`` exists, and hydrate persisted terminal runs."""
        self._data_dir: Path = data_dir
        self._runs_dir: Path = data_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock: threading.Lock = threading.Lock()
        self._jobs: dict[uuid.UUID, Job] = {}
        self._active_by_kind: dict[str, Job] = {}
        self._stops: dict[uuid.UUID, _StopSignal] = {}
        self._finalized: dict[uuid.UUID, threading.Event] = {}
        self._hydrate()

    def _hydrate(self) -> None:
        """Load persisted terminal run records oldest-to-newest, capped at the 50 most recent."""
        hydrated: list[Job] = []
        for record_path in self._runs_dir.glob("*.json"):
            job = _load_run_record(record_path)
            if job is not None:
                hydrated.append(job)
        hydrated.sort(key=_record_sort_key)
        for job in hydrated[-_MAX_HYDRATED_RUNS:]:
            self._jobs[job.id] = job

    async def submit(
        self,
        kind: str,
        run_fn: Callable[[ProgressReporter], dict[str, object] | None],
    ) -> Job:
        """Schedule ``run_fn`` for execution and return its :class:`Job` immediately.

        ``run_fn`` may return a ``dict``; when it does (and the job
        succeeds), the runner stashes it on ``Job.result`` so
        ``GET /api/jobs/{id}`` surfaces it. ``None`` (or a non-dict) is
        the "no payload" signal.

        Raises:
            JobConflictError: A job of the same ``kind`` is already queued
                or running on this runner.
        """
        with self._lock:
            active = self._active_by_kind.get(kind)
            if active is not None and active.status not in _TERMINAL_STATUSES:
                msg = f"concurrent job of kind {kind!r} already in progress (job_id={active.id})"
                raise JobConflictError(msg)
            job = Job(id=uuid.uuid4(), kind=kind)
            self._jobs[job.id] = job
            self._active_by_kind[kind] = job
            stop = _StopSignal()
            self._stops[job.id] = stop
            self._finalized[job.id] = threading.Event()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _ = self._executor.submit(self._run, job, run_fn, stop)
        else:
            _ = loop.run_in_executor(self._executor, self._run, job, run_fn, stop)
        return job

    def get(self, job_id: uuid.UUID) -> Job:
        """Return the live :class:`Job` for ``job_id``.

        Raises:
            KeyError: ``job_id`` does not match any submitted job.
        """
        with self._lock:
            if job_id not in self._jobs:
                msg = f"job {job_id} not found"
                raise KeyError(msg)
            return self._jobs[job_id]

    def list(self) -> list[Job]:
        """Return all jobs in newest-first order (most recently submitted first)."""
        with self._lock:
            return list(reversed(list(self._jobs.values())))

    def is_finalized(self, job_id: uuid.UUID) -> bool:
        """True once the worker thread has persisted the job's final run record.

        ``pause``/``cancel`` mark a job terminal synchronously, but the worker
        may still be flushing its last checkpoint. The SSE stream waits on this
        so a client's post-``done`` refresh sees the persisted artifacts.
        Unknown ids (e.g. hydrated records) count as finalized.
        """
        with self._lock:
            event = self._finalized.get(job_id)
            return True if event is None else event.is_set()

    def cancel(self, job_id: uuid.UUID) -> None:
        """Cooperatively cancel a queued or running job.

        Sets the stop event, marks the job ``"cancelled"`` immediately,
        and stamps ``finished_at`` so callers can poll for terminal state
        without waiting for the worker thread to return. Subsequent
        ``report()`` calls from the worker become no-ops; the run record
        is persisted when ``run_fn`` returns.

        Cancellation is cooperative: the runner cannot force ``run_fn`` to
        stop. It only signals; ``run_fn`` must check (via the report
        callable's ``stop_requested()``, which returns ``True`` once set)
        or be naturally short.

        Raises:
            KeyError: ``job_id`` does not match any submitted job.
        """
        self._request_stop(job_id, "cancelled")

    def pause(self, job_id: uuid.UUID) -> None:
        """Cooperatively pause a queued or running job.

        Identical to :meth:`cancel` except the job takes the terminal
        status ``"paused"`` instead of ``"cancelled"``. A resumable pass
        (``library.scan_library``) checkpoints at the file boundary where
        ``stop_requested()`` returns ``True``, so the partial index is
        persisted and a later job resumes from it.

        Raises:
            KeyError: ``job_id`` does not match any submitted job.
        """
        self._request_stop(job_id, "paused")

    def _request_stop(self, job_id: uuid.UUID, status: JobStatus) -> None:
        """Set the job's stop signal and terminal status; persist immediately.

        Shared by :meth:`cancel` and :meth:`pause`; a no-op when the job is
        already terminal.
        """
        with self._lock:
            if job_id not in self._jobs:
                msg = f"job {job_id} not found"
                raise KeyError(msg)
            job = self._jobs[job_id]
            if job.status in _TERMINAL_STATUSES:
                return
            stop = self._stops[job_id]
            stop.reason = status
            stop.event.set()
            job.status = status
            job.finished_at = datetime.now(UTC)
            self._persist(job)

    def _run(
        self,
        job: Job,
        run_fn: Callable[[ProgressReporter], dict[str, object] | None],
        stop: _StopSignal,
    ) -> None:
        """Worker thread entry: transitions Job queued -> running -> terminal, then persists."""
        with self._lock:
            if stop.event.is_set():
                # Cancelled/paused before the worker thread actually started.
                self._persist(job)
                self._finalized[job.id].set()
                return
            job.started_at = datetime.now(UTC)
            if job.status == "queued":
                job.status = "running"

        run_result: object = None
        try:
            run_result = run_fn(_JobReport(job, stop, self._lock))
        except Exception as exc:  # noqa: BLE001 — run_fn is an injected callable; any failure is a job failure
            with self._lock:
                if job.status not in _TERMINAL_STATUSES:
                    job.status = "failed"
                    job.error = str(exc)
        finally:
            with self._lock:
                self._finalize(job, stop, run_result)

    def _finalize(self, job: Job, stop: _StopSignal, run_result: object) -> None:
        """Apply terminal-state transitions + result stashing under the lock; then persist.

        Extracted from :meth:`_run` so the worker-thread entry stays under
        the complexity threshold. ``run_result`` is ``run_fn``'s return
        value: a dict stashes the job's payload (``check_url`` returns a
        ``CheckResult``); a non-dict or ``None`` is the "no payload" signal.
        When the stop signal fired, the job keeps the reason's status
        (``"cancelled"`` / ``"paused"``) rather than flipping to succeeded.
        """
        if job.status not in _TERMINAL_STATUSES:
            if stop.event.is_set():
                job.status = stop.reason
            else:
                job.status = "succeeded"
                job.progress = 1.0
        if job.status == "succeeded" and isinstance(run_result, dict):
            job.result = cast("dict[str, object]", run_result)
        if job.finished_at is None:
            job.finished_at = datetime.now(UTC)
        self._persist(job)
        self._finalized[job.id].set()

    def _persist(self, job: Job) -> None:
        """Atomically write the Job's full state as JSON under ``<data_dir>/runs``."""
        record_path = self._runs_dir / f"{job.id}.json"
        payload: dict[str, object] = {
            "id": str(job.id),
            "kind": job.kind,
            "status": job.status,
            "progress": job.progress,
            "log_lines": list(job.log_lines),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "error": job.error,
            "result": job.result,
            "detail": job.detail,
        }
        tmp_path = record_path.with_suffix(".json.tmp")
        _ = tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _ = tmp_path.replace(record_path)
