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
from typing import TYPE_CHECKING, Literal, cast

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
_TERMINAL_STATUSES: frozenset[JobStatus] = frozenset({"succeeded", "failed", "cancelled"})

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class JobConflictError(RuntimeError):
    """Raised when a new job of the same kind is submitted while another is active."""


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


class JobRunner:
    """Thread-pool-backed job scheduler with per-kind serialization and run-record persistence.

    Each ``submit(kind, run_fn)`` schedules ``run_fn`` on a thread pool.
    Concurrent jobs of the same ``kind`` are rejected with
    :class:`JobConflictError`; jobs of different kinds run in parallel up
    to ``max_workers``. Cancellation is cooperative — :meth:`cancel` sets
    an event, marks the job ``"cancelled"`` immediately, and stamps
    ``finished_at``; subsequent ``report()`` calls become no-ops, and the
    run record is persisted as soon as ``run_fn`` returns.
    """

    def __init__(self, data_dir: Path, *, max_workers: int = 2) -> None:
        """Create the runner and ensure ``<data_dir>/runs`` exists."""
        self._data_dir: Path = data_dir
        self._runs_dir: Path = data_dir / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock: threading.Lock = threading.Lock()
        self._jobs: dict[uuid.UUID, Job] = {}
        self._active_by_kind: dict[str, Job] = {}
        self._cancel_events: dict[uuid.UUID, threading.Event] = {}

    async def submit(
        self,
        kind: str,
        run_fn: Callable[[Callable[[float, str], None]], dict[str, object] | None],
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
            cancel_event = threading.Event()
            self._cancel_events[job.id] = cancel_event

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _ = self._executor.submit(self._run, job, run_fn, cancel_event)
        else:
            _ = loop.run_in_executor(self._executor, self._run, job, run_fn, cancel_event)
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

    def cancel(self, job_id: uuid.UUID) -> None:
        """Cooperatively cancel a queued or running job.

        Sets the cancel event, marks the job ``"cancelled"`` immediately,
        and stamps ``finished_at`` so callers can poll for terminal state
        without waiting for the worker thread to return. Subsequent
        ``report()`` calls from the worker become no-ops; the run record
        is persisted when ``run_fn`` returns.

        Cancellation is cooperative: the runner cannot force ``run_fn`` to
        stop. It only signals; ``run_fn`` must check (via the report
        callable, which no-ops during cancellation) or be naturally short.

        Raises:
            KeyError: ``job_id`` does not match any submitted job.
        """
        with self._lock:
            if job_id not in self._jobs:
                msg = f"job {job_id} not found"
                raise KeyError(msg)
            job = self._jobs[job_id]
            if job.status in _TERMINAL_STATUSES:
                return
            self._cancel_events[job_id].set()
            job.status = "cancelled"
            job.finished_at = datetime.now(UTC)
            self._persist(job)

    def _run(
        self,
        job: Job,
        run_fn: Callable[[Callable[[float, str], None]], dict[str, object] | None],
        cancel_event: threading.Event,
    ) -> None:
        """Worker thread entry: transitions Job queued -> running -> terminal, then persists."""
        with self._lock:
            if cancel_event.is_set():
                # Cancelled before the worker thread actually started.
                self._persist(job)
                return
            job.started_at = datetime.now(UTC)
            if job.status == "queued":
                job.status = "running"

        def report(progress: float, message: str) -> None:
            with self._lock:
                if cancel_event.is_set():
                    return
                job.progress = progress
                job.log_lines.append(message)

        run_result: object = None
        try:
            run_result = run_fn(report)
        except Exception as exc:  # noqa: BLE001 — run_fn is an injected callable; any failure is a job failure
            with self._lock:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.error = str(exc)
        finally:
            with self._lock:
                self._finalize(job, cancel_event, run_result)

    def _finalize(self, job: Job, cancel_event: threading.Event, run_result: object) -> None:
        """Apply terminal-state transitions + result stashing under the lock; then persist.

        Extracted from :meth:`_run` so the worker-thread entry stays under
        the complexity threshold. ``run_result`` is ``run_fn``'s return
        value: a dict stashes the job's payload (``check_url`` returns a
        ``CheckResult``); a non-dict or ``None`` is the "no payload" signal.
        """
        if job.status not in _TERMINAL_STATUSES:
            if cancel_event.is_set():
                job.status = "cancelled"
            else:
                job.status = "succeeded"
                job.progress = 1.0
        if job.status == "succeeded" and isinstance(run_result, dict):
            job.result = cast("dict[str, object]", run_result)
        if job.finished_at is None:
            job.finished_at = datetime.now(UTC)
        self._persist(job)

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
        }
        tmp_path = record_path.with_suffix(".json.tmp")
        _ = tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _ = tmp_path.replace(record_path)
