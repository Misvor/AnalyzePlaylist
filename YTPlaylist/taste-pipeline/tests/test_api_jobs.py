"""Tests for taste_pipeline.web.routes_jobs: POST/GET/SSE endpoints over JobRunner.

TDD-first for todo-5 (job control + SSE progress endpoints). All tests use
``tmp_path`` for a fresh ``data_dir`` and inject a ``FakeJobRunner`` onto
``app.state.runner`` so no real thread pool or asyncio loop is involved.
SSE tests use ``client.stream("GET", url)`` with ``iter_lines()`` -- the
stream is closed by the ``with`` block when the generator returns.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from taste_pipeline.config import Config, load_config
from taste_pipeline.web import create_app
from taste_pipeline.web.jobs import JobRunner
from taste_pipeline.web.routes_jobs import JOB_FACTORIES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string (forward slashes avoid backslash escapes on Windows)."""
    return path.as_posix()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal valid config TOML under tmp_path and return its path."""
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    body = (
        f'like_library_dir = "{_toml_path(like_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _make_config(tmp_path: Path) -> Config:
    """Build a real Config via load_config from a tmp TOML file."""
    return load_config(_write_config(tmp_path))


class _FakeJob:
    """A synchronous stand-in for the real Job dataclass -- drives state in a background thread."""

    def __init__(
        self,
        job_id: uuid.UUID,
        kind: str,
        run_fn: Callable[[Callable[[float, str], None]], None] | None = None,
    ) -> None:
        self.id = job_id
        self.kind = kind
        self.status = "queued"
        self.progress = 0.0
        self.log_lines: list[str] = []
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self._run_fn = run_fn
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self.start()

    def start(self) -> None:
        """Spawn the background worker thread that calls the injected run_fn."""
        self._thread = threading.Thread(target=self._drive, daemon=True)
        self._thread.start()

    def _drive(self) -> None:
        with self._lock:
            if self._cancel_event.is_set():
                self.status = "cancelled"
                self.finished_at = datetime.now(UTC)
                return
            self.started_at = datetime.now(UTC)
            self.status = "running"

        def report(progress: float, message: str) -> None:
            with self._lock:
                if self._cancel_event.is_set():
                    return
                self.progress = progress
                self.log_lines.append(message)

        try:
            assert self._run_fn is not None
            self._run_fn(report)
        except Exception as exc:  # noqa: BLE001 -- fake runner mirrors JobRunner contract
            with self._lock:
                if self.status != "cancelled":
                    self.status = "failed"
                    self.error = str(exc)
        finally:
            with self._lock:
                if self.status not in ("succeeded", "failed", "cancelled"):
                    self.status = "succeeded"
                    self.progress = 1.0
                if self.finished_at is None:
                    self.finished_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Mark the job cancelled (the FakeJob also wakes its run_fn by no-op'ing report)."""
        with self._lock:
            self._cancel_event.set()
            if self.status not in ("succeeded", "failed", "cancelled"):
                self.status = "cancelled"
                self.finished_at = datetime.now(UTC)


class FakeJobRunner:
    """Async stand-in for JobRunner -- same public shape, no real executor."""

    def __init__(self) -> None:
        self._jobs: dict[uuid.UUID, _FakeJob] = {}
        self._active_by_kind: dict[str, _FakeJob] = {}
        self.submit_calls: list[tuple[str, Callable[[Callable[[float, str], None]], None]]] = []

    async def submit(
        self,
        kind: str,
        run_fn: Callable[[Callable[[float, str], None]], None],
    ) -> _FakeJob:
        """Create a new _FakeJob; raise if an active job of the same kind exists."""
        self.submit_calls.append((kind, run_fn))
        active = self._active_by_kind.get(kind)
        if active is not None and active.status not in ("succeeded", "failed", "cancelled"):
            msg = f"concurrent job of kind {kind!r} already in progress (job_id={active.id})"
            raise RuntimeError(msg)
        job = _FakeJob(uuid.uuid4(), kind, run_fn)
        self._jobs[job.id] = job
        self._active_by_kind[kind] = job
        return job

    def get(self, job_id: uuid.UUID) -> _FakeJob:
        """Return the live _FakeJob for job_id; raise KeyError if unknown."""
        if job_id not in self._jobs:
            msg = f"job {job_id} not found"
            raise KeyError(msg)
        return self._jobs[job_id]

    def list(self) -> list[_FakeJob]:
        """Return all jobs in newest-first order."""
        return list(reversed(list(self._jobs.values())))


def _build_client_with_fake_runner(
    tmp_path: Path,
) -> tuple[TestClient, FakeJobRunner]:
    """Construct an app, inject a FakeJobRunner on app.state, and return (client, runner)."""
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    runner = FakeJobRunner()
    app.state.runner = runner
    return TestClient(app), runner


def _wait_for_terminal(job: _FakeJob, *, timeout_s: float = 2.0) -> _FakeJob:
    """Spin-wait until the FakeJob reaches a terminal status; raise on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    msg = f"job {job.id} did not reach terminal state within {timeout_s}s"
    raise AssertionError(msg)


def _submit(
    runner: FakeJobRunner,
    kind: str,
    run_fn: Callable[[Callable[[float, str], None]], None],
) -> _FakeJob:
    """Drive an async FakeJobRunner.submit from a synchronous test."""
    return asyncio.run(runner.submit(kind, run_fn))


def test_post_jobs_creates_job_and_returns_201_with_json(tmp_path: Path) -> None:
    # Given an app with an injected FakeJobRunner and a registered "feed" factory
    client, _runner = _build_client_with_fake_runner(tmp_path)

    def slow_feed_run(report: Callable[[float, str], None]) -> None:
        time.sleep(0.5)
        report(1.0, "done")

    JOB_FACTORIES["feed"] = slow_feed_run
    try:
        response = client.post("/api/jobs", json={"kind": "feed"})

        # Then 201 + JSON body with id (uuid string), kind, and non-terminal status
        assert response.status_code == 201
        assert response.headers["content-type"].startswith("application/json")
        payload = response.json()
        assert payload["kind"] == "feed"
        assert payload["status"] in ("queued", "running")
        uuid.UUID(payload["id"])  # must parse as a uuid
    finally:
        JOB_FACTORIES.pop("feed", None)


def test_post_jobs_unknown_kind_returns_422(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES (no factories registered)
    client, _runner = _build_client_with_fake_runner(tmp_path)
    JOB_FACTORIES.pop("bogus", None)  # ensure not registered

    # When POSTing an unknown kind
    response = client.post("/api/jobs", json={"kind": "bogus"})

    # Then 422 (Pydantic Literal validation rejects unknown kinds)
    assert response.status_code == 422


def test_post_jobs_missing_kind_returns_422(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES
    client, _runner = _build_client_with_fake_runner(tmp_path)

    # When POSTing an empty body
    response = client.post("/api/jobs", json={})

    # Then 422
    assert response.status_code == 422


def test_get_jobs_returns_list_newest_first(tmp_path: Path) -> None:
    # Given an app with two jobs submitted (different kinds so no conflict)
    client, runner = _build_client_with_fake_runner(tmp_path)

    def trivial(report: Callable[[float, str], None]) -> None:
        report(1.0, "done")

    job1 = _submit(runner, "feed", trivial)
    job2 = _submit(runner, "metadata", trivial)
    _wait_for_terminal(job1)
    _wait_for_terminal(job2)

    # When listing jobs
    response = client.get("/api/jobs")

    # Then 200 + array of length 2 with newest first
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["id"] == str(job2.id)
    assert payload[1]["id"] == str(job1.id)


def test_get_jobs_by_id_returns_single_job(tmp_path: Path) -> None:
    # Given an app with one submitted job
    client, runner = _build_client_with_fake_runner(tmp_path)
    job = _submit(runner, "feed", lambda report: report(1.0, "ok"))

    # When fetching by id
    response = client.get(f"/api/jobs/{job.id}")

    # Then 200 + the same job
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == str(job.id)
    assert payload["kind"] == "feed"

    # And unknown id returns 404
    response_404 = client.get(f"/api/jobs/{uuid.uuid4()}")
    assert response_404.status_code == 404


def _collect_sse_lines(client: TestClient, url: str, *, timeout_s: float = 5.0) -> list[str]:
    """Open an SSE stream, collect every line until the server closes, return the lines."""
    lines: list[str] = []
    with client.stream("GET", url) as response:
        end_at = time.monotonic() + timeout_s
        for line in response.iter_lines():
            lines.append(line)
            if line.startswith("event: done") or time.monotonic() > end_at:
                break
    return lines


def test_get_jobs_events_sse_streams_progress_then_done(tmp_path: Path) -> None:
    # Given an app with a FakeJobRunner whose run_fn reports 0.5 then sleeps 0.05 then reports 1.0
    client, runner = _build_client_with_fake_runner(tmp_path)

    def slow_reporting(report: Callable[[float, str], None]) -> None:
        report(0.5, "halfway")
        time.sleep(0.05)
        report(1.0, "done")

    job = _submit(runner, "feed", slow_reporting)

    # When opening the SSE stream
    lines = _collect_sse_lines(client, f"/api/jobs/{job.id}/events")

    # Then at least one progress line and a final done line are present
    progress_lines = [ln for ln in lines if '"progress"' in ln]
    done_lines = [ln for ln in lines if '"done"' in ln]
    assert progress_lines, f"no progress events in SSE stream: {lines!r}"
    assert done_lines, f"no done event in SSE stream: {lines!r}"
    assert any('"halfway"' in ln for ln in progress_lines), (
        f"no 'halfway' progress message in stream: {progress_lines!r}"
    )
    last_done = done_lines[-1]
    payload = json.loads(last_done.removeprefix("data:"))
    assert payload.get("event") == "done"
    assert payload.get("status") == "succeeded"


def test_get_jobs_events_sse_on_failed_job_ends_with_done_failed(tmp_path: Path) -> None:
    # Given an app with a FakeJobRunner whose run_fn raises immediately
    client, runner = _build_client_with_fake_runner(tmp_path)

    def failing(report: Callable[[float, str], None]) -> None:
        raise RuntimeError("kaboom")

    job = _submit(runner, "feed", failing)

    # When opening the SSE stream
    lines = _collect_sse_lines(client, f"/api/jobs/{job.id}/events")

    # Then the last done event has status=failed and mentions the error
    done_lines = [ln for ln in lines if '"done"' in ln]
    assert done_lines, f"no done event in SSE stream: {lines!r}"
    last_done = done_lines[-1]
    payload = json.loads(last_done.removeprefix("data:"))
    assert payload.get("event") == "done"
    assert payload.get("status") == "failed"
    assert payload.get("error") is not None
    assert "kaboom" in payload["error"]


def test_post_jobs_uses_runner_from_app_state(tmp_path: Path) -> None:
    # Given an app whose state.runner is a FakeJobRunner
    client, runner = _build_client_with_fake_runner(tmp_path)

    def factory(report: Callable[[float, str], None]) -> None:
        report(1.0, "ok")

    JOB_FACTORIES["feed"] = factory
    try:
        # When POSTing a job
        response = client.post("/api/jobs", json={"kind": "feed"})

        # Then runner.submit was called exactly once with kind="feed" and the registered factory
        assert response.status_code == 201
        assert len(runner.submit_calls) == 1
        submitted_kind, submitted_run_fn = runner.submit_calls[0]
        assert submitted_kind == "feed"
        assert submitted_run_fn is factory
    finally:
        JOB_FACTORIES.pop("feed", None)


def test_create_app_attaches_real_job_runner_to_state(tmp_path: Path) -> None:
    # Given a vanilla create_app (no FakeJobRunner injected)
    cfg = _make_config(tmp_path)
    app = create_app(cfg)

    # Then app.state.runner is a real JobRunner bound to the config's data_dir
    assert isinstance(app.state.runner, JobRunner)
    assert app.state.runner._data_dir == cfg.data_dir
