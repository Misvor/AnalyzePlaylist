"""Tests for taste_pipeline.web.jobs: Job dataclass + JobRunner thread-pool scheduler.

The runner is pure infrastructure (no pipeline imports), so tests use
``tmp_path`` for an isolated ``data_dir`` and lightweight fakes for ``run_fn``.
Async tests use ``pytest-asyncio`` (``@pytest.mark.asyncio`` per test).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import TYPE_CHECKING

import pytest

from taste_pipeline.web.jobs import Job, JobConflictError, JobRunner, ProgressReporter

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


async def _wait_for_terminal(
    runner: JobRunner,
    job_id: uuid.UUID,
    *,
    timeout_s: float = 3.0,
) -> Job:
    """Poll ``runner.get`` until the Job is terminal; raise AssertionError on timeout.

    Uses ``asyncio.sleep`` so we yield to the event loop and never block
    it (adversarial ``hung_or_long_commands`` guard).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job.status in ("succeeded", "failed", "cancelled", "paused"):
            return job
        await asyncio.sleep(0.005)
    msg = f"job {job_id} did not reach terminal state within {timeout_s}s"
    raise AssertionError(msg)


@pytest.mark.asyncio
async def test_submit_returns_job_with_unique_uuid_and_kind(tmp_path: Path) -> None:
    # Given a fresh JobRunner over an empty tmp data_dir
    runner = JobRunner(tmp_path)

    # When submitting a slow job (worker stays in run_fn long enough for us to observe the pre-terminal state)
    def slow_run(report: Callable[[float, str], None]) -> None:
        time.sleep(0.3)

    job = await runner.submit("feed", slow_run)

    # Then the returned Job has a UUID id, the right kind, and a non-terminal status
    assert isinstance(job.id, uuid.UUID)
    assert job.kind == "feed"
    assert job.status in ("queued", "running")
    assert job.error is None

    # Cleanup: let the worker finish so pytest doesn't leave a dangling thread
    await _wait_for_terminal(runner, job.id)


@pytest.mark.asyncio
async def test_job_runs_to_succeeded_with_progress_one_and_log_lines(tmp_path: Path) -> None:
    # Given a runner with a run_fn that reports progress and messages
    runner = JobRunner(tmp_path)

    def run_fn(report: Callable[[float, str], None]) -> None:
        report(0.5, "half")
        report(1.0, "done")

    # When submitting and waiting for the job to reach a terminal state
    job = await runner.submit("feed", run_fn)
    final = await _wait_for_terminal(runner, job.id)

    # Then the job succeeded, progress is 1.0, both log lines are captured, and
    # started_at / finished_at are both populated
    assert final.status == "succeeded"
    assert final.progress == 1.0
    assert "half" in final.log_lines
    assert "done" in final.log_lines
    assert final.error is None
    assert final.started_at is not None
    assert final.finished_at is not None


@pytest.mark.asyncio
async def test_job_failure_path_sets_status_failed_with_error(tmp_path: Path) -> None:
    # Given a runner with a run_fn that raises RuntimeError("boom")
    runner = JobRunner(tmp_path)

    def run_fn(report: Callable[[float, str], None]) -> None:
        raise RuntimeError("boom")

    # When submitting and waiting
    job = await runner.submit("feed", run_fn)
    final = await _wait_for_terminal(runner, job.id)

    # Then the job is failed, the error message contains "boom", and the
    # runner itself did not crash (subsequent submit/list/get still work)
    assert final.status == "failed"
    assert final.error is not None
    assert "boom" in final.error
    # Smoke check the runner survived
    assert runner.get(job.id) is final
    assert isinstance(runner.list(), list)


@pytest.mark.asyncio
async def test_concurrent_same_kind_submit_is_rejected(tmp_path: Path) -> None:
    # Given a runner with an in-flight job of kind "feed" (sleeps ~0.5s)
    runner = JobRunner(tmp_path)

    def slow_run(report: Callable[[float, str], None]) -> None:
        time.sleep(0.5)

    await runner.submit("feed", slow_run)
    await asyncio.sleep(0.05)  # let the first job enter the running state

    # When submitting another job of the same kind
    # Then a JobConflictError is raised whose message names the kind
    with pytest.raises(JobConflictError) as exc_info:
        await runner.submit("feed", lambda report: None)
    assert "feed" in str(exc_info.value)
    assert "concurrent" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_run_record_persists_as_json_under_data_dir_runs(tmp_path: Path) -> None:
    # Given a runner with a successful job
    runner = JobRunner(tmp_path)

    def run_fn(report: Callable[[float, str], None]) -> None:
        report(1.0, "done")

    job = await runner.submit("feed", run_fn)
    await _wait_for_terminal(runner, job.id)

    # When looking for the persisted run record
    record_path = tmp_path / "runs" / f"{job.id}.json"

    # Then the JSON file exists, parses cleanly, and contains every required
    # field with the values we expect
    assert record_path.exists()
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    for field_name in (
        "id",
        "kind",
        "status",
        "progress",
        "started_at",
        "finished_at",
        "error",
        "log_lines",
    ):
        assert field_name in payload, f"missing field {field_name!r}"
    assert payload["id"] == str(job.id)
    assert payload["kind"] == "feed"
    assert payload["status"] == "succeeded"
    assert payload["progress"] == 1.0
    assert payload["log_lines"] == ["done"]
    assert payload["error"] is None
    assert payload["started_at"] is not None
    assert payload["finished_at"] is not None


def test_job_detail_defaults_to_none() -> None:
    # Given a freshly constructed Job
    job = Job(id=uuid.uuid4(), kind="index")

    # Then its structured detail payload is unset
    assert job.detail is None


@pytest.mark.asyncio
async def test_report_detail_is_persisted_and_hydrated(tmp_path: Path) -> None:
    # Given a runner whose run_fn reports a structured detail payload
    runner = JobRunner(tmp_path)
    detail: dict[str, object] = {
        "current": "a.flac",
        "processed": 0,
        "total": 3,
        "remaining": 3,
        "upcoming": ["b.flac", "c.flac"],
        "elapsed_seconds": 0.1,
        "eta_seconds": None,
    }

    def run_fn(report: ProgressReporter) -> None:
        report(0.0, "indexed 0/3: a.flac", detail=detail)

    # When the job runs to completion
    job = await runner.submit("index", run_fn)
    await _wait_for_terminal(runner, job.id)

    # Then a later report without detail leaves the last payload in place
    assert runner.get(job.id).detail == detail
    # And the persisted run record round-trips the detail object
    payload = json.loads((tmp_path / "runs" / f"{job.id}.json").read_text(encoding="utf-8"))
    assert payload["detail"] == detail

    # When a fresh runner hydrates that record from disk
    hydrated = JobRunner(tmp_path).get(job.id)

    # Then the detail payload is restored
    assert hydrated.detail == detail


@pytest.mark.asyncio
async def test_cancel_sets_status_to_cancelled(tmp_path: Path) -> None:
    # Given a runner with a slow running job (sleeps ~1s so we can cancel mid-flight)
    runner = JobRunner(tmp_path)

    def slow_run(report: Callable[[float, str], None]) -> None:
        time.sleep(1.0)

    job = await runner.submit("kind1", slow_run)
    await asyncio.sleep(0.05)  # let the worker thread enter run_fn

    # When cancelling the job
    runner.cancel(job.id)

    # Then status transitions to "cancelled" immediately and finished_at is set
    final = runner.get(job.id)
    assert final.status == "cancelled"
    assert final.finished_at is not None

    # And once the worker thread returns, the run record is persisted as cancelled
    persisted = await _wait_for_terminal(runner, job.id, timeout_s=3.0)
    assert persisted.status == "cancelled"
    record_path = tmp_path / "runs" / f"{job.id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"


@pytest.mark.asyncio
async def test_list_returns_submitted_jobs_newest_first(tmp_path: Path) -> None:
    # Given a runner with two jobs submitted in sequence (different kinds so no conflict)
    runner = JobRunner(tmp_path)
    job1 = await runner.submit("kind1", lambda report: None)
    await asyncio.sleep(0.01)  # ensure submission order is observable
    job2 = await runner.submit("kind2", lambda report: None)

    # When listing jobs
    listed = runner.list()

    # Then both are present and newest-first
    assert len(listed) == 2
    assert listed[0].id == job2.id
    assert listed[1].id == job1.id


@pytest.mark.asyncio
async def test_get_returns_job_by_id(tmp_path: Path) -> None:
    # Given a runner with a submitted job
    runner = JobRunner(tmp_path)
    job = await runner.submit("feed", lambda report: None)

    # When getting by the job's id
    # Then the SAME Job instance is returned (live reference, not a copy)
    assert runner.get(job.id) is job

    # And an unknown id raises KeyError (not None, not empty)
    with pytest.raises(KeyError):
        runner.get(uuid.uuid4())


@pytest.mark.asyncio
async def test_each_job_has_its_own_report_callable(tmp_path: Path) -> None:
    """Stale-state guard: report() in one job's run_fn writes only to that job's log_lines.

    Two jobs of different kinds run interleaved on the thread pool. Each
    call to ``report`` must mutate the bound Job's state, not some shared
    or stale target.
    """
    # Given two concurrent jobs of different kinds whose report() messages are
    # namespaced by kind
    runner = JobRunner(tmp_path)

    def run_a(report: Callable[[float, str], None]) -> None:
        report(0.5, "from-a-1")
        time.sleep(0.15)
        report(1.0, "from-a-2")

    def run_b(report: Callable[[float, str], None]) -> None:
        time.sleep(0.05)
        report(1.0, "from-b-1")

    job_a = await runner.submit("kind-a", run_a)
    job_b = await runner.submit("kind-b", run_b)

    # When both jobs complete
    await _wait_for_terminal(runner, job_a.id)
    await _wait_for_terminal(runner, job_b.id)

    # Then each job's log_lines only contains messages from its own run_fn
    final_a = runner.get(job_a.id)
    final_b = runner.get(job_b.id)
    assert final_a.log_lines, "job A should have at least one log line"
    assert final_b.log_lines, "job B should have at least one log line"
    assert all(line.startswith("from-a") for line in final_a.log_lines)
    assert all(line.startswith("from-b") for line in final_b.log_lines)
    assert not any(line.startswith("from-b") for line in final_a.log_lines)
    assert not any(line.startswith("from-a") for line in final_b.log_lines)


def _write_run_record(runs_dir: Path, payload: dict[str, object], file_name: str = "") -> uuid.UUID:
    """Write one run-record JSON under ``runs_dir`` and return the job id it encodes."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.UUID(str(payload["id"]))
    (runs_dir / (file_name or f"{job_id}.json")).write_text(json.dumps(payload), encoding="utf-8")
    return job_id


def _terminal_record(job_id: uuid.UUID, status: str = "succeeded") -> dict[str, object]:
    """A well-formed persisted terminal run record matching ``JobRunner._persist``."""
    return {
        "id": str(job_id),
        "kind": "index",
        "status": status,
        "progress": 1.0,
        "log_lines": ["indexed 3/3", "indexed 3 files"],
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "error": None,
        "result": None,
        "detail": None,
    }


def test_runner_hydrates_persisted_terminal_run_record(tmp_path: Path) -> None:
    # Given a data_dir with one persisted terminal run record
    job_id = uuid.uuid4()
    _ = _write_run_record(tmp_path / "runs", _terminal_record(job_id, "succeeded"))

    # When constructing a JobRunner over that data_dir
    runner = JobRunner(tmp_path)

    # Then the hydrated job is returned by list() with its full state
    listed = runner.list()
    assert len(listed) == 1, f"expected 1 hydrated job, got {listed!r}"
    assert listed[0].id == job_id
    assert listed[0].kind == "index"
    assert listed[0].status == "succeeded"
    assert listed[0].progress == 1.0
    assert listed[0].log_lines == ["indexed 3/3", "indexed 3 files"]
    assert listed[0].detail is None
    assert listed[0].started_at is not None
    assert listed[0].finished_at is not None


def test_runner_hydrates_detail_from_run_record(tmp_path: Path) -> None:
    # Given a persisted terminal record carrying a structured detail payload
    job_id = uuid.uuid4()
    record = _terminal_record(job_id)
    record["detail"] = {"current": "last.flac", "processed": 2, "total": 3}
    _ = _write_run_record(tmp_path / "runs", record)

    # When constructing a JobRunner over that data_dir
    runner = JobRunner(tmp_path)

    # Then the hydrated job exposes the detail payload
    assert runner.list()[0].detail == {"current": "last.flac", "processed": 2, "total": 3}


def test_runner_skips_malformed_run_records_without_raising(tmp_path: Path) -> None:
    # Given a runs dir with malformed JSON and a record missing required fields
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    (runs_dir / "missing.json").write_text(json.dumps({"kind": "feed"}), encoding="utf-8")

    # When constructing a JobRunner
    runner = JobRunner(tmp_path)

    # Then the bad records are ignored and construction did not raise
    assert runner.list() == []


def test_runner_ignores_non_terminal_run_records(tmp_path: Path) -> None:
    # Given a persisted record whose status is not terminal
    job_id = uuid.uuid4()
    payload = _terminal_record(job_id)
    payload["status"] = "running"
    _ = _write_run_record(tmp_path / "runs", payload)

    # When constructing a JobRunner
    runner = JobRunner(tmp_path)

    # Then the non-terminal record is not hydrated (it is not a finished run)
    assert runner.list() == []


@pytest.mark.asyncio
async def test_pause_sets_status_paused_and_persists(tmp_path: Path) -> None:
    # Given a runner with a slow running job we can pause mid-flight
    runner = JobRunner(tmp_path)

    def slow_run(report: Callable[[float, str], None]) -> None:
        time.sleep(0.3)

    job = await runner.submit("index", slow_run)
    await asyncio.sleep(0.05)  # let the worker thread enter run_fn

    # When pausing the job
    runner.pause(job.id)

    # Then the status is "paused" with finished_at stamped immediately
    paused = runner.get(job.id)
    assert paused.status == "paused"
    assert paused.finished_at is not None

    # And the paused status is persisted right away (pause persists synchronously)
    payload = json.loads((tmp_path / "runs" / f"{job.id}.json").read_text(encoding="utf-8"))
    assert payload["status"] == "paused"

    # And once the worker returns, the finalize path keeps it paused (not succeeded)
    final = await _wait_for_terminal(runner, job.id, timeout_s=3.0)
    assert final.status == "paused"
    payload_after = json.loads((tmp_path / "runs" / f"{job.id}.json").read_text(encoding="utf-8"))
    assert payload_after["status"] == "paused"


@pytest.mark.asyncio
async def test_report_stop_requested_is_false_then_true_after_pause(tmp_path: Path) -> None:
    # Given a runner whose run_fn records report.stop_requested() before and after a pause
    runner = JobRunner(tmp_path)
    seen: dict[str, bool] = {}
    started = threading.Event()
    finished = threading.Event()

    def run_fn(report: ProgressReporter) -> None:
        try:
            seen["before"] = report.stop_requested()
            started.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if report.stop_requested():
                    seen["after"] = True
                    return
                time.sleep(0.005)
            seen["after"] = False
        finally:
            finished.set()

    job = await runner.submit("kind-pause", run_fn)
    assert started.wait(timeout=1.0), "run_fn did not start"

    # Then the stop signal is unset while the job is running
    assert seen["before"] is False

    # When pausing
    runner.pause(job.id)

    # Then the stop signal becomes observable and the job ends paused
    assert finished.wait(timeout=2.0), "run_fn did not observe the stop signal"
    final = await _wait_for_terminal(runner, job.id, timeout_s=3.0)
    assert final.status == "paused"
    assert seen.get("after") is True


@pytest.mark.asyncio
async def test_report_stop_requested_is_true_after_cancel(tmp_path: Path) -> None:
    # Given a runner whose run_fn polls report.stop_requested()
    runner = JobRunner(tmp_path)
    seen: dict[str, bool] = {}
    started = threading.Event()
    finished = threading.Event()

    def run_fn(report: ProgressReporter) -> None:
        try:
            started.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if report.stop_requested():
                    seen["after"] = True
                    return
                time.sleep(0.005)
            seen["after"] = False
        finally:
            finished.set()

    job = await runner.submit("kind-cancel", run_fn)
    assert started.wait(timeout=1.0), "run_fn did not start"

    # When cancelling (existing behavior)
    runner.cancel(job.id)

    # Then the stop signal is observed and the job stays cancelled
    assert finished.wait(timeout=2.0), "run_fn did not observe the stop signal"
    final = await _wait_for_terminal(runner, job.id, timeout_s=3.0)
    assert final.status == "cancelled"
    assert seen.get("after") is True


@pytest.mark.asyncio
async def test_run_fn_honoring_stop_requested_ends_paused_not_succeeded(tmp_path: Path) -> None:
    # Given a run_fn that stops cooperatively once pause is requested, then
    # tries one more report() (which must be a no-op after the stop).
    runner = JobRunner(tmp_path)
    started = threading.Event()
    finished = threading.Event()

    def run_fn(report: ProgressReporter) -> None:
        report(0.1, "start")
        started.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not report.stop_requested():
            time.sleep(0.005)
        report(0.5, "after-stop (should be ignored)")
        finished.set()

    job = await runner.submit("index", run_fn)
    assert started.wait(timeout=1.0), "run_fn did not start"

    # When pausing and letting run_fn return of its own accord
    runner.pause(job.id)
    assert finished.wait(timeout=2.0), "run_fn did not return after pause"

    # Then finalize preserves "paused" (not succeeded, not cancelled) and the
    # post-stop report was discarded.
    final = await _wait_for_terminal(runner, job.id, timeout_s=3.0)
    assert final.status == "paused"
    assert final.status not in ("succeeded", "cancelled")
    assert "after-stop (should be ignored)" not in final.log_lines
