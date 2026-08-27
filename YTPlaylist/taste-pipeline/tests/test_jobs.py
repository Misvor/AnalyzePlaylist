"""Tests for taste_pipeline.web.jobs: Job dataclass + JobRunner thread-pool scheduler.

The runner is pure infrastructure (no pipeline imports), so tests use
``tmp_path`` for an isolated ``data_dir`` and lightweight fakes for ``run_fn``.
Async tests use ``pytest-asyncio`` (``@pytest.mark.asyncio`` per test).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING

import pytest

from taste_pipeline.web.jobs import Job, JobConflictError, JobRunner

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
        if job.status in ("succeeded", "failed", "cancelled"):
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
