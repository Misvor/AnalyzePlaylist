"""Tests for taste_pipeline.web.routes_jobs: POST/GET/SSE endpoints over JobRunner.

TDD-first for todo-5 (job control + SSE progress endpoints). All tests use
``tmp_path`` for a fresh ``data_dir`` and inject a ``FakeJobRunner`` onto
``app.state.runner`` so no real thread pool or asyncio loop is involved.
SSE tests use ``client.stream("GET", url)`` with ``iter_lines()`` -- the
stream is closed by the ``with`` block when the generator returns.

The ``FakeJobRunner`` / ``_FakeJob`` / ``_build_client_with_fake_runner`` /
``_wait_for_terminal`` / ``_submit`` helpers live in ``tests/conftest.py``
(shared with the triage-route test files via the same module-level
import pattern) so the duplicate-helper bloat does not return.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import TYPE_CHECKING

from conftest import (
    _build_client_with_fake_runner,
    _client,
    _make_config,
    _submit,
    _wait_for_terminal,
)
from taste_pipeline.web import create_app
from taste_pipeline.web.jobs import JobRunner
from taste_pipeline.web.routes_jobs import JOB_FACTORIES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.testclient import TestClient


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


def test_dashboard_run_buttons_use_hx_encoding_json_for_api_jobs(tmp_path: Path) -> None:
    # Given an app built from a valid config (real route rendering the dashboard)
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then every run button must carry hx-encoding="json" so POST /api/jobs
    # receives a JSON body (Pydantic JobCreateRequest) instead of form-encoded.
    assert response.status_code == 200
    body = response.text

    expected_buttons = (
        ("run-feed", '"kind":"feed"'),
        ("run-metadata", '"kind":"metadata"'),
        ("run-download", '"kind":"download"'),
        ("run-index", '"kind":"index"'),
    )
    hx_encoding_marker = 'hx-encoding="json"'
    for button_id, expected_kind_marker in expected_buttons:
        pattern = r'<button\b[^>]*\bid="' + button_id + r'"[^>]*>.*?</button>'
        match = re.search(pattern, body, flags=re.DOTALL)
        assert match is not None, f"button {button_id!r} missing from dashboard"
        block = match.group(0)
        assert hx_encoding_marker in block, (
            f"button {button_id!r} missing {hx_encoding_marker}; "
            f"htmx will send form-encoded body and POST /api/jobs returns 422"
        )
        assert 'hx-post="/api/jobs"' in block, f"button {button_id!r} no longer posts to /api/jobs"
        assert expected_kind_marker in block, (
            f"button {button_id!r} hx-vals does not contain {expected_kind_marker!r}"
        )

    actual_count = body.count(hx_encoding_marker)
    assert actual_count == 4, f"expected exactly 4 {hx_encoding_marker} attributes, found {actual_count}"
