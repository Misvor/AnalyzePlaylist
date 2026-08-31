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

from fastapi.testclient import TestClient

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


def test_post_jobs_form_encoded_body_returns_422(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES (no factories registered yet)
    client, _runner = _build_client_with_fake_runner(tmp_path)

    # When POSTing a form-encoded body (the default htmx encoding without hx-encoding="json")
    response = client.post("/api/jobs", data={"kind": "feed"})

    # Then 422 -- the endpoint is JSON-only by contract; form-encoded bodies
    # are rejected with a clear error so dashboard regressions surface fast.
    # Locks the "manual JSON parsing" contract: backend does not depend on
    # python-multipart, does not need to support form bodies.
    assert response.status_code == 422
    payload = response.json()
    assert "invalid JSON body" in payload["detail"], f"expected invalid-JSON error message, got {payload!r}"


def test_post_jobs_non_string_kind_returns_422(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES
    client, _runner = _build_client_with_fake_runner(tmp_path)

    # When POSTing a JSON body where kind is not a string
    response = client.post("/api/jobs", json={"kind": 42})

    # Then 422 with a "must be a string" message
    assert response.status_code == 422
    assert "string" in response.json()["detail"]


def test_post_jobs_non_object_body_returns_422(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES
    client, _runner = _build_client_with_fake_runner(tmp_path)

    # When POSTing a JSON body that is a top-level array, not an object
    response = client.post("/api/jobs", json=["feed"])

    # Then 422 -- the body must be a JSON object
    assert response.status_code == 422
    assert "JSON object" in response.json()["detail"]


def test_post_jobs_empty_body_returns_422(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES
    client, _runner = _build_client_with_fake_runner(tmp_path)

    # When POSTing with no body at all
    response = client.post("/api/jobs")

    # Then 422 (request.json() raises JSONDecodeError on empty body)
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
        time.sleep(0.2)
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


def test_dashboard_run_buttons_use_hx_ext_json_enc_for_api_jobs(tmp_path: Path) -> None:
    # Given an app built from a valid config (real route rendering the dashboard)
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then every run button must carry hx-ext="json-enc" so POST /api/jobs
    # receives a JSON body (via the json-enc htmx extension). Without this
    # extension, htmx 2.x falls back to application/x-www-form-urlencoded and
    # the endpoint returns 422.
    assert response.status_code == 200
    body = response.text

    expected_buttons = (
        ("run-feed", '"kind":"feed"'),
        ("run-metadata", '"kind":"metadata"'),
        ("run-download", '"kind":"download"'),
        ("run-index", '"kind":"index"'),
        ("run-calibrate", '"kind":"calibrate"'),
    )
    json_enc_marker = 'hx-ext="json-enc"'
    for button_id, expected_kind_marker in expected_buttons:
        pattern = r'<button\b[^>]*\bid="' + button_id + r'"[^>]*>.*?</button>'
        match = re.search(pattern, body, flags=re.DOTALL)
        assert match is not None, f"button {button_id!r} missing from dashboard"
        block = match.group(0)
        assert json_enc_marker in block, (
            f"button {button_id!r} missing {json_enc_marker}; "
            f"htmx will send form-encoded body and POST /api/jobs returns 422"
        )
        assert 'hx-post="/api/jobs"' in block, f"button {button_id!r} no longer posts to /api/jobs"
        assert expected_kind_marker in block, (
            f"button {button_id!r} hx-vals does not contain {expected_kind_marker!r}"
        )

    actual_count = body.count(json_enc_marker)
    assert actual_count == 5, f"expected exactly 5 {json_enc_marker} attributes, found {actual_count}"


def test_base_template_loads_htmx_and_json_enc_extension(tmp_path: Path) -> None:
    # Given an app with the dashboard rendered
    client = _client(tmp_path)

    # When requesting the dashboard (which extends base.html)
    response = client.get("/")
    body = response.text

    # Then base.html must load both htmx and the json-enc extension,
    # in that order (json-enc depends on the global `htmx` symbol
    # being defined when its script executes).
    assert response.status_code == 200
    htmx_idx = body.find("htmx.min.js")
    json_enc_idx = body.find("htmx-json-enc.js")
    assert htmx_idx != -1, "base.html does not load htmx.min.js"
    assert json_enc_idx != -1, "base.html does not load htmx-json-enc.js"
    assert htmx_idx < json_enc_idx, (
        "htmx-json-enc.js must be loaded AFTER htmx.min.js (the extension's IIFE references window.htmx)"
    )


def test_static_vendor_htmx_json_enc_endpoint_serves_file(tmp_path: Path) -> None:
    # Given an app
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)

    # When GETting the vendored json-enc extension
    response = client.get("/static/vendor/htmx-json-enc.js")

    # Then it serves with a JavaScript content-type and the extension
    # body that calls htmx.defineExtension("json-enc", ...)
    assert response.status_code == 200
    assert "javascript" in response.headers.get("content-type", "").lower()
    body = response.text
    assert 'htmx.defineExtension("json-enc"' in body or "htmx.defineExtension('json-enc'" in body
    assert "JSON.stringify" in body
    assert "application/json" in body


def test_post_jobs_check_url_creates_job_with_kind_check_url(tmp_path: Path) -> None:
    # Given an app with an empty JOB_FACTORIES and a registered slow check_url factory
    client, _runner = _build_client_with_fake_runner(tmp_path)

    def slow_check_url_run(report: Callable[[float, str], None]) -> dict[str, object]:
        time.sleep(0.3)
        report(1.0, "done")
        return {"verdict": "matches", "score": 0.82}

    JOB_FACTORIES["check_url"] = slow_check_url_run
    try:
        # When POSTing to /api/jobs with kind=check_url
        response = client.post("/api/jobs", json={"kind": "check_url"})

        # Then 201 + JSON body with kind=check_url and a non-terminal status
        assert response.status_code == 201, (
            f"check_url kind must be accepted, got {response.status_code}: {response.text}"
        )
        payload = response.json()
        assert payload["kind"] == "check_url"
        assert payload["status"] in ("queued", "running"), (
            f"slow factory should leave the job non-terminal at submit time; got {payload['status']!r}"
        )
    finally:
        JOB_FACTORIES.pop("check_url", None)


def test_get_jobs_returns_check_url_job_with_result_field(
    tmp_path: Path,
) -> None:
    # Given a real JobRunner with a check_url factory that returns a CheckResult dict
    import asyncio  # noqa: PLC0415 -- lazy: test-only
    import time as _time  # noqa: PLC0415 -- lazy: test-only

    from taste_pipeline.web.jobs import JobRunner  # noqa: PLC0415 -- lazy: under test

    runner = JobRunner(tmp_path)

    def check_url_run(report: Callable[[float, str], None]) -> dict[str, object]:
        report(1.0, "done")
        return {
            "verdict": "matches",
            "score": 0.82,
            "top_k": [{"score": 0.82, "library_track_id": "track_0.flac"}],
            "threshold_keep": 0.75,
            "threshold_skip": 0.25,
            "human_readable": "Matches your taste (0.82 >= 0.75)",
        }

    job = asyncio.run(runner.submit("check_url", check_url_run))

    # When waiting for terminal state
    deadline = _time.monotonic() + 3.0
    while _time.monotonic() < deadline:
        current = runner.get(job.id)
        if current.status in ("succeeded", "failed", "cancelled"):
            break
        _time.sleep(0.01)
    final = runner.get(job.id)

    # Then the result field is populated with the returned dict
    assert final.status == "succeeded"
    assert final.result is not None, "result field must be populated after a successful check_url job"
    assert final.result["verdict"] == "matches"
    assert final.result["score"] == 0.82
