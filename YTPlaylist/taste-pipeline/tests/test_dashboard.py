"""Tests for taste_pipeline.web: Dashboard page (run controls + run history + live progress).

TDD-first for todo-8. All tests use ``tmp_path`` for a fresh ``data_dir`` and
build a real ``create_app`` over a real ``Config`` so the ``GET /`` route
renders the dashboard through the real Jinja2Templates/StaticFiles stack.
No real network, no real YouTube, no real JobRunner: the real ``JobRunner``
from todo-4 is harmless on an empty ``data_dir`` (no jobs until something
posts to ``/api/jobs``), so the dashboard's history table is empty in the
default fixture.

The dashboard's contract:
- extends ``base.html`` (so nav IDs, status bar, htmx script, app.css are
  inherited),
- contains five run buttons (one per pipeline pass kind + calibrate) wired with
  ``hx-post="/api/jobs"`` + ``hx-vals='{"kind":"<feed|metadata|download|index>"}'``,
- contains a ``<table id="run-history">`` for htmx to populate from
  ``GET /api/jobs`` (the actual row population is a client-side htmx swap
  in the browser, not server-rendered),
- contains a visible error handler for the "concurrent same-kind job
  rejected" response (htmx:responseError listener + #concurrent-error
  element),
- contains an inline script that wires ``EventSource('/api/jobs/{id}/events')``
  to running jobs (via a function re-invoked on htmx:afterSwap),
- does NOT auto-start any job on page load.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from taste_pipeline.config import Config, load_config
from taste_pipeline.web import create_app

if TYPE_CHECKING:
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


def _client(tmp_path: Path) -> TestClient:
    """Build a TestClient over a real Config-derived app."""
    return TestClient(create_app(_make_config(tmp_path)))


def _extract_hx_vals(html: str, button_id: str) -> str | None:
    """Extract the value of the ``hx-vals`` attribute from the button with the given id.

    The attribute may appear before or after the ``id=`` on the same tag,
    so the helper tries both orderings. The hx-vals value is JSON; the
    outer quote type is single in our templates and the JSON inside uses
    double quotes, so the capture must allow the opposite quote type to
    appear inside the value.
    """
    pattern_id_first = r'<button\b[^>]*\bid="' + re.escape(button_id) + r'"[^>]*\bhx-vals=([\'"])(.+?)\1'
    match = re.search(pattern_id_first, html, flags=re.DOTALL)
    if match is None:
        pattern_hxvals_first = (
            r'<button\b[^>]*\bhx-vals=([\'"])(.+?)\1[^>]*\bid="' + re.escape(button_id) + r'"'
        )
        match = re.search(pattern_hxvals_first, html, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(2)


def _button_exists(html: str, button_id: str, *, expected_label: str) -> bool:
    """Return True iff a <button id={button_id}> exists with the given label text and hx-post=/api/jobs.

    The label check is done against the entire button's outer HTML (so the
    label substring is inside the same <button> element), and hx-post is
    verified independently to lock the URL.
    """
    pattern = r'<button\b[^>]*\bid="' + re.escape(button_id) + r'"[^>]*>.*?</button>'
    match = re.search(pattern, html, flags=re.DOTALL)
    if match is None:
        return False
    block = match.group(0)
    return expected_label in block and 'hx-post="/api/jobs"' in block


def test_get_root_renders_dashboard_with_five_run_buttons(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the response is 200 HTML and contains one button per kind, each
    # wired to POST /api/jobs with the right kind in hx-vals, and each with
    # the human-readable label from the plan.
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text

    expected_buttons = (
        ("run-feed", "Pass A — Feed", '"kind":"feed"'),
        ("run-metadata", "Pass B — Metadata", '"kind":"metadata"'),
        ("run-download", "Pass C — Download", '"kind":"download"'),
        ("run-index", "Build Index", '"kind":"index"'),
        ("run-calibrate", "Calibrate weights", '"kind":"calibrate"'),
    )
    for button_id, expected_label, expected_kind_marker in expected_buttons:
        assert _button_exists(body, button_id, expected_label=expected_label), (
            f"button {button_id!r} missing or misconfigured: "
            f"label {expected_label!r} or hx-post=/api/jobs absent"
        )
        hx_vals = _extract_hx_vals(body, button_id)
        assert hx_vals is not None, f"button {button_id!r} has no hx-vals attribute"
        assert expected_kind_marker in hx_vals, (
            f"button {button_id!r} hx-vals {hx_vals!r} does not contain {expected_kind_marker!r}"
        )


def test_get_root_renders_run_history_table_with_columns(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the response is 200 HTML and contains a <table id="run-history">
    # with a <thead> whose columns cover kind / status / progress / started /
    # finished / error (matches the Job JSON shape from routes_jobs._job_to_payload).
    assert response.status_code == 200
    body = response.text
    assert '<table id="run-history">' in body, "run-history <table> with id=run-history missing"
    # Locate the run-history table and inspect its thead.
    table_match = re.search(r'<table id="run-history">(.*?)</table>', body, flags=re.DOTALL)
    assert table_match is not None, "could not isolate run-history table"
    table_inner = table_match.group(1)
    thead_match = re.search(r"<thead\b[^>]*>(.*?)</thead>", table_inner, flags=re.DOTALL)
    assert thead_match is not None, "run-history table has no <thead>"
    thead = thead_match.group(1).lower()
    for col in ("kind", "status", "progress", "started", "finished", "error"):
        assert col in thead, f"run-history thead missing column {col!r}"


def test_get_root_does_not_auto_start_jobs(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the rendered HTML MUST NOT contain any hx-trigger="load" on an
    # element that posts to /api/jobs -- the dashboard renders an empty
    # history table that the user populates by clicking run buttons.
    body = response.text
    # The dangerous pattern: hx-trigger="load" near hx-post="/api/jobs" (or
    # hx-get="/api/jobs"). Simpler invariant: no hx-trigger="load" on the
    # page at all, because the only safe auto-load would be a status panel
    # refresh which the spec does not require.
    assert 'hx-trigger="load"' not in body, (
        f'dashboard auto-fires on load; found hx-trigger="load". body excerpt:\n{body[:500]}'
    )
    # Belt-and-suspenders: no <body> hx-get=/api/jobs on load.
    assert not re.search(r'<body[^>]*hx-(get|post)="/api/jobs"[^>]*hx-trigger="load"', body), (
        "dashboard <body> auto-fetches /api/jobs on load"
    )


def test_dashboard_renders_app_name_in_title(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the <title> contains the app name "Taste Pipeline" (so the
    # browser tab and any bookmarks carry the product name, not just the
    # page-specific prefix).
    assert response.status_code == 200
    title_match = re.search(r"<title>(.*?)</title>", response.text, flags=re.DOTALL)
    assert title_match is not None, "dashboard has no <title> element"
    title = title_match.group(1)
    assert "Taste Pipeline" in title, f"dashboard <title> {title!r} does not contain 'Taste Pipeline'"


def test_dashboard_uses_base_template(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the rendered HTML inherits the base shell structure: top action
    # bar, main nav, content block, bottom status bar. The dashboard extends
    # base.html so these IDs must be present in the final document.
    assert response.status_code == 200
    body = response.text
    for required_id in ("action-bar", "main-nav", "content", "status-bar"):
        assert f'id="{required_id}"' in body, (
            f"dashboard missing inherited base.html element id={required_id!r}"
        )
    # The four nav anchors must also be present (so the dashboard is
    # navigable from any other page that links to the same nav).
    for nav_id in ("nav-dashboard", "nav-triage", "nav-index", "nav-settings"):
        assert nav_id in body, f"dashboard missing nav anchor {nav_id!r}"


def test_dashboard_includes_htmx_and_handles_concurrent_rejection_visibly(
    tmp_path: Path,
) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the response includes the vendored htmx script (inherited from
    # base.html's <head>) AND a handler for the "concurrent same-kind job
    # rejected" response. The handler is the bit that turns a 422 from
    # /api/jobs into a visible error badge instead of a silent failure.
    assert response.status_code == 200
    body = response.text
    # Vendored htmx is loaded by base.html
    assert "/static/vendor/htmx.min.js" in body, (
        "dashboard does not load the vendored htmx bundle from base.html"
    )
    # The handler: either a htmx:responseError listener is present OR a
    # #concurrent-error placeholder is present. We assert BOTH because the
    # full contract is: "the response handler exists AND it has somewhere
    # to render the error". A page that only has one half is incomplete.
    assert "htmx:responseError" in body, (
        "dashboard has no htmx:responseError listener; concurrent same-kind rejections will be silent"
    )
    assert 'id="concurrent-error"' in body, (
        "dashboard has no #concurrent-error element for visible rejection messaging"
    )


def test_dashboard_event_source_setup_for_running_jobs(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the dashboard page
    response = client.get("/")

    # Then the response includes inline JS that wires EventSource to the
    # SSE endpoint for running jobs. The contract is: a function is
    # defined (attachJobStreamers) AND EventSource is constructed against
    # the /api/jobs/{id}/events URL AND htmx:afterSwap is used to
    # re-invoke the wiring after the table is populated. Any of those
    # three pieces missing means live progress is broken.
    assert response.status_code == 200
    body = response.text
    # 1. The wiring function must be defined.
    assert "attachJobStreamers" in body, (
        "dashboard has no attachJobStreamers function; SSE live progress is unwired"
    )
    # 2. The SSE endpoint URL must be constructed by JS (browser-side
    # template literal, string concat, or hx-vals -- any of those will
    # carry the /events suffix and the /api/jobs prefix).
    assert "EventSource" in body, "dashboard has no EventSource call; SSE live progress is unwired"
    assert "/api/jobs/" in body, (
        "dashboard has no /api/jobs/ URL in any script; SSE live progress endpoint unreachable"
    )
    assert "/events" in body, (
        "dashboard has no /events URL in any script; SSE live progress endpoint unreachable"
    )
    # 3. The wiring must re-fire after htmx populates the table.
    assert "htmx:afterSwap" in body, (
        "dashboard does not subscribe to htmx:afterSwap; new rows in the "
        "history table will not be wired to live progress"
    )
