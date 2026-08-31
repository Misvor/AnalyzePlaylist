"""Tests for the Check Track page (templates/check.html) and the base nav lockup.

The check page is a Jinja2 template that extends base.html and includes
a drag-drop zone, a URL input, and a result panel. These tests lock
the DOM contract: every element id the inline JS looks up MUST be
present in the rendered HTML, and the SSE EventSource pattern must be
wired (the URL flow runs as a job).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from taste_pipeline.config import Config, load_config
from taste_pipeline.web import create_app

if TYPE_CHECKING:
    from pathlib import Path


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string."""
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


def test_get_check_returns_200_and_renders_drop_zone(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When GETting /check
    response = client.get("/check")

    # Then 200 + the drop-zone element is present
    assert response.status_code == 200, (
        f"GET /check must return 200, got {response.status_code}: {response.text}"
    )
    body = response.text
    assert 'id="drop-zone"' in body, "check page must include the drop-zone element"


def test_get_check_includes_url_input_field(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/check")

    body = response.text
    assert 'id="url-input"' in body, "check page must include the URL input field"
    assert 'id="check-url-btn"' in body, "check page must include the Check URL button"


def test_get_check_includes_required_event_source_handler_for_sse(tmp_path: Path) -> None:
    """The URL flow uses SSE; the inline JS must wire EventSource('/api/jobs/...').

    The check page opens an EventSource for the job id returned by
    POST /api/check/url and listens for ``progress`` + ``done`` events.
    The test asserts the pattern is in the rendered page; without it
    the URL flow would silently never advance past ``submitting``.
    """
    client = _client(tmp_path)

    response = client.get("/check")

    body = response.text
    assert "EventSource" in body, "check page must use EventSource for SSE progress"
    assert "/api/jobs/" in body, "check page must reference the /api/jobs/{id}/events endpoint"
    assert "/api/check/url" in body, "check page must POST to /api/check/url"
    assert "/api/check/file" in body, "check page must POST to /api/check/file"


def test_get_check_uses_base_template(tmp_path: Path) -> None:
    """The check page extends base.html; the nav ids from base must appear in the rendered output.

    Locks the contract that adding ``{% extends "base.html" %}`` to
    check.html surfaces the same nav ids the other pages do (Dashboard,
    Triage, Check, Index, Settings). The new Check nav id is also
    asserted here so the nav check + the page check live in the same
    test surface.
    """
    client = _client(tmp_path)

    response = client.get("/check")

    body = response.text
    # Base nav ids
    for nav_id in ("nav-dashboard", "nav-triage", "nav-check", "nav-index", "nav-settings"):
        assert f'id="{nav_id}"' in body, f"missing nav id {nav_id!r} in check page"
    # Status bar (rendered from base.html's env.globals)
    assert 'id="status-bar"' in body
