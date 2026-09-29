"""Tests for the dedicated Index page (templates/index.html) + base nav wiring.

The Index page shows live, structured progress for the like-library index
scan: the song currently being embedded, the upcoming queue, processed /
total / remaining counts, elapsed time, ETA, a progress bar, and a log.
These tests lock the DOM contract (every element id the inline JS looks
up must exist), the SSE wiring (unnamed ``data: {"event": ...}`` frames
dispatched via ``onmessage``), and the server-rendered status panel fed
by ``build_index_status``.
"""

from __future__ import annotations

import json
import re
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


def _seed_manifest(tmp_path: Path, entries: int) -> None:
    """Write a valid library manifest with ``entries`` rows under data_dir/index."""
    index_dir = tmp_path / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest = {f"track_{i}.flac": {"mtime": 1.0, "size": 100, "vector_id": i} for i in range(entries)}
    (index_dir / "library_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_get_index_returns_200_and_extends_base(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When GETting /index
    response = client.get("/index")

    # Then it is 200 HTML inheriting the base shell (nav, content, status bar)
    assert response.status_code == 200, f"GET /index must return 200, got {response.status_code}"
    assert "text/html" in response.headers["content-type"]
    body = response.text
    for required_id in ("action-bar", "main-nav", "content", "status-bar"):
        assert f'id="{required_id}"' in body, f"index page missing base id {required_id!r}"
    for nav_id in ("nav-dashboard", "nav-triage", "nav-check", "nav-index", "nav-settings"):
        assert f'id="{nav_id}"' in body, f"index page missing nav id {nav_id!r}"


def test_nav_index_href_points_at_index_route(tmp_path: Path) -> None:
    # Given any page that extends base.html
    client = _client(tmp_path)

    # When requesting the index page
    body = client.get("/index").text

    # Then the #nav-index anchor's href is /index (not the dashboard root),
    # with the id and href allowed in either attribute order.
    match = re.search(r'<a\b[^>]*\bid="nav-index"[^>]*>', body, flags=re.DOTALL)
    assert match is not None, "base nav is missing the #nav-index anchor"
    anchor = match.group(0)
    assert 'href="/index"' in anchor, f"#nav-index href is not /index: {anchor!r}"


def test_index_page_contains_every_frozen_element_id(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When GETting /index
    body = client.get("/index").text

    # Then every element id the inline JS touches exists in the document
    frozen_ids = (
        "index-build-btn",
        "index-status-badge",
        "index-progress",
        "index-progress-fill",
        "index-current",
        "index-counts",
        "index-elapsed",
        "index-eta",
        "index-queue",
        "index-queue-list",
        "index-log",
        "index-error",
        "index-status-count",
        "index-status-last-scan",
        "index-status-lib",
        "index-status-thresholds",
    )
    for element_id in frozen_ids:
        assert f'id="{element_id}"' in body, f"index page missing frozen element id {element_id!r}"


def test_index_page_wires_sse_and_build_button(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When GETting /index
    body = client.get("/index").text

    # Then the page has a Build Index button, references the jobs API + SSE
    # endpoint, and branches on the JSON `event` field via onmessage (the
    # server frames are unnamed `data: {"event": ...}` payloads).
    assert 'id="index-build-btn"' in body
    assert "Build Index" in body
    assert "EventSource" in body, "index page must use EventSource for SSE progress"
    assert "/api/jobs" in body, "index page must reference the jobs API"
    assert "/events" in body, "index page must reference the SSE events endpoint"
    assert "onmessage" in body, "index page must use es.onmessage (unnamed frames)"
    assert "data.event === 'progress'" in body, "index page must handle the progress event"
    assert "data.event === 'done'" in body, "index page must handle the done event"


def test_index_page_renders_status_panel_values(tmp_path: Path) -> None:
    # Given a config whose library manifest already has two indexed tracks
    config = _make_config(tmp_path)
    _seed_manifest(tmp_path, entries=2)
    client = TestClient(create_app(config))

    # When GETting /index
    body = client.get("/index").text

    # Then the status panel server-renders the build_index_status values
    assert "Indexed: 2 tracks" in body, "status panel must show the manifest count"
    assert str(config.like_library_dir) in body, "status panel must show the like-library path"
    assert "not calibrated" in body, "status panel must show the uncalibrated state"


def test_index_page_has_batch_pause_and_coverage_controls(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When GETting /index
    body = client.get("/index").text

    # Then the batch-size input, pause button, and coverage line all exist
    for element_id in ("index-batch-size", "index-pause-btn", "index-status-coverage"):
        assert f'id="{element_id}"' in body, f"index page missing element id {element_id!r}"

    # And the page POSTs to the pause endpoint and treats paused as terminal
    assert "/pause" in body, "index page must reference the pause endpoint"
    assert "paused" in body, "index page must know about the paused status"
    assert "data.event === 'progress'" in body, "index page must handle the progress event"
    assert "data.event === 'done'" in body, "index page must handle the done event"
    assert "onmessage" in body, "index page must use es.onmessage (unnamed frames)"


def test_index_page_shows_compute_device_in_status_panel(tmp_path: Path) -> None:
    # Given an app built from a valid config (device defaults to "auto")
    client = _client(tmp_path)

    # When GETting /index
    body = client.get("/index").text

    # Then the status panel server-renders the device span AND the
    # refreshStatusPanel script re-reads it from /api/index.
    assert 'id="index-status-device"' in body, "index status panel missing the #index-status-device span"
    assert "Device: auto" in body, "index status panel must server-render the configured device value"
    assert "status.device" in body, (
        "refreshStatusPanel must refresh the device span from the /api/index payload"
    )
