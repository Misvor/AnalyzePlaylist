"""Tests for taste_pipeline.web templates and vendored static assets.

These tests lock the base HTML shell contract: semantic structure (header/nav/
main/footer), 4 nav anchors (Dashboard / Triage / Index / Settings), the
bottom status bar showing the default server host:port, the desktop-app CSS
aesthetic, and the vendored htmx 2.x bundle (no CDN reference).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from taste_pipeline.config import Config, load_config
from taste_pipeline.web import create_app


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


def test_get_root_returns_base_shell_with_nav_and_status_bar(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the index page
    response = client.get("/")

    # Then the response is 200 HTML containing the base shell structure
    assert response.status_code == 200
    body = response.text
    assert "<html" in body
    # App name in the title or h1
    assert "Taste Pipeline" in body
    # Semantic top action bar
    assert 'id="action-bar"' in body
    # Main nav element with all 5 expected nav IDs (Dashboard / Triage / Check / Index / Settings)
    assert 'id="main-nav"' in body
    for nav_id in ("nav-dashboard", "nav-triage", "nav-check", "nav-index", "nav-settings"):
        assert nav_id in body, f"missing nav id: {nav_id}"
    # Main content block
    assert 'id="content"' in body
    # Bottom status bar with the default server host:port from Config (127.0.0.1:8741)
    assert 'id="status-bar"' in body
    assert "127.0.0.1" in body
    assert "8741" in body


def test_static_app_css_returns_200_with_desktop_styles(tmp_path: Path) -> None:
    # Given an app built from a valid config
    client = _client(tmp_path)

    # When requesting the app stylesheet
    response = client.get("/static/app.css")

    # Then the static mount serves it as CSS with desktop-app aesthetic rules
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    body = response.text
    # System font stack (no Google Fonts / external CSS)
    assert "system-ui" in body
    # 1px border rule somewhere in the stylesheet
    assert "1px solid" in body
    # Per-row progress bar class (consumed by dashboard table rows in todo 8)
    assert ".progress-bar" in body


def test_static_vendor_htmx_returns_200_with_no_cdn_reference(tmp_path: Path) -> None:
    # Given an app built from a valid config and the vendored htmx bundle on disk
    client = _client(tmp_path)

    # When requesting the vendored htmx bundle
    response = client.get("/static/vendor/htmx.min.js")

    # Then the bundle is served as JavaScript, is non-trivial in size,
    # contains the htmx signature, and references NO external URLs.
    assert response.status_code == 200
    ct = response.headers["content-type"]
    assert ct.startswith(("application/javascript", "text/javascript")), ct
    body = response.text
    assert len(body) > 1000, f"vendored htmx body too small: {len(body)} bytes"
    # htmx signature: the unminified bundle contains the literal string "htmx"
    assert "htmx" in body
    # No CDN reference inside the vendored bundle itself.
    assert "https://" not in body, "vendored htmx.min.js must not reference any https:// URL"


def test_base_template_contains_no_external_urls(tmp_path: Path) -> None:
    # Given the committed base.html template
    base_html_path = (
        Path(__file__).resolve().parents[1] / "src" / "taste_pipeline" / "web" / "templates" / "base.html"
    )

    # When reading the file from disk
    content = base_html_path.read_text(encoding="utf-8")

    # Then the template references NO external URLs (no CDN, no Google Fonts, no remote scripts).
    assert "https://" not in content, (
        f"base.html must not reference any https:// URL (offending content):\n{content}"
    )
    assert "http://" not in content, (
        f"base.html must not reference any http:// URL (offending content):\n{content}"
    )
    # Sanity: it still references the local static assets.
    assert "/static/app.css" in content
    assert "/static/vendor/htmx.min.js" in content


def test_status_bar_uses_config_web_host_and_port(tmp_path: Path) -> None:
    # Given a tmp config with non-default web_host + web_port
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    body = (
        f'like_library_dir = "{_toml_path(like_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
        f'web_host = "192.0.2.1"\n'
        f"web_port = 9999\n"
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    client = TestClient(create_app(load_config(config_path)))

    # When GETting the index page
    response = client.get("/")

    # Then the status bar renders the CUSTOM host:port from config, NOT the hardcoded default
    assert response.status_code == 200
    body_text = response.text
    assert "Server: 192.0.2.1:9999" in body_text, (
        f"status bar should render custom config web_host:web_port; got body:\n{body_text}"
    )
    # And the hardcoded default is absent (a regression to a literal would show it)
    assert "Server: 127.0.0.1:8741" not in body_text, (
        f"status bar must not contain the hardcoded default; got body:\n{body_text}"
    )
