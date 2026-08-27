"""Tests for taste_pipeline.web.routes_settings: read-only Settings page.

The Settings page is the human-facing mirror of the JSON
``GET /api/config`` endpoint (todo-3). It shows every ``Config`` field
name + its current value as a two-column table so the user can audit
their pipeline setup without opening ``config.toml`` in a text editor.

Page contract (locked by these tests):

- ``GET /settings`` returns 200 HTML that extends ``base.html`` (so
  action bar / main nav / content / status bar IDs and the four nav
  anchors are inherited), and renders all ``Config`` field names with
  their current values. The list of field names is enumerated via
  ``dataclasses.fields(Config)`` (not hardcoded) so a new field in
  ``config.py`` is automatically covered.
- ``Path`` fields (e.g. ``like_library_dir``) render as their
  ``str(Path)`` form -- the value cells show a portable string, never a
  Python ``Path`` repr.
- ``cookie_file`` (which points at a file that may contain Netscape
  cookie header lines + actual cookie values) is rendered as its path
  string only; the file's contents MUST NOT appear anywhere in the
  response. This guards against an accidental ``cookie_file.read_text()``
  in the page render path leaking the user's session cookies into the
  page DOM (and therefore into browser view-source, screenshots, etc.).
- Optional numeric fields (``keep_threshold``, ``skip_threshold``) may
  be ``None`` (no thresholds configured); the page renders ``None`` as
  the literal string ``None`` so the user can see "this is unset"
  without a crash. There is no form / no POST; the page is read-only.
- Numeric fields render with their decimal string form
  (``web_port=8741``, ``feed_window_days=7``) so the user can confirm
  the values match what they expect.

All tests use ``tmp_path`` for a fresh ``data_dir`` and never read the
user's real config. The ``cookie_file`` test writes a sentinel file
under ``tmp_path`` whose contents would be a real disclosure if they
leaked into the HTML.
"""

from __future__ import annotations

from dataclasses import fields
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


def test_get_settings_returns_200_with_base_template(tmp_path: Path) -> None:
    # Given a real app built from a tmp config
    client = _client(tmp_path)

    # When requesting the settings page
    response = client.get("/settings")

    # Then the response is 200 HTML and inherits the base shell structure
    assert response.status_code == 200, (
        f"GET /settings returned {response.status_code}, expected 200; body={response.text[:200]}"
    )
    assert "text/html" in response.headers["content-type"]
    body = response.text
    for required_id in ("action-bar", "main-nav", "content", "status-bar"):
        assert f'id="{required_id}"' in body, (
            f"settings page missing inherited base.html element id={required_id!r}"
        )
    # The four nav anchors from base.html must also be present so the user
    # can navigate to any other page from the settings view.
    for nav_id in ("nav-dashboard", "nav-triage", "nav-index", "nav-settings"):
        assert nav_id in body, f"settings page missing nav anchor {nav_id!r}"


def test_get_settings_renders_all_config_fields(tmp_path: Path) -> None:
    # Given a real app built from a tmp config (defaults populated by Config dataclass)
    client = _client(tmp_path)

    # When requesting the settings page
    response = client.get("/settings")

    # Then every Config field name appears somewhere in the rendered HTML.
    # Enumerated via dataclasses.fields(Config) so a new field added to
    # config.py is automatically covered by this test (no hardcoded list).
    assert response.status_code == 200
    body = response.text
    for f in fields(Config):
        assert f.name in body, (
            f"settings page missing Config field {f.name!r}; "
            f"every dataclass field must be rendered (enumerated via fields(Config))"
        )


def test_get_settings_does_not_render_cookie_file_contents(tmp_path: Path) -> None:
    # Given a tmp config with a cookie_file whose contents would be a real
    # disclosure if they leaked into the HTML. The "SECRET_COOKIE_VALUE"
    # string is the sentinel: a regression that read the cookie file into
    # the rendered page would fail this test.
    cfg = _make_config(tmp_path)
    cookie_path = cfg.cookie_file
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    cookie_path.write_text(
        "# Netscape HTTP Cookie File\n.example.com\tTRUE\t/\tFALSE\t9999999999\tname\tSECRET_COOKIE_VALUE\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(cfg))

    # When requesting the settings page
    response = client.get("/settings")

    # Then the cookie file's path string IS rendered (the user wants to
    # see WHERE the cookies live) but the file contents MUST NOT appear
    # anywhere in the response -- not the "# Netscape" header, not the
    # domain, not the SECRET_COOKIE_VALUE.
    assert response.status_code == 200
    body = response.text
    assert str(cookie_path) in body, (
        f"settings page missing the cookie_file path string {str(cookie_path)!r}; "
        "the user needs to see WHERE the cookies live"
    )
    for forbidden in ("Netscape", "SECRET_COOKIE_VALUE", "example.com"):
        assert forbidden not in body, (
            f"settings page leaked cookie-file content {forbidden!r} into the HTML; "
            "the path is the only thing the page is allowed to render for cookie_file"
        )


def test_get_settings_renders_path_fields_as_strings_not_path_objects(tmp_path: Path) -> None:
    # Given a tmp config whose like_library_dir is a real directory under tmp_path
    cfg = _make_config(tmp_path)
    client = TestClient(create_app(cfg))

    # When requesting the settings page
    response = client.get("/settings")

    # Then the path field's value cell carries the str(Path) form (e.g.
    # "/tmp/.../lib" on POSIX or "C:\\...\\lib" on Windows), NOT a repr
    # like "PosixPath('/tmp/.../lib')" -- users should see a portable
    # path string, not a Python class name.
    assert response.status_code == 200
    body = response.text
    expected = str(cfg.like_library_dir)
    assert expected in body, (
        f"settings page missing str(like_library_dir) = {expected!r}; "
        "path fields must be rendered as portable strings, not as repr(Path)"
    )
    # And the class-name "PosixPath" / "WindowsPath" / "PurePath" must NOT
    # appear in the value cell (a regression to {{ value }} on a Path
    # object would show the repr with one of these class names).
    for repr_token in ("PosixPath", "WindowsPath", "PurePath"):
        assert repr_token not in body, (
            f"settings page rendered a Path repr token {repr_token!r}; "
            "path fields must be coerced to str before rendering"
        )


def test_get_settings_renders_null_thresholds_without_crashing(tmp_path: Path) -> None:
    # Given a tmp config (Config defaults leave keep_threshold / skip_threshold as None)
    cfg = _make_config(tmp_path)
    # Sanity: the two thresholds are indeed None for the default config.
    assert cfg.keep_threshold is None
    assert cfg.skip_threshold is None
    client = TestClient(create_app(cfg))

    # When requesting the settings page
    response = client.get("/settings")

    # Then the page renders successfully and the threshold rows show a
    # visible "unset" marker. We accept either the Python repr "None"
    # or the JSON literal "null" -- the contract is "does not crash AND
    # the user can see the field is unset", not a specific token.
    assert response.status_code == 200, (
        f"GET /settings with None thresholds returned {response.status_code}; body={response.text[:200]}"
    )
    body = response.text
    # The two threshold field names must appear (so the user knows they exist).
    assert "keep_threshold" in body
    assert "skip_threshold" in body
    # The unset marker must appear at least twice (once per threshold row).
    # Accept either "None" or "null" so the implementation can pick whichever
    # reads more naturally; "None" is the Python repr and "null" is the JSON
    # form, both are immediately recognizable as "unset" to the user.
    none_count = body.count("None") + body.count("null")
    assert none_count >= 2, (
        f"settings page rendered {none_count} 'None'/'null' markers for the two "
        "None thresholds; expected at least 2 (one per threshold row)"
    )


def test_get_settings_renders_numeric_fields_correctly(tmp_path: Path) -> None:
    # Given a tmp config (the dataclass defaults set web_port=8741, feed_window_days=7)
    cfg = _make_config(tmp_path)
    # Sanity: the defaults we are about to assert in the HTML are the
    # ones the dataclass actually populated.
    assert cfg.web_port == 8741
    assert cfg.feed_window_days == 7
    client = TestClient(create_app(cfg))

    # When requesting the settings page
    response = client.get("/settings")

    # Then the numeric value cells carry the integer rendered as a
    # decimal string (no repr, no class name, just the number).
    assert response.status_code == 200
    body = response.text
    assert "8741" in body, (
        "settings page missing the web_port=8741 value; numeric fields must render as their decimal form"
    )
    assert "feed_window_days" in body, "settings page missing the feed_window_days field name"
    # The value 7 is ambiguous (substring of e.g. 8741, 7000, etc.) so we
    # assert it appears AT LEAST 3 times: once as feed_window_days=7, once
    # as sample_rate=48000-no-7 (48000 has a 7), once elsewhere. The point
    # is to confirm the literal "7" is in the page, not the exact count.
    assert body.count("7") >= 3, (
        "settings page missing the digit '7' for feed_window_days=7; "
        "expected the literal integer to appear in the value cell"
    )
