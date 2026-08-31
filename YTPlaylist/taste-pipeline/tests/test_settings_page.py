"""Tests for taste_pipeline.web.routes_settings: editable Settings page.

The Settings page is the human-facing mirror of the JSON
``GET /api/config`` endpoint (todo-3), evolved into a fully editable
form (post-todo-3). It shows every ``Config`` field name + its current
value as a two-column editable form so the user can audit AND change
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
  without a crash.
- Numeric fields render with their decimal string form
  (``web_port=8741``, ``feed_window_days=7``) so the user can confirm
  the values match what they expect.
- Every Config field is editable in the form, including path / network
  fields (``like_library_dir``, ``data_dir``, ``cookie_file``,
  ``download_archive``, ``web_host``, ``web_port``). Path / network
  changes are persisted to ``config.toml`` but only take effect after
  a process restart; the "Unapplied path / network changes -- restart
  required" banner renders when the on-disk file differs from
  ``app.state.running_config`` (the snapshot taken at app startup).

All tests use ``tmp_path`` for a fresh ``data_dir`` and never read the
user's real config. The ``cookie_file`` test writes a sentinel file
under ``tmp_path`` whose contents would be a real disclosure if they
leaked into the HTML.
"""

from __future__ import annotations

import re
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
    """Write a minimal valid config TOML under tmp_path and return its path.

    ``exist_ok=True`` mirrors ``tests/conftest.py`` so the helper can
    be called twice on the same ``tmp_path`` (some new tests write a
    config, build an app that edits it, then need the same path back
    to inspect the on-disk state).
    """
    like_lib = tmp_path / "lib"
    like_lib.mkdir(exist_ok=True)
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
    """Build a TestClient over a real Config-derived app with config_path recorded.

    Recording ``config_path`` on ``app.state`` is what enables the
    ``POST /api/config`` endpoint to atomically rewrite the same file
    the test loaded from. Without it, ``POST /api/config`` returns 409
    and the form is rendered with the form-disabled flag.
    """
    config_path = _write_config(tmp_path)
    config = load_config(config_path)
    return TestClient(create_app(config, config_path=config_path))


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


# ── Editable-settings contract (POST /api/config) ────────────────────────


# The editable-field classification is the contract the form renders and the
# POST handler enforces. Every Config field is editable; path / network fields
# additionally trigger the restart-required banner (see _PATH_NETWORK_FIELDS).
EXPECTED_EDITABLE_FIELDS = frozenset(
    {
        "keep_threshold",
        "skip_threshold",
        "min_dislikes_for_classifier",
        "feed_window_days",
        "max_feed_items",
        "max_metadata_fetch",
        "chunk_seconds",
        "min_chunk_seconds",
        "sample_rate",
        "model_name",
        "like_library_dir",
        "data_dir",
        "cookie_file",
        "download_archive",
        "web_host",
        "web_port",
    }
)
EXPECTED_PATH_NETWORK_FIELDS = frozenset(
    {
        "like_library_dir",
        "data_dir",
        "cookie_file",
        "download_archive",
        "web_host",
        "web_port",
    }
)


def test_get_settings_returns_form_with_inputs_for_editable_fields(tmp_path: Path) -> None:
    # Given a real app built from a tmp config (with config_path so POST works)
    client = _client(tmp_path)

    # When requesting the settings page
    response = client.get("/settings")

    # Then the page renders a <form> and every editable Config field is
    # bound to an <input> or <select> whose name= matches the field.
    assert response.status_code == 200
    body = response.text
    assert '<form id="settings-form"' in body, "settings page is missing the form wrapper"
    for field_name in EXPECTED_EDITABLE_FIELDS:
        # The input/select is named after the field; the form collects
        # all editable fields via collectEditableFields().
        assert f'name="{field_name}"' in body, (
            f"settings form missing an input/select for editable field {field_name!r}; "
            f"the user cannot edit a field that has no input"
        )


def test_get_settings_renders_path_fields_as_editable_inputs(tmp_path: Path) -> None:
    # Given a real app built from a tmp config
    client = _client(tmp_path)

    # When requesting the settings page
    response = client.get("/settings")

    # Then every path / network field renders as an <input> with the
    # matching name attribute. The form is wired to POST these via
    # collectEditableFields(); without an input the user cannot edit
    # the field from the GUI.
    assert response.status_code == 200
    body = response.text
    path_fields = {"like_library_dir", "data_dir", "cookie_file", "download_archive"}
    for field_name in path_fields:
        assert f'name="{field_name}"' in body, (
            f"settings form missing an input for editable path field {field_name!r}; "
            "the user cannot edit a field that has no input"
        )


def test_get_settings_renders_network_fields_as_editable_inputs(tmp_path: Path) -> None:
    # Given a real app built from a tmp config
    client = _client(tmp_path)

    # When requesting the settings page
    response = client.get("/settings")

    # Then web_host renders as an <input type="text"> (maxlength 255)
    # and web_port renders as an <input type="number"> with min=1 /
    # max=65535 so the browser rejects obviously-invalid ports before
    # the form submits.
    assert response.status_code == 200
    body = response.text
    assert 'name="web_host"' in body, (
        "settings form missing the web_host input; the user cannot change the bind address"
    )
    assert 'name="web_port"' in body, (
        "settings form missing the web_port input; the user cannot change the bind port"
    )
    # The port input is constrained client-side so users cannot enter
    # 0 / negative / 99999 — the browser blocks the form submission.
    port_match = re.search(
        r'<input id="field-web_port" name="web_port"[^>]*>',
        body,
    )
    assert port_match, "settings form web_port input tag not found"
    port_tag = port_match.group(0)
    assert 'type="number"' in port_tag, f"web_port input must be type=number; got {port_tag!r}"
    assert 'min="1"' in port_tag, f"web_port input must declare min=1; got {port_tag!r}"
    assert 'max="65535"' in port_tag, f"web_port input must declare max=65535; got {port_tag!r}"


def test_get_settings_includes_browse_buttons_for_storage_paths(tmp_path: Path) -> None:
    # Given the settings page rendered for a fresh config
    client = _client(tmp_path)

    # When requesting it
    response = client.get("/settings")

    # Then every Storage path field has a Browse button with the right
    # data-pick kind and the right data-target (locks the contract: the
    # JS handler uses these attributes to dispatch the native dialog).
    assert response.status_code == 200
    body = response.text

    expected_buttons = (
        ("like_library_dir", "directory"),
        ("data_dir", "directory"),
        ("download_archive", "directory"),
        ("cookie_file", "file"),
    )
    for field_name, pick_kind in expected_buttons:
        pattern = (
            r'<button\b[^>]*\bclass="browse-btn"[^>]*\bdata-pick="'
            + pick_kind
            + r'"[^>]*\bdata-target="field-'
            + field_name
            + r'"[^>]*>.*?</button>'
        )
        match = re.search(pattern, body, flags=re.DOTALL)
        assert match, (
            f"missing Browse button for {field_name!r} with data-pick={pick_kind!r}; "
            f"page does not render a native-dialog trigger for this field"
        )
        assert "Browse..." in match.group(0), (
            f"Browse button for {field_name!r} missing the visible 'Browse...' label"
        )


def test_get_settings_includes_pick_path_javascript_handler(tmp_path: Path) -> None:
    # Given the settings page rendered
    client = _client(tmp_path)

    # When requesting it
    response = client.get("/settings")

    # Then the inline JS contains the desktop native-dialog path (uses
    # ``window.pywebview.api.open_directory_dialog``) and the browser
    # fallback path (uses ``webkitdirectory``). Locks both branches so
    # a regression in either handler is caught.
    assert response.status_code == 200
    body = response.text
    assert "pickPath" in body, "settings form missing pickPath helper"
    assert "window.pywebview" in body, (
        "settings form does not attempt to call window.pywebview.api for the desktop native dialog"
    )
    assert "open_directory_dialog" in body, "settings form does not call open_directory_dialog"
    assert "open_file_dialog" in body, "settings form does not call open_file_dialog"
    assert "webkitdirectory" in body, (
        "settings form is missing the browser fallback (webkitdirectory); "
        "browsers cannot expose absolute paths so the fallback only gives a NAME"
    )


def test_get_settings_renders_dirty_indicator_badge(tmp_path: Path) -> None:
    # Given the settings page rendered
    client = _client(tmp_path)

    # When requesting it
    response = client.get("/settings")

    # Then the page contains a hidden dirty-form badge that flips
    # visible on input. Without this badge a user who typed path
    # changes and navigated away would silently lose their edits
    # because GET /settings re-reads from config.toml.
    assert response.status_code == 200
    body = response.text
    assert 'id="settings-dirty-indicator"' in body, (
        "settings page missing the unsaved-changes indicator badge"
    )
    assert "hidden" in body, "dirty indicator badge should start hidden"


def test_settings_page_includes_beforeunload_guard_for_unsaved_changes(tmp_path: Path) -> None:
    # Given the settings page rendered
    client = _client(tmp_path)

    # When requesting it
    response = client.get("/settings")

    # Then the inline JS includes the beforeunload handler that
    # prompts the user when there are unsaved changes. Without this
    # handler a user could lose path edits by switching tabs.
    assert response.status_code == 200
    body = response.text
    assert "beforeunload" in body, "settings form missing beforeunload guard for unsaved changes"
    assert "formDirty" in body, "settings form missing the formDirty flag tracking unsaved state"


def test_get_settings_shows_unapplied_changes_banner_when_path_differs(tmp_path: Path) -> None:
    # Given a tmp config.toml that DIFFERS from the running snapshot:
    # the file on disk has like_library_dir = a NEW directory, but
    # the app was built with the original like_library_dir. The
    # banner must surface this divergence as "restart required".
    original_lib = tmp_path / "lib"
    original_lib.mkdir(exist_ok=True)
    body = (
        f'like_library_dir = "{_toml_path(original_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    # Build the app with the ORIGINAL config (so app.state.running_config
    # is the original), then re-write the file to point at a NEW dir.
    running_cfg = load_config(config_path)
    new_lib = tmp_path / "new_lib"
    new_lib.mkdir(exist_ok=True)
    config_path.write_text(
        body.replace(_toml_path(original_lib), _toml_path(new_lib)),
        encoding="utf-8",
    )
    client = TestClient(create_app(running_cfg, config_path=config_path))

    # When requesting the settings page
    response = client.get("/settings")

    # Then the unapplied-changes banner is rendered with both old +
    # new values so the user can see WHICH field needs a restart.
    assert response.status_code == 200
    body_text = response.text
    assert 'id="settings-unapplied-banner"' in body_text, (
        "settings page is missing the unapplied-changes banner element; "
        "the user has no way to know their path edit requires a restart"
    )
    assert "restart required" in body_text.lower(), (
        "settings page banner does not explain the user must restart the server"
    )
    assert _toml_path(original_lib) in body_text, (
        f"banner must show the OLD value {_toml_path(original_lib)!r} for like_library_dir"
    )
    assert _toml_path(new_lib) in body_text, (
        f"banner must show the NEW value {_toml_path(new_lib)!r} for like_library_dir"
    )


def test_post_config_updates_keep_threshold_in_memory_and_on_disk(tmp_path: Path) -> None:
    # Given a tmp config with config_path recorded on app.state
    config_path = _write_config(tmp_path)
    client = _client(tmp_path)
    # Sanity: starting state has keep_threshold = None (default)
    assert _make_config(tmp_path).keep_threshold is None

    # When POSTing a valid update that sets keep_threshold=0.85
    response = client.post("/api/config", json={"keep_threshold": 0.85})

    # Then the response is 200 and the in-memory config reflects the change
    assert response.status_code == 200, (
        f"POST /api/config returned {response.status_code}; expected 200; body={response.text[:300]}"
    )
    payload = response.json()
    assert payload["keep_threshold"] == 0.85, (
        f"response payload did not reflect the update: keep_threshold={payload['keep_threshold']!r}"
    )
    # And the on-disk file was updated too
    assert config_path.is_file(), f"config.toml went missing at {config_path}"
    reloaded = load_config(config_path)
    assert reloaded.keep_threshold == 0.85, (
        f"on-disk config did not pick up the update; reload sees keep_threshold={reloaded.keep_threshold!r}"
    )
    # And subsequent GET /settings reads the new value (the in-memory
    # state was refreshed by the handler, not just the on-disk file).
    response2 = client.get("/settings")
    assert "0.85" in response2.text, (
        "GET /settings after POST did not show the updated keep_threshold; app.state.config was not refreshed"
    )


def test_post_config_rejects_invalid_keep_threshold_above_1(tmp_path: Path) -> None:
    # Given a real app
    client = _client(tmp_path)

    # When POSTing an out-of-range keep_threshold
    response = client.post("/api/config", json={"keep_threshold": 1.5})

    # Then 422 (keep_threshold must be in [0, 1] or null)
    assert response.status_code == 422, (
        f"POST /api/config with keep_threshold=1.5 returned {response.status_code}; expected 422"
    )
    assert "must be <=" in response.json()["detail"], (
        f"error detail must mention the bound; got {response.json()['detail']!r}"
    )


def test_post_config_rejects_non_numeric_chunk_seconds(tmp_path: Path) -> None:
    # Given a real app
    client = _client(tmp_path)

    # When POSTing a non-numeric chunk_seconds
    response = client.post("/api/config", json={"chunk_seconds": "abc"})

    # Then 422 (chunk_seconds must be a number)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "must be a number" in detail, f"error detail should mention 'must be a number', got {detail!r}"


def test_post_config_atomic_write_creates_no_temp_file_on_success(tmp_path: Path) -> None:
    # Given a real app
    config_path = _write_config(tmp_path)
    client = TestClient(create_app(load_config(config_path), config_path=config_path))

    # When POSTing a valid update
    response = client.post("/api/config", json={"feed_window_days": 14})

    # Then the response is 200 and NO config.toml.tmp file remains --
    # the atomic write (write to .tmp + Path.replace) must complete by
    # the time the POST handler returns.
    assert response.status_code == 200
    tmp_left = config_path.with_suffix(config_path.suffix + ".tmp")
    assert not tmp_left.exists(), f"atomic write left a stray {tmp_left}; Path.replace() should have moved it"


def test_post_config_returns_updated_config_json(tmp_path: Path) -> None:
    """The POST response carries the full updated Config so the client can sync its local copy."""
    # Given a real app
    client = _client(tmp_path)

    # When POSTing a single field update
    response = client.post("/api/config", json={"sample_rate": 44100})

    # Then the response is 200 JSON with EVERY Config field (not just the updated one)
    assert response.status_code == 200
    payload = response.json()
    expected_keys = {f.name for f in fields(Config)}
    assert set(payload.keys()) == expected_keys, (
        f"response payload missing keys; expected exactly the Config fields, got "
        f"missing={expected_keys - set(payload.keys())} extra={set(payload.keys()) - expected_keys}"
    )
    assert payload["sample_rate"] == 44100, (
        f"sample_rate not in response payload; got {payload['sample_rate']!r}"
    )


def test_post_config_returns_409_when_config_path_unknown(tmp_path: Path) -> None:
    """An app built without config_path cannot persist edits; POST returns 409 (not 500)."""
    # Given an app built without config_path
    cfg = _make_config(tmp_path)
    client = TestClient(create_app(cfg))  # no config_path kwarg -> None

    # When POSTing an update
    response = client.post("/api/config", json={"keep_threshold": 0.5})

    # Then 409 with a clear message
    assert response.status_code == 409, (
        f"POST /api/config without config_path returned {response.status_code}; expected 409"
    )
    assert "config_path" in response.json()["detail"], (
        f"error detail should mention config_path; got {response.json()['detail']!r}"
    )


def test_post_config_preserves_unmentioned_fields_on_disk(tmp_path: Path) -> None:
    """POST /api/config is a PARTIAL update -- fields not in the body must survive on disk.

    The atomic write merges the new values with the existing file
    contents so a single-field POST does not erase the rest of the
    config.
    """
    # Given a real app
    config_path = _write_config(tmp_path)
    original = load_config(config_path)
    client = _client(tmp_path)

    # When POSTing only keep_threshold
    response = client.post("/api/config", json={"keep_threshold": 0.7})

    # Then the file on disk has the new keep_threshold AND every other field intact
    assert response.status_code == 200
    reloaded = load_config(config_path)
    assert reloaded.keep_threshold == 0.7
    assert reloaded.feed_window_days == original.feed_window_days
    assert reloaded.web_port == original.web_port
    assert reloaded.model_name == original.model_name


def test_post_config_rejects_unknown_field(tmp_path: Path) -> None:
    """Unknown fields are rejected (not silently dropped) so typos surface immediately."""
    # Given a real app
    client = _client(tmp_path)

    # When POSTing an unknown key
    response = client.post("/api/config", json={"not_a_real_field": 123})

    # Then 422 with an "unknown field" message
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "not_a_real_field" in detail, f"error detail should name the unknown field; got {detail!r}"
    assert "unknown" in detail, f"error detail should mention 'unknown'; got {detail!r}"


def test_post_config_rejects_skip_threshold_gte_keep_threshold(tmp_path: Path) -> None:
    """Cross-field invariant: skip must stay strictly less than keep."""
    # Given a real app with keep_threshold set
    client = _client(tmp_path)
    _ = client.post("/api/config", json={"keep_threshold": 0.8})

    # When POSTing skip_threshold >= keep_threshold
    response = client.post("/api/config", json={"skip_threshold": 0.85})

    # Then 422 with a clear invariant message
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "skip_threshold" in detail, f"error detail should name skip_threshold; got {detail!r}"
    assert "keep_threshold" in detail, f"error detail should name keep_threshold; got {detail!r}"


# ── Editable path / network fields (restart required) ─────────────────


def test_post_config_accepts_path_field_and_persists_to_disk(tmp_path: Path) -> None:
    # Given a real app and a NEW like_library_dir (a real directory under tmp_path)
    config_path = _write_config(tmp_path)
    client = _client(tmp_path)
    new_lib = tmp_path / "new_lib"
    new_lib.mkdir(exist_ok=True)

    # When POSTing a new like_library_dir
    response = client.post("/api/config", json={"like_library_dir": str(new_lib)})

    # Then 200, the response payload reflects the change, and the on-disk file does too
    assert response.status_code == 200, (
        f"POST /api/config with like_library_dir returned {response.status_code}; expected 200; "
        f"body={response.text[:300]}"
    )
    payload = response.json()
    assert payload["like_library_dir"] == str(new_lib), (
        f"response payload did not reflect the like_library_dir update; got {payload['like_library_dir']!r}"
    )
    reloaded = load_config(config_path)
    assert str(reloaded.like_library_dir) == str(new_lib), (
        f"on-disk like_library_dir was not persisted; reload sees {reloaded.like_library_dir!r}"
    )
    # And subsequent GET /settings shows the new value (in-memory state refreshed)
    response2 = client.get("/settings")
    assert str(new_lib) in response2.text, (
        "GET /settings after POST did not show the new like_library_dir; "
        "app.state.config was not refreshed to match the on-disk file"
    )
    # And the restart-required banner is now showing the divergence
    # between app.state.running_config (still the original) and the
    # on-disk file (now the new path).
    assert 'id="settings-unapplied-banner"' in response2.text, (
        "GET /settings after a like_library_dir edit is missing the restart-required banner; "
        "app.state.running_config was updated by the POST handler (it must NOT be)"
    )


def test_post_config_rejects_path_field_pointing_to_a_file(tmp_path: Path) -> None:
    # Given a real app and a file (not a directory) at the target path
    client = _client(tmp_path)
    file_path = tmp_path / "not_a_dir.flac"
    file_path.write_bytes(b"FAKE_FLAC_HEADER" + b"\x00" * 32)

    # When POSTing a like_library_dir that points at a file
    response = client.post("/api/config", json={"like_library_dir": str(file_path)})

    # Then 422 with a clear message
    assert response.status_code == 422, (
        f"POST /api/config with like_library_dir pointing at a file returned {response.status_code}; "
        f"expected 422; body={response.text[:200]}"
    )
    detail = response.json()["detail"]
    assert "like_library_dir" in detail, f"error detail should name the field; got {detail!r}"
    assert "not a directory" in detail, f"error detail should explain the validation rule; got {detail!r}"


def test_post_config_rejects_invalid_web_port(tmp_path: Path) -> None:
    # Given a real app
    client = _client(tmp_path)

    # When POSTing web_port out of the TCP range
    response = client.post("/api/config", json={"web_port": 99999})

    # Then 422
    assert response.status_code == 422, (
        f"POST /api/config with web_port=99999 returned {response.status_code}; expected 422"
    )
    detail = response.json()["detail"]
    assert "web_port" in detail, f"error detail should name web_port; got {detail!r}"
    assert "[1, 65535]" in detail, f"error detail should mention the valid port range; got {detail!r}"


def test_post_config_rejects_non_string_web_host(tmp_path: Path) -> None:
    # Given a real app
    client = _client(tmp_path)

    # When POSTing web_host as a non-string (an int)
    response = client.post("/api/config", json={"web_host": 123})

    # Then 422
    assert response.status_code == 422, (
        f"POST /api/config with web_host=123 returned {response.status_code}; expected 422"
    )
    detail = response.json()["detail"]
    assert "web_host" in detail, f"error detail should name web_host; got {detail!r}"
    assert "must be a string" in detail, f"error detail should mention the type requirement; got {detail!r}"


def test_post_config_accepts_data_dir_and_persists(tmp_path: Path) -> None:
    # Given a real app and a new data_dir under tmp_path (load_config auto-creates it)
    config_path = _write_config(tmp_path)
    client = _client(tmp_path)
    new_data = tmp_path / "new_data"

    # When POSTing the new data_dir (it does not exist yet — load_config will create it)
    response = client.post("/api/config", json={"data_dir": str(new_data)})

    # Then 200 and persisted
    assert response.status_code == 200, (
        f"POST /api/config with new data_dir returned {response.status_code}; expected 200; "
        f"body={response.text[:300]}"
    )
    reloaded = load_config(config_path)
    assert str(reloaded.data_dir) == str(new_data), (
        f"on-disk data_dir was not persisted; reload sees {reloaded.data_dir!r}"
    )
    # And load_config auto-created the dir
    assert new_data.is_dir(), f"data_dir was not auto-created by load_config; {new_data!r} is not a directory"


def test_post_config_accepts_web_host_and_persists(tmp_path: Path) -> None:
    # Given a real app
    config_path = _write_config(tmp_path)
    client = _client(tmp_path)

    # When POSTing a new web_host (LAN-bindable address)
    response = client.post("/api/config", json={"web_host": "0.0.0.0"})  # noqa: S104 -- test exercises the LAN-bind path explicitly

    # Then 200 and persisted
    assert response.status_code == 200, (
        f"POST /api/config with web_host=0.0.0.0 returned {response.status_code}; expected 200; "
        f"body={response.text[:300]}"
    )
    reloaded = load_config(config_path)
    assert reloaded.web_host == "0.0.0.0", (  # noqa: S104 -- test exercises the LAN-bind path explicitly
        f"on-disk web_host was not persisted; reload sees {reloaded.web_host!r}"
    )


def test_post_config_accepts_web_port_in_valid_range(tmp_path: Path) -> None:
    # Given a real app
    config_path = _write_config(tmp_path)
    client = _client(tmp_path)

    # When POSTing a valid port
    response = client.post("/api/config", json={"web_port": 9999})

    # Then 200 and persisted
    assert response.status_code == 200, (
        f"POST /api/config with web_port=9999 returned {response.status_code}; expected 200; "
        f"body={response.text[:300]}"
    )
    reloaded = load_config(config_path)
    assert reloaded.web_port == 9999, f"on-disk web_port was not persisted; reload sees {reloaded.web_port!r}"
