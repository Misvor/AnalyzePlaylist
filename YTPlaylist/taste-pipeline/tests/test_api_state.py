"""Tests for taste_pipeline.web.api: read-only config + state summary endpoints."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from taste_pipeline.config import Config, load_config
from taste_pipeline.state import StateStore
from taste_pipeline.web import create_app

if TYPE_CHECKING:
    from pathlib import Path


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string (forward slashes avoid backslash escapes on Windows)."""
    return path.as_posix()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal valid config TOML under tmp_path and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _make_app(tmp_path: Path) -> Config:
    """Build a real app config from a tmp TOML; return the Config for further use."""
    return load_config(_write_config(tmp_path))


def test_get_api_config_returns_all_config_fields(tmp_path: Path) -> None:
    # Given an app built from a valid tmp config
    app = create_app(_make_app(tmp_path))
    client = TestClient(app)

    # When requesting the config endpoint
    response = client.get("/api/config")

    # Then the response is 200 JSON whose keys are exactly the Config dataclass field names
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert set(payload.keys()) == {f.name for f in fields(Config)}


def test_get_api_config_serializes_path_fields_as_strings(tmp_path: Path) -> None:
    # Given an app built from a valid tmp config (the cookie_file has real text in it)
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    app = create_app(_make_app(tmp_path))
    client = TestClient(app)

    # When requesting the config endpoint
    response = client.get("/api/config")
    payload = response.json()

    # Then every path-typed field is serialized as a string (not a Path / not file contents)
    for path_field in ("like_library_dir", "data_dir", "cookie_file", "download_archive"):
        assert isinstance(payload[path_field], str), (
            f"{path_field} is not a string: {type(payload[path_field])}"
        )
    # And the cookie_file value is the configured PATH, not the file's contents
    assert payload["cookie_file"] == str(cookie_file)
    assert "Netscape" not in payload["cookie_file"]


def test_get_api_state_summary_returns_empty_counts_for_empty_state(tmp_path: Path) -> None:
    # Given an app built from a valid tmp config (data_dir created but never seeded)
    app = create_app(_make_app(tmp_path))
    client = TestClient(app)

    # When requesting the state summary endpoint
    response = client.get("/api/state/summary")

    # Then the response is 200 JSON with zero counts grouped by table
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"seen": {}, "downloads": {}}


def test_get_api_state_summary_returns_correct_counts_after_seeding(tmp_path: Path) -> None:
    # Given an app built from a valid tmp config whose data_dir is seeded via the real StateStore
    cfg = _make_app(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)
    store = StateStore(cfg.data_dir)
    try:
        assert store.record_seen("vid1", stage="listed") is True
        assert store.record_seen("vid2", stage="listed") is True
        assert store.record_seen("vid2", stage="listed") is False  # duplicate is a no-op
        audio = tmp_path / "data" / "vid3.flac"
        assert store.record_download("vid3", audio_path=audio) is True
    finally:
        store.close()

    # When requesting the state summary endpoint
    response = client.get("/api/state/summary")

    # Then the counts reflect the seeding (2 listed seen rows, 1 downloaded row)
    assert response.status_code == 200
    assert response.json() == {"seen": {"listed": 2}, "downloads": {"downloaded": 1}}


def test_get_api_state_summary_uses_data_dir_from_app_config(tmp_path: Path) -> None:
    # Given TWO distinct configs with their own data_dirs; the app is built from config A
    cfg_a = _make_app(tmp_path)
    cfg_b = _make_app(tmp_path / "other")
    app = create_app(cfg_a)
    client = TestClient(app)
    # Seed config B's data_dir with a row that should NEVER appear in the app's view
    store_b = StateStore(cfg_b.data_dir)
    try:
        store_b.record_seen("should_not_appear", stage="listed")
    finally:
        store_b.close()
    # Seed config A's data_dir with the row that SHOULD appear
    store_a = StateStore(cfg_a.data_dir)
    try:
        store_a.record_seen("from_a", stage="listed")
    finally:
        store_a.close()

    # When requesting the state summary endpoint on the app built from config A
    response = client.get("/api/state/summary")

    # Then only the row from config A's data_dir is visible (proves the endpoint
    # reads from app.state.config.data_dir, not from a hard-coded path or B's data_dir)
    assert response.status_code == 200
    payload = response.json()
    assert payload["seen"] == {"listed": 1}
    assert "from_a" not in str(payload) or payload["seen"].get("listed") == 1
    # And the B-only row is provably absent
    assert all(stage == "listed" and count == 1 for stage, count in payload["seen"].items())


def test_get_api_state_summary_aggregates_multiple_stages(tmp_path: Path) -> None:
    # Given an app whose data_dir has been seeded with rows across multiple stages
    cfg = _make_app(tmp_path)
    app = create_app(cfg)
    client = TestClient(app)
    store = StateStore(cfg.data_dir)
    try:
        store.record_seen("a", stage="listed")
        store.record_seen("b", stage="metadata")
        store.record_seen("c", stage="metadata")
        store.record_download("a", audio_path=tmp_path / "data" / "a.flac", stage="downloaded")
        store.record_download("b", audio_path=tmp_path / "data" / "b.flac", stage="transcoded")
    finally:
        store.close()

    # When requesting the state summary endpoint
    payload = client.get("/api/state/summary").json()

    # Then counts are grouped per stage for each table independently
    assert payload["seen"] == {"listed": 1, "metadata": 2}
    assert payload["downloads"] == {"downloaded": 1, "transcoded": 1}
