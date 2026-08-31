"""Tests for taste_pipeline.web.routes_index: read-only library index status endpoint.

The endpoint exposes the on-disk state of the like-library embedding
index (no model loading, no scan trigger) so the dashboard can show the
user how fresh their index is without spinning up the CLAP embedder.

Endpoint contract (locked by these tests):

- ``GET /api/index`` returns JSON ``{count, last_scan, like_library_path,
  like_library_track_count}``:
    - ``count``: number of tracks in the index (``len(manifest)`` when
      ``<data_dir>/index/library_manifest.json`` is present, else ``0``).
    - ``last_scan``: ISO-8601 UTC string of the manifest file's mtime,
      or ``None`` when the manifest does not exist.
    - ``like_library_path``: ``str(config.like_library_dir)``.
    - ``like_library_track_count``: recursive count of audio files
      (``.flac``, ``.mp3``, ``.opus``, ``.m4a``, ``.ogg``) under
      ``config.like_library_dir``.
- The handler MUST NOT import or load the CLAP model; the module-level
  import set of ``taste_pipeline.web.routes_index`` is checked by a
  guardrail test.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

import taste_pipeline.web.routes_index as routes_index_module
from taste_pipeline.config import load_config
from taste_pipeline.web import create_app
from taste_pipeline.web.routes_index import build_index_status

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


def _make_client(tmp_path: Path) -> TestClient:
    """Build a real FastAPI app from a tmp TOML; return a TestClient ready to call."""
    cfg = load_config(_write_config(tmp_path))
    return TestClient(create_app(cfg))


def _seed_manifest(data_dir: Path, n: int) -> Path:
    """Write a manifest with N synthetic entries under ``<data_dir>/index/``; return the manifest path."""
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, float | int]] = {}
    for i in range(n):
        manifest[f"track_{i:03d}.flac"] = {
            "mtime": 1700000000.0 + i,
            "size": 1024 * (i + 1),
            "vector_id": i,
        }
    manifest_path = index_dir / "library_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def test_get_api_index_returns_zero_count_when_no_index_files(tmp_path: Path) -> None:
    """GET /api/index returns 200 with count=0 and last_scan=None when no index files exist."""
    # Given a freshly-built app whose data_dir has no index files
    client = _make_client(tmp_path)

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then it is 200 JSON with count=0 and last_scan=None
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["count"] == 0
    assert payload["last_scan"] is None


def test_get_api_index_returns_count_from_manifest_when_present(tmp_path: Path) -> None:
    """Given a manifest with N entries, GET /api/index returns count == N (manifest length)."""
    # Given a config whose data_dir/index/library_manifest.json holds N=5 entries
    cfg = load_config(_write_config(tmp_path))
    _ = _seed_manifest(cfg.data_dir, n=5)
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the count matches the manifest length
    assert response.status_code == 200
    assert response.json()["count"] == 5


def test_get_api_index_returns_last_scan_from_manifest_mtime(tmp_path: Path) -> None:
    """Given a manifest, last_scan is the mtime rendered as ISO-8601 UTC (within 1s tolerance)."""
    # Given a config whose manifest was written a moment ago
    cfg = load_config(_write_config(tmp_path))
    manifest_path = _seed_manifest(cfg.data_dir, n=3)
    # The mtime is whatever the OS recorded at write; capture it just before reading.
    expected_mtime = manifest_path.stat().st_mtime
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then last_scan is an ISO-8601 UTC string within 1s of the manifest's mtime
    assert response.status_code == 200
    last_scan_str = response.json()["last_scan"]
    assert isinstance(last_scan_str, str), f"last_scan must be a string, got {type(last_scan_str)}"
    parsed = datetime.fromisoformat(last_scan_str)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    delta = abs(parsed.timestamp() - expected_mtime)
    assert delta < 1.0, f"last_scan {last_scan_str} differs from mtime {expected_mtime} by {delta:.3f}s"


def test_get_api_index_does_not_load_clap_model(tmp_path: Path) -> None:
    """Guardrail: routes_index must not import transformers or torch (no CLAP loading on the read path)."""
    # The module-level imports of routes_index are the only thing the handler can reach
    module_vars = vars(routes_index_module)

    # When we inspect the module's imported names, neither transformers nor torch is present
    forbidden = {"transformers", "torch"}
    for name in forbidden:
        assert name not in module_vars, (
            f"routes_index imported {name!r} -- this loads the CLAP model on the read path; "
            "the index status endpoint must stay read-only"
        )


def test_get_api_index_like_library_path_matches_config(tmp_path: Path) -> None:
    """Given a config whose like_library_dir is tmp_path/lib, the response's path equals its string form."""
    # Given a fresh config (like_library_dir = tmp_path/lib)
    cfg = load_config(_write_config(tmp_path))
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then like_library_path equals str(config.like_library_dir)
    assert response.status_code == 200
    assert response.json()["like_library_path"] == str(cfg.like_library_dir)


def test_get_api_index_like_library_track_count_counts_audio_files(tmp_path: Path) -> None:
    """Given N audio files of various supported extensions under like_library_dir, the count matches N."""
    # Given a like-library containing 5 audio files (mixed extensions) + 1 non-audio file
    cfg = load_config(_write_config(tmp_path))
    lib = cfg.like_library_dir
    extensions = [".flac", ".mp3", ".opus", ".m4a", ".ogg"]
    for i, ext in enumerate(extensions):
        (lib / f"song_{i:02d}{ext}").write_bytes(b"fake-audio")
    # A non-audio file MUST NOT be counted
    (lib / "notes.txt").write_text("not audio", encoding="utf-8")
    # Files in subdirectories also count (recursive glob)
    (lib / "subdir").mkdir()
    (lib / "subdir" / "nested.flac").write_bytes(b"fake-audio-nested")
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the count is 6 (5 in root + 1 in subdir); the notes.txt is excluded
    assert response.status_code == 200
    assert response.json()["like_library_track_count"] == 6


def test_get_api_index_response_shape_has_all_five_fields(tmp_path: Path) -> None:
    """The response payload always has exactly the canonical 5 keys."""
    # Given a fresh config (no index, empty lib)
    client = _make_client(tmp_path)

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the response shape is the canonical 5-key dict
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "count",
        "last_scan",
        "like_library_path",
        "like_library_track_count",
        "thresholds",
    }


def test_get_api_index_count_does_not_require_vectors_file(tmp_path: Path) -> None:
    """If only the manifest is present (vectors file missing), count is still len(manifest)."""
    # Given a manifest with N=3 entries but no vectors file
    cfg = load_config(_write_config(tmp_path))
    _ = _seed_manifest(cfg.data_dir, n=3)
    # Deliberately do NOT write library_vectors.npy
    vectors_path = cfg.data_dir / "index" / "library_vectors.npy"
    if vectors_path.exists():
        vectors_path.unlink()
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then count still equals N (manifest is the source of truth for the read-only status)
    assert response.status_code == 200
    assert response.json()["count"] == 3


def test_get_api_index_handles_malformed_manifest_gracefully(tmp_path: Path) -> None:
    """If the manifest is malformed JSON, the endpoint returns 200 with count=0 (not 500)."""
    # Given a manifest file containing invalid JSON
    cfg = load_config(_write_config(tmp_path))
    index_dir = cfg.data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "library_manifest.json").write_text("not valid json{", encoding="utf-8")
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the endpoint reports a zero count instead of crashing
    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_get_api_index_zero_track_count_when_lib_empty(tmp_path: Path) -> None:
    """Given an empty like_library_dir, like_library_track_count is 0."""
    # Given a config whose like_library_dir exists but is empty
    cfg = load_config(_write_config(tmp_path))
    # like_library_dir was already created by _write_config; leave it empty
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the track count is 0
    assert response.status_code == 200
    assert response.json()["like_library_track_count"] == 0


def test_get_api_index_does_not_read_vector_matrix_bytes_on_request(tmp_path: Path) -> None:
    """Guardrail: a real vectors.npy must NOT be opened by the read-only endpoint."""
    # Given a manifest AND a vectors file in place; we then replace vectors.npy with a sentinel
    cfg = load_config(_write_config(tmp_path))
    _ = _seed_manifest(cfg.data_dir, n=2)
    vectors_path = cfg.data_dir / "index" / "library_vectors.npy"
    sentinel = b"THIS_IS_NOT_VALID_NUMPY_BYTES" * 100
    vectors_path.write_bytes(sentinel)
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then it still returns 200 and count from the manifest (no numpy.load call)
    assert response.status_code == 200
    assert response.json()["count"] == 2
    # And the sentinel bytes are unchanged (no read-or-rewrite of vectors file)
    assert vectors_path.read_bytes() == sentinel


def test_get_api_index_is_case_insensitive_on_audio_extensions(tmp_path: Path) -> None:
    """Given audio files with mixed-case extensions, the track count still picks them up."""
    # Given like_library_dir with .FLAC, .Mp3, .OpUs files
    cfg = load_config(_write_config(tmp_path))
    lib = cfg.like_library_dir
    (lib / "a.FLAC").write_bytes(b"x")
    (lib / "b.Mp3").write_bytes(b"x")
    (lib / "c.OpUs").write_bytes(b"x")
    (lib / "d.m4A").write_bytes(b"x")
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then all 4 are counted
    assert response.status_code == 200
    assert response.json()["like_library_track_count"] == 4


def test_get_api_index_last_scan_iso8601_has_utc_suffix(tmp_path: Path) -> None:
    """Given a manifest, the last_scan string parses as a timezone-aware datetime in UTC."""
    # Given a config with a freshly-written manifest
    cfg = load_config(_write_config(tmp_path))
    _ = _seed_manifest(cfg.data_dir, n=1)
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then last_scan parses as a timezone-aware ISO-8601 timestamp
    last_scan_str = response.json()["last_scan"]
    parsed = datetime.fromisoformat(last_scan_str)
    assert parsed.tzinfo is not None, (
        f"last_scan {last_scan_str!r} is naive; expected a UTC-suffixed ISO-8601 string"
    )


def test_build_index_status_helper_matches_endpoint_payload(tmp_path: Path) -> None:
    """The dashboard-side helper returns the same payload as the /api/index JSON endpoint."""
    # Given a config with a seeded manifest + some audio files
    cfg = load_config(_write_config(tmp_path))
    _ = _seed_manifest(cfg.data_dir, n=4)
    (cfg.like_library_dir / "track_a.flac").write_bytes(b"x")
    (cfg.like_library_dir / "track_b.mp3").write_bytes(b"x")

    # When we call build_index_status directly AND the JSON endpoint
    helper_payload = build_index_status(cfg)
    endpoint_payload = TestClient(create_app(cfg)).get("/api/index").json()

    # Then the two payloads are byte-equal: same keys, same values
    assert helper_payload == endpoint_payload


def test_get_api_index_returns_none_thresholds_when_not_calibrated(tmp_path: Path) -> None:
    """When no thresholds.json exists, the response's thresholds field is None."""
    # Given a config whose data_dir has no thresholds.json
    cfg = load_config(_write_config(tmp_path))
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the thresholds field is None (the dashboard renders "(not calibrated)")
    assert response.status_code == 200
    payload = response.json()
    assert "thresholds" in payload, "response must always carry the thresholds key"
    assert payload["thresholds"] is None, (
        f"thresholds must be None when no thresholds.json exists, got {payload['thresholds']!r}"
    )


def test_get_api_index_returns_thresholds_when_calibrated(tmp_path: Path) -> None:
    """When thresholds.json exists, the response's thresholds field carries the parsed dict."""
    # Given a config whose data_dir has a valid thresholds.json
    cfg = load_config(_write_config(tmp_path))
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "keep_threshold": 0.81,
                "skip_threshold": 0.19,
                "computed_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(cfg))

    # When requesting the index status endpoint
    response = client.get("/api/index")

    # Then the thresholds field carries the keep/skip pair (computed_at is stripped)
    assert response.status_code == 200
    payload = response.json()
    assert payload["thresholds"] == {"keep_threshold": 0.81, "skip_threshold": 0.19}, (
        f"thresholds field must round-trip the keep/skip values; got {payload['thresholds']!r}"
    )
