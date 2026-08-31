"""Tests for taste_pipeline.web.routes_check: file drag-drop + URL job endpoints.

The check route module mounts three endpoints:

- ``POST /api/check/file`` -- synchronous, base64-encoded audio file.
- ``POST /api/check/url`` -- job runner entry point (kind=check_url).
- ``GET /check`` -- render the check page.

All tests use ``tmp_path`` for an isolated ``data_dir`` and inject a
``FakeJobRunner`` for the URL job (the file endpoint is synchronous and
needs no runner). The library-empty guard is locked by a sentinel test:
the library manifest is absent from ``<data_dir>/index`` for a fresh
``tmp_path``, so the index_count check returns 0 and the file endpoint
returns 503.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING

import numpy as np
from fastapi.testclient import TestClient

from conftest import _build_client_with_fake_runner, _client, _make_config
from taste_pipeline.library import LibraryIndex
from taste_pipeline.web import create_app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from taste_pipeline.config import Config as ConfigType


def _seed_library_manifest(
    config: ConfigType,
    *,
    track_count: int = 3,
) -> None:
    """Write a synthetic ``library_manifest.json`` under ``config.data_dir/index``.

    The vectors file is left absent on purpose -- the routes_check
    endpoint only reads the manifest (mirrors the ``routes_index`` read
    pattern). The check_url factory re-runs the real ``scan_library`` on
    the URL path; tests that exercise the URL endpoint rely on the
    full pipeline, not this seed.
    """
    index_dir = config.data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, float | int]] = {}
    for i in range(track_count):
        manifest[f"track_{i}.flac"] = {"mtime": 0.0, "size": 0, "vector_id": i}
    _ = (index_dir / "library_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _install_fake_embed_and_scan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    library_count: int,
    track_similarity_seed: int = 42,
) -> None:
    """Monkey-patch ``embed.embed_track`` and ``library.scan_library`` for check_audio_file.

    Returns a deterministic ``(n, 512)`` library matrix of unit vectors
    and a query embedding with a known alignment (cosine ~0.95 to the
    first library row) so the verdict is reproducible across runs.
    """
    from taste_pipeline import calibrate, embed, library  # noqa: PLC0415 -- lazy: under test

    rng = np.random.default_rng(seed=track_similarity_seed)
    raw = rng.standard_normal((library_count, 512)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    matrix = (raw / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)

    target = matrix[0]
    noise = rng.standard_normal(512).astype(np.float32)
    noise = noise - float(noise @ target) * target
    noise = noise / max(float(np.linalg.norm(noise)), 1e-8)
    alpha = np.float32(0.95)
    track_emb = (alpha * target + np.sqrt(np.float32(1.0) - alpha * alpha) * noise).astype(np.float32)

    def fake_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        return track_emb

    def fake_scan_library(_config: ConfigType, _embedder: object) -> LibraryIndex:
        return LibraryIndex(
            _vectors=matrix,
            _manifest={
                f"track_{i}.flac": {"mtime": 0.0, "size": 0, "vector_id": i} for i in range(library_count)
            },
        )

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "scan_library", fake_scan_library)
    monkeypatch.setattr(calibrate, "load_thresholds", lambda _data_dir: None)


def test_post_check_file_with_valid_audio_returns_check_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given an app with a seeded library manifest (so index_count > 0)
    # and a monkey-patched embedder / scanner.
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=3)
    _install_fake_embed_and_scan(monkeypatch, library_count=3)
    client = TestClient(create_app(cfg))

    payload_bytes = b"FAKE_AUDIO_BYTES"
    encoded = base64.b64encode(payload_bytes).decode("ascii")

    # When POSTing to /api/check/file
    response = client.post(
        "/api/check/file",
        json={"filename": "test.flac", "content_b64": encoded},
    )

    # Then 200 + JSON body with the verdict + score fields
    assert response.status_code == 200, (
        f"valid base64 audio + non-empty library must return 200, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["verdict"] in {"matches", "uncertain", "doesn't_match", "no_library", "error"}
    assert "score" in data
    assert "top_k" in data
    assert "human_readable" in data
    assert isinstance(data["top_k"], list)
    # And no calibration was loaded -> verdict is "uncertain" but score is computed
    assert data["verdict"] == "uncertain"


def test_post_check_file_rejects_non_audio_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=1)
    _install_fake_embed_and_scan(monkeypatch, library_count=1)
    client = TestClient(create_app(cfg))

    payload_bytes = b"hello world"
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    response = client.post(
        "/api/check/file",
        json={"filename": "test.txt", "content_b64": encoded},
    )

    assert response.status_code == 422, (
        f"non-audio extension must return 422, got {response.status_code}: {response.text}"
    )
    detail = response.json().get("detail", "")
    assert "extension" in detail.lower(), f"detail should mention extension; got {detail!r}"

    response_zip = client.post(
        "/api/check/file",
        json={"filename": "song.zip", "content_b64": encoded},
    )
    assert response_zip.status_code == 422


def test_post_check_file_rejects_invalid_base64(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=1)
    _install_fake_embed_and_scan(monkeypatch, library_count=1)
    client = TestClient(create_app(cfg))

    response = client.post(
        "/api/check/file",
        json={"filename": "test.flac", "content_b64": "!!!not_valid_base64!!!"},
    )

    assert response.status_code == 422
    detail = response.json().get("detail", "")
    assert "base64" in detail.lower(), f"detail should mention base64; got {detail!r}"


def test_post_check_file_returns_503_when_library_empty(tmp_path: Path) -> None:
    client = _client(tmp_path)

    encoded = base64.b64encode(b"data").decode("ascii")
    response = client.post(
        "/api/check/file",
        json={"filename": "test.flac", "content_b64": encoded},
    )

    assert response.status_code == 503, (
        f"empty library must return 503, got {response.status_code}: {response.text}"
    )
    detail = response.json().get("detail", "")
    assert "Build Index" in detail, f"detail should name the fix; got {detail!r}"


def test_post_check_file_cleans_up_tmp_file_on_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The tmp file must be removed even when the embedder raises."""
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=1)

    from taste_pipeline import embed  # noqa: PLC0415 -- lazy: under test

    def raising_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        raise RuntimeError("simulated_embed_failure")

    monkeypatch.setattr(embed, "embed_track", raising_embedder)
    client = TestClient(create_app(cfg))

    encoded = base64.b64encode(b"data").decode("ascii")
    response = client.post(
        "/api/check/file",
        json={"filename": "test.flac", "content_b64": encoded},
    )

    assert response.status_code == 500
    tmp_dir = cfg.data_dir / "tmp"
    if tmp_dir.is_dir():
        leftovers = list(tmp_dir.iterdir())
        assert not leftovers, f"tmp dir should be empty after a failed check; leftover files: {leftovers}"


def test_post_check_file_rejects_empty_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=1)
    _install_fake_embed_and_scan(monkeypatch, library_count=1)
    client = TestClient(create_app(cfg))

    response = client.post(
        "/api/check/file",
        json={"filename": "", "content_b64": base64.b64encode(b"data").decode("ascii")},
    )

    assert response.status_code == 422
    detail = response.json().get("detail", "")
    assert "filename" in detail.lower()


def test_post_check_url_returns_201_and_creates_job(tmp_path: Path) -> None:
    client, runner = _build_client_with_fake_runner(tmp_path)
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=1)

    response = client.post(
        "/api/check/url",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )

    assert response.status_code == 201, (
        f"valid URL must return 201, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data["kind"] == "check_url"
    assert data["status"] in {"queued", "running"}
    assert any(call[0] == "check_url" for call in runner.submit_calls), (
        f"runner.submit was not called with kind=check_url; calls: {runner.submit_calls}"
    )


def test_post_check_url_rejects_empty_url(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    _seed_library_manifest(cfg, track_count=1)
    client, _ = _build_client_with_fake_runner(tmp_path)

    response = client.post("/api/check/url", json={})

    assert response.status_code == 422
    detail = response.json().get("detail", "")
    assert "url" in detail.lower()


def test_post_check_url_returns_503_when_library_empty(tmp_path: Path) -> None:
    client, _ = _build_client_with_fake_runner(tmp_path)

    response = client.post(
        "/api/check/url",
        json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
    )

    assert response.status_code == 503
    detail = response.json().get("detail", "")
    assert "Build Index" in detail


def test_get_check_renders_drag_drop_zone_and_url_input(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/check")

    assert response.status_code == 200
    body = response.text
    assert 'id="drop-zone"' in body, "check page must have the drop-zone element"
    assert 'id="file-input"' in body, "check page must have the file input"
    assert 'id="url-input"' in body, "check page must have the URL input field"
    assert 'id="check-url-btn"' in body, "check page must have the Check URL button"
    assert 'id="check-result"' in body, "check page must have the result panel"
    assert "EventSource" in body, "check page must use EventSource for SSE progress"
