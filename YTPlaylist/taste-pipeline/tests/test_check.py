"""Tests for taste_pipeline.check: track similarity scoring against the library index.

The check module is the gatekeeper for the Check Track page's verdict.
Two surfaces:

- :func:`check_track_against_library` -- pure numpy function. Takes a
  query vector + library matrix + ids + thresholds; returns a
  :class:`CheckResult`. Never touches disk.
- :func:`check_audio_file` -- thin I/O wrapper: embeds one audio file
  (single-track path), loads the persisted library index, loads
  calibration thresholds, then delegates to the pure function. Mirrors
  :mod:`taste_pipeline.calibrate`'s lazy-import pattern so tests can
  monkey-patch ``embed.embed_track`` and ``library.load_index`` and
  observe the patches.

No real CLAP model is loaded in any test -- the embedder and index loader
are monkey-patched to return deterministic synthetic ``(n, 512)`` and
``(512,)`` matrices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from taste_pipeline.check import (
    CheckResult,
    check_audio_file,
    check_track_against_library,
)
from taste_pipeline.config import Config, load_config
from taste_pipeline.library import LibraryIndex

if TYPE_CHECKING:
    from pathlib import Path

    from taste_pipeline.config import Config as ConfigType


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal valid config TOML under tmp_path and return its path."""
    like_lib = tmp_path / "lib"
    like_lib.mkdir(exist_ok=True)
    body = (
        f'like_library_dir = "{like_lib.as_posix()}"\n'
        f'data_dir = "{(tmp_path / "data").as_posix()}"\n'
        f'cookie_file = "{(tmp_path / "cookies.txt").as_posix()}"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _make_config(tmp_path: Path) -> Config:
    """Build a real Config via load_config from a tmp TOML file."""
    return load_config(_write_minimal_config(tmp_path))


def _unit_vectors(n: int, d: int = 512, seed: int = 42) -> np.ndarray:
    """Return ``n`` random unit-norm ``(d,)`` float32 vectors."""
    rng = np.random.default_rng(seed=seed)
    raw = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)


def _aligned_vector(target: np.ndarray, seed: int = 1) -> np.ndarray:
    """Return a unit vector with a known similarity to ``target``.

    Builds ``alpha * target + sqrt(1-alpha**2) * orthogonal_noise``
    with ``alpha = 0.95`` -- the resulting cosine similarity to ``target``
    is exactly 0.95 (when the noise is orthogonal).
    """
    rng = np.random.default_rng(seed=seed)
    d = target.shape[0]
    noise = rng.standard_normal(d).astype(np.float32)
    noise = noise - float(noise @ target) * target  # orthogonalize against target
    noise = noise / max(float(np.linalg.norm(noise)), 1e-8)
    alpha = np.float32(0.95)
    combined = alpha * target + np.sqrt(np.float32(1.0) - alpha * alpha) * noise
    return (combined / np.linalg.norm(combined)).astype(np.float32)


def test_check_track_against_library_no_library_returns_no_library_verdict() -> None:
    # Given an empty library matrix (no indexed tracks)
    track = _unit_vectors(1)[0]
    empty_matrix = np.empty((0, 512), dtype=np.float32)
    ids: list[str] = []

    # When checking the track
    result = check_track_against_library(track, empty_matrix, ids, thresholds=None)

    # Then the verdict is "no_library" and the summary names the fix
    assert result.verdict == "no_library"
    assert result.score == 0.0
    assert result.top_k == []
    assert "Build Index" in result.human_readable


def test_check_track_against_library_with_thresholds_matches_high_similarity() -> None:
    # Given a library with 5 random tracks and a query that is highly similar to the first
    matrix = _unit_vectors(5, seed=10)
    ids = [f"track_{i}.flac" for i in range(5)]
    query = _aligned_vector(matrix[0])  # cosine similarity 0.95 to track_0
    thresholds = {"keep_threshold": 0.5, "skip_threshold": 0.2}

    # When checking
    result = check_track_against_library(query, matrix, ids, thresholds)

    # Then verdict is "matches" and the score is at or above keep_threshold
    assert result.verdict == "matches", (
        f"high-similarity query against a known threshold should match, got {result!r}"
    )
    assert result.score >= thresholds["keep_threshold"]
    assert result.threshold_keep == 0.5
    assert result.threshold_skip == 0.2
    # And the top match in the top-K is the aligned one
    assert result.top_k[0][1] == "track_0.flac"
    assert result.top_k[0][0] >= thresholds["keep_threshold"]


def test_check_track_against_library_with_thresholds_doesnt_match_low_similarity() -> None:
    # Given a library with 5 random unit vectors and a query with cosine < skip
    matrix = _unit_vectors(5, seed=20)
    ids = [f"track_{i}.flac" for i in range(5)]
    # A query that's nearly orthogonal to every library row: take the mean,
    # then build a vector whose max cosine is bounded by construction.
    rng = np.random.default_rng(seed=21)
    noise = rng.standard_normal(512).astype(np.float32)
    noise = noise / np.linalg.norm(noise)
    # Subtract the projection onto every library row to make it nearly orthogonal
    for row in matrix:
        noise = noise - float(noise @ row) * row
    noise = noise / max(float(np.linalg.norm(noise)), 1e-8)
    thresholds = {"keep_threshold": 0.95, "skip_threshold": 0.9}

    # When checking
    result = check_track_against_library(noise, matrix, ids, thresholds)

    # Then verdict is "doesn't_match"
    assert result.verdict == "doesn't_match", (
        f"orthogonalized query against high-threshold band should be doesn't_match, got {result!r}"
    )
    assert result.score < thresholds["skip_threshold"]


def test_check_track_against_library_with_thresholds_uncertain_middle_similarity() -> None:
    # Given a library with 5 random tracks and a query whose max similarity
    # falls strictly between skip and keep (the in-between band)
    matrix = _unit_vectors(5, seed=30)
    ids = [f"track_{i}.flac" for i in range(5)]
    rng = np.random.default_rng(seed=31)
    query = rng.standard_normal(512).astype(np.float32)
    query = query / np.linalg.norm(query)
    # Compute the actual max cosine against the library
    actual_max = float((query.reshape(1, -1) @ matrix.T).max())
    # Pick thresholds that straddle the actual max -- they MUST put the query in the middle
    skip = max(0.0, actual_max - 0.05)
    keep = min(1.0, actual_max + 0.05)
    if skip >= keep:
        pytest.skip(f"test setup could not bracket the actual max cosine ({actual_max:.4f})")
    thresholds = {"keep_threshold": keep, "skip_threshold": skip}

    # When checking
    result = check_track_against_library(query, matrix, ids, thresholds)

    # Then verdict is "uncertain"
    assert result.verdict == "uncertain", (
        f"middle-band query should be uncertain, got {result!r} (thresholds {thresholds})"
    )
    assert thresholds["skip_threshold"] <= result.score < thresholds["keep_threshold"]


def test_check_track_against_library_no_calibration_returns_uncertain() -> None:
    # Given a library + high-similarity query + thresholds=None (no calibration)
    matrix = _unit_vectors(3, seed=40)
    ids = [f"track_{i}.flac" for i in range(3)]
    query = _aligned_vector(matrix[0])

    # When checking
    result = check_track_against_library(query, matrix, ids, thresholds=None)

    # Then verdict is "uncertain" (no calibration to apply)
    assert result.verdict == "uncertain"
    assert result.threshold_keep is None
    assert result.threshold_skip is None
    assert "no calibration" in result.human_readable.lower()


def test_check_track_against_library_top_k_returns_highest_first() -> None:
    # Given a library with 10 tracks; query is highly aligned to track_3
    matrix = _unit_vectors(10, seed=50)
    ids = [f"track_{i:02d}.flac" for i in range(10)]
    query = _aligned_vector(matrix[3])

    # When checking with top_k=5
    result = check_track_against_library(query, matrix, ids, thresholds=None, top_k=5)

    # Then the top-K list has 5 entries, sorted by similarity descending,
    # and the FIRST entry is track_3 (the aligned one)
    assert len(result.top_k) == 5
    similarities = [s for s, _ in result.top_k]
    assert similarities == sorted(similarities, reverse=True), (
        f"top_k must be sorted descending; got {similarities}"
    )
    assert result.top_k[0][1] == "track_03.flac", (
        f"first top-K entry must be the aligned track_03; got {result.top_k[0]}"
    )


def test_check_track_against_library_top_k_clamped_to_library_size() -> None:
    # Given a 3-track library and a query, with top_k=10 (larger than library)
    matrix = _unit_vectors(3, seed=60)
    ids = [f"track_{i}.flac" for i in range(3)]

    # When checking with top_k=10
    result = check_track_against_library(matrix[0], matrix, ids, thresholds=None, top_k=10)

    # Then the top-K list is clamped to the library size (3)
    assert len(result.top_k) == 3, (
        f"top_k must be clamped to library size when larger than n, got {len(result.top_k)} entries"
    )


def test_check_audio_file_embeds_query_and_loads_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: check_audio_file -> embed query + load index + load_thresholds -> compare.

    Verifies the integration contract: ``check_audio_file`` embeds ONLY the
    query track and reads the persisted index via :func:`library.load_index`
    (looked up by attribute, not cached at import time). The embedder and
    loader are monkey-patched so no real CLAP model is loaded.
    """
    cfg = _make_config(tmp_path)

    # Synthetic library: 4 random tracks.
    vectors = _unit_vectors(4, seed=70)
    index = LibraryIndex(
        _vectors=vectors,
        _manifest={f"track_{i}.flac": {"mtime": 0.0, "size": 0, "vector_id": i} for i in range(4)},
    )
    fake_track_embedding = _aligned_vector(vectors[0])
    embed_calls: list[Path] = []

    def fake_embedder(path: Path, _config: ConfigType) -> np.ndarray:
        embed_calls.append(path)
        return fake_track_embedding

    load_calls: list[Config] = []

    def fake_load_index(config: Config) -> LibraryIndex:
        load_calls.append(config)
        return index

    from taste_pipeline import calibrate, embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "load_index", fake_load_index)
    # Stub load_thresholds to return None (no calibration).
    monkeypatch.setattr(calibrate, "load_thresholds", lambda _data_dir: None)

    # When running check on a (fake) audio path
    audio_path = tmp_path / "fake.flac"
    audio_path.write_bytes(b"FAKE_AUDIO_BYTES")
    result = check_audio_file(audio_path, cfg)

    # Then: load_index was called exactly once with the config
    assert len(load_calls) == 1, f"load_index called {len(load_calls)} times, expected 1"
    assert load_calls[0] is cfg, "load_index received the wrong config"
    # And: only the query track was embedded -- the library is NOT re-embedded
    assert embed_calls == [audio_path], f"expected only the query to be embedded, got {embed_calls!r}"

    # And: the result is a verdict (no calibration -> uncertain)
    assert isinstance(result, CheckResult)
    # No thresholds loaded -> uncertain verdict
    assert result.verdict == "uncertain"


def test_check_audio_file_returns_no_library_when_count_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty like-library (n=0) -> verdict='no_library' from the I/O wrapper too."""
    cfg = _make_config(tmp_path)

    # Empty LibraryIndex (n=0): no_library expected
    empty_vectors = np.empty((0, 512), dtype=np.float32)
    index = LibraryIndex(_vectors=empty_vectors, _manifest={})

    def fake_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        return _unit_vectors(1)[0]

    def fake_load_index(_config: Config) -> LibraryIndex:
        return index

    from taste_pipeline import calibrate, embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "load_index", fake_load_index)
    monkeypatch.setattr(calibrate, "load_thresholds", lambda _data_dir: None)

    audio_path = tmp_path / "fake.flac"
    audio_path.write_bytes(b"FAKE_AUDIO_BYTES")
    result = check_audio_file(audio_path, cfg)

    assert result.verdict == "no_library"
    assert result.top_k == []


def test_check_audio_file_propagates_thresholds_from_calibrate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration thresholds loaded by check_audio_file are honored by the verdict."""
    cfg = _make_config(tmp_path)

    # Library with 3 tracks
    vectors = _unit_vectors(3, seed=80)
    index = LibraryIndex(
        _vectors=vectors,
        _manifest={f"track_{i}.flac": {"mtime": 0.0, "size": 0, "vector_id": i} for i in range(3)},
    )
    fake_track_embedding = _aligned_vector(vectors[0])  # cosine ~0.95 to vectors[0]

    def fake_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        return fake_track_embedding

    def fake_load_index(_config: Config) -> LibraryIndex:
        return index

    # Thresholds calibrated with VERY low keep (so 0.95 always matches) and high skip
    fake_thresholds = {"keep_threshold": 0.5, "skip_threshold": 0.2}

    from taste_pipeline import calibrate, embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "load_index", fake_load_index)
    monkeypatch.setattr(calibrate, "load_thresholds", lambda _data_dir: fake_thresholds)

    audio_path = tmp_path / "fake.flac"
    audio_path.write_bytes(b"FAKE_AUDIO_BYTES")
    result = check_audio_file(audio_path, cfg)

    # The verdict must reflect the calibrated thresholds
    assert result.threshold_keep == 0.5
    assert result.threshold_skip == 0.2
    assert result.verdict == "matches", (
        f"high-similarity query with keep_threshold=0.5 should match, got {result!r}"
    )


def test_check_audio_file_uses_row_order_not_sorted_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The library ids are taken in row (vector_id) order, not sorted manifest order.

    After an incremental index build the manifest insertion order need not be
    globally sorted, so ``sorted(manifest)`` would desync ids from vector rows.
    The manifest here is deliberately inserted with ``a.flac`` (vector_id 1)
    before ``z.flac`` (vector_id 0); the query is aligned to row 0, so the top
    match MUST resolve to ``z.flac`` when ids follow row order.
    """
    cfg = _make_config(tmp_path)

    vectors = _unit_vectors(2, seed=90)
    index = LibraryIndex(
        _vectors=vectors,
        _manifest={
            "a.flac": {"mtime": 0.0, "size": 0, "vector_id": 1},
            "z.flac": {"mtime": 0.0, "size": 0, "vector_id": 0},
        },
    )
    query = _aligned_vector(vectors[0])  # cosine ~0.95 to row 0 (z.flac)

    def fake_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        return query

    def fake_load_index(_config: Config) -> LibraryIndex:
        return index

    from taste_pipeline import calibrate, embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "load_index", fake_load_index)
    monkeypatch.setattr(calibrate, "load_thresholds", lambda _data_dir: None)

    audio_path = tmp_path / "fake.flac"
    audio_path.write_bytes(b"FAKE_AUDIO_BYTES")
    result = check_audio_file(audio_path, cfg)

    # Row order is [z.flac, a.flac]; the aligned row-0 track wins the top slot.
    assert result.top_k[0][1] == "z.flac", (
        f"top match must follow row order (z.flac at vector_id 0), got {result.top_k[0]}"
    )


def test_check_module_does_not_import_heavy_dependencies_at_module_load() -> None:
    """Guardrail: importing check must NOT load transformers or torch at module load.

    The dashboard hits this module's helpers on the read path. A
    module-level import of embed / library / calibrate would pull
    transformers + torch into the request handler's import set, which
    is unacceptable for a fast read-only endpoint.
    """
    import taste_pipeline.check as check_module  # noqa: PLC0415 -- lazy: under test

    forbidden = {"transformers", "torch"}
    module_vars = vars(check_module)
    for name in forbidden:
        assert name not in module_vars, (
            f"check module imported {name!r} at module load; this loads the CLAP model "
            f"on the dashboard's read path"
        )
