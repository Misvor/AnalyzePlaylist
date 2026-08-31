"""Tests for taste_pipeline.calibrate: threshold calibration pipeline.

The calibrate module is the gatekeeper for the "Calibrate weights"
button on the dashboard. It walks the like-library through the real
CLAP embedder (in production) and derives ``keep_threshold`` /
``skip_threshold`` from the pairwise cosine-similarity distribution.

Contract locked by these tests:

- :func:`compute_thresholds` returns a dict with both keys for any input
  shape (degenerate inputs get defensive defaults instead of raising).
- :func:`persist_thresholds` / :func:`load_thresholds` round-trip
  correctly; an absent file loads as ``None`` (not an exception).
- :func:`run_calibration` integrates with :mod:`taste_pipeline.embed`
  and :mod:`taste_pipeline.library` via lazy imports so tests can
  monkey-patch both modules without the calibrate module caching its
  own references.
- :func:`run_calibration` reports progress through the injected
  ``report`` callback at the documented milestones (0.0 / 0.3 / 0.9 / 1.0).
- :func:`run_calibration` persists the thresholds under
  ``<data_dir>/thresholds.json`` and returns the same dict.

No real CLAP model is loaded in any test -- the embedder is monkey-patched
to return a deterministic synthetic ``(n, 512)`` matrix.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest  # noqa: TC002 -- pytest used at runtime (pytest.raises) and as a type annotation

from taste_pipeline import calibrate as calibrate_module
from taste_pipeline.calibrate import (
    _DEFAULT_KEEP_THRESHOLD,
    _DEFAULT_SKIP_THRESHOLD,
    _THRESHOLDS_FILE_NAME,
    compute_thresholds,
    load_thresholds,
    persist_thresholds,
    run_calibration,
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


def _unit_vectors(n: int, d: int = 512) -> np.ndarray:
    """Return ``n`` random unit-norm ``(d,)`` float32 vectors.

    Used to build a synthetic embedding matrix with non-trivial pairwise
    similarity distribution; the vectors are L2-normalized so the cosine
    similarity equals the dot product.
    """
    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)


def test_compute_thresholds_returns_dict_with_keep_and_skip() -> None:
    # Given a (10, 512) synthetic embedding matrix
    vectors = _unit_vectors(10)

    # When computing thresholds with the default quantiles
    result = compute_thresholds(vectors)

    # Then the dict has both keys, both as floats, both in [0, 1]
    assert set(result.keys()) == {"keep_threshold", "skip_threshold"}, (
        f"compute_thresholds must return exactly {{keep_threshold, skip_threshold}}, got {set(result.keys())}"
    )
    for key in ("keep_threshold", "skip_threshold"):
        assert isinstance(result[key], float), f"{key} must be a float, got {type(result[key])}"
        # Cosine similarity ranges from -1 to 1 (not [0, 1]); the quantile
        # of an upper-triangular matrix is therefore in [-1, 1] too.
        assert -1.0 <= result[key] <= 1.0, f"{key}={result[key]} must be in [-1, 1] (cosine similarity range)"
    # keep_quantile (0.75) > skip_quantile (0.25) -> keep_threshold >= skip_threshold
    assert result["keep_threshold"] >= result["skip_threshold"], (
        f"keep_threshold ({result['keep_threshold']}) must be >= skip_threshold "
        f"({result['skip_threshold']}) for the default quantiles 0.75/0.25"
    )


def test_compute_thresholds_with_small_library_returns_defaults() -> None:
    # Given a (1, 512) vector -- too small for pairwise similarity (n < 2)
    vectors = _unit_vectors(1)

    # When computing thresholds
    result = compute_thresholds(vectors)

    # Then the defensive defaults are returned (the two keys exist; the
    # caller doesn't have to branch on "no calibration possible").
    assert result == {
        "keep_threshold": _DEFAULT_KEEP_THRESHOLD,
        "skip_threshold": _DEFAULT_SKIP_THRESHOLD,
    }, (
        f"n=1 must yield defensive defaults "
        f"(keep={_DEFAULT_KEEP_THRESHOLD}, skip={_DEFAULT_SKIP_THRESHOLD}), got {result}"
    )

    # And n=0 also returns the defensive defaults -- the function must
    # never raise on an empty matrix (the dashboard can show "not
    # calibrated" before the user has run the index pass).
    empty = np.empty((0, 512), dtype=np.float32)
    assert compute_thresholds(empty) == {
        "keep_threshold": _DEFAULT_KEEP_THRESHOLD,
        "skip_threshold": _DEFAULT_SKIP_THRESHOLD,
    }


def test_persist_thresholds_writes_valid_json_at_expected_path(tmp_path: Path) -> None:
    # Given a thresholds dict and a fresh tmp data dir
    data_dir = tmp_path / "data"
    thresholds = {"keep_threshold": 0.8, "skip_threshold": 0.2}

    # When persisting
    out_path = persist_thresholds(data_dir, thresholds)

    # Then the path is data_dir/thresholds.json and the file is parseable JSON
    assert out_path == data_dir / _THRESHOLDS_FILE_NAME, (
        f"persist_thresholds must return data_dir/{_THRESHOLDS_FILE_NAME}, got {out_path}"
    )
    assert out_path.is_file(), f"thresholds.json was not written at {out_path}"
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["keep_threshold"] == 0.8, (
        f"persisted keep_threshold lost its value: got {loaded['keep_threshold']}"
    )
    assert loaded["skip_threshold"] == 0.2, (
        f"persisted skip_threshold lost its value: got {loaded['skip_threshold']}"
    )
    # And the file carries a computed_at ISO timestamp (the UI uses it).
    assert "computed_at" in loaded, (
        "thresholds.json missing computed_at; the UI needs this for 'Calibrated N hours ago'"
    )
    parsed = loaded["computed_at"]
    # ISO-8601 timestamps parse without raising; sanity-check the prefix.
    assert isinstance(parsed, str), f"computed_at is not a string: {parsed!r}"
    assert parsed.startswith("20"), f"computed_at is not an ISO-8601 string: {parsed!r}"


def test_persist_thresholds_creates_data_dir_if_missing(tmp_path: Path) -> None:
    # Given a data_dir that does NOT exist yet
    data_dir = tmp_path / "fresh" / "data"
    assert not data_dir.exists()

    # When persisting
    _ = persist_thresholds(data_dir, {"keep_threshold": 0.7, "skip_threshold": 0.1})

    # Then the data_dir was created (calibration is the first write
    # in some deployments; missing dir would otherwise raise FileNotFoundError).
    assert data_dir.is_dir(), f"persist_thresholds did not create {data_dir}"


def test_load_thresholds_returns_none_when_file_missing(tmp_path: Path) -> None:
    # Given an empty data_dir with no thresholds.json
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # When loading
    result = load_thresholds(data_dir)

    # Then None is returned (caller branches on "not calibrated")
    assert result is None, f"load_thresholds must return None for a missing file, got {result!r}"


def test_load_thresholds_round_trips_persisted_values(tmp_path: Path) -> None:
    # Given a persisted thresholds.json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _ = persist_thresholds(data_dir, {"keep_threshold": 0.82, "skip_threshold": 0.18})

    # When loading
    result = load_thresholds(data_dir)

    # Then both keys round-trip exactly (computed_at is dropped, but the
    # threshold values themselves are preserved bit-for-bit).
    assert result is not None
    assert result["keep_threshold"] == 0.82, f"keep_threshold lost precision: got {result['keep_threshold']}"
    assert result["skip_threshold"] == 0.18, f"skip_threshold lost precision: got {result['skip_threshold']}"


def test_load_thresholds_returns_none_for_malformed_json(tmp_path: Path) -> None:
    # Given a malformed thresholds.json (e.g. calibration crashed mid-write)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / _THRESHOLDS_FILE_NAME).write_text("not valid json{", encoding="utf-8")

    # When loading
    result = load_thresholds(data_dir)

    # Then None is returned -- "not calibrated" surface state, not a 500.
    assert result is None, f"load_thresholds must treat malformed JSON as 'missing', got {result!r}"


def test_run_calibration_calls_scan_library_with_embedder_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end calibration: scan -> compute -> persist, all mocked.

    Verifies the integration contract: ``run_calibration`` calls
    :func:`library.scan_library` with the real ``embed.embed_track``
    embedder (looked up by attribute, not cached at import time), and
    persists the computed thresholds under ``config.data_dir``. The
    embedder and scan_library are monkey-patched to return a synthetic
    ``LibraryIndex`` so no real CLAP model is loaded.
    """

    cfg = _make_config(tmp_path)

    # Build the synthetic LibraryIndex the patched scan_library will return.
    vectors = _unit_vectors(8)
    index = LibraryIndex(_vectors=vectors, _manifest={f"track_{i}.flac": {} for i in range(8)})

    # The fake embedder is the object the calibrate module looks up via
    # ``embed.embed_track`` at call time. Asserting identity inside the
    # patched scan_library closure confirms the calibrate module passed
    # the real embedder through (not its own reference).
    def fake_embedder(path: Path, config: ConfigType) -> np.ndarray:
        return vectors[0]

    # Patch scan_library + embed.embed_track. The calibrate module imports
    # ``from taste_pipeline import embed, library`` INSIDE run_calibration,
    # so the monkey-patches must land on those source module attrs.
    from taste_pipeline import embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    scan_calls: list[tuple[Config, object]] = []

    def fake_scan_library(config: Config, embedder: object) -> LibraryIndex:
        scan_calls.append((config, embedder))
        # The fake embedder MUST be the one the calibrate module passed --
        # this is the integration seam the lazy-import pattern protects.
        assert embedder is fake_embedder, (
            f"calibrate passed a different embedder than embed.embed_track "
            f"(got {embedder!r}); the monkey-patch contract is broken"
        )
        return index

    monkeypatch.setattr(library, "scan_library", fake_scan_library)

    # And capture the progress report calls (they should land at 0.0 / 0.3 / 0.9 / 1.0).
    reports: list[tuple[float, str]] = []

    # When running calibration
    result = run_calibration(cfg, lambda progress, message: reports.append((progress, message)))

    # Then: scan_library was called exactly once with the patched embedder
    assert len(scan_calls) == 1, f"scan_library called {len(scan_calls)} times, expected 1"
    assert scan_calls[0][0] is cfg, "scan_library received the wrong config"

    # And: the thresholds JSON was persisted at <data_dir>/thresholds.json
    thresholds_path = cfg.data_dir / _THRESHOLDS_FILE_NAME
    assert thresholds_path.is_file(), f"calibration did not persist thresholds.json at {thresholds_path}"

    # And: the returned dict matches what's in the file
    loaded = load_thresholds(cfg.data_dir)
    assert loaded is not None, "persisted thresholds.json could not be reloaded"
    assert result == loaded, (
        f"run_calibration return value {result!r} does not match loaded thresholds.json {loaded!r}"
    )

    # And: progress was reported at the documented milestones
    progress_values = [p for p, _ in reports]
    assert progress_values[0] == 0.0, f"first progress must be 0.0, got {progress_values}"
    assert progress_values[-1] == 1.0, f"last progress must be 1.0, got {progress_values}"
    assert 0.3 in progress_values, f"0.3 milestone missing from progress reports: {progress_values}"
    assert 0.9 in progress_values, f"0.9 milestone missing from progress reports: {progress_values}"
    # And the human-readable message for the embedded count is plausible
    messages = [m for _, m in reports]
    assert any("8 tracks" in m for m in messages), (
        f"report message for the embedded-count milestone must mention the track count "
        f"(8), got messages {messages!r}"
    )


def test_run_calibration_returns_defaults_for_empty_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given an empty like-library (n=0), calibration returns defensive defaults and persists them."""

    cfg = _make_config(tmp_path)

    # Empty LibraryIndex (n=0): defensive defaults expected
    empty_vectors = np.empty((0, 512), dtype=np.float32)
    index = LibraryIndex(_vectors=empty_vectors, _manifest={})

    def fake_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        return np.zeros(512, dtype=np.float32)

    def fake_scan_library(_config: Config, _embedder: object) -> LibraryIndex:
        return index

    from taste_pipeline import embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "scan_library", fake_scan_library)

    result = run_calibration(cfg, lambda _p, _m: None)

    # Defensive defaults are returned and persisted.
    assert result == {
        "keep_threshold": _DEFAULT_KEEP_THRESHOLD,
        "skip_threshold": _DEFAULT_SKIP_THRESHOLD,
    }, f"empty library must yield defensive defaults, got {result}"
    persisted = load_thresholds(cfg.data_dir)
    assert persisted is not None
    assert persisted == result, f"persisted thresholds {persisted!r} do not match returned {result!r}"


def test_run_calibration_propagates_embedder_shape_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration re-raises scan_library's ValueError when the embedder shape is wrong.

    The runner converts any exception into ``status="failed"``; this test
    proves the calibrate module does NOT swallow ValueError -- the user
    sees a clear error message instead of a silent zero-track index.
    """
    import pytest  # noqa: PLC0415 -- fixture import

    cfg = _make_config(tmp_path)

    def fake_embedder(_path: Path, _config: ConfigType) -> np.ndarray:
        return np.zeros(512, dtype=np.float32)

    def fake_scan_library(_config: Config, _embedder: object) -> LibraryIndex:
        message = "embedder returned shape (513,) for /tmp/foo.flac, expected (512,)"
        raise ValueError(message)

    from taste_pipeline import embed, library  # noqa: PLC0415 -- lazy: under test

    monkeypatch.setattr(embed, "embed_track", fake_embedder)
    monkeypatch.setattr(library, "scan_library", fake_scan_library)

    with pytest.raises(ValueError, match="embedder returned shape"):
        run_calibration(cfg, lambda _p, _m: None)

    # And no thresholds.json was written (no partial state)
    assert not (cfg.data_dir / _THRESHOLDS_FILE_NAME).exists(), (
        "calibration must not persist thresholds when scan_library raised"
    )


def test_calibrate_module_does_not_import_heavy_dependencies_at_module_load() -> None:
    """Guardrail: importing calibrate must NOT load transformers or torch at module import.

    The dashboard hits this module's helpers on the index-status read path
    (load_thresholds). A module-level import of embed / library would pull
    transformers + torch into the request handler's import set, which is
    unacceptable for a fast read-only endpoint.
    """
    # The module namespace carries only the names the module itself
    # defines; importing embed or library would have populated the
    # module's namespace with their top-level symbols.
    module_vars = vars(calibrate_module)
    forbidden = {"transformers", "torch"}
    for name in forbidden:
        assert name not in module_vars, (
            f"calibrate module imported {name!r} at module load; "
            f"this loads the CLAP model on the dashboard's index-status read path"
        )
