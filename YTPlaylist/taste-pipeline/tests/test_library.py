"""Tests for taste_pipeline.library: incremental scan, vector reuse, re-embed, and deletion purge."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from taste_pipeline import library as library_module
from taste_pipeline.config import Config
from taste_pipeline.library import ScanProgress, load_index, scan_library

if TYPE_CHECKING:
    from pathlib import Path

_EMBED_DIM = 512


def _make_config(tmp_path: Path) -> Config:
    """A Config with an empty like library and a data dir under ``tmp_path``."""
    like_dir = tmp_path / "like"
    like_dir.mkdir()
    data_dir = tmp_path / "data"
    (data_dir / "index").mkdir(parents=True)
    return Config(
        like_library_dir=like_dir,
        data_dir=data_dir,
        cookie_file=tmp_path / "cookies.txt",
        download_archive=data_dir / "yt-dlp-archive.txt",
    )


def _write_audio(root: Path, rel_path: str) -> Path:
    """Write a fake audio file under ``root`` (creating parents) and return its path."""
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake audio: " + rel_path.encode("utf-8"))
    return path


def _fake_embedder(path: Path, config: Config) -> np.ndarray:
    """Deterministic offline embedder: a seeded 512-dim vector derived from the path."""
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed).standard_normal(_EMBED_DIM).astype(np.float32)


class _CountingEmbedder:
    """Wraps the fake embedder and records every path it is asked to embed."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path: Path, config: Config) -> np.ndarray:
        self.calls.append(path)
        return _fake_embedder(path, config)


def test_scan_empty_library_produces_empty_index(tmp_path: Path) -> None:
    # Given a like library with no files at all
    config = _make_config(tmp_path)

    # When scanning
    index = scan_library(config, _fake_embedder)

    # Then the index is empty but well-formed
    assert index.count() == 0
    assert index.vectors().shape == (0, _EMBED_DIM)
    assert index.vectors().dtype == np.float32
    assert index.manifest() == {}


def test_scan_three_audio_files_produces_three_rows(tmp_path: Path) -> None:
    # Given three audio files, one nested and one with an uppercase extension
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    _write_audio(like_dir, "a.mp3")
    _write_audio(like_dir, "sub/b.FLAC")
    _write_audio(like_dir, "c.wav")

    # When scanning
    index = scan_library(config, _fake_embedder)

    # Then every file is indexed once, keyed by POSIX relative path
    assert index.count() == 3
    assert index.vectors().shape == (3, _EMBED_DIM)
    manifest = index.manifest()
    assert set(manifest) == {"a.mp3", "c.wav", "sub/b.FLAC"}
    assert sorted(entry["vector_id"] for entry in manifest.values()) == [0, 1, 2]
    for rel_path, entry in manifest.items():
        stat = (like_dir / rel_path).stat()
        assert entry["mtime"] == stat.st_mtime
        assert entry["size"] == stat.st_size


def test_non_audio_files_are_ignored(tmp_path: Path) -> None:
    # Given a mix of audio and non-audio files in the library
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    _write_audio(like_dir, "track.ogg")
    _write_audio(like_dir, "notes.txt")
    _write_audio(like_dir, "art/cover.jpg")

    # When scanning
    index = scan_library(config, _fake_embedder)

    # Then only the audio file is indexed
    assert index.count() == 1
    assert set(index.manifest()) == {"track.ogg"}


def test_rescan_unchanged_library_reuses_vectors(tmp_path: Path) -> None:
    # Given a library that has already been scanned once
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.m4a", "c.flac"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()
    first = scan_library(config, embedder)
    assert len(embedder.calls) == 3

    # When rescanning without any changes
    embedder.calls.clear()
    second = scan_library(config, embedder)

    # Then nothing is re-embedded and the stored vectors are identical
    assert embedder.calls == []
    assert second.count() == 3
    assert np.array_equal(second.vectors(), first.vectors())
    assert second.manifest() == first.manifest()


def test_added_file_is_the_only_row_embedded(tmp_path: Path) -> None:
    # Given a scanned library of three files
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.m4a", "c.flac"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()
    first = scan_library(config, embedder)

    # When a fourth file appears and the library is rescanned
    new_path = _write_audio(like_dir, "d.wav")
    embedder.calls.clear()
    second = scan_library(config, embedder)

    # Then exactly the new file is embedded and the row count grows by one
    assert embedder.calls == [new_path]
    assert second.count() == first.count() + 1 == 4
    assert set(second.manifest()) == {"a.mp3", "b.m4a", "c.flac", "d.wav"}


def test_changed_mtime_triggers_reembed_of_that_file(tmp_path: Path) -> None:
    # Given a scanned library
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.m4a", "c.flac"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()
    first = scan_library(config, embedder)

    # When one file's mtime changes (same content, fresh timestamp)
    changed = like_dir / "b.m4a"
    stat = changed.stat()
    os.utime(changed, (stat.st_atime, stat.st_mtime + 10))
    embedder.calls.clear()
    second = scan_library(config, embedder)

    # Then only that file is re-embedded; the row count is unchanged
    assert embedder.calls == [changed]
    assert second.count() == 3
    assert second.manifest()["b.m4a"]["mtime"] != first.manifest()["b.m4a"]["mtime"]


def test_deleted_file_is_purged_from_manifest_and_vectors(tmp_path: Path) -> None:
    # Given a scanned library of three files
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.m4a", "c.flac"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()
    first = scan_library(config, embedder)
    assert first.count() == 3

    # When one file is deleted and the library is rescanned
    (like_dir / "b.m4a").unlink()
    embedder.calls.clear()
    second = scan_library(config, embedder)

    # Then its row is purged: no re-embedding, one fewer vector, no manifest entry
    assert embedder.calls == []
    assert second.count() == 2
    assert second.vectors().shape == (2, _EMBED_DIM)
    assert set(second.manifest()) == {"a.mp3", "c.flac"}
    persisted = json.loads((config.data_dir / "index" / "library_manifest.json").read_text(encoding="utf-8"))
    assert set(persisted) == {"a.mp3", "c.flac"}
    persisted_vectors = np.load(config.data_dir / "index" / "library_vectors.npy")
    assert persisted_vectors.shape == (2, _EMBED_DIM)


def test_scan_writes_only_under_data_dir(tmp_path: Path) -> None:
    # Given a library with audio files
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "sub/b.flac"):
        _write_audio(like_dir, name)
    before = sorted(p.relative_to(like_dir).as_posix() for p in like_dir.rglob("*"))

    # When scanning
    scan_library(config, _fake_embedder)

    # Then the index artifacts exist under data_dir and the library is untouched
    assert (config.data_dir / "index" / "library_manifest.json").is_file()
    assert (config.data_dir / "index" / "library_vectors.npy").is_file()
    after = sorted(p.relative_to(like_dir).as_posix() for p in like_dir.rglob("*"))
    assert after == before


def test_embedder_wrong_shape_raises_value_error(tmp_path: Path) -> None:
    # Given an embedder that returns a vector of the wrong dimensionality
    config = _make_config(tmp_path)
    _write_audio(config.like_library_dir, "a.mp3")

    def bad_embedder(path: Path, cfg: Config) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    # When scanning, the mismatch is reported rather than silently stored
    with pytest.raises(ValueError, match="expected"):
        scan_library(config, bad_embedder)


def test_scan_reports_scan_progress_once_per_file(tmp_path: Path) -> None:
    # Given a library with three audio files
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.flac", "c.wav"):
        _write_audio(like_dir, name)
    records: list[ScanProgress] = []

    # When scanning with an on_progress callback
    scan_library(config, _fake_embedder, on_progress=records.append)

    # Then the callback fired once per file, at the START of each embed:
    # `processed` counts finished files (0-based), `current` names the file
    # about to be embedded, and `remaining` still includes it.
    assert len(records) == 3, f"expected one progress record per file, got {records!r}"
    assert [record.processed for record in records] == [0, 1, 2]
    assert [record.current for record in records] == ["a.mp3", "b.flac", "c.wav"]
    assert all(record.total == 3 for record in records)
    assert [record.remaining for record in records] == [3, 2, 1]
    # `upcoming` holds the POSIX rel paths after `current`, in scan order.
    assert records[0].upcoming == ("b.flac", "c.wav")
    assert records[1].upcoming == ("c.wav",)
    assert records[2].upcoming == ()


def test_scan_upcoming_is_capped_at_ten(tmp_path: Path) -> None:
    # Given a library with twelve audio files (more than the cap)
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for index in range(12):
        _write_audio(like_dir, f"t{index:02d}.mp3")
    records: list[ScanProgress] = []

    # When scanning with an on_progress callback
    scan_library(config, _fake_embedder, on_progress=records.append)

    # Then the first record's upcoming list is capped at 10 while `remaining`
    # still reports the true count, and the tail of the scan drains to empty.
    assert len(records) == 12
    first = records[0]
    assert first.processed == 0
    assert first.current == "t00.mp3"
    assert first.remaining == 12
    assert len(first.upcoming) == 10
    assert first.upcoming[0] == "t01.mp3"
    assert first.upcoming[-1] == "t10.mp3"
    assert records[-1].upcoming == ()


def test_scan_empty_library_does_not_report_progress(tmp_path: Path) -> None:
    # Given a library with no audio files
    config = _make_config(tmp_path)
    records: list[ScanProgress] = []

    # When scanning with an on_progress callback
    scan_library(config, _fake_embedder, on_progress=records.append)

    # Then the callback never fires (there are no files to process)
    assert records == []


def test_scan_checkpoints_are_readable_via_load_index(tmp_path: Path) -> None:
    # Given a library with three audio files
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.flac", "c.wav"):
        _write_audio(like_dir, name)

    # When scanning
    scanned = scan_library(config, _fake_embedder)

    # Then the checkpoint artifacts exist and load_index returns the same index
    assert (config.data_dir / "index" / "library_manifest.json").is_file()
    assert (config.data_dir / "index" / "library_vectors.npy").is_file()
    loaded = load_index(config)
    assert loaded.count() == scanned.count() == 3
    assert np.array_equal(loaded.vectors(), scanned.vectors())
    assert loaded.manifest() == scanned.manifest()
    assert loaded.ids() == scanned.ids()


def test_load_index_missing_artifacts_returns_empty(tmp_path: Path) -> None:
    # Given a config whose index directory has no artifacts
    config = _make_config(tmp_path)

    # When loading the index
    index = load_index(config)

    # Then an empty, well-formed index is returned (no walk, no error)
    assert index.count() == 0
    assert index.vectors().shape == (0, _EMBED_DIM)
    assert index.manifest() == {}
    assert index.ids() == []
    assert index.complete is True


def test_load_index_never_calls_the_embedder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a persisted index
    config = _make_config(tmp_path)
    _write_audio(config.like_library_dir, "a.mp3")
    scan_library(config, _fake_embedder)

    # When the embedder path is poisoned and load_index is called
    def exploding_embedder(*_args: object, **_kwargs: object) -> np.ndarray:
        message = "load_index must never embed"
        raise AssertionError(message)

    monkeypatch.setattr("taste_pipeline.library._embed_one", exploding_embedder)

    # Then load_index still succeeds without touching the embedder
    index = load_index(config)
    assert index.count() == 1


def test_resume_after_max_new_files_embeds_only_remaining(tmp_path: Path) -> None:
    # Given a four-file library, of which the first two were indexed in a prior batch
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.mp3", "c.mp3", "d.mp3"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()
    first = scan_library(config, embedder, max_new_files=2)
    assert len(embedder.calls) == 2
    assert first.complete is False

    # When the scan is resumed with another two-file batch
    embedder.calls.clear()
    second = scan_library(config, embedder, max_new_files=2)

    # Then the already-indexed files are reused and only the remaining two embedded
    assert [path.name for path in embedder.calls] == ["c.mp3", "d.mp3"]
    assert second.count() == 4
    # And ids() are in row (walk) order, not re-sorted manifest order
    assert second.ids() == ["a.mp3", "b.mp3", "c.mp3", "d.mp3"]


def test_should_stop_returns_partial_index_that_is_readable(tmp_path: Path) -> None:
    # Given a four-file library and a stop signal that fires after two files
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.mp3", "c.mp3", "d.mp3"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()
    calls = {"count": 0}

    def should_stop() -> bool:
        calls["count"] += 1
        return calls["count"] >= 2

    # When scanning with the stop signal
    index = scan_library(config, embedder, should_stop=should_stop)

    # Then the partial index is returned with complete=False ...
    assert index.complete is False
    assert index.count() == 2
    # ... and it was checkpointed: load_index sees the same two rows
    reloaded = load_index(config)
    assert reloaded.count() == 2
    assert reloaded.ids() == index.ids() == ["a.mp3", "b.mp3"]


def test_max_new_files_returns_partial_index_that_is_readable(tmp_path: Path) -> None:
    # Given a four-file library and a batch cap of two
    config = _make_config(tmp_path)
    like_dir = config.like_library_dir
    for name in ("a.mp3", "b.mp3", "c.mp3", "d.mp3"):
        _write_audio(like_dir, name)
    embedder = _CountingEmbedder()

    # When scanning with max_new_files=2
    index = scan_library(config, embedder, max_new_files=2)

    # Then exactly two files were embedded, the rest waiting for the next batch
    assert len(embedder.calls) == 2
    assert index.count() == 2
    assert index.complete is False
    assert load_index(config).count() == 2


def test_torn_checkpoint_truncates_orphan_vector_rows(tmp_path: Path) -> None:
    # Given a torn checkpoint: the vectors file has MORE rows than the manifest
    # (vectors-first publish means the vector tail is always unreferenced).
    config = _make_config(tmp_path)
    index_dir = config.data_dir / "index"
    manifest = {
        "a.mp3": {"mtime": 1.0, "size": 10, "vector_id": 0},
        "b.mp3": {"mtime": 2.0, "size": 20, "vector_id": 1},
    }
    (index_dir / "library_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    orphan_tail = np.arange(3 * _EMBED_DIM, dtype=np.float32).reshape(3, _EMBED_DIM)
    np.save(index_dir / "library_vectors.npy", orphan_tail)

    # When loading the index
    loaded = load_index(config)

    # Then it recovers by truncating the orphan tail to the manifest length
    assert loaded.count() == 2
    assert loaded.ids() == ["a.mp3", "b.mp3"]
    assert np.array_equal(loaded.vectors(), orphan_tail[:2])


def test_load_index_returns_empty_when_vectors_short_of_manifest(tmp_path: Path) -> None:
    # Given an unrecoverable pair: fewer vector rows than manifest entries
    config = _make_config(tmp_path)
    index_dir = config.data_dir / "index"
    manifest = {
        "a.mp3": {"mtime": 1.0, "size": 10, "vector_id": 0},
        "b.mp3": {"mtime": 2.0, "size": 20, "vector_id": 1},
    }
    (index_dir / "library_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(index_dir / "library_vectors.npy", np.zeros((1, _EMBED_DIM), dtype=np.float32))

    # When loading
    loaded = load_index(config)

    # Then the index is reported empty rather than returning garbage
    assert loaded.count() == 0


def test_opus_files_are_indexed(tmp_path: Path) -> None:
    # Given a .opus file (routes_index counts it, so library must agree)
    config = _make_config(tmp_path)
    _write_audio(config.like_library_dir, "track.opus")

    # When scanning
    index = scan_library(config, _fake_embedder)

    # Then it is indexed like any other audio file
    assert ".opus" in library_module._AUDIO_EXTENSIONS
    assert index.count() == 1
    assert index.ids() == ["track.opus"]
