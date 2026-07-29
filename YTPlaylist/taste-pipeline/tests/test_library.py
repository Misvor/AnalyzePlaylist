"""Tests for taste_pipeline.library: incremental scan, vector reuse, re-embed, and deletion purge."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from taste_pipeline.config import Config
from taste_pipeline.library import scan_library

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
