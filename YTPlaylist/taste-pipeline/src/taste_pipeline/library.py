"""Like-library embedding index: incremental scan with deletion purge.

``scan_library`` walks ``config.like_library_dir`` recursively for audio
files (``.flac``/``.mp3``/``.m4a``/``.wav``/``.ogg``, case-insensitive) and
maintains two artifacts under ``<data_dir>/index``:

- ``library_manifest.json``: relative path -> ``{mtime, size, vector_id}``
- ``library_vectors.npy``: float32 ``(n, 512)`` matrix, row ``vector_id``
  belongs to the manifest entry with the same id.

Re-scans are incremental: a file whose path, mtime, and size are unchanged
keeps its stored vector; new or changed files are embedded through the
injected ``embedder`` callable; rows for deleted files are purged from both
artifacts. The embedder is injectable so this module never imports
``taste_pipeline.embed`` (built in parallel); the pipeline passes
``embed.embed_track`` at call time. The like library itself is read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from taste_pipeline.config import Config

_EMBED_DIM: Final[int] = 512
_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({".flac", ".mp3", ".m4a", ".wav", ".ogg"})
_INDEX_SUBDIR: Final[str] = "index"
_MANIFEST_NAME: Final[str] = "library_manifest.json"
_VECTORS_NAME: Final[str] = "library_vectors.npy"

ManifestEntry = dict[str, float | int]
Manifest = dict[str, ManifestEntry]


@dataclass(frozen=True, slots=True, eq=False)
class LibraryIndex:
    """Immutable snapshot of the like-library embedding index after a scan."""

    _vectors: np.ndarray = field(repr=False)
    _manifest: Manifest = field(repr=False)

    def vectors(self) -> np.ndarray:
        """The float32 ``(count, 512)`` embedding matrix, aligned with :meth:`manifest` vector_ids."""
        return self._vectors

    def count(self) -> int:
        """Number of indexed tracks (rows in the vector matrix)."""
        return int(self._vectors.shape[0])

    def manifest(self) -> Manifest:
        """A copy of the manifest mapping relative path -> ``{mtime, size, vector_id}``."""
        return {rel_path: dict(entry) for rel_path, entry in self._manifest.items()}


def scan_library(config: Config, embedder: Callable[[Path, Config], np.ndarray]) -> LibraryIndex:
    """Scan the like library and incrementally refresh its embedding index.

    Args:
        config: Pipeline configuration; reads ``like_library_dir`` (never
            written to) and persists the index under ``data_dir / "index"``.
        embedder: Callable embedding one audio file, signature
            ``(path, config) -> np.ndarray`` of shape ``(512,)``.

    Returns:
        The refreshed :class:`LibraryIndex`.

    Raises:
        ValueError: The embedder returned a vector that is not 512-dimensional.
    """
    index_dir = config.data_dir / _INDEX_SUBDIR
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / _MANIFEST_NAME
    vectors_path = index_dir / _VECTORS_NAME

    old_manifest, old_vectors = _load_index(manifest_path, vectors_path)
    scanned = _scan_audio_files(config.like_library_dir)

    rows: list[np.ndarray] = []
    manifest: Manifest = {}
    for rel_path, (mtime, size) in scanned.items():
        entry = old_manifest.get(rel_path)
        if entry is not None and entry["mtime"] == mtime and entry["size"] == size:
            row = np.asarray(old_vectors[int(entry["vector_id"])], dtype=np.float32)
        else:
            row = _embed_one(embedder, config.like_library_dir / rel_path, config)
        manifest[rel_path] = {"mtime": mtime, "size": size, "vector_id": len(rows)}
        rows.append(row)

    vectors = np.stack(rows) if rows else np.empty((0, _EMBED_DIM), dtype=np.float32)
    np.save(vectors_path, vectors)
    _ = manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return LibraryIndex(_vectors=vectors, _manifest=manifest)


def _scan_audio_files(root: Path) -> dict[str, tuple[float, int]]:
    """Walk ``root`` recursively; map POSIX relative path -> (mtime, size) for audio files, sorted by path."""
    found: dict[str, tuple[float, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in _AUDIO_EXTENSIONS:
            continue
        stat = path.stat()
        found[path.relative_to(root).as_posix()] = (stat.st_mtime, stat.st_size)
    return found


def _load_index(manifest_path: Path, vectors_path: Path) -> tuple[Manifest, np.ndarray]:
    """Load the persisted manifest and vectors; an incomplete pair counts as an empty index."""
    if not (manifest_path.is_file() and vectors_path.is_file()):
        return {}, np.empty((0, _EMBED_DIM), dtype=np.float32)
    manifest = cast("Manifest", json.loads(manifest_path.read_text(encoding="utf-8")))
    vectors = np.asarray(np.load(vectors_path), dtype=np.float32)
    return manifest, vectors


def _embed_one(embedder: Callable[[Path, Config], np.ndarray], path: Path, config: Config) -> np.ndarray:
    """Embed one audio file; coerce to float32 and require shape ``(512,)``."""
    vector = np.asarray(embedder(path, config), dtype=np.float32)
    if vector.shape != (_EMBED_DIM,):
        message = f"embedder returned shape {vector.shape} for {path}, expected ({_EMBED_DIM},)"
        raise ValueError(message)
    return vector
