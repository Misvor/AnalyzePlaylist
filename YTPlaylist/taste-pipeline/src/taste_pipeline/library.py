"""Like-library embedding index: incremental, checkpointed, and resumable scan.

``scan_library`` walks ``config.like_library_dir`` recursively for audio
files (``.flac``/``.mp3``/``.m4a``/``.wav``/``.ogg``/``.opus``,
case-insensitive) and maintains two artifacts under
``<data_dir>/index``:

- ``library_manifest.json``: relative path -> ``{mtime, size, vector_id}``
- ``library_vectors.npy``: float32 ``(n, 512)`` matrix, row ``vector_id``
  belongs to the manifest entry with the same id.

Re-scans are incremental: a file whose path, mtime, and size are unchanged
keeps its stored vector; new or changed files are embedded through the
injected ``embedder`` callable; rows for deleted files are purged from both
artifacts. The embedder is injectable so this module never imports
``taste_pipeline.embed`` (built in parallel); the pipeline passes
``embed.embed_track`` at call time. The like library itself is read-only.

The build is check-pointable: both artifacts are published atomically
(vectors first, manifest second) at a time/volume cadence, so a pause or
crash never loses completed work and a later scan resumes from the
manifest. :func:`load_index` reads the persisted pair without walking or
embedding, recovering from a torn checkpoint by truncating orphan vector
rows.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from taste_pipeline.config import Config

_EMBED_DIM: Final[int] = 512
_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({".flac", ".mp3", ".m4a", ".wav", ".ogg", ".opus"})
_INDEX_SUBDIR: Final[str] = "index"
_MANIFEST_NAME: Final[str] = "library_manifest.json"
_VECTORS_NAME: Final[str] = "library_vectors.npy"
_TMP_SUFFIX: Final[str] = ".tmp"
_MAX_UPCOMING: Final[int] = 10

# Checkpoint cadence: whichever of the two thresholds is reached first.
_CHECKPOINT_EVERY_N: Final[int] = 200

# Atomic-replace retry: on Windows a reader may hold the target file open,
# so ``os.replace`` can raise ``PermissionError`` transiently.
_REPLACE_ATTEMPTS: Final[int] = 5
_REPLACE_RETRY_SLEEP_S: Final[float] = 0.05

# Reader retry: a concurrent checkpoint publishes vectors before the
# manifest, so a short bounded retry lets the manifest catch up.
_READ_RETRIES: Final[int] = 5
_READ_RETRY_SLEEP_S: Final[float] = 0.01

ManifestEntry = dict[str, float | int]
Manifest = dict[str, ManifestEntry]


@dataclass(frozen=True, slots=True)
class ScanProgress:
    """One structured progress snapshot, emitted at the START of embedding each file.

    Attributes:
        processed: Files finished so far (``0`` on the first emit).
        total: Total audio files in this scan.
        current: POSIX rel path of the file about to be embedded.
        remaining: ``total - processed`` (includes the current file).
        upcoming: POSIX rel paths after ``current``, capped at
            :data:`_MAX_UPCOMING` (10).
    """

    processed: int
    total: int
    current: str
    remaining: int
    upcoming: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class LibraryIndex:
    """Immutable snapshot of the like-library embedding index.

    Attributes:
        complete: ``True`` when the scan covered the whole library; ``False``
            when it stopped early via ``should_stop`` / ``max_new_files``
            (the persisted pair is still a valid, resumable partial index).
    """

    _vectors: np.ndarray = field(repr=False)
    _manifest: Manifest = field(repr=False)
    complete: bool = True

    def vectors(self) -> np.ndarray:
        """The float32 ``(count, 512)`` embedding matrix, aligned with :meth:`ids` and vector_ids."""
        return self._vectors

    def count(self) -> int:
        """Number of indexed tracks (rows in the vector matrix)."""
        return int(self._vectors.shape[0])

    def manifest(self) -> Manifest:
        """A copy of the manifest mapping relative path -> ``{mtime, size, vector_id}``."""
        return {rel_path: dict(entry) for rel_path, entry in self._manifest.items()}

    def ids(self) -> list[str]:
        """Relative paths in row (``vector_id``) order, aligned with :meth:`vectors` rows."""
        return sorted(self._manifest, key=self._row_order_key)

    def _row_order_key(self, rel_path: str) -> int:
        """Sort key resolving a manifest entry to its ``vector_id`` row position."""
        return int(self._manifest[rel_path].get("vector_id", 0))


def load_index(config: Config) -> LibraryIndex:
    """Read the persisted index pair; never walks the library or embeds anything.

    A missing, corrupt, or unrecoverable pair yields an empty
    :class:`LibraryIndex` (0 rows). A torn checkpoint -- vectors published
    but the manifest not yet replaced -- is recovered by truncating the
    orphan vector tail, which is safe because those rows are unreferenced.

    Args:
        config: Pipeline configuration; reads the artifacts under
            ``config.data_dir / "index"``.

    Returns:
        The persisted :class:`LibraryIndex` (empty when unavailable).
    """
    index_dir = config.data_dir / _INDEX_SUBDIR
    manifest, vectors = _read_index_with_recovery(
        index_dir / _MANIFEST_NAME,
        index_dir / _VECTORS_NAME,
    )
    return LibraryIndex(_vectors=vectors, _manifest=manifest)


def scan_library(  # noqa: PLR0913 -- frozen resumable-scan interface (progress + stop + batch + cadence)
    config: Config,
    embedder: Callable[[Path, Config], np.ndarray],
    *,
    on_progress: Callable[[ScanProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_new_files: int | None = None,
    checkpoint_interval_s: float = 60.0,
) -> LibraryIndex:
    """Scan the like library and incrementally refresh its embedding index.

    The index is checkpointed atomically at each file boundary once either
    ``checkpoint_interval_s`` has elapsed or 200 new files have been
    embedded, plus a final checkpoint. A checkpoint is written AFTER the
    row and manifest entry are recorded, so stopping right after an embed
    still persists it.

    Args:
        config: Pipeline configuration; reads ``like_library_dir`` (never
            written to) and persists the index under ``data_dir / "index"``.
        embedder: Callable embedding one audio file, signature
            ``(path, config) -> np.ndarray`` of shape ``(512,)``.
        on_progress: Optional callback invoked once per scanned file, at
            the START of processing it, with a :class:`ScanProgress` record
            whose ``current`` names the file about to be embedded. Never
            called when there are no files. ``None`` (the default) disables
            progress reporting.
        should_stop: Optional callback checked at each file boundary; when
            it returns ``True`` the scan checkpoints and stops with a
            partial result (``complete`` is ``False``).
        max_new_files: Optional cap on the number of NEW/changed files
            embedded this run; once reached the scan checkpoints and stops
            with a partial result (``complete`` is ``False``). Cached files
            are never counted.
        checkpoint_interval_s: Minimum wall-clock seconds between
            checkpoints. Default ``60.0``.

    Returns:
        The refreshed :class:`LibraryIndex`; ``complete`` is ``False`` when
        the scan stopped early.

    Raises:
        ValueError: The embedder returned a vector that is not 512-dimensional.
    """
    index_dir = config.data_dir / _INDEX_SUBDIR
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / _MANIFEST_NAME
    vectors_path = index_dir / _VECTORS_NAME

    previous = load_index(config)
    old_manifest = previous.manifest()
    old_vectors = previous.vectors()
    scanned = _scan_audio_files(config.like_library_dir)

    rows: list[np.ndarray] = []
    manifest: Manifest = {}
    rel_paths = list(scanned)
    new_embeds = 0
    new_since_checkpoint = 0
    last_checkpoint = time.monotonic()
    stopped_early = False

    for index, rel_path in enumerate(rel_paths):
        if on_progress is not None:
            on_progress(_scan_progress(rel_paths, index))
        mtime, size = scanned[rel_path]
        entry = old_manifest.get(rel_path)
        if entry is not None and entry["mtime"] == mtime and entry["size"] == size:
            row = np.asarray(old_vectors[int(entry["vector_id"])], dtype=np.float32)
        else:
            row = _embed_one(embedder, config.like_library_dir / rel_path, config)
            new_embeds += 1
            new_since_checkpoint += 1
        # vector_id == row position: the manifest is rebuilt fresh each scan in
        # walk order, so every checkpoint renumbers in row order (deletion purge
        # is then free -- only files in the current walk survive).
        manifest[rel_path] = {"mtime": mtime, "size": size, "vector_id": len(rows)}
        rows.append(row)

        cadence_due = (
            time.monotonic() - last_checkpoint >= checkpoint_interval_s
            or new_since_checkpoint >= _CHECKPOINT_EVERY_N
        )
        stop_now = (should_stop is not None and should_stop()) or (
            max_new_files is not None and new_embeds >= max_new_files
        )
        if cadence_due or stop_now:
            _ = _checkpoint(vectors_path, manifest_path, rows, manifest)
            last_checkpoint = time.monotonic()
            new_since_checkpoint = 0
        if stop_now:
            stopped_early = True
            break

    vectors = _checkpoint(vectors_path, manifest_path, rows, manifest)
    return LibraryIndex(_vectors=vectors, _manifest=manifest, complete=not stopped_early)


def _scan_progress(files: list[str], index: int) -> ScanProgress:
    """Build the record emitted at the start of processing ``files[index]``."""
    total = len(files)
    return ScanProgress(
        processed=index,
        total=total,
        current=files[index],
        remaining=total - index,
        upcoming=tuple(files[index + 1 : index + 1 + _MAX_UPCOMING]),
    )


def _scan_audio_files(root: Path) -> dict[str, tuple[float, int]]:
    """Walk ``root`` recursively; map POSIX relative path -> (mtime, size) for audio files, sorted by path."""
    found: dict[str, tuple[float, int]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in _AUDIO_EXTENSIONS:
            continue
        stat = path.stat()
        found[path.relative_to(root).as_posix()] = (stat.st_mtime, stat.st_size)
    return found


def _empty_vectors() -> np.ndarray:
    """A well-formed ``(0, 512)`` float32 matrix for the empty index."""
    return np.empty((0, _EMBED_DIM), dtype=np.float32)


def _read_index_once(manifest_path: Path, vectors_path: Path) -> tuple[Manifest, np.ndarray]:
    """Read the persisted manifest and vectors once; a malformed/missing pair reads as empty."""
    if not (manifest_path.is_file() and vectors_path.is_file()):
        return {}, _empty_vectors()
    try:
        loaded = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
        raw = np.load(vectors_path)  # pyright: ignore[reportAny] -- numpy stub gap
    except (OSError, ValueError, json.JSONDecodeError):
        return {}, _empty_vectors()
    if not isinstance(loaded, dict):
        return {}, _empty_vectors()
    vectors = np.asarray(raw, dtype=np.float32)
    if vectors.shape[1:] != (_EMBED_DIM,):
        return {}, _empty_vectors()
    return cast("Manifest", loaded), vectors


def _index_is_consistent(manifest: Manifest, vectors: np.ndarray) -> bool:
    """True when ``len(manifest) == rows`` and the vector_ids are a permutation of ``0..n-1``."""
    row_count = int(vectors.shape[0])  # pyright: ignore[reportAny] -- numpy stub gap
    if len(manifest) != row_count:
        return False
    seen: set[int] = set()
    for entry in manifest.values():
        vector_id = int(entry.get("vector_id", -1))
        if vector_id < 0 or vector_id >= row_count or vector_id in seen:
            return False
        seen.add(vector_id)
    return len(seen) == row_count


def _read_index_with_recovery(manifest_path: Path, vectors_path: Path) -> tuple[Manifest, np.ndarray]:
    """Read + validate the pair with a bounded retry, then recover a torn vectors-first checkpoint."""
    manifest, vectors = _read_index_once(manifest_path, vectors_path)
    for _ in range(_READ_RETRIES - 1):
        if _index_is_consistent(manifest, vectors):
            return manifest, vectors
        time.sleep(_READ_RETRY_SLEEP_S)
        manifest, vectors = _read_index_once(manifest_path, vectors_path)
    if _index_is_consistent(manifest, vectors):
        return manifest, vectors
    # Vectors-first publish means a torn checkpoint leaves vectors a superset of
    # the manifest; the orphan tail is unreferenced, so truncation is safe.
    if vectors.shape[0] > len(manifest):
        truncated = vectors[: len(manifest)]
        if _index_is_consistent(manifest, truncated):
            return manifest, truncated
    return {}, _empty_vectors()


def _checkpoint(
    vectors_path: Path, manifest_path: Path, rows: list[np.ndarray], manifest: Manifest
) -> np.ndarray:
    """Publish a checkpoint atomically: vectors first, then manifest; returns the matrix."""
    vectors = np.stack(rows) if rows else _empty_vectors()
    _save_vectors(vectors_path, vectors)
    _save_manifest(manifest_path, manifest)
    return vectors


def _save_vectors(vectors_path: Path, vectors: np.ndarray) -> None:
    """Write ``vectors`` to a tmp file (file handle so numpy does not append ``.npy``) then replace."""
    tmp_path = vectors_path.with_name(vectors_path.name + _TMP_SUFFIX)
    with tmp_path.open("wb") as handle:
        np.save(handle, vectors)
    _replace_with_retry(tmp_path, vectors_path)


def _save_manifest(manifest_path: Path, manifest: Manifest) -> None:
    """Write the manifest to a tmp file then atomically replace the published one."""
    tmp_path = manifest_path.with_name(manifest_path.name + _TMP_SUFFIX)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _ = tmp_path.write_text(payload, encoding="utf-8")
    _replace_with_retry(tmp_path, manifest_path)


def _replace_with_retry(source: Path, target: Path) -> None:
    """Atomically replace ``target`` with ``source``, retrying on ``PermissionError`` (Windows reader)."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            _ = source.replace(target)
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_SLEEP_S)
        else:
            return


def _embed_one(embedder: Callable[[Path, Config], np.ndarray], path: Path, config: Config) -> np.ndarray:
    """Embed one audio file; coerce to float32 and require shape ``(512,)``."""
    vector = np.asarray(embedder(path, config), dtype=np.float32)
    if vector.shape != (_EMBED_DIM,):
        message = f"embedder returned shape {vector.shape} for {path}, expected ({_EMBED_DIM},)"
        raise ValueError(message)
    return vector
