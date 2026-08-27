"""Read-only library index status endpoint for the taste-pipeline web UI.

Exposes the on-disk state of the like-library embedding index at
``GET /api/index`` without loading the CLAP model or scanning the
library. The dashboard renders this server-side on initial page load
so the user can see how fresh their index is and how many tracks it
contains.

Response shape (locked by ``tests/test_api_index.py``):

- ``count``: ``len(manifest)`` when ``<data_dir>/index/library_manifest.json``
  is present and valid JSON, else ``0``. Malformed manifests yield
  ``count == 0`` (fail-open) so a corrupt file does not break the UI.
- ``last_scan``: ISO-8601 UTC string of the manifest file's mtime when
  the manifest exists, else ``None``.
- ``like_library_path``: ``str(config.like_library_dir)``.
- ``like_library_track_count``: recursive count of audio files
  (``.flac``/``.mp3``/``.opus``/``.m4a``/``.ogg``, case-insensitive)
  under ``config.like_library_dir``.

This module intentionally has no dependency on ``taste_pipeline.embed``
or ``taste_pipeline.library.scan_library`` -- loading the CLAP model on
a status read would be wasteful and the dashboard needs to know the
"never scanned yet" case (``count == 0, last_scan is None``) which the
scan code path does not naturally express.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

from fastapi import APIRouter, FastAPI, Request

if TYPE_CHECKING:
    from pathlib import Path

    from taste_pipeline.config import Config

router = APIRouter(prefix="/api")

# Layout constants kept in sync with ``taste_pipeline.library`` so the
# read path does not need to import that module. If the layout ever
# changes, the library integration tests will catch drift.
_INDEX_SUBDIR: Final[str] = "index"
_MANIFEST_NAME: Final[str] = "library_manifest.json"
_AUDIO_EXTENSIONS: Final[frozenset[str]] = frozenset({".flac", ".mp3", ".opus", ".m4a", ".ogg"})


def _config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.config`` to a typed ``Config`` for basedpyright."""
    return cast("Config", cast("FastAPI", request.app).state.config)


def build_index_status(config: Config) -> dict[str, object]:
    """Compute the index-status payload for ``config`` without going through FastAPI.

    Exposed as a module-level helper so the dashboard's ``GET /`` handler
    can render the panel server-side on page load with the same payload
    the ``/api/index`` JSON endpoint returns. Keeping the computation in
    one place means a future change to the response shape is a single
    edit, not a coordinated update to two handlers.
    """
    manifest_path = config.data_dir / _INDEX_SUBDIR / _MANIFEST_NAME
    return {
        "count": _read_manifest_count(manifest_path),
        "last_scan": _manifest_mtime_iso(manifest_path),
        "like_library_path": str(config.like_library_dir),
        "like_library_track_count": _count_audio_files(config.like_library_dir),
    }


def _read_manifest_count(manifest_path: Path) -> int:
    """Return ``len(json.loads(manifest_path))`` or ``0`` on any read/parse failure.

    Fail-open: a missing or malformed manifest reports ``count == 0``
    (which matches the "never scanned yet" UI state) instead of raising
    500. Manifest entries map ``rel_path -> {mtime, size, vector_id}``,
    so the count is the number of indexed tracks.
    """
    if not manifest_path.is_file():
        return 0
    try:
        loaded = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(loaded, dict):
        return 0
    return len(cast("dict[str, object]", loaded))


def _manifest_mtime_iso(manifest_path: Path) -> str | None:
    """Return the manifest's mtime as an ISO-8601 UTC string, or ``None`` when the manifest is absent."""
    if not manifest_path.is_file():
        return None
    mtime = manifest_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=UTC).isoformat()


def _count_audio_files(root: Path) -> int:
    """Recursively count files under ``root`` whose suffix (case-insensitive) is in ``_AUDIO_EXTENSIONS``.

    Missing directories count as ``0`` rather than raising -- the
    handler is read-only and must not crash on a freshly-built config
    whose ``like_library_dir`` exists but is empty (or whose ``rglob``
    race-loses the first scan).
    """
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.casefold() in _AUDIO_EXTENSIONS)


@router.get("/index")
async def get_index_status(request: Request) -> dict[str, object]:
    """Return the current library index status as JSON (read-only, no model loading).

    The response is shaped exactly as ``tests/test_api_index.py`` asserts:

    - ``count`` -- ``len(manifest)`` if the manifest file exists and
      parses as a JSON object, else ``0``.
    - ``last_scan`` -- ISO-8601 UTC timestamp of the manifest's mtime,
      or ``None`` when the manifest does not exist.
    - ``like_library_path`` -- ``str(config.like_library_dir)``.
    - ``like_library_track_count`` -- recursive count of audio files in
      ``config.like_library_dir`` (``_AUDIO_EXTENSIONS``, case-insensitive).
    """
    cfg = _config_from_request(request)
    return build_index_status(cfg)
