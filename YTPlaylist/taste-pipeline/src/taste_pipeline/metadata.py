"""Metadata pass-B: bounded per-video full-metadata fetch.

Pass A only sees yt-dlp's flat feed entries; pass B re-extracts each new
video individually to get the full info dict (Art Track ``track``/``artist``/
``album``, categories, description). The fetch is bounded by
``config.max_metadata_fetch`` per run because each video is one network
round-trip; ids beyond the bound stay at stage ``listed`` and are picked up
on the next run. Private/deleted videos fail with ``DownloadError`` and are
skipped so one bad id cannot abort the batch.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Final

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from taste_pipeline.config import Config
from taste_pipeline.state import StateStore

_LOGGER: Final = logging.getLogger(__name__)
_WATCH_URL: Final[str] = "https://www.youtube.com/watch?v={video_id}"


@dataclass(frozen=True, slots=True)
class TrackMetadata:
    """Full per-video metadata, normalized to the fields the taste pipeline needs."""

    video_id: str
    title: str | None
    uploader: str | None
    uploader_id: str | None
    channel_id: str | None
    upload_date: str | None
    timestamp: int | None
    categories: list[str] | None
    duration: float | None
    description: str | None
    track: str | None
    artist: str | None
    album: str | None
    thumbnail_url: str | None
    webpage_url: str | None


def metadata_to_json(meta: TrackMetadata) -> dict[str, object]:
    """Serialize ``meta`` to a plain JSON-compatible dict."""
    return asdict(meta)


def metadata_from_json(raw: Mapping[str, object]) -> TrackMetadata:
    """Rebuild a :class:`TrackMetadata` from a dict produced by :func:`metadata_to_json`."""
    return TrackMetadata(**dict(raw))  # pyright: ignore[reportArgumentType]


def _ydl_options(config: Config) -> dict[str, object]:
    """Build the yt-dlp parameters for a read-only full extraction of one video."""
    return {
        "skip_download": True,
        "cookiefile": str(config.cookie_file),
        "proxy": config.proxy,
        "quiet": True,
        "no_warnings": True,
    }


def _opt_str(value: object) -> str | None:
    """Return ``value`` when it is a string, else None (missing/garbled fields)."""
    return value if isinstance(value, str) else None


def _opt_int(value: object) -> int | None:
    """Return ``value`` as an int when it is a real number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _opt_float(value: object) -> float | None:
    """Return ``value`` as a float when it is a real number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _opt_str_list(value: object) -> list[str] | None:
    """Return ``value`` as a list of strings when it is a list, else None."""
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str)]


def _to_track_metadata(info: Mapping[str, object]) -> TrackMetadata | None:
    """Normalize one full info dict; entries without a usable video id are dropped."""
    video_id = _opt_str(info.get("id"))
    if not video_id:
        return None
    return TrackMetadata(
        video_id=video_id,
        title=_opt_str(info.get("title")),
        uploader=_opt_str(info.get("uploader")),
        uploader_id=_opt_str(info.get("uploader_id")),
        channel_id=_opt_str(info.get("channel_id")),
        upload_date=_opt_str(info.get("upload_date")),
        timestamp=_opt_int(info.get("timestamp")),
        categories=_opt_str_list(info.get("categories")),
        duration=_opt_float(info.get("duration")),
        description=_opt_str(info.get("description")),
        track=_opt_str(info.get("track")),
        artist=_opt_str(info.get("artist")),
        album=_opt_str(info.get("album")),
        thumbnail_url=_opt_str(info.get("thumbnail")),
        webpage_url=_opt_str(info.get("webpage_url")),
    )


def fetch_metadata(video_ids: Sequence[str], config: Config, state: StateStore) -> list[TrackMetadata]:
    """Fetch full metadata for at most ``config.max_metadata_fetch`` ids.

    Input order is feed order (newest first) and is preserved in the result.
    Every successfully fetched id is recorded in the seen-history with stage
    ``metadata``; ids beyond the bound are left at stage ``listed`` for the
    next run. A video whose extraction fails with ``DownloadError`` (private,
    deleted, region-locked) is logged and skipped without aborting the batch.

    Args:
        video_ids: Candidate ids in feed order (newest first).
        config: Pipeline configuration (cookie file, fetch bound).
        state: Seen-history store shared with the other passes.

    Returns:
        Normalized metadata for the fetched videos, in input order.
    """
    bound = config.max_metadata_fetch
    selected = list(video_ids[:bound])
    deferred = len(video_ids) - len(selected)
    if deferred > 0:
        _LOGGER.info("deferring %d ids to next run (bound %d)", deferred, bound)
    # See feed.py: typeshed's yt-dlp stub mistypes params we rely on; the
    # options dict carries the runtime values yt-dlp documents.
    results: list[TrackMetadata] = []
    with YoutubeDL(_ydl_options(config)) as ydl:  # pyright: ignore[reportArgumentType]
        for video_id in selected:
            try:
                info = ydl.extract_info(_WATCH_URL.format(video_id=video_id), download=False)
            except DownloadError as exc:
                _LOGGER.warning("skipping %s: metadata fetch failed: %s", video_id, exc)
                continue
            meta = _to_track_metadata(info)
            if meta is None:
                _LOGGER.warning("skipping %s: extractor returned no usable video id", video_id)
                continue
            _ = state.record_seen(meta.video_id, stage="metadata")
            results.append(meta)
    return results
