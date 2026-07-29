"""Subscriptions feed pass-A: flat listing with client-side windowing and dedupe.

yt-dlp performs the flat ``:ytsubs`` extraction and drops shorts/live entries
via ``match_filter``; the date window and the seen-history dedupe run in this
module because ``--dateafter`` is a no-op on flat entries and the download
archive is download-time state (read-only during listing).
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from typing_extensions import override
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, match_filter_func

from taste_pipeline.config import Config
from taste_pipeline.state import StateStore

_SUBSCRIPTIONS_FEED: Final[str] = ":ytsubs"
_MATCH_FILTER: Final[str] = "!is_live & !post_live & !was_live & original_url!*=/shorts/"
_BOT_CHECK_MARKER: Final[str] = "Sign in to confirm"
_SECONDS_PER_DAY: Final[int] = 86400


@dataclass(frozen=True, slots=True)
class CookieExpiredError(Exception):
    """Raised when YouTube answers the subscriptions listing with a login/bot check."""

    detail: str

    @override
    def __str__(self) -> str:
        """Render the failure with a pointer to the cookie re-export runbook."""
        return (
            "YouTube rejected the subscriptions listing with a login/bot check "
            f"({self.detail}). The cookie file has likely expired; re-export "
            "cookies.txt as described in docs/cookies.md and retry."
        )


@dataclass(frozen=True, slots=True)
class FeedItem:
    """One flat subscriptions-feed entry, normalized to the fields pass A needs."""

    video_id: str
    title: str | None
    uploader: str | None
    channel_id: str | None
    url: str | None
    timestamp: int | None
    duration: float | None


def _ydl_options(config: Config) -> dict[str, object]:
    """Build the yt-dlp parameters for a read-only flat listing of the subscriptions feed."""
    return {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "cookiefile": str(config.cookie_file),
        "extractor_args": {"youtubetab": ["approximate_date"]},
        "playlistend": config.max_feed_items,
        "match_filter": match_filter_func(_MATCH_FILTER),
        "quiet": True,
        "no_warnings": True,
    }


def _opt_str(value: object) -> str | None:
    """Return ``value`` when it is a string, else None (missing/garbled flat fields)."""
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


def _to_feed_item(entry: Mapping[str, object]) -> FeedItem | None:
    """Normalize one flat entry; entries without a usable video id are dropped."""
    video_id = _opt_str(entry.get("id"))
    if not video_id:
        return None
    return FeedItem(
        video_id=video_id,
        title=_opt_str(entry.get("title")),
        uploader=_opt_str(entry.get("uploader")),
        channel_id=_opt_str(entry.get("channel_id")),
        url=_opt_str(entry.get("url")),
        timestamp=_opt_int(entry.get("timestamp")),
        duration=_opt_float(entry.get("duration")),
    )


def fetch_feed(config: Config, state: StateStore) -> list[FeedItem]:
    """List the subscriptions feed and return the items that are new and inside the window.

    yt-dlp drops live/shorts entries during extraction (``match_filter``);
    this function then drops entries older than ``config.feed_window_days``
    (entries without a timestamp are kept, fail-open — they are re-checked on
    the next run) and entries already present in the seen-history. Every
    listed id is recorded with stage ``listed``, so a re-run of the same feed
    yields zero new items.

    Args:
        config: Pipeline configuration (cookie file, window, listing bound).
        state: Seen-history store shared with the later passes.

    Returns:
        New in-window feed items, in feed order (newest first).

    Raises:
        CookieExpiredError: yt-dlp hit YouTube's login/bot check.
        DownloadError: Any other yt-dlp extraction failure.
    """
    try:
        # typeshed's yt-dlp stub mistypes two params we rely on (skip_download as
        # str, extractor_args values as Mapping); yt-dlp's runtime and docs take
        # a bool / list-of-args, so the options dict is built with the real values.
        with YoutubeDL(_ydl_options(config)) as ydl:  # pyright: ignore[reportArgumentType]
            info = ydl.extract_info(_SUBSCRIPTIONS_FEED, download=False)
    except DownloadError as exc:
        if _BOT_CHECK_MARKER in str(exc):
            raise CookieExpiredError(detail=str(exc)) from exc
        raise
    entries = info.get("entries") or []
    items = [item for entry in entries if (item := _to_feed_item(entry)) is not None]
    fresh = [item for item in items if not state.already_seen(item.video_id)]
    for item in items:
        _ = state.record_seen(item.video_id, stage="listed")
    cutoff = time.time() - config.feed_window_days * _SECONDS_PER_DAY
    return [item for item in fresh if item.timestamp is None or item.timestamp >= cutoff]
