"""Tests for taste_pipeline.feed: flat listing, client-side windowing, seen-history dedupe."""

import json
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import cast

import pytest
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from taste_pipeline.config import Config
from taste_pipeline.feed import CookieExpiredError, fetch_feed
from taste_pipeline.state import StateStore

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "flat_feed.json"
WINDOW_DAYS = 7
SECONDS_PER_DAY = 86400
SURVIVOR_IDS = ["song0000001", "note0000005"]
LISTED_IDS = ["song0000001", "old00000002", "note0000005"]
FILTERED_IDS = ["short000003", "live0000004"]


def _load_flat_entries() -> list[dict[str, object]]:
    """Load the recorded flat feed, converting ``timestamp_age_days`` into absolute timestamps."""
    # json.loads is the untyped boundary; the fixture's shape is fixed by this repo.
    raw = cast("list[dict[str, object]]", json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
    now = int(time.time())
    for entry in raw:
        age_days = entry.pop("timestamp_age_days", None)
        if isinstance(age_days, (int, float)):
            entry["timestamp"] = now - int(age_days) * SECONDS_PER_DAY
    return raw


def _patch_extractor(
    monkeypatch: pytest.MonkeyPatch, entries: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Replay ``entries`` through a fake extractor; return the captured YoutubeDL params.

    yt-dlp applies ``params['match_filter']`` to flat entries with
    ``incomplete=True`` (see ``YoutubeDL.__process_iterable_entry``); the fake
    does the same so the live/shorts filter string is exercised for real.
    """
    captured: list[dict[str, object]] = []

    def fake_extract_info(self: YoutubeDL, url: str, *, download: bool = False) -> dict[str, object]:
        _ = download
        captured.append(dict(self.params))
        # match_filter_func's product takes (info_dict, incomplete); the typeshed
        # stub unions it with a 1-arg form, which no real call site can satisfy.
        match_filter = cast(
            "Callable[[Mapping[str, object], bool], str | None] | None", self.params.get("match_filter")
        )
        # yt-dlp's match filter returns None for kept entries and a reason
        # string for rejected ones (see YoutubeDL._match_entry); flat entries
        # are matched with incomplete=True (YoutubeDL.__process_iterable_entry).
        incomplete = True
        kept = [entry for entry in entries if match_filter is None or match_filter(entry, incomplete) is None]
        return {"_type": "playlist", "id": url, "title": "Subscriptions", "entries": kept}

    monkeypatch.setattr(YoutubeDL, "extract_info", fake_extract_info)
    return captured


def _patch_raising_extractor(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """Monkeypatch the extractor to raise ``DownloadError(message)`` like a network-side failure."""

    def fake_extract_info(self: YoutubeDL, url: str, *, download: bool = False) -> dict[str, object]:
        _ = (self, url, download)
        raise DownloadError(message)

    monkeypatch.setattr(YoutubeDL, "extract_info", fake_extract_info)


def _read_seen_rows(db_path: Path) -> list[tuple[str, str, str]]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT video_id, first_seen, stage FROM seen").fetchall()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    data_dir = tmp_path / "data"
    return Config(
        like_library_dir=tmp_path,
        data_dir=data_dir,
        cookie_file=tmp_path / "cookies.txt",
        download_archive=data_dir / "yt-dlp-archive.txt",
        feed_window_days=WINDOW_DAYS,
    )


@pytest.fixture
def state(config: Config) -> Iterator[StateStore]:
    store = StateStore(config.data_dir)
    yield store
    store.close()


def test_fetch_feed_returns_only_new_in_window_items(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given a recorded flat feed replayed through the fake extractor
    _ = _patch_extractor(monkeypatch, _load_flat_entries())

    # When fetching the feed
    items = fetch_feed(config, state)

    # Then only the in-window song and the fail-open (timestamp=None) entry survive;
    # the out-of-window entry is dropped by the date window and the shorts/live
    # entries were dropped by yt-dlp's match filter inside the extractor.
    assert [item.video_id for item in items] == SURVIVOR_IDS


def test_fetch_feed_normalizes_flat_entry_fields(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given the recorded flat feed
    _ = _patch_extractor(monkeypatch, _load_flat_entries())

    # When fetching the feed
    items = fetch_feed(config, state)

    # Then fields are parsed, timestamp is an int, and the timestamp-less entry stays None
    by_id = {item.video_id: item for item in items}
    song = by_id["song0000001"]
    assert song.title == "Veorra - Blindside"
    assert song.uploader == "Veorra"
    assert song.channel_id == "UC1111111111111111111111"
    assert song.url == "https://www.youtube.com/watch?v=song0000001"
    assert isinstance(song.timestamp, int)
    assert song.duration == 214.0
    assert by_id["note0000005"].timestamp is None


def test_fetch_feed_records_every_listed_id_as_seen(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given the recorded flat feed
    _ = _patch_extractor(monkeypatch, _load_flat_entries())

    # When fetching the feed
    _ = fetch_feed(config, state)

    # Then every id the extractor handed us is recorded with stage 'listed',
    # while shorts/live ids never reached our code (dropped inside yt-dlp)
    for video_id in LISTED_IDS:
        assert state.already_seen(video_id)
    for video_id in FILTERED_IDS:
        assert not state.already_seen(video_id)
    rows = _read_seen_rows(state.db_path)
    assert sorted(row[0] for row in rows) == sorted(LISTED_IDS)
    assert {row[2] for row in rows} == {"listed"}
    assert all("T" in row[1] for row in rows)  # ISO 8601 UTC first_seen


def test_fetch_feed_second_run_yields_zero_new_items(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given the recorded flat feed has already been listed once
    entries = _load_flat_entries()
    _ = _patch_extractor(monkeypatch, entries)
    first = fetch_feed(config, state)
    assert [item.video_id for item in first] == SURVIVOR_IDS
    first_seen_before = {row[0]: row[1] for row in _read_seen_rows(state.db_path)}

    # When the same feed is listed again with the same state store
    second = fetch_feed(config, state)

    # Then nothing is new and the seen table is unchanged (INSERT OR IGNORE)
    assert second == []
    first_seen_after = {row[0]: row[1] for row in _read_seen_rows(state.db_path)}
    assert first_seen_after == first_seen_before


def test_fetch_feed_configures_yt_dlp_for_flat_readonly_listing(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given the fake extractor capturing the params YoutubeDL was built with
    captured = _patch_extractor(monkeypatch, _load_flat_entries())

    # When fetching the feed
    _ = fetch_feed(config, state)

    # Then yt-dlp is configured for a flat, read-only, cookie-authenticated listing
    assert len(captured) == 1
    params = captured[0]
    assert params["extract_flat"] == "in_playlist"
    assert params["skip_download"] is True
    assert params["cookiefile"] == str(config.cookie_file)
    assert params["playlistend"] == config.max_feed_items
    assert params["extractor_args"] == {"youtubetab": ["approximate_date"]}
    assert params["quiet"] is True
    assert params["no_warnings"] is True
    assert callable(params["match_filter"])
    # Guardrails: windowing/early-exit/dedupe live in our code, not yt-dlp flags
    assert "daterange" not in params
    assert "dateafter" not in params
    assert "download_archive" not in params


def test_fetch_feed_login_check_raises_cookie_expired_error(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given yt-dlp failing with YouTube's login/bot check
    _patch_raising_extractor(monkeypatch, "Sign in to confirm you're not a bot")

    # When fetching the feed, Then the pipeline points at the cookie runbook
    with pytest.raises(CookieExpiredError) as exc_info:
        _ = fetch_feed(config, state)
    assert "docs/cookies.md" in str(exc_info.value)


def test_fetch_feed_other_download_error_propagates(
    monkeypatch: pytest.MonkeyPatch, config: Config, state: StateStore
) -> None:
    # Given yt-dlp failing with a non-login error
    _patch_raising_extractor(monkeypatch, "HTTP Error 503: Service Unavailable")

    # When fetching the feed, Then the original error propagates unchanged
    with pytest.raises(DownloadError, match="503"):
        _ = fetch_feed(config, state)
