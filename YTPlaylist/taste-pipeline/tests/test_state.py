"""Tests for taste_pipeline.state: SQLite-backed seen/download tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from taste_pipeline.state import StateStore

if TYPE_CHECKING:
    from pathlib import Path


def test_state_counts_empty_state_returns_empty_subdicts(tmp_path: Path) -> None:
    # Given a fresh StateStore with no rows recorded
    store = StateStore(tmp_path)
    try:
        # When asking for grouped counts
        counts = store.state_counts()
    finally:
        store.close()

    # Then both sub-dicts are empty (no rows => no groups)
    assert counts == {"seen": {}, "downloads": {}}


def test_state_counts_groups_seen_rows_by_stage(tmp_path: Path) -> None:
    # Given a StateStore with three seen rows across two stages
    store = StateStore(tmp_path)
    try:
        store.record_seen("vid1", stage="listed")
        store.record_seen("vid2", stage="listed")
        store.record_seen("vid3", stage="metadata")
    finally:
        pass  # close below; keep variable alive for the assert

    # When asking for grouped counts
    counts = store.state_counts()
    store.close()

    # Then seen has the expected per-stage counts and downloads is still empty
    assert counts["seen"] == {"listed": 2, "metadata": 1}
    assert counts["downloads"] == {}


def test_state_counts_groups_downloads_by_stage(tmp_path: Path) -> None:
    # Given a StateStore with two download rows (default stage + explicit override)
    store = StateStore(tmp_path)
    try:
        store.record_download("vid1", audio_path=tmp_path / "a.flac")
        store.record_download("vid2", audio_path=tmp_path / "b.flac", stage="transcoded")
    finally:
        pass

    # When asking for grouped counts
    counts = store.state_counts()
    store.close()

    # Then downloads has the expected per-stage counts and seen is still empty
    assert counts["downloads"] == {"downloaded": 1, "transcoded": 1}
    assert counts["seen"] == {}


def test_state_counts_repeat_calls_return_consistent_results(tmp_path: Path) -> None:
    # Given a StateStore that is queried twice without further writes in between
    store = StateStore(tmp_path)
    try:
        store.record_seen("vid1", stage="listed")
        first = store.state_counts()
        second = store.state_counts()
    finally:
        store.close()

    # Then both calls return the same counts (idempotent read, no row-count drift)
    assert first == second == {"seen": {"listed": 1}, "downloads": {}}


def test_state_counts_does_not_count_duplicate_seen_inserts(tmp_path: Path) -> None:
    # Given a StateStore where the same video_id is recorded twice (idempotent inserts)
    store = StateStore(tmp_path)
    try:
        store.record_seen("vid1", stage="listed")
        store.record_seen("vid1", stage="listed")
    finally:
        pass

    # When asking for grouped counts
    counts = store.state_counts()
    store.close()

    # Then duplicates collapse to a single row (primary key enforcement)
    assert counts["seen"] == {"listed": 1}
