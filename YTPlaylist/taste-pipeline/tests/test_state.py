"""Tests for taste_pipeline.state: SQLite-backed seen/download tracking."""

from __future__ import annotations

from pathlib import Path

from taste_pipeline.state import StateStore


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


def test_get_download_record_returns_none_when_row_absent(tmp_path: Path) -> None:
    # Given a fresh StateStore with no download rows
    store = StateStore(tmp_path)
    try:
        # When asking for a video_id that was never recorded
        result = store.get_download_record("never_recorded")
    finally:
        store.close()

    # Then None is returned (the caller treats this as a 404 case)
    assert result is None


def test_get_download_record_returns_stage_and_audio_path(tmp_path: Path) -> None:
    # Given a StateStore with one downloaded row at a known path
    store = StateStore(tmp_path)
    try:
        audio_path = tmp_path / "song.flac"
        _ = store.record_download("vid1", audio_path=audio_path, stage="kept")
    finally:
        pass

    # When asking for that video_id
    result = store.get_download_record("vid1")
    store.close()

    # Then the (stage, audio_path) tuple round-trips through the table
    assert result is not None
    stage, stored_path = result
    assert stage == "kept"
    assert Path(stored_path) == audio_path


def test_get_download_record_returns_latest_path_after_replace(tmp_path: Path) -> None:
    # Given a video_id whose downloads row was REPLACEd with a new path (triage moved the file)
    store = StateStore(tmp_path)
    try:
        original_path = tmp_path / "inbox" / "old.flac"
        new_path = tmp_path / "keep" / "new.flac"
        _ = store.record_download("vid1", audio_path=original_path)
        _ = store.record_download("vid1", audio_path=new_path, stage="kept")
    finally:
        pass

    # When asking for that video_id
    result = store.get_download_record("vid1")
    store.close()

    # Then the returned audio_path is the latest one (REPLACE semantics)
    assert result is not None
    _stage, stored_path = result
    assert Path(stored_path) == new_path
