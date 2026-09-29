"""Tests for taste_pipeline.web.job_factories: each pipeline pass as a JobRunner run_fn.

TDD-first for todo-6 (pipeline passes wired as job kinds). Every pipeline
boundary is monkey-patched at the SOURCE module path (e.g.
``taste_pipeline.feed.fetch_feed``) — never on a locally cached reference —
so the factories' ``feed.fetch_feed(...)`` call shape is what the patch
redirects. The tests build a real :class:`Config` via ``load_config`` and a
real :class:`StateStore` against a per-test ``tmp_path``; no real network
calls, no shared state across tests.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from yt_dlp.utils import DownloadError

from taste_pipeline.config import Config, load_config
from taste_pipeline.download import DownloadedTrack
from taste_pipeline.feed import CookieExpiredError, FeedItem
from taste_pipeline.library import LibraryIndex, ScanProgress
from taste_pipeline.metadata import TrackMetadata
from taste_pipeline.state import StateStore
from taste_pipeline.web import job_factories
from taste_pipeline.web.jobs import JobRunner

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable
    from pathlib import Path


def _toml_path(p: Path) -> str:
    """Render a Path as a TOML-safe string (forward slashes for Windows)."""
    return p.as_posix()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal valid config TOML under ``tmp_path`` and return its path."""
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    body = (
        f'like_library_dir = "{_toml_path(like_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
    )
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _make_config(tmp_path: Path) -> Config:
    """Build a real :class:`Config` from a tmp TOML via :func:`load_config`."""
    return load_config(_write_config(tmp_path))


async def _wait_for_terminal(runner: JobRunner, job_id: uuid.UUID, *, timeout_s: float = 3.0) -> object:
    """Poll ``runner.get`` until the Job is terminal; raise on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.005)
    msg = f"job {job_id} did not reach terminal state within {timeout_s}s"
    raise AssertionError(msg)


def _make_feed_item(video_id: str = "v1", *, timestamp: int | None = None) -> FeedItem:
    """Build a real :class:`FeedItem` with predictable fields for assertions."""
    return FeedItem(
        video_id=video_id,
        title=f"title-{video_id}",
        uploader=f"uploader-{video_id}",
        channel_id=None,
        url=None,
        timestamp=timestamp,
        duration=None,
    )


def _make_meta(video_id: str = "v1") -> TrackMetadata:
    """Build a real TrackMetadata with empty fields (no is_song rule fires by default)."""
    return TrackMetadata(
        video_id=video_id,
        title=f"title-{video_id}",
        uploader=f"uploader-{video_id}",
        uploader_id=None,
        channel_id=None,
        upload_date=None,
        timestamp=None,
        categories=None,
        duration=None,
        description=None,
        track=None,
        artist=None,
        album=None,
        thumbnail_url=None,
        webpage_url=None,
    )


def _make_fake_library_index(*, n: int = 3) -> LibraryIndex:
    """Build a real :class:`LibraryIndex` with ``n`` rows of zero vectors for tests."""
    vectors = np.zeros((n, 512), dtype=np.float32)
    manifest: dict[str, dict[str, float | int]] = {
        f"track_{i}.flac": {"mtime": 1.0, "size": 100, "vector_id": i} for i in range(n)
    }
    return LibraryIndex(_vectors=vectors, _manifest=manifest)


@pytest.mark.asyncio
async def test_feed_factory_calls_fetch_feed_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a real Config and StateStore, plus a monkey-patched fetch_feed returning 2 items
    config = _make_config(tmp_path)
    state = StateStore(tmp_path / "data")
    items = [_make_feed_item("v1"), _make_feed_item("v2")]

    captured: dict[str, object] = {}
    report_calls: list[tuple[float, str]] = []

    def fake_fetch_feed(cfg: Config, st: StateStore) -> list[FeedItem]:
        captured["config"] = cfg
        captured["state"] = st
        return items

    monkeypatch.setattr("taste_pipeline.feed.fetch_feed", fake_fetch_feed)
    factory = job_factories.make_feed_factory(config, state)
    runner = JobRunner(tmp_path / "data")

    # Wrap the factory to capture report() calls (without mutating the closure we passed in)
    captured_run_fn: Callable[[Callable[[float, str], None]], None] = factory

    # When submitting the feed factory and waiting for terminal
    job = await runner.submit("feed", captured_run_fn)
    final = await _wait_for_terminal(runner, job.id)
    # Reach into the runner to read the report() calls the wrapper made
    _ = report_calls  # placeholder: the assertions below are on the live Job state

    # Then the job succeeded, progress reached 1.0, fetch_feed was called once with (config, state)
    assert final.status == "succeeded"
    assert final.progress == 1.0
    assert captured["config"] is config
    assert captured["state"] is state
    # And report() was called at least twice (once per item) -- log_lines is the captured history
    assert len(final.log_lines) >= 2, f"expected >=2 log lines (one per item), got {final.log_lines!r}"
    assert any("v1" in line for line in final.log_lines)
    assert any("v2" in line for line in final.log_lines)


@pytest.mark.asyncio
async def test_feed_factory_swallows_cookie_expired_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a monkey-patched fetch_feed that raises CookieExpiredError
    config = _make_config(tmp_path)
    state = StateStore(tmp_path / "data")

    def fake_fetch_feed(cfg: Config, st: StateStore) -> list[FeedItem]:
        raise CookieExpiredError(detail="Sign in to confirm you are not a bot")

    monkeypatch.setattr("taste_pipeline.feed.fetch_feed", fake_fetch_feed)
    factory = job_factories.make_feed_factory(config, state)
    runner = JobRunner(tmp_path / "data")

    # When submitting
    job = await runner.submit("feed", factory)
    final = await _wait_for_terminal(runner, job.id)

    # Then the job ends in failed state with a readable error mentioning "login"
    assert final.status == "failed"
    assert final.error is not None
    assert "login" in final.error.lower(), f"expected 'login' in error, got {final.error!r}"


@pytest.mark.asyncio
async def test_metadata_factory_calls_fetch_metadata_then_is_song(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a real Config/StateStore, monkey-patched fetch_metadata + is_song
    config = _make_config(tmp_path)
    state = StateStore(tmp_path / "data")
    metas = [_make_meta("v1"), _make_meta("v2")]

    fetch_metadata_mock = MagicMock(return_value=metas)
    is_song_mock = MagicMock(
        side_effect=lambda meta: (True, "rule_p1") if meta.video_id == "v1" else (False, "none")
    )

    monkeypatch.setattr("taste_pipeline.metadata.fetch_metadata", fetch_metadata_mock)
    monkeypatch.setattr("taste_pipeline.detect.is_song", is_song_mock)

    factory = job_factories.make_metadata_factory(config, state, ids=["v1", "v2"])
    runner = JobRunner(tmp_path / "data")

    # When submitting
    job = await runner.submit("metadata", factory)
    final = await _wait_for_terminal(runner, job.id)

    # Then both pipeline calls happened, the job succeeded, progress is 1.0,
    # and the run record log_lines include detection results (rule_p1 + none)
    assert final.status == "succeeded"
    assert final.progress == 1.0
    assert fetch_metadata_mock.call_count == 1
    # is_song was called once per fetched meta (2 total)
    assert is_song_mock.call_count == 2
    log_blob = "\n".join(final.log_lines)
    assert "rule_p1" in log_blob, f"expected 'rule_p1' in log lines, got {final.log_lines!r}"
    assert "rule=none" in log_blob, f"expected 'rule=none' in log lines, got {final.log_lines!r}"


@pytest.mark.asyncio
async def test_metadata_factory_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a monkey-patched fetch_metadata that raises DownloadError
    config = _make_config(tmp_path)
    state = StateStore(tmp_path / "data")

    def fake_fetch_metadata(ids: list[str], cfg: Config, st: StateStore) -> list[TrackMetadata]:
        msg = "net fail"
        raise DownloadError(msg)

    monkeypatch.setattr("taste_pipeline.metadata.fetch_metadata", fake_fetch_metadata)
    factory = job_factories.make_metadata_factory(config, state, ids=["v1"])
    runner = JobRunner(tmp_path / "data")

    # When submitting
    job = await runner.submit("metadata", factory)
    final = await _wait_for_terminal(runner, job.id)

    # Then the job is failed and the error mentions "net fail"
    assert final.status == "failed"
    assert final.error is not None
    assert "net fail" in final.error, f"expected 'net fail' in error, got {final.error!r}"


@pytest.mark.asyncio
async def test_download_factory_calls_download_songs_with_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a real Config/StateStore, two real candidate TrackMetadatas, and a
    # monkey-patched download_songs that records its args
    config = _make_config(tmp_path)
    state = StateStore(tmp_path / "data")
    candidates = [_make_meta("v1"), _make_meta("v2")]

    captured: dict[str, object] = {}

    def fake_download_songs(cands: list[TrackMetadata], cfg: Config, st: StateStore) -> list[DownloadedTrack]:
        captured["candidates"] = cands
        captured["config"] = cfg
        captured["state"] = st
        # Return DownloadedTrack-likes; the factory only reads .meta.video_id for report
        inbox = tmp_path / "inbox"
        inbox.mkdir(exist_ok=True)
        return [DownloadedTrack(audio_path=inbox / "a.flac", meta=cand) for cand in cands]

    monkeypatch.setattr("taste_pipeline.download.download_songs", fake_download_songs)
    factory = job_factories.make_download_factory(config, state, candidates=candidates)
    runner = JobRunner(tmp_path / "data")

    # When submitting
    job = await runner.submit("download", factory)
    final = await _wait_for_terminal(runner, job.id)

    # Then download_songs was called once with the candidates list, and the job succeeded
    assert final.status == "succeeded"
    assert final.progress == 1.0
    assert captured["candidates"] is candidates
    assert captured["config"] is config
    assert captured["state"] is state
    # And report() was called once per downloaded track
    assert any("v1" in line for line in final.log_lines)
    assert any("v2" in line for line in final.log_lines)


@pytest.mark.asyncio
async def test_index_factory_calls_scan_library_with_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a real Config, a monkey-patched embed.embed_track and library.scan_library
    config = _make_config(tmp_path)

    fake_embed_track = MagicMock(return_value=np.zeros(512, dtype=np.float32))
    fake_index = _make_fake_library_index(n=3)

    captured: dict[str, object] = {}

    def fake_scan_library(
        cfg: Config,
        embedder: Callable[[object, object], np.ndarray],
        *,
        on_progress: Callable[[ScanProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        max_new_files: int | None = None,
    ) -> LibraryIndex:
        captured["config"] = cfg
        captured["embedder"] = embedder
        captured["on_progress"] = on_progress
        captured["should_stop"] = should_stop
        captured["max_new_files"] = max_new_files
        return fake_index

    monkeypatch.setattr("taste_pipeline.embed.embed_track", fake_embed_track)
    monkeypatch.setattr("taste_pipeline.library.scan_library", fake_scan_library)
    factory = job_factories.make_index_factory(config)
    runner = JobRunner(tmp_path / "data")

    # When submitting
    job = await runner.submit("index", factory)
    final = await _wait_for_terminal(runner, job.id)

    # Then scan_library was called with (config, embedder) where embedder is the
    # (patched) embed.embed_track, and the job succeeded
    assert final.status == "succeeded"
    assert final.progress == 1.0
    assert captured["config"] is config
    assert captured["embedder"] is fake_embed_track
    # And the pause/batch wiring is forwarded to scan_library
    assert callable(captured["should_stop"]), "index factory must forward report.stop_requested"
    assert captured["max_new_files"] is None, "no batch_size means the whole library"
    # And the run record mentions the indexed count
    assert any("3" in line for line in final.log_lines), (
        f"expected indexed count '3' in log lines, got {final.log_lines!r}"
    )


@pytest.mark.asyncio
async def test_index_factory_forwards_structured_progress_in_detail_and_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a fake scan_library that drives the on_progress callback with ScanProgress records
    config = _make_config(tmp_path)
    fake_embed_track = MagicMock(return_value=np.zeros(512, dtype=np.float32))
    fake_index = _make_fake_library_index(n=3)
    files = ["a.flac", "b.flac", "c.flac"]

    def fake_scan_library(
        cfg: Config,
        embedder: Callable[[object, object], np.ndarray],
        *,
        on_progress: Callable[[ScanProgress], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        max_new_files: int | None = None,
    ) -> LibraryIndex:
        assert on_progress is not None, "index factory must pass an on_progress callback"
        total = len(files)
        for k, name in enumerate(files, start=1):
            processed = k - 1
            on_progress(
                ScanProgress(
                    processed=processed,
                    total=total,
                    current=name,
                    remaining=total - processed,
                    upcoming=tuple(files[k:]),
                )
            )
        return fake_index

    monkeypatch.setattr("taste_pipeline.embed.embed_track", fake_embed_track)
    monkeypatch.setattr("taste_pipeline.library.scan_library", fake_scan_library)
    factory = job_factories.make_index_factory(config)
    runner = JobRunner(tmp_path / "data")

    # When submitting
    job = await runner.submit("index", factory)
    final = await _wait_for_terminal(runner, job.id)

    # Then the job's log_lines contain every intermediate indexed k/n: <file>
    # line, not just the final file count
    assert final.status == "succeeded"
    log_blob = "\n".join(final.log_lines)
    assert "indexed 0/3: a.flac" in log_blob, f"expected 'indexed 0/3: a.flac' in {final.log_lines!r}"
    assert "indexed 1/3: b.flac" in log_blob, f"expected 'indexed 1/3: b.flac' in {final.log_lines!r}"
    assert "indexed 2/3: c.flac" in log_blob, f"expected 'indexed 2/3: c.flac' in {final.log_lines!r}"

    # And the final detail (left in place after the closing report) describes
    # the structured progress payload the Index page consumes.
    detail = final.detail
    assert detail is not None, "index factory must attach a structured detail payload"
    assert detail["current"] == "c.flac"
    assert detail["processed"] == 2
    assert detail["total"] == 3
    assert detail["remaining"] == 1
    assert detail["upcoming"] == []
    assert isinstance(detail["elapsed_seconds"], float)
    assert detail["eta_seconds"] is not None, "eta_seconds is computed once processed > 0"


@pytest.mark.asyncio
async def test_register_factories_populates_all_four_kinds(
    tmp_path: Path,
) -> None:
    # Given an empty target dict and a real Config/StateStore
    config = _make_config(tmp_path)
    state = StateStore(tmp_path / "data")
    target: dict[str, Callable[[Callable[[float, str], None]], None]] = {}

    # When register_factories is called
    job_factories.register_factories(target, config, state)

    # Then all five kinds are populated with a callable factory
    # (we do NOT invoke the factories here -- the production metadata + download
    # factories re-fetch the feed, which would hit the real network without
    # the per-test monkey-patches used in the other tests; each factory's
    # behavior is covered by the make_X_factory tests above)
    assert set(target) == {"feed", "metadata", "download", "index", "calibrate"}
    for kind, factory in target.items():
        assert callable(factory), f"factory for {kind!r} is not callable: {factory!r}"
