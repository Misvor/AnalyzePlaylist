"""Job-runner factories: each pipeline pass wrapped as a ``run_fn(report)`` closure.

This module is the bridge between :class:`~taste_pipeline.web.jobs.JobRunner`
(infrastructure, no pipeline imports) and the four pipeline passes
(``feed`` / ``metadata`` / ``download`` / ``index``). Each ``make_X_factory``
returns a closure with the same shape :class:`JobRunner.submit` expects::

    Callable[[Callable[[float, str], None]], None]

so the closure can be plugged straight into :data:`JOB_FACTORIES` in
:mod:`taste_pipeline.web.routes_jobs`.

Design notes
------------

- **Imports are inside the factory bodies.** The pipeline modules (``feed``,
  ``metadata``, ``detect``, ``download``, ``embed``, ``library``) are imported
  only when the factory's ``run_fn`` is invoked, not at module import. This
  keeps ``import taste_pipeline.web.job_factories`` cheap and — more
  importantly — makes :func:`pytest.MonkeyPatch.setattr` on the source
  modules (``taste_pipeline.feed.fetch_feed`` etc.) redirect every call,
  because the factory looks up ``feed.fetch_feed`` at call time against the
  same module object the test patched.
- **Config + StateStore are baked into the closure** at ``make_X_factory``
  call time. Tests can build a factory directly with their own ``tmp_path``
  Config/StateStore; production wires them via :func:`register_factories` in
  the POST handler.
- **Metadata and download cross-pass handoff: re-fetch.** The metadata
  factory re-fetches the feed to get input ids when none are supplied; the
  download factory re-fetches metadata + re-runs :func:`detect.is_song` to
  build its candidate list when no candidates are supplied. The state store's
  ``already_seen`` / ``record_seen`` / ``record_download`` stages make the
  re-runs idempotent — the feed pass marks stage ``listed``, the metadata
  pass marks stage ``metadata`` and skips already-listed ones via
  ``max_metadata_fetch`` bounds, the download pass records the FLAC path and
  yt-dlp's ``download_archive`` blocks re-downloads. This avoids the
  alternative "persist detection results in the run record" path which
  couples the dashboard to the run-record JSON schema.
- **Failures propagate.** Each factory does NOT catch pipeline-specific
  exceptions (``CookieExpiredError``, ``DownloadError``, etc.). The runner's
  blanket ``except Exception`` in :meth:`JobRunner._run` turns any raise
  into ``status="failed"`` with ``str(exc)`` as the ``error`` message —
  ``CookieExpiredError.__str__`` already includes a readable pointer to the
  cookie re-export runbook; ``DownloadError("net fail")`` already includes
  the failure cause. Catching+re-raising would just duplicate that work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from yt_dlp import YoutubeDL

if TYPE_CHECKING:
    from taste_pipeline.config import Config
    from taste_pipeline.metadata import TrackMetadata
    from taste_pipeline.state import StateStore

RunFn = Callable[[Callable[[float, str], None]], "dict[str, object] | None"]


def make_feed_factory(config: Config, state: StateStore) -> RunFn:
    """Build a ``feed`` factory: call :func:`taste_pipeline.feed.fetch_feed` and report per item.

    The returned closure calls :func:`fetch_feed` and reports progress
    ``index/total`` for each :class:`~taste_pipeline.feed.FeedItem` returned.
    An empty feed still produces a successful job (the runner's ``finally``
    block sets progress to 1.0 and emits zero log lines).
    """
    from taste_pipeline import feed  # noqa: PLC0415 -- lazy: monkey-patch works because of module-attr lookup

    def run_fn(report: Callable[[float, str], None]) -> None:
        items = feed.fetch_feed(config, state)
        total = len(items)
        for index, item in enumerate(items, start=1):
            progress = index / total if total else 1.0
            report(progress, f"listed {item.video_id}")

    return run_fn


def make_metadata_factory(config: Config, state: StateStore, ids: list[str] | None = None) -> RunFn:
    """Build a ``metadata`` factory: fetch metadata and run :func:`detect.is_song`, reporting per item.

    Args:
        config: Pipeline configuration.
        state: State store (used by :func:`fetch_metadata` to record stage ``metadata``).
        ids: Video ids to fetch; when ``None`` the factory re-fetches the feed
            (at job run time) to discover the current ``listed`` ids. Tests
            pass an explicit list to keep the test isolated from the feed pass.
    """
    from taste_pipeline import detect, metadata  # noqa: PLC0415 -- lazy: monkey-patch works

    def run_fn(report: Callable[[float, str], None]) -> None:
        if ids is None:
            from taste_pipeline import feed  # noqa: PLC0415 -- lazy: re-fetch only when needed

            fresh_ids = [item.video_id for item in feed.fetch_feed(config, state)]
        else:
            fresh_ids = ids
        metas = metadata.fetch_metadata(fresh_ids, config, state)
        total = len(metas)
        for index, meta in enumerate(metas, start=1):
            _is_song, rule_id = detect.is_song(meta)
            progress = index / total if total else 1.0
            report(progress, f"meta {meta.video_id} rule={rule_id}")

    return run_fn


def make_download_factory(
    config: Config,
    state: StateStore,
    candidates: list[TrackMetadata] | None = None,
) -> RunFn:
    """Build a ``download`` factory: pass metas to :func:`download.download_songs`, reporting per track.

    Args:
        config: Pipeline configuration.
        state: State store (used by :func:`download_songs` to record stage ``downloaded``).
        candidates: Song-candidate :class:`TrackMetadata` instances to download.
            When ``None`` the factory re-fetches the feed + metadata and
            re-runs :func:`detect.is_song` (at job run time) to discover
            candidates. Tests pass an explicit list so ``download_songs`` can
            be monkey-patched cleanly without the upstream passes running.
    """
    from taste_pipeline import download  # noqa: PLC0415 -- lazy: monkey-patch works

    def run_fn(report: Callable[[float, str], None]) -> None:
        if candidates is None:
            from taste_pipeline import detect, feed, metadata  # noqa: PLC0415 -- lazy re-fetch

            fresh_ids = [item.video_id for item in feed.fetch_feed(config, state)]
            fresh_metas = metadata.fetch_metadata(fresh_ids, config, state)
            resolved: list[TrackMetadata] = [meta for meta in fresh_metas if detect.is_song(meta)[0]]
        else:
            resolved = candidates
        results = download.download_songs(resolved, config, state)
        total = len(results)
        for index, result in enumerate(results, start=1):
            progress = index / total if total else 1.0
            report(progress, f"downloaded {result.meta.video_id}")

    return run_fn


def make_index_factory(config: Config) -> RunFn:
    """Build an ``index`` factory: run :func:`library.scan_library` with the real :func:`embed.embed_track`.

    The embedder is looked up via ``embed.embed_track`` at ``run_fn`` call
    time (not at factory-build time) so a test that monkey-patches
    ``taste_pipeline.embed.embed_track`` before submitting sees its fake
    vector — the factory does not cache a reference.
    """
    from taste_pipeline import embed, library  # noqa: PLC0415 -- lazy: monkey-patch works

    def run_fn(report: Callable[[float, str], None]) -> None:
        embedder = embed.embed_track
        index = library.scan_library(config, embedder)
        report(1.0, f"indexed {index.count()} files")

    return run_fn


def make_calibrate_factory(config: Config, state: StateStore) -> RunFn:
    """Build a ``calibrate`` factory: derive keep/skip thresholds from the like-library.

    Delegates to :func:`taste_pipeline.calibrate.run_calibration`, which
    itself imports :mod:`taste_pipeline.embed` and
    :mod:`taste_pipeline.library` lazily inside the function. Tests
    monkey-patch ``taste_pipeline.embed.embed_track`` and
    ``taste_pipeline.library.scan_library`` before submitting; the
    calibrate module looks up those names by attribute at call time so
    the patches are observed.

    The :class:`StateStore` argument is part of the factory contract
    (every production factory takes ``(config, state)``) but the
    calibration pass does not record any state store stages — the
    ``keep``/``skip`` decisions happen later in the download/triage
    flow.
    """
    from taste_pipeline import calibrate  # noqa: PLC0415 -- lazy: monkey-patch works

    def run_fn(report: Callable[[float, str], None]) -> None:
        # ``state`` is part of the factory contract (every production
        # factory takes ``(config, state)``) but the calibration pass
        # does not record any state store stages -- the keep/skip
        # decisions happen later in the download/triage flow.
        _ = state
        _ = calibrate.run_calibration(config, report)

    return run_fn


def make_check_url_factory(config: Config, state: StateStore, url: str) -> RunFn:
    """Build a ``check_url`` factory: fetch → metadata → download → embed → compare.

    The full pipeline runs inside one ``run_fn`` so the user gets a single
    background job (and SSE progress stream) for the whole flow. Returns a
    :class:`CheckResult` dict via the ``run_fn`` return value; the runner
    stashes it on ``job.result`` so ``GET /api/jobs/{id}`` surfaces it.

    Pipeline stages:

    1. Extract the ``video_id`` from the URL via yt-dlp (no download).
    2. Fetch the metadata for that video (pass-B).
    3. Run :func:`detect.is_song`; bail with a verdict if not a song.
    4. Download the FLAC into ``<data_dir>/inbox`` (pass-C).
    5. Call :func:`taste_pipeline.check.check_audio_file` for the embed + compare.
    6. Return the :class:`CheckResult` as a plain dict.

    The check pass itself loads calibration thresholds via
    :func:`taste_pipeline.calibrate.load_thresholds` (no manual scan --
    the check factory does not need to re-derive thresholds). All
    pipeline modules are imported lazily inside the closure so a test
    that ``monkeypatch.setattr``s ``taste_pipeline.metadata.fetch_metadata``,
    ``taste_pipeline.download.download_songs``,
    ``taste_pipeline.detect.is_song``, or
    ``taste_pipeline.check.check_audio_file`` observes the patches.
    """
    from taste_pipeline.detect import is_song  # noqa: PLC0415 -- lazy: monkey-patch works

    def run_fn(report: Callable[[float, str], None]) -> dict[str, object]:
        from taste_pipeline import check, download, metadata  # noqa: PLC0415 -- lazy re-fetch

        report(0.0, "parsing URL")
        with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        video_id = str(info.get("id") or "")
        if not video_id:
            message = f"could not parse a video id from {url!r}"
            raise ValueError(message)

        report(0.1, "fetching metadata")
        metas = metadata.fetch_metadata([video_id], config, state)
        if not metas:
            message = f"no metadata returned for {video_id}"
            raise ValueError(message)
        track_meta = metas[0]

        report(0.3, "checking if it's a song")
        is_song_result = is_song(track_meta)[0]
        if not is_song_result:
            message = f"{video_id} is not a song (skipping download)"
            raise ValueError(message)

        report(0.5, "downloading")
        downloaded = download.download_songs([track_meta], config, state)
        if not downloaded:
            message = f"download produced no file for {video_id}"
            raise ValueError(message)

        report(0.8, "computing embedding and comparing")
        verdict = check.check_audio_file(downloaded[0].audio_path, config)
        report(1.0, "complete")
        return {
            "verdict": verdict.verdict,
            "score": verdict.score,
            "top_k": [{"score": score, "library_track_id": track_id} for score, track_id in verdict.top_k],
            "threshold_keep": verdict.threshold_keep,
            "threshold_skip": verdict.threshold_skip,
            "human_readable": verdict.human_readable,
        }

    return run_fn


def register_factories(
    target: dict[str, RunFn],
    config: Config,
    state: StateStore,
) -> None:
    """Populate ``target`` with the production factories bound to ``(config, state)``.

    The caller (the POST handler in :mod:`taste_pipeline.web.routes_jobs`)
    guards with ``if not JOB_FACTORIES`` so a test that already populated
    ``target`` with fakes is left untouched. Tests of this module prefer to
    call :func:`make_feed_factory` etc. directly with explicit ``ids`` /
    ``candidates`` to keep the test surface tight.
    """
    target["feed"] = make_feed_factory(config, state)
    target["metadata"] = make_metadata_factory(config, state)
    target["download"] = make_download_factory(config, state)
    target["index"] = make_index_factory(config)
    target["calibrate"] = make_calibrate_factory(config, state)
