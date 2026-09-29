"""Pytest configuration: --runslow flag and slow-marker handling.

Also hosts shared test helpers used by the triage-route test files
(``test_inbox_listing.py``, ``test_triage_actions.py``, ``test_audio_route.py``)
and the job-runner test files (``test_api_jobs.py``). These are plain
module-level functions so the test files can ``from conftest import _helper``
without any fixture decoration; pytest auto-loads this file for every test
collection so the imports just work.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.testclient import TestClient

    from taste_pipeline.config import Config


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ── Triage-route test helpers (shared by test_inbox_listing / test_triage_actions / test_audio_route) ──


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string (forward slashes avoid backslash escapes on Windows)."""
    return path.as_posix()


def _write_config(tmp_path: Path) -> Path:
    """Write a minimal valid config TOML under tmp_path and return its path.

    ``exist_ok=True`` is necessary because tests that already constructed a
    Config (e.g. to seed the data_dir) call this helper a second time to
    build a second app -- both app builds share the same ``like_lib/`` dir
    under tmp_path. Config validation also expects ``like_library_dir`` to
    exist; we never read its contents, the empty dir is enough.
    """
    like_lib = tmp_path / "lib"
    like_lib.mkdir(exist_ok=True)
    body = (
        f'like_library_dir = "{_toml_path(like_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _make_config(tmp_path: Path) -> Config:
    """Build a real Config via load_config from a tmp TOML file."""
    from taste_pipeline.config import load_config  # noqa: PLC0415 -- lazy: keep conftest dep-free at import

    return load_config(_write_config(tmp_path))


def _seed_inbox(
    data_dir: Path,
    tracks: list[tuple[str, str, str]],
) -> list[Path]:
    """Write fake FLACs + sidecars for the given tracks and record them in the state store.

    Args:
        data_dir: Pipeline data dir (already has ``inbox/`` from config load).
        tracks: Each entry is ``(video_id, title, uploader)``.

    Returns:
        The list of absolute paths to the written FLAC files (in input order).
    """
    from taste_pipeline.state import StateStore  # noqa: PLC0415 -- lazy: conftest stays dep-free at import

    inbox = data_dir / "inbox"
    store = StateStore(data_dir)
    try:
        paths: list[Path] = []
        for video_id, title, uploader in tracks:
            # Filename embeds the [video_id] marker the pipeline uses for lookup.
            safe_title = re.sub(r"[^A-Za-z0-9_-]", "_", title)[:40]
            audio_path = inbox / f"{safe_title} [{video_id}].flac"
            _ = audio_path.write_bytes(b"FAKE_FLAC_HEADER" + b"\x00" * 32)
            sidecar = audio_path.with_name(audio_path.name + ".meta.json")
            sidecar_payload = {
                "video_id": video_id,
                "title": title,
                "uploader": uploader,
                "uploader_id": None,
                "channel_id": None,
                "upload_date": None,
                "timestamp": None,
                "categories": None,
                "duration": None,
                "description": None,
                "track": title,
                "artist": uploader,
                "album": None,
                "thumbnail_url": None,
                "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
            }
            _ = sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
            _ = store.record_download(video_id, audio_path)
            paths.append(audio_path)
    finally:
        store.close()
    return paths


def _client(tmp_path: Path) -> TestClient:
    """Build a TestClient over a real Config-derived app."""
    from fastapi.testclient import TestClient  # noqa: PLC0415 -- lazy: conftest stays dep-free at import

    from taste_pipeline.web import create_app  # noqa: PLC0415 -- lazy: conftest stays dep-free at import

    return TestClient(create_app(_make_config(tmp_path)))


# ── Job-runner test helpers (shared by test_api_jobs.py) ──


class _FakeJob:
    """A synchronous stand-in for the real Job dataclass -- drives state in a background thread."""

    def __init__(
        self,
        job_id: uuid.UUID,
        kind: str,
        run_fn: Callable[[Callable[[float, str], None]], object] | None = None,
    ) -> None:
        self.id = job_id
        self.kind = kind
        self.status = "queued"
        self.progress = 0.0
        self.log_lines: list[str] = []
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self.error: str | None = None
        self.result: dict[str, object] | None = None
        self.detail: dict[str, object] | None = None
        self._run_fn = run_fn
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self.start()

    def start(self) -> None:
        """Spawn the background worker thread that calls the injected run_fn."""
        self._thread = threading.Thread(target=self._drive, daemon=True)
        self._thread.start()

    def _drive(self) -> None:
        with self._lock:
            if self._cancel_event.is_set():
                self.status = "cancelled"
                self.finished_at = datetime.now(UTC)
                return
            self.started_at = datetime.now(UTC)
            self.status = "running"

        def report(progress: float, message: str, detail: dict[str, object] | None = None) -> None:
            with self._lock:
                if self._cancel_event.is_set():
                    return
                self.progress = progress
                self.log_lines.append(message)
                if detail is not None:
                    self.detail = detail

        try:
            assert self._run_fn is not None
            run_result = self._run_fn(report)
            if isinstance(run_result, dict):
                self.result = run_result
        except Exception as exc:  # noqa: BLE001 -- fake runner mirrors JobRunner contract
            with self._lock:
                if self.status != "cancelled":
                    self.status = "failed"
                    self.error = str(exc)
        finally:
            with self._lock:
                if self.status not in ("succeeded", "failed", "cancelled"):
                    self.status = "succeeded"
                    self.progress = 1.0
                if self.finished_at is None:
                    self.finished_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Mark the job cancelled (the FakeJob also wakes its run_fn by no-op'ing report)."""
        with self._lock:
            self._cancel_event.set()
            if self.status not in ("succeeded", "failed", "cancelled"):
                self.status = "cancelled"
                self.finished_at = datetime.now(UTC)


class FakeJobRunner:
    """Async stand-in for JobRunner -- same public shape, no real executor."""

    def __init__(self) -> None:
        self._jobs: dict[uuid.UUID, _FakeJob] = {}
        self._active_by_kind: dict[str, _FakeJob] = {}
        self.submit_calls: list[tuple[str, Callable[[Callable[[float, str], None]], object]]] = []

    async def submit(
        self,
        kind: str,
        run_fn: Callable[[Callable[[float, str], None]], object],
    ) -> _FakeJob:
        """Create a new _FakeJob; raise if an active job of the same kind exists."""
        self.submit_calls.append((kind, run_fn))
        active = self._active_by_kind.get(kind)
        if active is not None and active.status not in ("succeeded", "failed", "cancelled"):
            msg = f"concurrent job of kind {kind!r} already in progress (job_id={active.id})"
            raise RuntimeError(msg)
        job = _FakeJob(uuid.uuid4(), kind, run_fn)
        self._jobs[job.id] = job
        self._active_by_kind[kind] = job
        return job

    def get(self, job_id: uuid.UUID) -> _FakeJob:
        """Return the live _FakeJob for job_id; raise KeyError if unknown."""
        if job_id not in self._jobs:
            msg = f"job {job_id} not found"
            raise KeyError(msg)
        return self._jobs[job_id]

    def list(self) -> list[_FakeJob]:
        """Return all jobs in newest-first order."""
        return list(reversed(list(self._jobs.values())))

    def is_finalized(self, job_id: uuid.UUID) -> bool:
        """The fake runner finalizes synchronously, so every job is already finalized."""
        _ = job_id
        return True


def _build_client_with_fake_runner(
    tmp_path: Path,
) -> tuple[TestClient, FakeJobRunner]:
    """Construct an app, inject a FakeJobRunner on app.state, and return (client, runner)."""
    from fastapi.testclient import TestClient  # noqa: PLC0415 -- lazy: conftest stays dep-free at import

    from taste_pipeline.web import create_app  # noqa: PLC0415 -- lazy: conftest stays dep-free at import

    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    runner = FakeJobRunner()
    app.state.runner = runner
    return TestClient(app), runner


def _wait_for_terminal(job: _FakeJob, *, timeout_s: float = 2.0) -> _FakeJob:
    """Spin-wait until the FakeJob reaches a terminal status; raise on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.005)
    msg = f"job {job.id} did not reach terminal state within {timeout_s}s"
    raise AssertionError(msg)


def _submit(
    runner: FakeJobRunner,
    kind: str,
    run_fn: Callable[[Callable[[float, str], None]], object],
) -> _FakeJob:
    """Drive an async FakeJobRunner.submit from a synchronous test."""
    import asyncio  # noqa: PLC0415 -- lazy: conftest stays dep-free at import

    return asyncio.run(runner.submit(kind, run_fn))
