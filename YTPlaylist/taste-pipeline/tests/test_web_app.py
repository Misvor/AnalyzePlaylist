"""Tests for taste_pipeline.web: FastAPI app factory and the __main__ entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from taste_pipeline.config import Config, ConfigError, load_config
from taste_pipeline.web import __main__ as web_main
from taste_pipeline.web import create_app

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _toml_path(path: Path) -> str:
    """Render a Path as a TOML-safe string (forward slashes avoid backslash escapes on Windows)."""
    return path.as_posix()


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    """Write a minimal valid config TOML under tmp_path and return its path."""
    like_lib = tmp_path / "lib"
    like_lib.mkdir()
    body = (
        f'like_library_dir = "{_toml_path(like_lib)}"\n'
        f'data_dir = "{_toml_path(tmp_path / "data")}"\n'
        f'cookie_file = "{_toml_path(tmp_path / "cookies.txt")}"\n'
    ) + extra
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def _make_config(tmp_path: Path, extra: str = "") -> Config:
    """Build a real Config via load_config from a tmp TOML file."""
    return load_config(_write_config(tmp_path, extra))


@dataclass
class _FakeThread:
    """Stands in for threading.Thread: records construction, start() never runs the target."""

    target: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    daemon: bool

    def start(self) -> None:
        """No-op: the test inspects the recorded target instead of running it."""


def test_create_app_returns_fastapi_and_stores_config_on_state(tmp_path: Path) -> None:
    # Given a valid Config built from a tmp TOML file
    cfg = _make_config(tmp_path)

    # When building the app via the factory
    app = create_app(cfg)

    # Then the factory returns a FastAPI instance with the injected config on app.state
    assert isinstance(app, FastAPI)
    assert app.state.config is cfg


def test_get_root_returns_html_shell(tmp_path: Path) -> None:
    # Given an app built from a valid config
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)

    # When requesting the index page
    response = client.get("/")

    # Then the response is 200 HTML containing the base template shell
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text


def test_static_css_is_served(tmp_path: Path) -> None:
    # Given an app built from a valid config
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)

    # When requesting the placeholder stylesheet
    response = client.get("/static/app.css")

    # Then the static mount serves it
    assert response.status_code == 200


def test_server_mode_runs_uvicorn_foreground_and_never_touches_webview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a valid config file and mocked uvicorn/webview boundaries
    config_path = _write_config(tmp_path)
    run_mock = MagicMock()
    create_window_mock = MagicMock()
    start_mock = MagicMock()
    monkeypatch.setattr(web_main.uvicorn, "run", run_mock)
    monkeypatch.setattr(web_main.webview, "create_window", create_window_mock)
    monkeypatch.setattr(web_main.webview, "start", start_mock)

    # When invoking the entry point with --server
    exit_code = web_main.main(["--server", "--config", str(config_path)])

    # Then uvicorn ran in the foreground bound to the config host/port
    assert exit_code == 0
    run_mock.assert_called_once()
    assert run_mock.call_args.kwargs["host"] == "127.0.0.1"
    assert run_mock.call_args.kwargs["port"] == 8741
    # And no window was ever created or started
    create_window_mock.assert_not_called()
    start_mock.assert_not_called()


def test_window_mode_runs_uvicorn_on_background_thread_and_webview_on_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given a valid config file and mocked uvicorn/webview/threading boundaries
    config_path = _write_config(tmp_path)
    run_mock = MagicMock()
    create_window_mock = MagicMock()
    start_mock = MagicMock()
    threads: list[_FakeThread] = []
    monkeypatch.setattr(web_main.uvicorn, "run", run_mock)
    monkeypatch.setattr(web_main.webview, "create_window", create_window_mock)
    monkeypatch.setattr(web_main.webview, "start", start_mock)

    def _record_thread(
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        daemon: bool = False,
    ) -> _FakeThread:
        thread = _FakeThread(target=target, args=args, kwargs=kwargs or {}, daemon=daemon)
        threads.append(thread)
        return thread

    monkeypatch.setattr(web_main.threading, "Thread", _record_thread)

    # When invoking the entry point in default (window) mode
    exit_code = web_main.main(["--config", str(config_path)])

    # Then uvicorn was scheduled on a daemon background thread (not run on the main thread)
    assert exit_code == 0
    run_mock.assert_not_called()
    assert len(threads) == 1
    assert threads[0].target is run_mock
    assert threads[0].kwargs["host"] == "127.0.0.1"
    assert threads[0].kwargs["port"] == 8741
    assert threads[0].daemon is True
    # And a pywebview window pointing at the server was created and started on the main thread
    create_window_mock.assert_called_once_with(
        "Taste Pipeline", "http://127.0.0.1:8741", width=1100, height=750
    )
    start_mock.assert_called_once_with()


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    # Given a --config path that does not exist
    missing = tmp_path / "absent.toml"

    # When invoking the entry point
    # Then load_config's ConfigError propagates (CLI exits non-zero)
    with pytest.raises(ConfigError):
        web_main.main(["--config", str(missing)])


def test_argparse_defaults_to_window_mode_and_optional_config(tmp_path: Path) -> None:
    # Given only a --config argument (no --server)
    config_path = _write_config(tmp_path)

    # When parsing arguments
    args = web_main._build_parser().parse_args(["--config", str(config_path)])

    # Then server mode is off and the config path is captured
    assert args.server is False
    assert args.config is not None
