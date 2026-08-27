"""Entry point: ``python -m taste_pipeline.web``.

Default mode serves the FastAPI app on ``config.web_host``/``config.web_port``
from a daemon background thread, then opens a pywebview window on the main
thread. ``--server`` runs uvicorn in the foreground instead (headless, for
tests and plain browsers). ``--config PATH`` is passed to
:func:`taste_pipeline.config.load_config`.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import uvicorn
import webview

from taste_pipeline.config import load_config
from taste_pipeline.web.app import create_app

_WINDOW_WIDTH: Final = 1100
_WINDOW_HEIGHT: Final = 750


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the web entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m taste_pipeline.web",
        description="Serve the Taste Pipeline web UI (pywebview window by default).",
    )
    _ = parser.add_argument(
        "--server",
        action="store_true",
        help="run uvicorn in the foreground (headless; no desktop window)",
    )
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to config.toml (passed to load_config)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, load config, and run either the server or the desktop window.

    Returns 0 when the server (or window loop) exits. ``ConfigError`` from
    ``load_config`` propagates, so the CLI exits non-zero on a bad config.
    """
    args = _build_parser().parse_args(argv)
    config_path = cast("Path | None", args.config)
    server_mode = cast("bool", args.server)
    config = load_config(config_path)
    app = create_app(config)
    host = config.web_host
    port = config.web_port
    if server_mode:
        uvicorn.run(app, host=host, port=port)
        return 0
    server_thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": host, "port": port},
        daemon=True,
    )
    server_thread.start()
    # pywebview's stub for create_window has an upstream Unknown in the url
    # parameter type; the inline directive below pins the third-party gap.
    _ = webview.create_window(  # pyright: ignore[reportUnknownMemberType]
        "Taste Pipeline",
        f"http://{host}:{port}",
        width=_WINDOW_WIDTH,
        height=_WINDOW_HEIGHT,
    )
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
