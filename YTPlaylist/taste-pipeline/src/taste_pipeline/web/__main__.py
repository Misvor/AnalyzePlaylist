"""Entry point: ``python -m taste_pipeline.web``.

Default mode serves the FastAPI app on ``config.web_host``/``config.web_port``
from a daemon background thread, then opens a pywebview window on the main
thread. ``--server`` runs uvicorn in the foreground instead (headless, for
tests and plain browsers). ``--config PATH`` is passed to
:func:`taste_pipeline.config.load_config`.

UX: when ``load_config`` raises :class:`~taste_pipeline.config.ConfigError`
because the config file is missing, this entry point prints a self-explanatory
fix (pass ``--config``, set ``$TASTE_PIPELINE_CONFIG``, or copy
``config.example.toml`` to ``config.toml`` in the cwd) and exits non-zero
instead of just re-raising the bare exception. Other ``ConfigError`` kinds
(malformed TOML, invalid field values) still propagate unchanged so the
caller sees the original error message. The fix lives here, not in
``load_config``, because ``load_config`` is also called from
``tests/conftest.py`` with a tmp path that always exists -- a "not found"
error from a test should not be silently rewritten into a UX message.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import uvicorn
import webview

from taste_pipeline.config import ConfigError, load_config
from taste_pipeline.web.app import create_app

_WINDOW_WIDTH: Final = 1100
_WINDOW_HEIGHT: Final = 750

_CONFIG_NOT_FOUND_HINT: Final = (
    "\nNo config.toml found in cwd.\n"
    "\n"
    "Fix:\n"
    "  1. Pass --config <path>\n"
    "  2. Set TASTE_PIPELINE_CONFIG=<path>\n"
    "  3. Copy config.example.toml to config.toml in your cwd\n"
    "     cp config.example.toml config.toml\n"
)


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
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        if "not found" in str(exc):
            _ = sys.stderr.write(f"{exc}\n{_CONFIG_NOT_FOUND_HINT}")
            return 2
        raise
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
