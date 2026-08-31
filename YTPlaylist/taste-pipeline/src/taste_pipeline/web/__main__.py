"""Entry point: ``python -m taste_pipeline.web``.

Default mode serves the FastAPI app on ``config.web_host``/``config.web_port``
from a daemon background thread, then opens a pywebview window on the main
thread. ``--server`` runs uvicorn in the foreground instead (headless, for
tests and plain browsers). ``--config PATH`` is passed to
:func:`taste_pipeline.config.load_config`.

In window mode the entry point also exposes two native dialog helpers
(:func:`open_directory_dialog`, :func:`open_file_dialog`) to the page via
``window.expose``, so the ``/settings`` page can launch the host OS file /
directory picker and receive the absolute path back. These functions are
no-ops when no pywebview window is registered (e.g. when the module is
imported in tests or ``--server`` mode), so importing this module is safe
in any context.

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
from typing import TYPE_CHECKING, Any, Final, cast

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


def open_directory_dialog() -> str | None:
    """Open a native OS directory picker on the live pywebview window.

    Returns the absolute path the user selected, or ``None`` if cancelled
    or no pywebview window is registered. Must be exposed via
    ``window.expose`` on the same thread as ``webview.create_window``
    (done in :func:`main`); the server thread cannot call this safely.
    """
    if not webview.windows:
        return None
    result = webview.windows[0].create_file_dialog(
        webview.FileDialog.FOLDER,
        directory=str(Path.home()),
        allow_multiple=False,
    )
    return result[0] if result else None


def open_file_dialog() -> str | None:
    """Open a native OS file picker on the live pywebview window.

    Returns the absolute path the user selected, or ``None`` if cancelled
    or no pywebview window is registered. See :func:`open_directory_dialog`
    for the threading contract.
    """
    if not webview.windows:
        return None
    result = webview.windows[0].create_file_dialog(
        webview.FileDialog.OPEN,
        directory=str(Path.home()),
        allow_multiple=False,
        file_types=("All files (*.*)",),
    )
    return result[0] if result else None


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
    app = create_app(config, config_path=config_path)
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
    # parameter type; `Any` + per-call ignores below pin the third-party gap.
    window: Any = webview.create_window(  # pyright: ignore[reportUnknownMemberType, reportExplicitAny]
        "Taste Pipeline",
        f"http://{host}:{port}",
        width=_WINDOW_WIDTH,
        height=_WINDOW_HEIGHT,
    )
    # Expose native dialog helpers to JS as window.pywebview.api.<name>().
    # Must run on the same thread as create_window (i.e. the main thread,
    # before webview.start) — pywebview's event loop marshals the call back
    # to the main thread only when the binding is set up here.
    window.expose(open_directory_dialog)  # pyright: ignore[reportAny]
    window.expose(open_file_dialog)  # pyright: ignore[reportAny]
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
