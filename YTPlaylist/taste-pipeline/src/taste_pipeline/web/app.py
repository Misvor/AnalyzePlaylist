"""FastAPI application factory for the taste-pipeline web UI.

The factory takes a validated :class:`~taste_pipeline.config.Config` (built
elsewhere by ``load_config``) so callers and tests can inject a temporary
``data_dir``. Static assets and Jinja2 templates are resolved relative to
this package directory, so the app works regardless of the process cwd.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from taste_pipeline.web.api import router as api_router
from taste_pipeline.web.jobs import JobRunner
from taste_pipeline.web.routes_check import router as check_router
from taste_pipeline.web.routes_index import build_index_status
from taste_pipeline.web.routes_index import router as index_router
from taste_pipeline.web.routes_jobs import router as jobs_router
from taste_pipeline.web.routes_jobs import run_row_context
from taste_pipeline.web.routes_settings import router as settings_router
from taste_pipeline.web.routes_triage import router as triage_router

if TYPE_CHECKING:
    from fastapi.responses import Response

    from taste_pipeline.config import Config

_PACKAGE_DIR: Final = Path(__file__).resolve().parent
_STATIC_DIR: Final = _PACKAGE_DIR / "static"
_TEMPLATES_DIR: Final = _PACKAGE_DIR / "templates"


def create_app(config: Config, *, config_path: Path | None = None) -> FastAPI:
    """Build the FastAPI app for an already-validated pipeline config.

    Mounts the package ``static/`` dir at ``/static``, registers the package
    ``templates/`` dir with ``Jinja2Templates``, stores ``config`` on
    ``app.state.config`` for later route modules, and serves the base HTML
    shell at ``GET /``.

    Args:
        config: Validated pipeline config (built by ``load_config``).
        config_path: Filesystem path the config was loaded from; required
            for ``POST /api/config`` to persist edits back to the same
            file. When ``None`` (the default), the settings editor
            refuses to write -- the page still renders, but the form
            is disabled and ``POST /api/config`` returns 409.
    """
    app = FastAPI(title="Taste Pipeline")
    app.state.config = config
    # Snapshot of the currently-running config; set once at app construction and NEVER updated
    # by POST /api/config. The settings page compares it to on-disk config.toml to detect
    # unapplied path / network changes (file != running = banner). Rebuilt on next restart.
    app.state.running_config = config
    app.state.config_path = config_path
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=_TEMPLATES_DIR)

    async def _index(request: Request) -> Response:
        """Render the dashboard with the index status panel and server-rendered run history."""
        config = cast("Config", app.state.config)
        index_status = build_index_status(config)
        runner = cast("JobRunner", app.state.runner)
        jobs = [run_row_context(job) for job in runner.list()]
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {"index_status": index_status, "jobs": jobs},
        )

    async def _index_page(request: Request) -> Response:
        """Render the dedicated Index page: index-scan status panel + live progress UI."""
        config = cast("Config", app.state.config)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"index_status": build_index_status(config)},
        )

    _ = app.get("/")(_index)
    _ = app.get("/index")(_index_page)
    app.include_router(api_router)
    app.include_router(index_router)
    app.include_router(jobs_router)
    app.include_router(settings_router)
    app.include_router(triage_router)
    app.include_router(check_router)
    app.state.runner = JobRunner(config.data_dir)
    app.state.templates = templates
    # env.globals is the Jinja2 seam that makes config values reachable from
    # every template render (status bar etc.) without per-render context wiring.
    # Jinja2's stubs narrow globals to a fixed value-type union; runtime accepts any value.
    templates.env.globals["server_host"] = config.web_host  # pyright: ignore[reportArgumentType]
    templates.env.globals["server_port"] = config.web_port  # pyright: ignore[reportArgumentType]
    return app
