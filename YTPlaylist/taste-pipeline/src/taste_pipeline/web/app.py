"""FastAPI application factory for the taste-pipeline web UI.

The factory takes a validated :class:`~taste_pipeline.config.Config` (built
elsewhere by ``load_config``) so callers and tests can inject a temporary
``data_dir``. Static assets and Jinja2 templates are resolved relative to
this package directory, so the app works regardless of the process cwd.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from taste_pipeline.web.api import router as api_router

if TYPE_CHECKING:
    from fastapi.responses import Response

    from taste_pipeline.config import Config

_PACKAGE_DIR: Final = Path(__file__).resolve().parent
_STATIC_DIR: Final = _PACKAGE_DIR / "static"
_TEMPLATES_DIR: Final = _PACKAGE_DIR / "templates"


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI app for an already-validated pipeline config.

    Mounts the package ``static/`` dir at ``/static``, registers the package
    ``templates/`` dir with ``Jinja2Templates``, stores ``config`` on
    ``app.state.config`` for later route modules, and serves the base HTML
    shell at ``GET /``.
    """
    app = FastAPI(title="Taste Pipeline")
    app.state.config = config
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=_TEMPLATES_DIR)

    async def _index(request: Request) -> Response:
        """Render the base HTML shell."""
        return templates.TemplateResponse(request, "base.html")

    _ = app.get("/")(_index)
    app.include_router(api_router)
    return app
