"""Read-only Settings page for the taste-pipeline web UI.

Renders every :class:`~taste_pipeline.config.Config` field as a
two-column table so the user can audit their pipeline setup from the
browser without opening ``config.toml`` in a text editor. The page is
deliberately a strict mirror of the JSON ``GET /api/config`` endpoint
(todo-3): same field set, same value coercion (Path -> str, scalars
pass through), no editing, no POST, no form state.

Endpoint:

- ``GET /settings`` -- render ``templates/settings.html`` with one row
  per Config field. Path fields are rendered as portable strings
  (forward-slash on POSIX, drive-letter on Windows) so the user sees
  the path they can paste into a shell, not a Python ``Path`` repr.
  Optional numeric fields (``keep_threshold``, ``skip_threshold``)
  render as ``None`` when unset so the user can see "this is unset"
  instead of a blank cell that might mean "default" or "broken".

Security: the deployment is loopback-only (``config.web_host =
127.0.0.1`` by default), so a CSRF layer is not required. The
``cookie_file`` field carries the file path string only -- the page
NEVER reads the cookie file. A future regression that called
``config.cookie_file.read_text()`` and inlined the contents into the
HTML would leak the user's session cookies into the page DOM (and
therefore into browser view-source, screenshots, etc.); the
``test_get_settings_does_not_render_cookie_file_contents`` test
locks this with a sentinel value.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from taste_pipeline.config import Config


router = APIRouter()

# CSS-friendly kind labels used by the template to pick a value-cell
# style (e.g. font-family for paths, italic for "None"). Path and None
# are handled by the caller before this dict is consulted; the entries
# cover the remaining scalar types Config carries.
_KIND_BY_TYPE: Final[dict[type, str]] = {
    bool: "bool",
    int: "int",
    float: "float",
    str: "str",
}


def _config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.config`` to a typed ``Config`` for basedpyright."""
    return cast("Config", cast("FastAPI", request.app).state.config)


def _templates_from_request(request: Request) -> Jinja2Templates:
    """Cast ``request.app.state.templates`` to a typed ``Jinja2Templates``.

    :func:`taste_pipeline.web.app.create_app` sets this attribute so
    route modules do not need to know the templates directory layout.
    """
    return cast("Jinja2Templates", cast("FastAPI", request.app).state.templates)


def _kind_name(value: object) -> str:
    """Return a CSS-friendly kind label for a scalar value (int / float / str / bool / other)."""
    for py_type, label in _KIND_BY_TYPE.items():
        if isinstance(value, py_type):
            return label
    return "other"


def build_settings_rows(config: Config) -> list[dict[str, object]]:
    """Serialize every ``Config`` field to one table row.

    Each row: ``{"name", "value", "kind"}``:

    - ``name`` -- the dataclass field name (e.g. ``"web_port"``).
    - ``value`` -- the user-facing value: ``str(Path)`` for ``Path``
      fields (portable, no ``PosixPath(...)`` repr); the literal
      string ``"None"`` for unset optional fields; the scalar itself
      for everything else.
    - ``kind`` -- one of ``"path"``, ``"none"``, ``"int"``, ``"float"``,
      ``"str"``, ``"bool"``, ``"other"``; the template uses this to
      pick a value-cell style.

    Rows appear in ``dataclasses.fields(Config)`` order (the order the
    dataclass declared them), so the page is deterministic across
    requests without needing a separate sort.
    """
    rows: list[dict[str, object]] = []
    for f in fields(config):
        value = cast("object", getattr(config, f.name))
        if isinstance(value, Path):
            rows.append({"name": f.name, "value": str(value), "kind": "path"})
        elif value is None:
            # Render the literal string "None" so the user sees an
            # explicit "unset" marker instead of a blank cell (which
            # would be ambiguous with a default empty string).
            rows.append({"name": f.name, "value": "None", "kind": "none"})
        else:
            rows.append({"name": f.name, "value": value, "kind": _kind_name(value)})
    return rows


@router.get("/settings", response_class=Response)
async def settings_page(request: Request) -> Response:
    """Render the read-only Settings page: one row per Config field.

    Pulls the rows from :func:`build_settings_rows` so the page payload
    and any future JSON endpoint (or test) can share the same
    single-source-of-truth serialization. The page itself extends
    ``base.html`` (so the nav, action bar, and status bar are inherited)
    and carries the small "edit config.toml + restart" hint at the top.
    """
    cfg = _config_from_request(request)
    rows = build_settings_rows(cfg)
    templates = _templates_from_request(request)
    return templates.TemplateResponse(request, "settings.html", {"rows": rows})
