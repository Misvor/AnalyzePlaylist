"""Settings page + POST /api/config for the taste-pipeline web UI.

Renders the :class:`~taste_pipeline.config.Config` as an editable form,
not the strict mirror the previous read-only page exposed. The form is
deliberately split into two groups:

- **Editable fields** (``keep_threshold``, ``skip_threshold``,
  ``min_dislikes_for_classifier``, ``feed_window_days``,
  ``max_feed_items``, ``max_metadata_fetch``, ``chunk_seconds``,
  ``min_chunk_seconds``, ``sample_rate``, ``model_name``): live
  updates; ``POST /api/config`` writes the new value to
  ``app.state.config`` AND atomically rewrites the on-disk
  ``config.toml``.

- **Read-only fields** (``like_library_dir``, ``data_dir``,
  ``cookie_file``, ``download_archive``, ``web_host``, ``web_port``):
  path / network knobs that require a process restart to take effect.
  The form renders them as static text with a "(restart required)"
  badge. ``POST /api/config`` rejects any of these keys in the
  request body with 422 -- the contract is "client cannot move
  paths while the process is running".

The atomic write is ``<config.toml>.tmp`` followed by
``Path.replace()``; if the process is killed mid-write the original
file is unchanged, and a future :func:`load_config` call sees the
original. Without ``.tmp + replace`` a SIGKILL between ``write()`` and
``fsync()`` would leave the user with a half-written config.toml --
unrecoverable without a manual editor pass.

After a successful edit, ``app.state.config`` is replaced with a new
``Config`` instance built from ``load_config(config_path)`` so the
running process picks up the new values immediately. Path edits go
through ``load_config`` too -- they are simply rejected by the POST
handler, so the new Config is always the same as the on-disk file
(``load_config`` is the single source of truth for "what does the file
look like right now").

The deployment is loopback-only (``config.web_host = 127.0.0.1`` by
default), so CSRF is not a concern here. A future regression that
introduced remote access must add a CSRF layer BEFORE removing this
comment.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from taste_pipeline.config import Config


router = APIRouter()

# Field classification: editable values can be changed at runtime via
# POST /api/config; readonly fields are paths / network knobs that need
# a restart. The mapping is a single source of truth -- the template
# consults it to decide which fields get an <input> and which get a
# static "(restart required)" badge, and the POST handler consults it
# to reject path edits.
_EDITABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "keep_threshold",
        "skip_threshold",
        "min_dislikes_for_classifier",
        "feed_window_days",
        "max_feed_items",
        "max_metadata_fetch",
        "chunk_seconds",
        "min_chunk_seconds",
        "sample_rate",
        "model_name",
    }
)
_READONLY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "like_library_dir",
        "data_dir",
        "cookie_file",
        "download_archive",
        "web_host",
        "web_port",
    }
)


def _config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.config`` to a typed ``Config`` for basedpyright."""
    return cast("Config", cast("FastAPI", request.app).state.config)


def _config_path_from_request(request: Request) -> Path | None:
    """Cast ``request.app.state.config_path`` to a typed ``Path | None`` for basedpyright."""
    return cast("Path | None", cast("FastAPI", request.app).state.config_path)


def _templates_from_request(request: Request) -> Jinja2Templates:
    """Cast ``request.app.state.templates`` to a typed ``Jinja2Templates``."""
    return cast("Jinja2Templates", cast("FastAPI", request.app).state.templates)


def build_settings_rows(config: Config) -> list[dict[str, object]]:
    """Serialize every ``Config`` field to one form row.

    Each row: ``{"name", "value", "kind", "editable"}``:

    - ``name`` -- the dataclass field name.
    - ``value`` -- the user-facing value: ``str(Path)`` for ``Path``
      fields; the literal string ``"None"`` for unset optional fields;
      the scalar itself for everything else.
    - ``kind`` -- one of ``"path"``, ``"none"``, ``"int"``, ``"float"``,
      ``"str"``, ``"bool"``, ``"other"``.
    - ``editable`` -- True iff the field is in :data:`_EDITABLE_FIELDS`
      (the template renders an ``<input>`` for these); False for
      read-only fields (template renders a static badge).

    Rows appear in ``dataclasses.fields(Config)`` order so the page is
    deterministic across requests.
    """
    rows: list[dict[str, object]] = []
    for f in fields(config):
        value = cast("object", getattr(config, f.name))
        if isinstance(value, Path):
            row_value: object = str(value)
            kind = "path"
        elif value is None:
            row_value = "None"
            kind = "none"
        elif isinstance(value, bool):
            row_value = value
            kind = "bool"
        elif isinstance(value, int):
            row_value = value
            kind = "int"
        elif isinstance(value, float):
            row_value = value
            kind = "float"
        elif isinstance(value, str):
            row_value = value
            kind = "str"
        else:
            row_value = str(value)
            kind = "other"
        rows.append(
            {
                "name": f.name,
                "value": row_value,
                "kind": kind,
                "editable": f.name in _EDITABLE_FIELDS,
            }
        )
    return rows


@router.get("/settings", response_class=Response)
async def settings_page(request: Request) -> Response:
    """Render the editable Settings page: one form row per Config field.

    Path / network fields render as static text with a "(restart
    required)" badge; everything else gets an ``<input>`` bound to the
    current value. The submit button POSTs only the editable keys (the
    read-only fields are not in the form payload), so the POST handler
    does not need to filter path keys out of the body -- they are
    absent by construction.
    """
    cfg = _config_from_request(request)
    rows = build_settings_rows(cfg)
    config_path = _config_path_from_request(request)
    templates = _templates_from_request(request)
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"rows": rows, "config_path": str(config_path) if config_path is not None else None},
    )


# ── POST /api/config ─────────────────────────────────────────────────


# Type aliases: the request body shape and the post-validate payload.
# Both are plain dict[str, object] (the JSON body is parsed manually,
# not via Pydantic models, matching the project's "manual validation"
# pattern from routes_jobs.py).


def _validate_int_field(name: str, raw: object, *, minimum: int | None = None) -> int | None:
    """Coerce ``raw`` to int (rejecting bool); return None on validation error."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        message = f"{name}: must be an int (got {type(raw).__name__})"
        raise _FieldValidationError(message)
    if minimum is not None and raw < minimum:
        message = f"{name}: must be >= {minimum} (got {raw})"
        raise _FieldValidationError(message)
    return raw


def _validate_float_field(
    name: str, raw: object, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    """Coerce ``raw`` to float (rejecting bool); apply optional bounds."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        message = f"{name}: must be a number (got {type(raw).__name__})"
        raise _FieldValidationError(message)
    result = float(raw)
    if minimum is not None and result < minimum:
        message = f"{name}: must be >= {minimum} (got {result})"
        raise _FieldValidationError(message)
    if maximum is not None and result > maximum:
        message = f"{name}: must be <= {maximum} (got {result})"
        raise _FieldValidationError(message)
    return result


def _validate_optional_float_field(
    name: str,
    raw: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Coerce ``raw`` to float or None; None passes through."""
    if raw is None:
        return None
    return _validate_float_field(name, raw, minimum=minimum, maximum=maximum)


def _validate_str_field(name: str, raw: object) -> str:
    """Coerce ``raw`` to a non-empty string."""
    if not isinstance(raw, str):
        message = f"{name}: must be a string (got {type(raw).__name__})"
        raise _FieldValidationError(message)
    if not raw:
        message = f"{name}: must be a non-empty string"
        raise _FieldValidationError(message)
    return raw


class _FieldValidationError(ValueError):
    """Internal exception type for one-field validation failures.

    Bubbled up to :func:`_parse_config_update` which collects them into
    a 422 response. Distinct from :class:`ValueError` so callers can
    wrap their own validation in ``try`` without catching ours.
    """


def _parse_config_update(
    payload: dict[str, object],
    editable_fields: frozenset[str],
    readonly_fields: frozenset[str],
) -> dict[str, object]:
    """Validate ``payload`` and return a coerced dict of changes.

    Args:
        payload: Parsed JSON body from ``POST /api/config``. Keys are
            Config field names; values are the raw JSON-typed values.
        editable_fields: Whitelist of fields the POST may update.
        readonly_fields: Fields that, if present in the body, cause a
            422 (the contract: client cannot move paths while the
            process is running).

    Returns:
        A dict ``{field_name: coerced_value}`` suitable for passing to
        :func:`dataclasses.replace` on a ``Config`` instance.

    Raises:
        _FieldValidationError: any field failed validation. Callers
            convert this into an HTTPException 422.
    """
    changes: dict[str, object] = {}
    for name, raw in payload.items():
        if name in readonly_fields:
            message = f"{name}: is read-only (path/network knob; restart required)"
            raise _FieldValidationError(message)
        if name not in editable_fields:
            message = f"{name}: unknown field"
            raise _FieldValidationError(message)
        if name in ("keep_threshold", "skip_threshold"):
            changes[name] = _validate_optional_float_field(name, raw, minimum=0.0, maximum=1.0)
        elif name == "min_dislikes_for_classifier":
            changes[name] = _validate_int_field(name, raw, minimum=0)
        elif name in ("feed_window_days", "max_feed_items", "max_metadata_fetch") or name == "sample_rate":
            changes[name] = _validate_int_field(name, raw, minimum=1)
        elif name in ("chunk_seconds", "min_chunk_seconds"):
            changes[name] = _validate_float_field(name, raw, minimum=0.0)
        elif name == "model_name":
            changes[name] = _validate_str_field(name, raw)
        else:  # pragma: no cover -- defensive guard for an unhandled editable field
            message = f"{name}: no validator registered (config drift; update routes_settings.py)"
            raise _FieldValidationError(message)
    return changes


def _cross_field_checks(
    current: Config,
    changes: dict[str, object],
) -> None:
    """Apply cross-field invariants (chunk < min_chunk; skip < keep) AFTER coercion.

    Mirrors the cross-field checks in :mod:`taste_pipeline.config` but
    on a partial-update dict, so a single edit cannot break invariants
    while leaving the rest of the config intact.
    """
    new_chunk = cast("float | int | None", changes.get("chunk_seconds"))
    new_min_chunk = cast("float | int | None", changes.get("min_chunk_seconds"))
    chunk: float | int | None = (
        float(new_chunk)
        if isinstance(new_chunk, (int, float)) and not isinstance(new_chunk, bool)
        else current.chunk_seconds
    )
    min_chunk: float | int | None = (
        float(new_min_chunk)
        if isinstance(new_min_chunk, (int, float)) and not isinstance(new_min_chunk, bool)
        else current.min_chunk_seconds
    )
    if isinstance(chunk, float) and isinstance(min_chunk, float) and min_chunk > chunk:
        message = f"min_chunk_seconds ({min_chunk}) must not exceed chunk_seconds ({chunk})"
        raise _FieldValidationError(message)

    new_keep = cast("float | int | None", changes.get("keep_threshold"))
    new_skip = cast("float | int | None", changes.get("skip_threshold"))
    keep: float | int | None = (
        float(new_keep)
        if isinstance(new_keep, (int, float)) and not isinstance(new_keep, bool)
        else current.keep_threshold
    )
    skip: float | int | None = (
        float(new_skip)
        if isinstance(new_skip, (int, float)) and not isinstance(new_skip, bool)
        else current.skip_threshold
    )
    if isinstance(keep, float) and isinstance(skip, float) and skip >= keep:
        message = f"skip_threshold ({skip}) must be strictly less than keep_threshold ({keep})"
        raise _FieldValidationError(message)


def _write_config_atomic(config_path: Path, payload: dict[str, object]) -> None:
    """Serialize ``payload`` to TOML and write ``config_path`` atomically.

    The write strategy: serialize the new TOML to ``<path>.tmp`` first,
    then ``Path.replace()`` onto the original. ``replace()`` is atomic on
    POSIX and Windows (NTFS) for same-volume replacements, so a
    mid-write crash leaves either the original file unchanged or the
    new file in place -- never a half-written config.toml.

    The ``config_path.parent`` directory is created first so a fresh
    deployment that has only just supplied a config path can still be
    persisted to.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    body = _render_toml(payload)
    tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
    _ = tmp_path.write_text(body, encoding="utf-8")
    _ = tmp_path.replace(config_path)


def _render_toml(payload: dict[str, object]) -> str:
    """Render ``payload`` as a sorted, indented TOML document.

    A tiny hand-rolled writer: the only values ``POST /api/config``
    ever sends are scalars (int / float / str / bool / None), and we
    need byte-stable output so a write-then-read round-trips to the
    same canonical form. Python's stdlib does not ship a TOML writer
    (only a reader), and the project deliberately avoids pulling in
    ``tomli_w`` for one endpoint.
    """
    lines: list[str] = []
    for key in sorted(payload):
        value = payload[key]
        lines.append(f"{key} = {_toml_literal(value)}")
    return "\n".join(lines) + "\n"


def _toml_literal(value: object) -> str:
    """Render a single TOML scalar literal: int / float / str / bool / None."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # ``repr`` always emits a valid TOML float (includes ".0" for whole numbers).
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if value is None:
        # JSON null isn't valid TOML; write a single-line comment so the
        # file remains parseable (the field will need to be set to a
        # real value before loading again).
        return '"__unset__"  # TOML has no null; unset via the form'
    message = f"unsupported TOML literal type: {type(value).__name__}"
    raise TypeError(message)


def _read_existing_config(config_path: Path) -> dict[str, object]:
    """Load the existing ``config.toml`` and return its raw key/value dict.

    Used to build the merge payload that ``_write_config_atomic``
    writes -- the POST only contains the EDITED fields, but the file
    must contain every field (missing required keys would fail
    :func:`load_config` on next load).
    """
    if not config_path.is_file():
        return {}
    loaded = cast("object", tomllib.loads(config_path.read_text(encoding="utf-8")))
    if not isinstance(loaded, dict):
        return {}
    return cast("dict[str, object]", loaded)


@router.post("/api/config")
async def update_config(request: Request) -> dict[str, object]:
    """Validate a partial-update body and persist it to ``config.toml``.

    The body is a JSON object ``{field_name: value, ...}`` containing
    only the fields being changed; read-only fields are rejected if
    present (422). On success:

    1. The in-memory :class:`Config` is replaced (``dataclasses.replace``
       + ``app.state.config = new_config``) so the running process
       picks up the new values for non-path fields.
    2. The on-disk ``config.toml`` is rewritten atomically
       (``.tmp + Path.replace``), preserving every field not in the
    3. The Jinja ``env.globals['server_host']`` /
       ``env.globals['server_port']`` are refreshed so the status bar
       picks up any future web_host / web_port change once those
       become editable.

    Returns the freshly-loaded Config as a dict (every field, including
    the unchanged ones, so the client can replace its local copy).

    Raises:
        HTTPException 409: ``config_path`` was ``None`` at app
            construction (the settings editor cannot persist without a
            known file path).
        HTTPException 422: body is not JSON, not an object, contains
            unknown / read-only keys, fails per-field validation, or
            breaks cross-field invariants (chunk < min_chunk,
            skip < keep).
    """
    app = cast("FastAPI", request.app)
    config_path = _config_path_from_request(request)
    if config_path is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="config_path is unknown to this app; settings editor cannot persist",
        )

    try:
        raw: object = await request.json()  # pyright: ignore[reportAny]
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"invalid JSON body: {exc.msg}",
        ) from exc
    if not isinstance(raw, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"request body must be a JSON object, got {type(raw).__name__}",
        )
    payload = cast("dict[str, object]", raw)

    current = _config_from_request(request)
    try:
        changes = _parse_config_update(payload, _EDITABLE_FIELDS, _READONLY_FIELDS)
        _cross_field_checks(current, changes)
    except _FieldValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    # Build the merged payload to write: existing values + changes.
    # We don't include fields that the POST handler rejected
    # (read-only / unknown) -- those were never in ``changes``.
    existing = _read_existing_config(config_path)
    merged: dict[str, object] = dict(existing)
    merged.update(changes)
    _write_config_atomic(config_path, merged)

    # Reload the config from disk so the in-memory state matches the
    # file exactly. This catches any round-trip normalization that the
    # writer applies (e.g. bool -> "true"/"false") and keeps the
    # in-memory config in sync with the on-disk file.
    from taste_pipeline.config import (  # noqa: PLC0415 -- lazy: avoid app-startup cost
        load_config as _load_config,
    )

    new_config = _load_config(config_path)
    app.state.config = new_config

    templates = _templates_from_request(request)
    templates.env.globals["server_host"] = new_config.web_host  # pyright: ignore[reportArgumentType]
    templates.env.globals["server_port"] = new_config.web_port  # pyright: ignore[reportArgumentType]

    return _config_to_response_dict(new_config)


def _config_to_response_dict(config: Config) -> dict[str, object]:
    """Render a ``Config`` instance as a JSON-safe dict for the POST response.

    Path fields become ``str(Path)`` (matches the GET-side helper and
    the on-disk TOML representation). Unset optional fields become
    ``None`` (the JSON ``null``), matching the form input contract.
    """
    out: dict[str, object] = {}
    for f in fields(config):
        value = cast("object", getattr(config, f.name))
        if isinstance(value, Path):
            out[f.name] = str(value)
        else:
            out[f.name] = value
    return out


# Re-export for type-checked access in tests; not part of the public API.
__all__ = [
    "_EDITABLE_FIELDS",
    "_READONLY_FIELDS",
    "build_settings_rows",
    "router",
]
