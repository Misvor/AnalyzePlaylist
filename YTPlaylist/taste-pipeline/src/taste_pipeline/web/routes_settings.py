"""Settings page + POST /api/config for the taste-pipeline web UI.

Renders the :class:`~taste_pipeline.config.Config` as an editable form.
Every field is editable from the GUI, but path / network knobs require
a process restart before their values take effect:

- **Live-edit fields** (``keep_threshold``, ``skip_threshold``,
  ``min_dislikes_for_classifier``, ``feed_window_days``,
  ``max_feed_items``, ``max_metadata_fetch``, ``chunk_seconds``,
  ``min_chunk_seconds``, ``sample_rate``, ``model_name``): the POST
  handler writes the new value to ``app.state.config`` AND atomically
  rewrites the on-disk ``config.toml``. The running process picks up
  the new value immediately (e.g. the next calibrated threshold is
  the new ``keep_threshold``).

- **Path / network fields** (``like_library_dir``, ``data_dir``,
  ``cookie_file``, ``download_archive``, ``web_host``, ``web_port``):
  the POST handler validates and persists the new value to
  ``config.toml``, and updates ``app.state.config`` so subsequent
  GETs render the new value in the form. The running server is
  still bound to the old port, has the old runner, and reads / writes
  the old paths -- the values only take effect after a process
  restart. The settings page detects this by comparing the
  on-disk file to ``app.state.running_config`` (the snapshot taken
  at app startup) and shows a "restart required" banner for every
  path / network field that has changed.

The page distinguishes the two groups visually with a "Storage" +
"Network" subsection inside the editable form; the legend explains
that changes there will only take effect after restart.

The atomic write is ``<config.toml>.tmp`` followed by
``Path.replace()``; if the process is killed mid-write the original
file is unchanged, and a future :func:`load_config` call sees the
original. Without ``.tmp + replace`` a SIGKILL between ``write()`` and
``fsync()`` would leave the user with a half-written config.toml --
unrecoverable without a manual editor pass.

After a successful edit, ``app.state.config`` is replaced with a new
``Config`` instance built from ``load_config(config_path)`` so the
running process picks up the new values for live-edit fields. Path
edits update ``app.state.config`` too (so subsequent GETs render the
new value in the form), but ``app.state.running_config`` is NOT
updated -- it remains the snapshot of what is CURRENTLY RUNNING,
which is what the banner compares against to detect unapplied path
changes. The next process restart will rebuild
``app.state.running_config`` from the freshly-loaded config, and the
banner disappears because file == running.

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
    from collections.abc import Callable

    from fastapi.templating import Jinja2Templates

    from taste_pipeline.config import Config


router = APIRouter()

# Upper bound of the TCP user-port range per RFC 6335 §3.
_TCP_PORT_MAX: Final[int] = 65535

# Field classification: editable values can be changed at runtime via
# POST /api/config. The classification is a single source of truth --
# the template consults it to decide which fields get an <input>, the
# POST handler consults it to decide which keys are accepted in the
# body, and the unapplied-changes detector consults the path/network
# subset to decide which fields to compare against the running
# snapshot.
#
# The split is "live-edit" vs "path / network":
#
# - live-edit: changes take effect immediately (calibration
#   thresholds, pipeline scalars, model name).
# - path / network: persisted to config.toml but only take effect
#   after a process restart (runner directories, network bind).
#
# The previous implementation split this into editable vs read-only
# (with paths rejected by the POST handler). That has changed:
# path / network fields ARE editable now, but the running process
# keeps using the old values until restart.
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
        "like_library_dir",
        "data_dir",
        "cookie_file",
        "download_archive",
        "web_host",
        "web_port",
    }
)
_PATH_NETWORK_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "like_library_dir",
        "data_dir",
        "cookie_file",
        "download_archive",
        "web_host",
        "web_port",
    }
)
# Empty for documentation purposes: every Config field is editable
# via the form. The POST handler still validates path / network
# fields (path existence checks, port range) and persists them to
# disk, but the in-memory config picks them up only after restart.
_READONLY_FIELDS: Final[frozenset[str]] = frozenset()


def _config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.config`` to a typed ``Config`` for basedpyright."""
    return cast("Config", cast("FastAPI", request.app).state.config)


def _running_config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.running_config`` to a typed ``Config`` for basedpyright.

    ``running_config`` is the snapshot of what the running process is
    CURRENTLY using -- it is set once at app construction and never
    updated by the POST handler. Comparing it to the on-disk
    ``config.toml`` is how the page detects unapplied path / network
    changes ("you saved a new port, but the server is still bound to
    the old one until restart").
    """
    return cast("Config", cast("FastAPI", request.app).state.running_config)


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
      (the template renders an ``<input>`` for these). The
      path / network fields are also editable -- the
      "restart required" banner is rendered separately, not as a
      per-row badge.

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


def _detect_unapplied_paths(running: Config, on_disk: dict[str, object]) -> list[dict[str, str]]:
    """Compute the list of path / network fields that differ between the running config and disk.

    For each field in :data:`_PATH_NETWORK_FIELDS`:

    - read the running value from ``running.<field>``;
    - read the on-disk value from ``on_disk.get(field)`` (a string
      from TOML, or ``None`` if the file doesn't set it -- in which
      case the next load will fall back to the dataclass default, so
      we use the running value as the comparison baseline);
    - normalize both to a portable forward-slash string (TOML-canonical
      on every OS). ``str(Path)`` on Windows uses backslashes, which
      do not byte-equal the forward-slash form that the TOML writer
      used, even when they refer to the same file;
    - if the normalized forms differ, emit one entry
      ``{"name", "old", "new"}``.

    Returns an empty list when the file is in sync with the running
    snapshot (the common case at app startup, and after a restart).
    """
    out: list[dict[str, str]] = []
    for f in fields(running):
        if f.name not in _PATH_NETWORK_FIELDS:
            continue
        running_value = cast("object", getattr(running, f.name))
        on_disk_raw = on_disk.get(f.name)
        # Missing-in-file means "use the dataclass default" on the next load,
        # which is the value the running config already has -- so no divergence.
        if on_disk_raw is None:
            continue
        # Normalize to forward-slash form: TOML values are stored verbatim,
        # but ``str(Path)`` on Windows uses backslashes. ``Path.as_posix()``
        # is the canonical comparison form on every platform.
        if isinstance(running_value, Path):
            running_str = running_value.as_posix()
            on_disk_str = Path(str(on_disk_raw)).as_posix()
        else:
            running_str = str(running_value)
            on_disk_str = str(on_disk_raw)
        if running_str != on_disk_str:
            out.append({"name": f.name, "old": running_str, "new": on_disk_str})
    return out


@router.get("/settings", response_class=Response)
async def settings_page(request: Request) -> Response:
    """Render the editable Settings page: one form row per Config field.

    Every field gets an ``<input>``. The submit button POSTs only the
    editable keys (path / network fields ARE in the form payload --
    they're editable -- but the ``collectEditableFields`` JS gathers
    them into the same JSON object as the live-edit fields).

    The "restart required" banner is rendered when the on-disk
    ``config.toml`` differs from ``app.state.running_config`` for any
    path / network field -- this is the signal that the user has
    saved new values that won't take effect until the process is
    restarted.
    """
    cfg = _config_from_request(request)
    running_cfg = _running_config_from_request(request)
    rows = build_settings_rows(cfg)
    config_path = _config_path_from_request(request)
    templates = _templates_from_request(request)
    on_disk = _read_existing_config(config_path) if config_path is not None else {}
    unapplied_paths = _detect_unapplied_paths(running_cfg, on_disk)
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "rows": rows,
            "config_path": str(config_path) if config_path is not None else None,
            "unapplied_paths": unapplied_paths,
        },
    )


# ── POST /api/config ─────────────────────────────────────────────────


# Type aliases: the request body shape and the post-validate payload.
# Both are plain dict[str, object] (the JSON body is parsed manually,
# not via Pydantic models, matching the project's "manual validation"
# pattern from routes_jobs.py).


def _validate_int_field(name: str, raw: object, *, minimum: int | None = None) -> int:
    """Coerce ``raw`` to int (rejecting bool); raise on validation error."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        message = f"{name}: must be an int (got {type(raw).__name__})"
        raise _FieldValidationError(message)
    if minimum is not None and raw < minimum:
        message = f"{name}: must be >= {minimum} (got {raw})"
        raise _FieldValidationError(message)
    return raw


def _validate_float_field(
    name: str, raw: object, *, minimum: float | None = None, maximum: float | None = None
) -> float:
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


def _validate_str_field(name: str, raw: object, *, max_length: int | None = None) -> str:
    """Coerce ``raw`` to a non-empty string; optional max length check."""
    if not isinstance(raw, str):
        message = f"{name}: must be a string (got {type(raw).__name__})"
        raise _FieldValidationError(message)
    if not raw:
        message = f"{name}: must be a non-empty string"
        raise _FieldValidationError(message)
    if max_length is not None and len(raw) > max_length:
        message = f"{name}: must be at most {max_length} characters (got {len(raw)})"
        raise _FieldValidationError(message)
    return raw


def _validate_like_library_dir_field(name: str, raw: object) -> str:
    """Coerce ``raw`` to a ``like_library_dir`` path string.

    ``like_library_dir`` must be an EXISTING directory at the time of
    validation -- ``load_config`` rejects a non-existent path with
    "does not exist or is not a directory" (the runner does not
    auto-create it; it only creates the data_dir subtree). If the
    path exists but is not a directory, reject (no silently-typing-a-
    file-into-the-library-dir).
    """
    s = _validate_str_field(name, raw)
    p = Path(s)
    if p.exists() and not p.is_dir():
        message = f"{name}: path exists but is not a directory ({s!r})"
        raise _FieldValidationError(message)
    if not p.exists():
        message = f"{name}: directory does not exist ({s!r}); the runner does not auto-create it"
        raise _FieldValidationError(message)
    return s


def _validate_data_dir_field(name: str, raw: object) -> str:
    """Coerce ``raw`` to a ``data_dir`` path string.

    ``data_dir`` is auto-created by ``load_config`` if missing
    (``data_dir.mkdir(parents=True, exist_ok=True)``); so the POST
    handler only needs to reject:

    - non-string input,
    - an existing path that is NOT a directory (a file pointing at
      the data location is nonsense),
    - a non-existent path whose parent does not exist (the runner
      could not create ``data_dir`` if its parent is missing).
    """
    s = _validate_str_field(name, raw)
    p = Path(s)
    if p.exists() and not p.is_dir():
        message = f"{name}: path exists but is not a directory ({s!r})"
        raise _FieldValidationError(message)
    if not p.exists() and not p.parent.exists():
        message = f"{name}: parent directory does not exist ({s!r})"
        raise _FieldValidationError(message)
    return s


def _validate_file_path_field(name: str, raw: object) -> str:
    """Coerce ``raw`` to a file path string; path may be absent (parent must exist).

    The file may be absent (yt-dlp will load it on first use; the
    file's existence is not the POST handler's responsibility). If it
    exists, it must be a regular file (not a directory). If it does
    NOT exist, its parent must exist (so the runner / yt-dlp can
    create it on first use).
    """
    s = _validate_str_field(name, raw)
    p = Path(s)
    if p.exists() and not p.is_file():
        message = f"{name}: path exists but is not a file ({s!r})"
        raise _FieldValidationError(message)
    if not p.exists() and not p.parent.exists():
        message = f"{name}: parent directory does not exist ({s!r})"
        raise _FieldValidationError(message)
    return s


def _validate_web_host_field(name: str, raw: object) -> str:
    """Coerce ``raw`` to a host string (non-empty, max 255 chars).

    Light validation: any non-empty string up to 255 chars is
    accepted (covers "0.0.0.0", "127.0.0.1", "localhost", IPv6 forms,
    hostnames). Full IP / hostname parsing is overkill for a
    loopback-only deployment.
    """
    return _validate_str_field(name, raw, max_length=255)


def _validate_web_port_field(name: str, raw: object) -> int:
    """Coerce ``raw`` to an int in ``[1, 65535]`` (TCP port range)."""
    v = _validate_int_field(name, raw)
    if v < 1 or v > _TCP_PORT_MAX:
        message = f"{name}: must be in [1, {_TCP_PORT_MAX}] (got {v})"
        raise _FieldValidationError(message)
    return v


class _FieldValidationError(ValueError):
    """Internal exception type for one-field validation failures.

    Bubbled up to :func:`_parse_config_update` which collects them into
    a 422 response. Distinct from :class:`ValueError` so callers can
    wrap their own validation in ``try`` without catching ours.
    """


def _validate_keep_skip(name: str, raw: object) -> float | None:
    """Bound the optional-float thresholds to ``[0.0, 1.0]``."""
    return _validate_optional_float_field(name, raw, minimum=0.0, maximum=1.0)


def _validate_int_min1(name: str, raw: object) -> int:
    """Bound the integer pipeline scalars to ``>= 1``."""
    return _validate_int_field(name, raw, minimum=1)


def _validate_int_min0(name: str, raw: object) -> int:
    """Bound the integer classifier trigger to ``>= 0``."""
    return _validate_int_field(name, raw, minimum=0)


def _validate_float_min0(name: str, raw: object) -> float:
    """Bound the float chunk sizes to ``>= 0.0``."""
    return _validate_float_field(name, raw, minimum=0.0)


# Per-field dispatch table: Config field name -> validator. New editable fields
# get a one-line entry here (and a matching entry in ``_EDITABLE_FIELDS``).
_FIELD_VALIDATORS: Final[dict[str, Callable[[str, object], object]]] = {
    "keep_threshold": _validate_keep_skip,
    "skip_threshold": _validate_keep_skip,
    "min_dislikes_for_classifier": _validate_int_min0,
    "feed_window_days": _validate_int_min1,
    "max_feed_items": _validate_int_min1,
    "max_metadata_fetch": _validate_int_min1,
    "sample_rate": _validate_int_min1,
    "chunk_seconds": _validate_float_min0,
    "min_chunk_seconds": _validate_float_min0,
    "model_name": _validate_str_field,
    "like_library_dir": _validate_like_library_dir_field,
    "data_dir": _validate_data_dir_field,
    "cookie_file": _validate_file_path_field,
    "download_archive": _validate_file_path_field,
    "web_host": _validate_web_host_field,
    "web_port": _validate_web_port_field,
}


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
            422. Currently empty (every Config field is editable);
            kept as a parameter so a future field can be marked
            truly read-only without touching the dispatch logic.

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
        validator = _FIELD_VALIDATORS.get(name)
        if validator is None:  # pragma: no cover -- defensive guard for unhandled editable field
            message = f"{name}: no validator registered (config drift; update routes_settings.py)"
            raise _FieldValidationError(message)
        changes[name] = validator(name, raw)
    return changes


def _cross_field_checks(
    current: Config,
    changes: dict[str, object],
) -> None:
    """Apply cross-field invariants (chunk < min_chunk; skip < keep) AFTER coercion.

    Mirrors the cross-field checks in :mod:`taste_pipeline.config` but
    on a partial-update dict, so a single edit cannot break invariants
    while leaving the rest of the config intact. Path / network
    fields have no cross-checks (they're independent).
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
    :func:`load_config` on next load). Also used by the GET handler to
    compare against the running snapshot for unapplied-changes
    detection.
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
    only the fields being changed. Path / network fields ARE accepted
    (with validation: directory paths must be dirs or have a parent,
    file paths must not point at directories, ``web_port`` must be in
    ``[1, 65535]``, ``web_host`` is a non-empty string). On success:

    1. The on-disk ``config.toml`` is rewritten atomically
       (``.tmp + Path.replace``), preserving every field not in the
       body.
    2. ``app.state.config`` is reloaded from disk so the in-memory
       state matches the file exactly -- subsequent GETs show the new
       values in the form.
    3. ``app.state.running_config`` is NOT updated: it stays at the
       snapshot from app startup. The next GET /settings detects the
       divergence and renders the "restart required" banner for every
       path / network field whose value differs between the two.
    4. The Jinja ``env.globals['server_host']`` /
       ``env.globals['server_port']`` are refreshed so the status bar
       picks up the new ``web_host`` / ``web_port`` once the process
       is restarted.

    Returns the freshly-loaded Config as a dict (every field, including
    the unchanged ones, so the client can replace its local copy).

    Raises:
        HTTPException 409: ``config_path`` was ``None`` at app
            construction (the settings editor cannot persist without a
            known file path).
        HTTPException 422: body is not JSON, not an object, contains
            unknown keys, fails per-field validation, or breaks
            cross-field invariants (chunk < min_chunk, skip < keep).
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
    # (unknown) -- those were never in ``changes``.
    existing = _read_existing_config(config_path)
    merged: dict[str, object] = dict(existing)
    merged.update(changes)
    _write_config_atomic(config_path, merged)

    # Reload the config from disk so the in-memory state matches the
    # file exactly. This catches any round-trip normalization that the
    # writer applies (e.g. bool -> "true"/"false") and keeps the
    # in-memory config in sync with the on-disk file. NOTE: this
    # updates app.state.config (so subsequent GETs show the new form
    # values) but does NOT update app.state.running_config -- the
    # running snapshot is what the banner compares against to detect
    # unapplied path / network changes.
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
    "_PATH_NETWORK_FIELDS",
    "_READONLY_FIELDS",
    "build_settings_rows",
    "router",
]
