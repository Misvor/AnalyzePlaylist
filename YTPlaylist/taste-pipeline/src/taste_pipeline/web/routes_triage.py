r"""Inbox triage endpoints: HTML page, action API, and audio streaming.

The triage workflow lets the user listen to downloaded FLACs in the inbox
and assign each one to keep/review/skip/dislike. Every action moves the
FLAC (and its sibling ``<file>.meta.json`` sidecar) into the matching
data subdir and updates ``downloads.stage`` via
:meth:`taste_pipeline.state.StateStore.record_download` (REPLACE
semantics -- the canonical ``audio_path`` always points at the current
location). The audio route streams the file bytes so the HTML5
``<audio>`` element can play it.

Mounts four endpoints in one router (no ``prefix=`` because the four
URLs share no common prefix):

- ``GET  /api/triage`` -- JSON list of inbox items (filename, title,
  uploader, audio_path, video_id) for htmx partials / scripted clients.
- ``POST /api/triage/{video_id}`` body ``{"action": "keep"|"review"|"skip"|"dislike"}``
  -- move the file + sidecar to ``<data_dir>/<action>/`` and REPLACE
  the ``downloads`` row's ``audio_path`` + ``stage``. Idempotent on
  re-post.
- ``GET  /audio/{rel_path:path}`` -- stream a file from
  ``<data_dir>/<rel_path>`` via ``FileResponse``. Path-traversal
  attempts (any ``..`` segment, or any resolved path that escapes
  ``data_dir``) are rejected with 400.
- ``GET  /triage`` -- render ``templates/triage.html`` with the inbox
  items so the user can browse + act.

Security: the deployment is loopback-only (``config.web_host =
127.0.0.1`` by default), so a token auth layer is not required. The
audio route's only attack surface is the path-traversal guard in
:func:`_safe_under_data_dir`. All access to mutable state goes through
:meth:`StateStore.record_download` (the existing REPLACE primitive --
no new SQL).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from taste_pipeline.config import Config
    from taste_pipeline.state import StateStore

router = APIRouter()

# Subdir name (== action verb) -> past-tense stage stored in downloads.stage.
# ``review`` is both the verb and the noun here; the rest get past-tense forms.
_ACTION_TO_STAGE: dict[str, str] = {
    "keep": "kept",
    "review": "review",
    "skip": "skipped",
    "dislike": "disliked",
}
_VALID_ACTIONS: frozenset[str] = frozenset(_ACTION_TO_STAGE.keys())


class _TriageActionRequest(BaseModel):
    """Body schema for ``POST /api/triage/{video_id}``: plain ``str`` action (validated by the handler).

    The handler rejects unknown actions with 400 (not 422) so the test
    contract is "POST with bogus action -> 400"; Pydantic Literal
    validation would yield 422 instead, which the user-facing dashboard
    surfaces differently.
    """

    action: str


def _config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.config`` to a typed ``Config`` for basedpyright."""
    return cast("Config", cast("FastAPI", request.app).state.config)


def _templates_from_request(request: Request) -> Jinja2Templates:
    """Cast ``request.app.state.templates`` to a typed ``Jinja2Templates``.

    :func:`taste_pipeline.web.app.create_app` sets this attribute so
    route modules do not need to know the templates directory layout.
    """
    return cast("Jinja2Templates", cast("FastAPI", request.app).state.templates)


def _state_store_from_request(request: Request) -> StateStore:
    """Return a typed ``StateStore`` for ``request.app.state.config.data_dir``."""
    from taste_pipeline.state import StateStore  # noqa: PLC0415 -- lazy import keeps startup cheap

    return StateStore(_config_from_request(request).data_dir)


def _read_sidecar(audio_path: Path) -> dict[str, object]:
    """Parse ``<audio_path>.meta.json`` if present, else return ``{}`` (fail-open).

    A missing sidecar is not an error: the row still appears in the
    triage UI with null title/uploader. A malformed sidecar (bad JSON,
    non-dict payload) is also fail-open -- the row appears with empty
    metadata but the FLAC itself is still servable.
    """
    sidecar = audio_path.with_name(audio_path.name + ".meta.json")
    if not sidecar.is_file():
        return {}
    try:
        loaded_raw = cast("object", json.loads(sidecar.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded_raw, dict):
        return {}
    return cast("dict[str, object]", loaded_raw)


def _video_id_from_filename(audio_path: Path) -> str | None:
    """Extract the ``[video_id]`` marker from a yt-dlp outtmpl-style filename.

    The pipeline's outtmpl embeds ``[video_id]`` at the end of the stem
    (download.py:28 ``_OUTTMPL_NAME = "%(uploader).80s - %(title).80s [%(id)s].%(ext)s"``).
    Returns ``None`` when no marker is found -- the file is still
    servable, but cannot be triaged by id (the POST endpoint would
    reject it with 404 anyway).
    """
    stem = audio_path.stem
    if "[" not in stem or "]" not in stem:
        return None
    return stem.rsplit("[", 1)[1].split("]", 1)[0] or None


def _inbox_items(data_dir: Path) -> list[dict[str, object]]:
    """Build the inbox listing by globbing ``<data_dir>/inbox/*.flac`` and merging sidecars.

    Each entry: ``{video_id, filename, title, uploader, audio_path}``.
    Files with no sidecar get ``None`` for title/uploader. Files with no
    ``[video_id]`` marker get ``None`` for video_id. Sorted by filename
    for stable display across requests.
    """
    inbox = data_dir / "inbox"
    if not inbox.is_dir():
        return []
    items: list[dict[str, object]] = []
    for audio_path in sorted(inbox.glob("*.flac")):
        sidecar = _read_sidecar(audio_path)
        items.append(
            {
                "video_id": _video_id_from_filename(audio_path) or sidecar.get("video_id"),
                "filename": audio_path.name,
                "title": sidecar.get("title"),
                "uploader": sidecar.get("uploader"),
                "audio_path": str(audio_path),
            }
        )
    return items


@router.get("/api/triage")
async def list_inbox(request: Request) -> list[dict[str, object]]:
    """Return the inbox items as JSON (title/uploader from sibling sidecars).

    Used by htmx partials and scripted clients that want the raw
    listing without rendering through ``templates/triage.html``.
    """
    cfg = _config_from_request(request)
    return _inbox_items(cfg.data_dir)


@router.post("/api/triage/{video_id}")
async def triage_action(request: Request, video_id: str, body: _TriageActionRequest) -> dict[str, object]:
    """Move ``video_id``'s FLAC + sidecar to the action's subdir and update its stage.

    Action verbs map to subdirs and stages:

    +---------+----------+----------+
    | action  | subdir   | stage    |
    +=========+==========+==========+
    | keep    | keep/    | kept     |
    +---------+----------+----------+
    | review  | review/  | review   |
    +---------+----------+----------+
    | skip    | skip/    | skipped  |
    +---------+----------+----------+
    | dislike | dislike/ | disliked |
    +---------+----------+----------+

    The state row is REPLACED via
    :meth:`StateStore.record_download` so ``downloads.audio_path``
    always points at the current location.

    Idempotent: re-posting the same action after the first move is a
    no-op (``audio_path.resolve() == new_path.resolve()`` -- the rename
    branch is skipped) and the state row is rewritten to the same
    values, which keeps the response shape stable.

    Errors:
        HTTPException 400 when ``body.action`` is not in ``_VALID_ACTIONS``.
        HTTPException 404 when no download row exists for ``video_id``.
    """
    if body.action not in _VALID_ACTIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown action {body.action!r}; expected one of {sorted(_VALID_ACTIONS)}",
        )
    cfg = _config_from_request(request)
    data_dir = cfg.data_dir

    store = _state_store_from_request(request)
    try:
        row = store.get_download_record(video_id)
    finally:
        store.close()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no download for video_id {video_id!r}",
        )
    _existing_stage, audio_path_str = row
    audio_path = Path(audio_path_str)
    if not audio_path.is_absolute():
        audio_path = (data_dir / audio_path).resolve()

    action = body.action
    target_dir = data_dir / action
    target_dir.mkdir(parents=True, exist_ok=True)  # safety: config.load_config already created it
    new_path = target_dir / audio_path.name

    # Skip the rename when the file is already in the target subdir (idempotent re-post).
    if audio_path.resolve() != new_path.resolve() and audio_path.exists():
        sidecar = audio_path.with_name(audio_path.name + ".meta.json")
        _ = audio_path.rename(new_path)
        if sidecar.exists():
            new_sidecar = new_path.with_name(new_path.name + ".meta.json")
            _ = sidecar.rename(new_sidecar)

    new_stage = _ACTION_TO_STAGE[action]
    store = _state_store_from_request(request)
    try:
        _ = store.record_download(video_id, new_path, stage=new_stage)
    finally:
        store.close()

    return {"ok": True, "new_stage": new_stage, "new_audio_path": str(new_path)}


def _safe_under_data_dir(rel_path: str, data_dir: Path) -> Path:
    """Resolve ``rel_path`` against ``data_dir`` and reject any traversal attempt.

    Two independent checks (defense in depth):

    1. Reject any ``..`` path segment + any absolute path on the raw input.
    2. Verify the resolved path stays under ``data_dir.resolve()`` via
       :meth:`Path.is_relative_to`. The first catches obvious attempts;
       the second catches symlink-driven escapes that the raw-segment
       check would miss.

    Raises:
        HTTPException 400 when either check fails.
    """
    raw = Path(rel_path)
    if raw.is_absolute():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="absolute paths not allowed",
        )
    if any(part == ".." for part in raw.parts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path traversal not allowed",
        )
    candidate = (data_dir / rel_path).resolve(strict=False)
    data_resolved = data_dir.resolve()
    if not (candidate == data_resolved or candidate.is_relative_to(data_resolved)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path escapes data_dir",
        )
    return candidate


@router.get("/audio/{rel_path:path}")
async def serve_audio(request: Request, rel_path: str) -> Response:
    """Stream a file from ``<data_dir>/<rel_path>`` with auto-detected media type.

    The ``rel_path`` MUST stay under ``<data_dir>`` (path-traversal
    guard in :func:`_safe_under_data_dir`). Returns 404 when the
    resolved file does not exist or is not a regular file.

    Note: the deployment is loopback-only (``config.web_host =
    127.0.0.1``), so external clients cannot reach this endpoint.
    """
    cfg = _config_from_request(request)
    safe = _safe_under_data_dir(rel_path, cfg.data_dir)
    if not safe.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"file not found: {rel_path}",
        )
    return FileResponse(safe)


@router.get("/triage", response_class=Response)
async def triage_page(request: Request) -> Response:
    """Render the triage page: an inbox table with audio players + 4 action buttons per row."""
    cfg = _config_from_request(request)
    items = _inbox_items(cfg.data_dir)
    templates = _templates_from_request(request)
    return templates.TemplateResponse(request, "triage.html", {"items": items})
