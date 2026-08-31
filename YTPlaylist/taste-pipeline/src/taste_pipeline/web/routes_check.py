"""Check-Track endpoints: file drag-and-drop + YouTube URL background job.

The Check Track feature lets the user score one track against the
like-library index. Two input modes:

- ``POST /api/check/file`` -- synchronous. The client base64-encodes the
  audio file and sends it as JSON. The server decodes, writes to a tmp
  path under ``<data_dir>/tmp``, runs :func:`taste_pipeline.check.check_audio_file`,
  deletes the tmp file, and returns the :class:`~taste_pipeline.check.CheckResult`.
  No ``python-multipart`` dependency is needed because the body is plain
  JSON (``{"filename": ..., "content_b64": ...}``).

- ``POST /api/check/url`` -- async via the job runner. The YouTube URL
  is queued as a ``check_url`` job that runs the full pipeline
  (yt-dlp parse → metadata fetch → download → embed → compare) in a
  background thread. The client polls ``GET /api/jobs/{id}`` (or
  watches ``GET /api/jobs/{id}/events``) until the job reaches a
  terminal state; the runner stashes the CheckResult on
  ``job.result`` so the GET returns it.

- ``GET /check`` -- the check page (Jinja2 template) with a drag-drop
  zone + URL input + a result panel.

Library-empty guard: both endpoints return 503 (``index.count() == 0``)
when the library has not been built yet. The error message names the
fix ("Run Build Index first") so the user does not need to read the
docs to recover.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import Response

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from taste_pipeline.config import Config
    from taste_pipeline.web.jobs import JobRunner

router = APIRouter()

# Allowed audio file extensions (lowercase, with leading dot). Synced with
# taste_pipeline.library so a file the embedder understands is also accepted
# by the upload endpoint. ``.opus`` is included because ffmpeg decodes it
# natively and yt-dlp's ``bestaudio`` can produce it.
_ALLOWED_AUDIO_EXTS: frozenset[str] = frozenset({".flac", ".mp3", ".wav", ".opus", ".m4a", ".ogg"})
_TMP_SUBDIR: str = "tmp"
_INDEX_SUBDIR: str = "index"
_LIBRARY_MANIFEST_NAME: str = "library_manifest.json"
_LIBRARY_EMPTY_DETAIL: str = "Library not indexed. Run Build Index first."
_TOP_K_TUPLE_LEN: int = 2  # (similarity, library_track_id)


def _config_from_request(request: Request) -> Config:
    """Cast ``request.app.state.config`` to a typed ``Config`` for basedpyright."""
    return cast("Config", cast("FastAPI", request.app).state.config)


def _templates_from_request(request: Request) -> Jinja2Templates:
    """Cast ``request.app.state.templates`` to a typed ``Jinja2Templates``."""
    return cast("Jinja2Templates", cast("FastAPI", request.app).state.templates)


def _runner_from_request(request: Request) -> JobRunner:
    """Cast ``request.app.state.runner`` to a typed ``JobRunner``."""
    return cast("JobRunner", cast("FastAPI", request.app).state.runner)


def _library_index_count(config: Config) -> int:
    """Return ``len(<data_dir>/index/library_manifest.json)`` or 0 when absent.

    A failed JSON parse yields ``0`` (fail-open): the check endpoint then
    returns the same 503 the empty-library case returns, instead of leaking
    a 500. The check is read-only -- never touches the vectors file -- so
    the dashboard's index-status read path remains cheap.
    """
    import json  # noqa: PLC0415 -- stdlib lazy: keeps module import cheap

    manifest_path = config.data_dir / _INDEX_SUBDIR / _LIBRARY_MANIFEST_NAME
    if not manifest_path.is_file():
        return 0
    try:
        loaded = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0
    if not isinstance(loaded, dict):
        return 0
    return len(cast("dict[str, object]", loaded))


def _check_result_to_payload(result: object) -> dict[str, object]:
    """Convert a :class:`CheckResult` (or its asdict form) to the JSON response payload.

    Centralizes the serialization so the file-upload endpoint and the URL
    job-factory return the same shape; the JS in ``check.html`` only has
    to know one shape.
    """
    if hasattr(result, "__dataclass_fields__"):
        payload = cast(
            "dict[str, object]",
            asdict(result),
        )
    elif isinstance(result, dict):
        payload = cast("dict[str, object]", result)
    else:
        payload = {}
    top_k_raw = payload.get("top_k", [])
    top_k_list: list[dict[str, object]] = []
    if isinstance(top_k_raw, list):
        for entry in top_k_raw:
            if isinstance(entry, dict):
                top_k_list.append(cast("dict[str, object]", entry))
            elif isinstance(entry, tuple) and len(entry) == _TOP_K_TUPLE_LEN:
                score, track_id = entry
                top_k_list.append({"score": float(score), "library_track_id": str(track_id)})
    payload["top_k"] = top_k_list
    return payload


def _decode_upload(content_b64: str, filename: str, tmp_dir: Path) -> Path:
    """Validate the upload, decode base64, write to ``tmp_dir/<uuid><ext>``, return the path.

    Raises:
        HTTPException 422 with a precise detail when the filename is empty,
        the body is not valid base64, or the extension is not in the
        allowlist.
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="filename must be a non-empty string",
        )
    if not content_b64:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="content_b64 must be a non-empty string",
        )
    extension = Path(filename).suffix.casefold()
    if extension not in _ALLOWED_AUDIO_EXTS:
        allowed = sorted(ext.lstrip(".") for ext in _ALLOWED_AUDIO_EXTS)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unsupported audio extension {extension!r}; expected one of {allowed}",
        )
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"content_b64 is not valid base64: {exc}",
        ) from exc
    tmp_dir.mkdir(parents=True, exist_ok=True)
    target = tmp_dir / f"{uuid.uuid4().hex}{extension}"
    _ = target.write_bytes(raw)
    return target


@router.post("/api/check/file")
async def check_file(request: Request) -> dict[str, object]:
    """Run the check pipeline against a base64-encoded audio file uploaded as JSON.

    Body shape: ``{"filename": "<original-name>", "content_b64": "<base64>"}``.

    Steps:

    1. Parse the JSON body manually (no Pydantic body model; mirrors
       ``routes_jobs.create_job`` -- avoids ``python-multipart``).
    2. Decode + write to a tmp file under ``<data_dir>/tmp``.
    3. Reject with 503 if the library index is empty (no calibration
       possible).
    4. Call :func:`taste_pipeline.check.check_audio_file` on the tmp path.
    5. Delete the tmp file (try/finally).
    6. Return the :class:`CheckResult` as JSON.

    Errors:
        422: empty filename, empty body, invalid base64, non-audio
            extension, malformed JSON, non-object body.
        503: library index is empty (the user must Build Index first).
        500: the embedder raised (e.g. ffmpeg could not decode the file).
    """
    import json  # noqa: PLC0415 -- stdlib lazy: keeps module import cheap

    from taste_pipeline.check import check_audio_file  # noqa: PLC0415 -- lazy: monkey-patch works

    try:
        raw_body = await request.json()  # pyright: ignore[reportAny]
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"invalid JSON body: {exc.msg}",
        ) from exc
    if not isinstance(raw_body, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"request body must be a JSON object, got {type(raw_body).__name__}",
        )
    body = cast("dict[str, object]", raw_body)

    filename_obj = body.get("filename", "")
    content_b64_obj = body.get("content_b64", "")
    if not isinstance(filename_obj, str) or not isinstance(content_b64_obj, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="filename and content_b64 must be strings",
        )

    cfg = _config_from_request(request)
    if _library_index_count(cfg) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_LIBRARY_EMPTY_DETAIL,
        )

    tmp_dir = cfg.data_dir / _TMP_SUBDIR
    tmp_path = _decode_upload(content_b64_obj, filename_obj, tmp_dir)
    try:
        result = check_audio_file(tmp_path, cfg)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"check failed: {exc}",
        ) from exc
    finally:
        with contextlib.suppress(OSError):
            _ = tmp_path.unlink()
    return _check_result_to_payload(result)


@router.post("/api/check/url", status_code=status.HTTP_201_CREATED)
async def check_url(request: Request) -> dict[str, object]:
    """Submit a ``check_url`` job: full pipeline (yt-dlp → metadata → download → embed → compare).

    Body shape: ``{"url": "https://www.youtube.com/watch?v=..."}``.

    The handler enqueues a ``check_url`` job via the existing
    ``JobRunner`` and ``JOB_FACTORIES`` registry; the factory closure
    carries the URL into the worker thread (so the handler does not
    have to persist it on the Job). The client watches progress via
    ``GET /api/jobs/{id}/events`` and reads the final :class:`CheckResult`
    from ``GET /api/jobs/{id}``'s ``result`` field.

    Errors:
        422: missing/empty ``url`` field, malformed JSON, non-object body.
        503: library index is empty (the user must Build Index first).
    """
    import json  # noqa: PLC0415 -- stdlib lazy: keeps module import cheap

    from taste_pipeline.web.job_factories import make_check_url_factory  # noqa: PLC0415 -- lazy: factory
    from taste_pipeline.web.routes_jobs import JOB_FACTORIES  # noqa: PLC0415 -- lazy: routes coupling

    try:
        raw_body = await request.json()  # pyright: ignore[reportAny]
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"invalid JSON body: {exc.msg}",
        ) from exc
    if not isinstance(raw_body, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"request body must be a JSON object, got {type(raw_body).__name__}",  # pyright: ignore[reportAny] -- narrowed by isinstance
        )
    body = cast("dict[str, object]", raw_body)
    url_obj = body.get("url", "")
    if not isinstance(url_obj, str) or not url_obj.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="url must be a non-empty string",
        )

    cfg = _config_from_request(request)
    if _library_index_count(cfg) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_LIBRARY_EMPTY_DETAIL,
        )

    runner = _runner_from_request(request)
    if "check_url" not in JOB_FACTORIES:
        from taste_pipeline.state import StateStore  # noqa: PLC0415 -- lazy construction per app

        state = StateStore(cfg.data_dir)
        JOB_FACTORIES["check_url"] = make_check_url_factory(cfg, state, url_obj)
    job = await runner.submit("check_url", JOB_FACTORIES["check_url"])

    # Inline _job_to_payload shape so we don't import routes_jobs (avoid
    # the cross-module call). Mirrors routes_jobs._job_to_payload.
    return {
        "id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "log_lines": list(job.log_lines),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error": job.error,
        "result": job.result,
    }


@router.get("/check", response_class=Response)
async def check_page(request: Request) -> Response:
    """Render the check page: drag-drop zone + URL input + result panel."""
    templates = _templates_from_request(request)
    return templates.TemplateResponse(request, "check.html", {})
