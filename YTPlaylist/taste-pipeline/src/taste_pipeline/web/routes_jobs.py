r"""Job control + SSE progress endpoints for the taste-pipeline web UI.

Mounts four read/write endpoints under ``/api/jobs``:

- ``POST /api/jobs`` body ``{"kind": "feed"|"metadata"|"download"|"index"|"calibrate"}``
  -> creates a new :class:`~taste_pipeline.web.jobs.Job` via the runner
  exposed on ``app.state.runner`` and returns the job JSON (status 201).
- ``GET /api/jobs`` -> list of all jobs newest-first (status 200).
- ``GET /api/jobs/{job_id}`` -> single job JSON (status 200; 404 on unknown id).
- ``GET /api/jobs/{job_id}/events`` -> Server-Sent Events stream
  (``text/event-stream``) emitting one ``data: {"event": "progress",
  "progress": float, "message": str}\\n\\n`` per progress update and a
  final ``data: {"event": "done", "status": str, "error": str|None}\\n\\n``
  when the job reaches a terminal state.

The mapping of pipeline pass kind to its ``run_fn`` is the module-level
dict :data:`JOB_FACTORIES`. Production wiring populates it lazily on
the first ``POST /api/jobs`` via :func:`taste_pipeline.web.job_factories.register_factories`
(so the real ``feed`` / ``metadata`` / ``download`` / ``index`` factories
are bound to the app's config + state store at call time). Tests may
inject fakes by setting keys directly (e.g. ``JOB_FACTORIES["feed"] = fake``)
or via ``monkeypatch.setattr(routes_jobs, "JOB_FACTORIES", {...})``; the
``if not JOB_FACTORIES`` guard in :func:`create_job` short-circuits the
production registration when the dict is already populated.

The runner itself is owned by the app: ``create_app`` sets
``app.state.runner = JobRunner(config.data_dir)`` so this module can
read it via ``request.app.state.runner`` (cast to satisfy
``basedpyright typeCheckingMode="all"`` which forbids raw ``Any``).

Body parsing for ``POST /api/jobs`` is manual via
:meth:`starlette.requests.Request.json` (which works on
``application/json`` bodies without requiring ``python-multipart`` --
unlike FastAPI's :class:`Form` parameter, which would need that extra
dep). Form-encoded bodies, empty bodies, and malformed JSON all return
422 -- the endpoint is intentionally JSON-only; the dashboard buttons
carry ``hx-encoding="json"`` as belt-and-suspenders for browsers where
htmx honors the attribute, but the backend does not depend on it.
"""

from __future__ import annotations

import asyncio
import json
import uuid as _uuid
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from taste_pipeline.config import Config
    from taste_pipeline.web.jobs import Job, JobRunner

router = APIRouter(prefix="/api")

# Module-level registry mapping pipeline-pass kind to its run_fn factory.
# Each factory has the signature
# ``Callable[[Callable[[float, str], None]], dict[str, object] | None]``
# -- the same shape :class:`~taste_pipeline.web.jobs.JobRunner.submit` expects.
# A factory may return a ``dict`` to stash a payload on ``Job.result``
# (the check_url factory does this with a CheckResult dict); ``None`` is
# the "no payload" signal. Populated by the pipeline-wiring task; tests
# register fakes directly.
JOB_FACTORIES: dict[str, Callable[[Callable[[float, str], None]], dict[str, object] | None]] = {}

_TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "cancelled"})

# Single source of truth for valid job kinds -- checked at runtime in
# :func:`create_job` against the parsed JSON body. Tests should NOT
# hardcode a hardcoded tuple; iterate this set instead so a future
# new kind is added by editing one place.
_KNOWN_KINDS: frozenset[str] = frozenset({"feed", "metadata", "download", "index", "calibrate", "check_url"})


def _job_to_payload(job: Job) -> dict[str, object]:
    """Serialize a :class:`Job` to a JSON-safe dict (UUID -> str, datetimes -> ISO 8601)."""
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


def _sse(payload: dict[str, object]) -> str:
    """Render one SSE event frame: a single ``data:`` line followed by a blank line."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_job(request: Request) -> dict[str, object]:
    """Submit a new job of the given ``kind``; returns the job JSON with status 201.

    The JSON body ``{"kind": "feed"|"metadata"|"download"|"index"}`` is
    parsed manually via :meth:`Request.json` (Starlette), which works
    on ``application/json`` bodies without needing ``python-multipart``.
    This is resilient to htmx clients with or without
    ``hx-encoding="json"``, ``fetch``, ``curl``, ``axios`` -- any client
    that sends a proper JSON body. Form-encoded bodies, empty bodies,
    and malformed JSON all return 422 -- the endpoint is JSON-only by
    contract.

    On the first request, lazily populates :data:`JOB_FACTORIES` with
    the real pipeline factories bound to the app's ``config`` and a
    fresh :class:`StateStore` over ``config.data_dir``. Tests register
    fakes directly into :data:`JOB_FACTORIES` (the dict stays
    non-empty, the ``if not JOB_FACTORIES`` guard short-circuits, the
    test's fake wins).

    Raises:
        HTTPException: 422 on any of the following:
            - body is not valid JSON
            - body is not a JSON object
            - ``kind`` is missing
            - ``kind`` is not a string
            - ``kind`` is not in :data:`_KNOWN_KINDS`
            - ``kind`` is a known kind but no factory is registered
              (defensive guard for misconfigured deployments;
              ``pragma: no cover``).
    """
    try:
        # Starlette's Request.json() is stubbed to return Any; the isinstance
        # check below narrows the union to dict after the non-dict branches raise.
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
    kind_raw = payload.get("kind")
    if not isinstance(kind_raw, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'kind' must be a string",
        )
    kind: str = kind_raw
    if kind not in _KNOWN_KINDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"unknown kind {kind!r}; valid kinds: {sorted(_KNOWN_KINDS)}",
        )

    app = cast("FastAPI", request.app)
    config = cast("Config", app.state.config)
    runner = cast("JobRunner", app.state.runner)
    if not JOB_FACTORIES:
        from taste_pipeline.state import StateStore  # noqa: PLC0415 -- lazy construction per app
        from taste_pipeline.web.job_factories import register_factories  # noqa: PLC0415 -- direct submodule

        state = StateStore(config.data_dir)
        register_factories(JOB_FACTORIES, config, state)
    factory = JOB_FACTORIES.get(kind)
    if factory is None:  # pragma: no cover -- defensive guard against unregistered _KNOWN_KINDS members
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"no factory registered for job kind {kind!r}",
        )
    job = await runner.submit(kind, factory)
    return _job_to_payload(job)


@router.get("/jobs")
async def list_jobs(request: Request) -> list[dict[str, object]]:
    """Return every job the runner has, newest-first."""
    app = cast("FastAPI", request.app)
    runner = cast("JobRunner", app.state.runner)
    return [_job_to_payload(job) for job in runner.list()]


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict[str, object]:
    """Return a single job by id; 404 if the id is unknown."""
    app = cast("FastAPI", request.app)
    runner = cast("JobRunner", app.state.runner)
    try:
        parsed_id = _parse_uuid(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    try:
        return _job_to_payload(runner.get(parsed_id))
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/events")
async def stream_job_events(request: Request, job_id: str) -> StreamingResponse:
    """Stream SSE events for a single job: one ``progress`` per state change, one ``done`` at terminal state.

    The generator polls ``runner.get(...)`` every ~50ms and emits a
    ``progress`` event whenever the job's ``progress`` or ``log_lines``
    changes. When the job reaches a terminal state, the generator emits
    a single ``done`` event and returns, which closes the SSE stream.
    Each yield yields control back to the event loop (via ``asyncio.sleep(0.05)``
    between checks) so uvicorn flushes the response chunk immediately --
    no client-side buffering.
    """
    app = cast("FastAPI", request.app)
    runner = cast("JobRunner", app.state.runner)
    try:
        parsed_id = _parse_uuid(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    try:
        _ = runner.get(parsed_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    async def _event_generator() -> AsyncIterator[str]:
        """Poll the job and emit SSE events; close after the ``done`` event."""
        last_progress: float = -1.0
        last_log_count: int = 0
        while True:
            current = runner.get(parsed_id)
            if len(current.log_lines) > last_log_count:
                # Emit a progress event for every newly appended log line.
                # The newest message is the most useful for the client UI.
                message = current.log_lines[-1]
                yield _sse(
                    {
                        "event": "progress",
                        "progress": current.progress,
                        "message": message,
                    }
                )
                last_log_count = len(current.log_lines)
                last_progress = current.progress
            elif current.progress != last_progress:
                # Progress changed without a log line -- still notify.
                yield _sse(
                    {
                        "event": "progress",
                        "progress": current.progress,
                        "message": "",
                    }
                )
                last_progress = current.progress
            if current.status in _TERMINAL_STATUSES:
                yield _sse(
                    {
                        "event": "done",
                        "status": current.status,
                        "error": current.error,
                    }
                )
                return
            # Yield control so uvicorn flushes; the 0.05s budget is a deliberate
            # balance between responsiveness (low) and CPU/idle cost (very low).
            await asyncio.sleep(0.05)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")


def _parse_uuid(raw: str) -> _uuid.UUID:
    """Parse a UUID string or raise :class:`ValueError` (mapped to 404 by callers)."""
    try:
        return _uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError) as exc:
        msg = f"job {raw} not found"
        raise ValueError(msg) from exc
