"""Read-only API endpoints for the taste-pipeline web UI.

Exposes the validated :class:`~taste_pipeline.config.Config` as JSON at
``GET /api/config`` and a per-stage row-count summary of the SQLite
state at ``GET /api/state/summary``. Both routes derive their inputs
exclusively from ``app.state.config`` so the factory's injected
configuration is the single source of truth.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, FastAPI, Request

from taste_pipeline.state import StateStore

if TYPE_CHECKING:
    from taste_pipeline.config import Config

router = APIRouter(prefix="/api")


def _config_to_payload(cfg: Config) -> dict[str, object]:
    """Serialize ``cfg`` to a JSON-safe dict: Path fields become str; scalars pass through."""
    payload: dict[str, object] = {}
    for f in fields(cfg):
        value = cast("object", getattr(cfg, f.name))
        payload[f.name] = str(value) if isinstance(value, Path) else value
    return payload


@router.get("/config")
async def get_config(request: Request) -> dict[str, object]:
    """Return every ``Config`` field as JSON (Path fields rendered as str, never file contents)."""
    app = cast("FastAPI", request.app)
    cfg = cast("Config", app.state.config)
    return _config_to_payload(cfg)


@router.get("/state/summary")
async def get_state_summary(request: Request) -> dict[str, dict[str, int]]:
    """Return ``{table: {stage: count}}`` for the seen-history and downloads tables."""
    app = cast("FastAPI", request.app)
    cfg = cast("Config", app.state.config)
    store = StateStore(cfg.data_dir)
    try:
        return store.state_counts()
    finally:
        store.close()
