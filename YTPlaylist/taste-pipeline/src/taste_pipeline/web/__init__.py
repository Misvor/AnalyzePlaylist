"""Web UI package: FastAPI app factory plus a pywebview desktop entry point."""

from taste_pipeline.web.app import create_app

__all__ = ["create_app"]
