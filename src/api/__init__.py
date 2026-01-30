"""FastAPI API module.

Exports the FastAPI application and routers.
"""

from src.api.app import app, create_app

__all__ = ["app", "create_app"]
