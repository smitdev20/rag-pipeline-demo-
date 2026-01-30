"""FastAPI application factory and configuration.

Main application entry point with lifespan management, middleware,
and router registration.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.chat import router as chat_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application startup and shutdown lifecycle.

    Handles resource initialization on startup and cleanup on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application while it runs.
    """
    # Startup
    logger.info("Starting RAG Chatbot API...")
    yield
    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    application = FastAPI(
        title="RAG Chatbot API",
        description=(
            "Retrieval-Augmented Generation API for intelligent document Q&A. "
            "Ingests PDF documents, generates embeddings, and provides context-aware "
            "responses through LLM orchestration. Supports streaming responses and "
            "session management."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    application.include_router(chat_router)

    @application.get("/health")
    async def health_check() -> dict[str, str]:
        """Check service health status."""
        return {"status": "healthy", "service": "rag-chatbot"}

    return application


app = create_app()
