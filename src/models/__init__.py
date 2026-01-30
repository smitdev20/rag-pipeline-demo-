"""Pydantic models for API requests and responses.

Provides type safety, validation, and automatic OpenAPI documentation.

Models:
    - ChatRequest: Incoming chat request payload
    - StreamChunk: Chunk of streamed response data
    - StreamStatus: Status enum for streaming updates
    - PDFUploadResponse: Response after PDF upload
"""

from src.models.schemas import (
    ChatRequest,
    PDFUploadResponse,
    StreamChunk,
    StreamStatus,
)

__all__ = [
    "ChatRequest",
    "PDFUploadResponse",
    "StreamChunk",
    "StreamStatus",
]
