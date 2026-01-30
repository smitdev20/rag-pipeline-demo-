"""Pydantic models for API requests and responses.

Provides type safety, validation, and automatic OpenAPI documentation.

Models:
    - ChatMessage: Individual message in conversation
    - ChatRequest: Incoming chat request payload
    - ChatResponse: Outgoing chat response
    - StreamChunk: Chunk of streamed response data
    - StatusUpdate: Status update during streaming
    - PDFUploadResponse: Response after PDF upload
"""

from src.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    PDFUploadResponse,
    Status,
    StatusUpdate,
    StreamChunk,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "PDFUploadResponse",
    "Status",
    "StatusUpdate",
    "StreamChunk",
]
