from enum import Enum

from pydantic import BaseModel, Field, field_validator


class StreamStatus(str, Enum):
    """Status values for streaming updates."""

    RECEIVED = "received"
    SEARCHING = "searching"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


class ChatRequest(BaseModel):
    """Request payload for chat completion endpoint.

    Attributes:
        message: User's question or prompt.
        session_id: Optional session for conversation continuity.
    """

    message: str = Field(..., min_length=1)
    session_id: str | None = None

    @field_validator("message", mode="before")
    @classmethod
    def strip_message(cls, v: str) -> str:
        """Strip whitespace from message before validation."""
        if isinstance(v, str):
            return v.strip()
        return v


class StreamChunk(BaseModel):
    """A chunk of streamed response data.

    Attributes:
        content: The text content of this chunk.
        done: Whether this is the final chunk.
        status: Current processing status (received, searching, generating, complete, error).
        error: Error message if something went wrong.
    """

    content: str
    done: bool
    status: StreamStatus | None = None
    error: str | None = None


class PDFUploadResponse(BaseModel):
    """Response after PDF upload processing.

    Attributes:
        filename: Name of the uploaded file.
        pages: Number of pages in the document.
        success: Whether the upload was successful.
        error: Error message if upload failed.
    """

    filename: str
    pages: int
    success: bool
    error: str | None = None
