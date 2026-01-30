from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    """Status values for streaming updates."""

    RECEIVED = "received"
    SEARCHING = "searching"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


class ChatMessage(BaseModel):
    """A single chat message in the conversation.

    Attributes:
        role: The speaker identifier (user, assistant, or system).
        content: The message text.
    """

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request payload for chat completion endpoint.

    Attributes:
        message: User's question or prompt.
        session_id: Optional session for conversation continuity.
    """

    message: str = Field(..., min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Response from the chatbot.

    Attributes:
        content: The assistant's generated answer.
        session_id: Session identifier for follow-up questions.
    """

    content: str
    session_id: str


class StreamChunk(BaseModel):
    """A chunk of streamed response data.

    Attributes:
        content: The text content of this chunk.
        done: Whether this is the final chunk.
        error: Error message if something went wrong.
    """

    content: str
    done: bool
    error: str | None = None


class StatusUpdate(BaseModel):
    """Status update during streaming operations.

    Attributes:
        status: Current processing status.
        message: Optional descriptive message.
    """

    status: Status
    message: str | None = None


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
