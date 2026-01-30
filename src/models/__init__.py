"""Pydantic models for API requests and responses.

Provides type safety, validation, and automatic OpenAPI documentation.

Models:
    - ChatMessage: Individual message in conversation
    - ChatRequest: Incoming chat request payload
    - ChatResponse: Outgoing chat response with sources
    - DocumentUpload: PDF upload metadata
    - SessionInfo: Chat session details
"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message in the conversation.

    Attributes:
        role: The speaker identifier (user, assistant, or system).
        content: The message text.
    """

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="The message content")


class ChatRequest(BaseModel):
    """Request payload for chat completion endpoint.

    Attributes:
        message: User's question or prompt.
        session_id: Optional session for conversation continuity.
    """

    message: str = Field(..., min_length=1, description="The user's message")
    session_id: str | None = Field(None, description="Session ID for conversation continuity")


class ChatResponse(BaseModel):
    """Response from the chatbot with source attribution.

    Attributes:
        response: The assistant's generated answer.
        session_id: Session identifier for follow-up questions.
        sources: Referenced document chunks.
    """

    response: str = Field(..., description="The assistant's response")
    session_id: str = Field(..., description="Session ID for this conversation")
    sources: list[str] = Field(default_factory=list, description="Source documents referenced")


class DocumentUpload(BaseModel):
    """Metadata for uploaded PDF document.

    Attributes:
        filename: Original file name.
        page_count: Number of pages processed.
        chunk_count: Number of text chunks created.
    """

    filename: str = Field(..., description="Original filename of the uploaded document")
    page_count: int = Field(..., ge=1, description="Number of pages in the document")
    chunk_count: int = Field(..., ge=1, description="Number of text chunks created")


class SessionInfo(BaseModel):
    """Information about a chat session.

    Attributes:
        session_id: Unique session identifier.
        message_count: Number of messages in session.
        created_at: Session creation timestamp.
    """

    session_id: str = Field(..., description="Unique session identifier")
    message_count: int = Field(..., ge=0, description="Number of messages in session")
    created_at: str = Field(..., description="ISO format timestamp of session creation")
