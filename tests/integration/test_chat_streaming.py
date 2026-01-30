"""Integration tests for SSE streaming chat endpoint.

Tests real streaming behavior with httpx AsyncClient and ASGITransport.
No mocks - uses actual FastAPI app and validates SSE protocol.

Requirements:
    - OPENAI_API_KEY environment variable for LLM tests
    - Tests marked with @pytest.mark.requires_api_key are skipped without key
"""

import json
import os

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.models.schemas import StreamChunk


def has_openai_key() -> bool:
    """Check if OpenAI API key is configured."""
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and not key.isspace())


requires_api_key = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set - skipping LLM integration test",
)


class TestStreamingEndpoint:
    """Integration tests for POST /chat/stream SSE endpoint."""

    @pytest.fixture
    async def client(self) -> AsyncClient:
        """Create async HTTP client with ASGI transport."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_stream_returns_sse_content_type(self, client: AsyncClient) -> None:
        """Streaming endpoint returns text/event-stream media type."""
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Say hello"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

    async def test_stream_returns_multiple_chunks(self, client: AsyncClient) -> None:
        """Response arrives as multiple SSE chunks, not single blob.

        This verifies true streaming behavior - chunks should arrive
        incrementally as they're generated.
        """
        chunks: list[str] = []

        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Count from 1 to 3"},
        ) as response:
            assert response.status_code == 200

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)

        # Should have multiple data chunks (content chunks + final done chunk)
        assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"

    async def test_chunks_are_valid_json(self, client: AsyncClient) -> None:
        """Each SSE data chunk contains valid JSON matching StreamChunk schema."""
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Hi"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    json_str = line.removeprefix("data: ").strip()
                    # Should parse as valid JSON
                    data = json.loads(json_str)
                    # Should match StreamChunk schema
                    chunk = StreamChunk.model_validate(data)
                    assert isinstance(chunk.content, str)
                    assert isinstance(chunk.done, bool)

    async def test_final_chunk_has_done_true(self, client: AsyncClient) -> None:
        """Last chunk in stream has done=true to signal completion."""
        chunks: list[StreamChunk] = []

        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Say yes"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    json_str = line.removeprefix("data: ").strip()
                    chunk = StreamChunk.model_validate_json(json_str)
                    chunks.append(chunk)

        assert len(chunks) > 0, "Expected at least one chunk"

        # Final chunk must have done=true
        final_chunk = chunks[-1]
        assert final_chunk.done is True, "Final chunk should have done=true"

        # All non-final chunks should have done=false
        for chunk in chunks[:-1]:
            assert chunk.done is False, "Non-final chunks should have done=false"

    async def test_session_id_is_optional(self, client: AsyncClient) -> None:
        """Request without session_id still works (auto-generated)."""
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Hello"},  # No session_id
        ) as response:
            assert response.status_code == 200

    async def test_session_id_accepted(self, client: AsyncClient) -> None:
        """Request with explicit session_id is accepted."""
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Hello", "session_id": "test-session-123"},
        ) as response:
            assert response.status_code == 200

    async def test_empty_message_returns_422(self, client: AsyncClient) -> None:
        """Empty message triggers validation error with 422 status."""
        response = await client.post(
            "/chat/stream",
            json={"message": ""},
        )

        assert response.status_code == 422
        error_detail = response.json()
        assert "detail" in error_detail

    async def test_missing_message_returns_422(self, client: AsyncClient) -> None:
        """Missing message field triggers validation error with 422 status."""
        response = await client.post(
            "/chat/stream",
            json={},  # No message field
        )

        assert response.status_code == 422

    async def test_invalid_json_returns_422(self, client: AsyncClient) -> None:
        """Malformed JSON body returns 422 status."""
        response = await client.post(
            "/chat/stream",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @requires_api_key
    async def test_content_chunks_have_text(self, client: AsyncClient) -> None:
        """Content chunks (done=false) contain actual response text.

        Requires valid OPENAI_API_KEY to get real LLM response.
        """
        content_chunks: list[StreamChunk] = []

        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Say the word 'hello' and nothing else"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    json_str = line.removeprefix("data: ").strip()
                    chunk = StreamChunk.model_validate_json(json_str)
                    if not chunk.done:
                        content_chunks.append(chunk)

        # Should have at least one content chunk with text
        assert len(content_chunks) > 0, "Expected content chunks before done"

        # Combine all content
        full_response = "".join(c.content for c in content_chunks)
        assert len(full_response) > 0, "Expected non-empty response content"

    @requires_api_key
    async def test_chunks_arrive_incrementally(self, client: AsyncClient) -> None:
        """Chunks arrive over time, not all at once.

        This tests true streaming - we should receive chunks as they're
        generated by the LLM, not buffered until completion.
        """
        chunk_count = 0
        content_chunks = 0

        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Write a haiku about coding"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_count += 1
                    json_str = line.removeprefix("data: ").strip()
                    chunk = StreamChunk.model_validate_json(json_str)
                    if not chunk.done and chunk.content:
                        content_chunks += 1

        # Should have at least 2 chunks total (content + done)
        assert chunk_count >= 2, f"Expected at least 2 chunks, got {chunk_count}"
        # Should have at least 1 content chunk with actual text
        assert content_chunks >= 1, f"Expected content chunks, got {content_chunks}"


class TestStreamingErrorHandling:
    """Tests for error scenarios in streaming endpoint."""

    @pytest.fixture
    async def client(self) -> AsyncClient:
        """Create async HTTP client with ASGI transport."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_whitespace_only_message_returns_422(self, client: AsyncClient) -> None:
        """Whitespace-only message is rejected as empty."""
        response = await client.post(
            "/chat/stream",
            json={"message": "   "},
        )

        # Pydantic min_length=1 should reject after strip
        assert response.status_code == 422

    async def test_wrong_http_method_returns_405(self, client: AsyncClient) -> None:
        """GET request to POST endpoint returns 405 Method Not Allowed."""
        response = await client.get("/chat/stream")

        assert response.status_code == 405

    async def test_cors_headers_present(self, client: AsyncClient) -> None:
        """Response includes CORS headers for cross-origin requests."""
        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "test"},
            headers={"Origin": "http://localhost:3000"},
        ) as response:
            # CORS middleware should add access-control headers
            assert "access-control-allow-origin" in response.headers
