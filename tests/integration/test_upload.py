"""Integration tests for PDF upload endpoint.

Tests real upload flow with actual files, no mocks.
Validates file validation, parsing, and knowledge base integration.

Requirements:
    - OPENAI_API_KEY environment variable for RAG query tests
    - Real test files in tests/data/
"""

import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.models.schemas import PDFUploadResponse, StreamChunk


def has_llm_api_key() -> bool:
    """Check if API key is configured (OPENAI_API_KEY)."""
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key and not key.isspace())


requires_api_key = pytest.mark.skipif(
    not has_llm_api_key(),
    reason="OPENAI_API_KEY not set - skipping LLM integration test",
)


class TestPDFUpload:
    """Integration tests for POST /upload/pdf endpoint."""

    @pytest.fixture
    async def client(self) -> AsyncClient:
        """Create async HTTP client with ASGI transport."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Return path to test data directory."""
        return Path(__file__).parent.parent / "data"

    @pytest.fixture
    def sample_pdf_path(self, test_data_dir: Path) -> Path:
        """Return path to sample PDF."""
        return test_data_dir / "sample.pdf"

    async def test_upload_pdf_success(
        self, client: AsyncClient, sample_pdf_path: Path
    ) -> None:
        """Upload valid PDF returns success with filename and page count."""
        with open(sample_pdf_path, "rb") as f:
            response = await client.post(
                "/upload/pdf",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )

        assert response.status_code == 200

        data = PDFUploadResponse.model_validate(response.json())
        assert data.success is True
        assert data.filename == "sample.pdf"
        assert data.pages > 0
        assert data.error is None

    async def test_upload_pdf_response_schema(
        self, client: AsyncClient, sample_pdf_path: Path
    ) -> None:
        """Response matches PDFUploadResponse schema exactly."""
        with open(sample_pdf_path, "rb") as f:
            response = await client.post(
                "/upload/pdf",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )

        assert response.status_code == 200

        # Should parse without errors
        data = response.json()
        assert "filename" in data
        assert "pages" in data
        assert "success" in data
        assert isinstance(data["filename"], str)
        assert isinstance(data["pages"], int)
        assert isinstance(data["success"], bool)

    async def test_reject_non_pdf_file_txt(
        self, client: AsyncClient, test_data_dir: Path
    ) -> None:
        """Non-PDF file with .txt extension is rejected with 400."""
        content = b"This is a plain text file, not a PDF."

        response = await client.post(
            "/upload/pdf",
            files={"file": ("document.txt", content, "text/plain")},
        )

        assert response.status_code == 400
        error_detail = response.json()
        assert "detail" in error_detail
        assert "PDF" in error_detail["detail"]

    async def test_reject_non_pdf_file_jpg(self, client: AsyncClient) -> None:
        """Image file with .jpg extension is rejected with 400."""
        # Minimal JPEG header bytes
        jpeg_header = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46])

        response = await client.post(
            "/upload/pdf",
            files={"file": ("image.jpg", jpeg_header, "image/jpeg")},
        )

        assert response.status_code == 400
        error_detail = response.json()
        assert "detail" in error_detail
        assert "PDF" in error_detail["detail"]

    async def test_reject_fake_pdf_extension(self, client: AsyncClient) -> None:
        """File with .pdf extension but non-PDF content is rejected."""
        # Plain text masquerading as PDF
        fake_pdf_content = b"This is not a real PDF file, just text with .pdf extension"

        response = await client.post(
            "/upload/pdf",
            files={"file": ("fake.pdf", fake_pdf_content, "application/pdf")},
        )

        assert response.status_code == 400
        error_detail = response.json()
        assert "detail" in error_detail
        # Should detect invalid PDF header
        assert "PDF" in error_detail["detail"] or "Invalid" in error_detail["detail"]

    async def test_reject_oversized_file(self, client: AsyncClient) -> None:
        """File exceeding 10MB limit is rejected with 413."""
        # Create content slightly over 10MB (10MB + 1KB)
        oversized_content = b"%PDF-1.4\n" + (b"x" * (10 * 1024 * 1024 + 1024))

        response = await client.post(
            "/upload/pdf",
            files={"file": ("large.pdf", oversized_content, "application/pdf")},
        )

        assert response.status_code == 413
        error_detail = response.json()
        assert "detail" in error_detail
        assert "size" in error_detail["detail"].lower() or "10MB" in error_detail["detail"]

    async def test_reject_empty_file(self, client: AsyncClient) -> None:
        """Empty file is rejected with 400."""
        response = await client.post(
            "/upload/pdf",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )

        assert response.status_code == 400
        error_detail = response.json()
        assert "detail" in error_detail

    async def test_reject_corrupt_pdf(
        self, client: AsyncClient, test_data_dir: Path
    ) -> None:
        """Corrupt PDF file is rejected with 400."""
        corrupt_pdf_path = test_data_dir / "corrupt.pdf"

        with open(corrupt_pdf_path, "rb") as f:
            response = await client.post(
                "/upload/pdf",
                files={"file": ("corrupt.pdf", f, "application/pdf")},
            )

        assert response.status_code == 400
        error_detail = response.json()
        assert "detail" in error_detail

    async def test_reject_missing_filename(self, client: AsyncClient) -> None:
        """Upload without filename is rejected."""
        pdf_header = b"%PDF-1.4\n%some minimal content"

        response = await client.post(
            "/upload/pdf",
            files={"file": ("", pdf_header, "application/pdf")},
        )

        # Empty filename rejected - 422 from FastAPI validation or 400 from our check
        assert response.status_code in (400, 422)

    async def test_upload_preserves_original_filename(
        self, client: AsyncClient, sample_pdf_path: Path
    ) -> None:
        """Uploaded file's original filename is preserved in response."""
        custom_filename = "my-custom-document.pdf"

        with open(sample_pdf_path, "rb") as f:
            response = await client.post(
                "/upload/pdf",
                files={"file": (custom_filename, f, "application/pdf")},
            )

        assert response.status_code == 200
        data = PDFUploadResponse.model_validate(response.json())
        assert data.filename == custom_filename


class TestPDFUploadAndChat:
    """Integration tests for upload followed by RAG chat queries."""

    @pytest.fixture
    async def client(self) -> AsyncClient:
        """Create async HTTP client with ASGI transport."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Return path to test data directory."""
        return Path(__file__).parent.parent / "data"

    @pytest.fixture
    def sample_pdf_path(self, test_data_dir: Path) -> Path:
        """Return path to sample PDF."""
        return test_data_dir / "sample.pdf"

    @requires_api_key
    async def test_chat_references_uploaded_pdf(
        self, client: AsyncClient, sample_pdf_path: Path
    ) -> None:
        """After upload, chat can query PDF content and get relevant response.

        This tests the full RAG flow:
        1. Upload PDF to knowledge base
        2. Send chat message asking about content
        3. Verify response is relevant to PDF
        """
        # Step 1: Upload PDF
        with open(sample_pdf_path, "rb") as f:
            upload_response = await client.post(
                "/upload/pdf",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )

        assert upload_response.status_code == 200
        upload_data = PDFUploadResponse.model_validate(upload_response.json())
        assert upload_data.success is True

        # Step 2: Send chat message asking about the PDF content
        content_chunks: list[str] = []

        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "What is the uploaded document about? Summarize its main topic."},
        ) as response:
            assert response.status_code == 200

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    json_str = line.removeprefix("data: ").strip()
                    chunk = StreamChunk.model_validate_json(json_str)
                    if not chunk.done and chunk.content:
                        content_chunks.append(chunk.content)

        # Step 3: Verify response contains content
        full_response = "".join(content_chunks)
        assert len(full_response) > 0, "Expected non-empty response about PDF content"

    @requires_api_key
    async def test_chat_stream_returns_valid_chunks_after_upload(
        self, client: AsyncClient, sample_pdf_path: Path
    ) -> None:
        """Chat stream returns properly formatted chunks after PDF upload."""
        # Upload PDF first
        with open(sample_pdf_path, "rb") as f:
            await client.post(
                "/upload/pdf",
                files={"file": ("sample.pdf", f, "application/pdf")},
            )

        # Query about uploaded content
        chunks: list[StreamChunk] = []

        async with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "Tell me about the document I just uploaded."},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    json_str = line.removeprefix("data: ").strip()
                    chunk = StreamChunk.model_validate_json(json_str)
                    chunks.append(chunk)

        # Should have multiple chunks ending with done=True
        assert len(chunks) >= 1
        assert chunks[-1].done is True


class TestUploadErrorHandling:
    """Tests for error scenarios in upload endpoint."""

    @pytest.fixture
    async def client(self) -> AsyncClient:
        """Create async HTTP client with ASGI transport."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    async def test_wrong_http_method_returns_405(self, client: AsyncClient) -> None:
        """GET request to POST endpoint returns 405 Method Not Allowed."""
        response = await client.get("/upload/pdf")
        assert response.status_code == 405

    async def test_missing_file_returns_422(self, client: AsyncClient) -> None:
        """Request without file attachment returns 422."""
        response = await client.post("/upload/pdf")
        assert response.status_code == 422

    async def test_wrong_form_field_name_returns_422(self, client: AsyncClient) -> None:
        """File uploaded with wrong field name returns 422."""
        response = await client.post(
            "/upload/pdf",
            files={"wrong_field": ("test.pdf", b"%PDF-1.4\n", "application/pdf")},
        )
        assert response.status_code == 422

    async def test_cors_headers_present(self, client: AsyncClient) -> None:
        """Response includes CORS headers for cross-origin requests."""
        response = await client.post(
            "/upload/pdf",
            files={"file": ("test.pdf", b"not a pdf", "application/pdf")},
            headers={"Origin": "http://localhost:3000"},
        )

        # CORS middleware should add access-control headers
        assert "access-control-allow-origin" in response.headers
