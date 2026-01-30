"""PDF upload endpoint for document ingestion.

Handles file upload, validation, parsing, and knowledge base storage.
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, status

from src.agent.chat_agent import get_agent_service
from src.models.schemas import PDFUploadResponse
from src.parsing.pdf_parser import MAX_FILE_SIZE, PDFParseError, parse_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

# 10MB limit matches pdf_parser constant
MAX_UPLOAD_SIZE = MAX_FILE_SIZE


def _validate_file_extension(filename: str | None) -> str:
    """Validate that file has .pdf extension.

    Args:
        filename: The uploaded filename.

    Returns:
        The validated filename.

    Raises:
        HTTPException: 400 if extension is invalid.
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    return filename


async def _read_and_validate_size(file: UploadFile) -> bytes:
    """Read file content and validate size.

    Args:
        file: The uploaded file.

    Returns:
        File content as bytes.

    Raises:
        HTTPException: 413 if file exceeds size limit.
    """
    content = await file.read()

    if len(content) > MAX_UPLOAD_SIZE:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({size_mb:.1f}MB) exceeds maximum allowed (10MB)",
        )

    return content


@router.post("/pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile) -> PDFUploadResponse:
    """Upload and process a PDF document.

    Accepts a PDF file, validates it, extracts text content using
    the PDF parser, and stores it in the knowledge base for RAG queries.

    Args:
        file: The uploaded PDF file (multipart/form-data).

    Returns:
        PDFUploadResponse with filename, page count, and success status.

    Raises:
        400: Invalid file (not PDF, empty, corrupt).
        413: File exceeds 10MB limit.
        500: Internal processing error.
    """
    # Validate extension
    filename = _validate_file_extension(file.filename)

    # Read and validate size
    content = await _read_and_validate_size(file)

    # Parse PDF
    try:
        pdf_content = parse_pdf(content)
    except PDFParseError as e:
        logger.warning(f"PDF parse error for {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Store in knowledge base
    try:
        agent_service = get_agent_service()
        await agent_service.add_document(
            content=pdf_content.text,
            name=filename,
            metadata=pdf_content.metadata,
        )
        logger.info(f"Successfully ingested PDF: {filename} ({pdf_content.pages} pages)")
    except Exception as e:
        logger.error(f"Failed to store document in knowledge base: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store document in knowledge base",
        ) from e

    return PDFUploadResponse(
        filename=filename,
        pages=pdf_content.pages,
        success=True,
    )
