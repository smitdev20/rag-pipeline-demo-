"""PDF parsing module using pypdf.

Extracts text content and metadata from PDF files with validation.
"""

import io
import logging

from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
PDF_MAGIC_BYTES = b"%PDF"


class PDFContent(BaseModel):
    """Extracted content from a PDF file.

    Attributes:
        text: Combined text content from all pages.
        pages: Total number of pages in the document.
        metadata: Document metadata (title, author, etc.).
    """

    text: str
    pages: int = Field(ge=0)
    metadata: dict[str, str | None]


class PDFParseError(Exception):
    """Raised when PDF parsing fails."""

    pass


def _validate_pdf_bytes(file_content: bytes) -> None:
    """Validate PDF file content before parsing.

    Args:
        file_content: Raw bytes of the PDF file.

    Raises:
        PDFParseError: If validation fails.
    """
    if not file_content:
        raise PDFParseError("Empty file provided")

    if len(file_content) > MAX_FILE_SIZE:
        size_mb = len(file_content) / (1024 * 1024)
        raise PDFParseError(f"File size ({size_mb:.1f}MB) exceeds maximum allowed (10MB)")

    if not file_content.lstrip()[:10].startswith(PDF_MAGIC_BYTES):
        raise PDFParseError("Invalid PDF: file does not start with PDF header")


def _extract_metadata(reader: PdfReader) -> dict[str, str | None]:
    """Extract metadata from PDF reader.

    Args:
        reader: Initialized PdfReader instance.

    Returns:
        Dictionary of metadata fields.
    """
    metadata: dict[str, str | None] = {}

    try:
        if reader.metadata:
            # Standard PDF metadata fields
            metadata["title"] = reader.metadata.get("/Title")
            metadata["author"] = reader.metadata.get("/Author")
            metadata["subject"] = reader.metadata.get("/Subject")
            metadata["creator"] = reader.metadata.get("/Creator")
            metadata["producer"] = reader.metadata.get("/Producer")

            # Handle dates (can be complex PDF date format)
            creation_date = reader.metadata.get("/CreationDate")
            if creation_date:
                metadata["creation_date"] = str(creation_date)

            mod_date = reader.metadata.get("/ModDate")
            if mod_date:
                metadata["modification_date"] = str(mod_date)
    except Exception as e:
        logger.warning(f"Failed to extract some metadata: {e}")

    # Filter out None values for cleaner output
    return {k: v for k, v in metadata.items() if v is not None}


def parse_pdf(file_content: bytes) -> PDFContent:
    """Parse a PDF file and extract its text content.

    Args:
        file_content: Raw bytes of the PDF file.

    Returns:
        PDFContent with extracted text, page count, and metadata.

    Raises:
        PDFParseError: If the file is invalid, too large, empty, or corrupt.
    """
    _validate_pdf_bytes(file_content)

    try:
        reader = PdfReader(io.BytesIO(file_content))
    except PdfReadError as e:
        raise PDFParseError(f"Corrupt or invalid PDF: {e}") from e
    except Exception as e:
        raise PDFParseError(f"Failed to read PDF: {e}") from e

    pages = len(reader.pages)
    if pages == 0:
        raise PDFParseError("PDF contains no pages")

    # Extract text from all pages
    text_parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"Failed to extract text from page {i + 1}: {e}")
            continue

    text = "\n\n".join(text_parts)

    if not text.strip():
        logger.warning("PDF contains no extractable text (may be scanned/image-based)")

    metadata = _extract_metadata(reader)

    return PDFContent(
        text=text,
        pages=pages,
        metadata=metadata,
    )
