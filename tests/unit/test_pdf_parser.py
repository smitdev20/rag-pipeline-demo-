"""Unit tests for PDF parser module."""

from pathlib import Path

import pytest
import pytest_check as check

from src.parsing.pdf_parser import MAX_FILE_SIZE, PDFParseError, parse_pdf

DATA_DIR = Path(__file__).parent.parent / "data"


class TestParsePdfValid:
    """Tests for successful PDF parsing."""

    def test_extracts_text_and_page_count(self) -> None:
        """Valid PDF returns text content and correct page count."""
        result = parse_pdf((DATA_DIR / "sample.pdf").read_bytes())

        check.greater(len(result.text), 0)
        check.is_in("Information security", result.text)
        check.equal(result.pages, 26)

    def test_returns_metadata_dict(self) -> None:
        """Valid PDF returns metadata as dict."""
        result = parse_pdf((DATA_DIR / "sample.pdf").read_bytes())

        check.is_instance(result.metadata, dict)

    def test_empty_page_pdf_succeeds(self) -> None:
        """PDF with empty pages parses without error."""
        result = parse_pdf((DATA_DIR / "empty.pdf").read_bytes())

        check.equal(result.pages, 1)


class TestParsePdfRejection:
    """Tests for PDF validation and rejection."""

    def test_rejects_empty_bytes(self) -> None:
        """Empty bytes raises PDFParseError."""
        with pytest.raises(PDFParseError, match="Empty file"):
            parse_pdf(b"")

    def test_rejects_non_pdf_file(self) -> None:
        """Non-PDF file raises PDFParseError."""
        with pytest.raises(PDFParseError, match="Invalid PDF"):
            parse_pdf((DATA_DIR / "corrupt.pdf").read_bytes())

    def test_rejects_oversized_file(self) -> None:
        """File over 10MB raises PDFParseError."""
        oversized = b"%PDF-1.4" + b"\x00" * (MAX_FILE_SIZE + 1)

        with pytest.raises(PDFParseError, match="exceeds maximum"):
            parse_pdf(oversized)

    def test_rejects_truncated_pdf(self) -> None:
        """Truncated PDF raises PDFParseError."""
        with pytest.raises(PDFParseError, match="Corrupt|Failed"):
            parse_pdf(b"%PDF-1.4\n1 0 obj\n<<")
