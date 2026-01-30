"""PDF parsing utilities for document processing.

Transforms documents into structured knowledge through text extraction,
chunking, and preprocessing.

Responsibilities:
    - PDF text extraction with pypdf
    - Document chunking with overlap for context preservation
    - Text cleaning and normalization
    - Metadata extraction (title, author, pages)

Output is optimized for RAG pipelines with clean text chunks ready for
embedding generation.
"""

from src.parsing.pdf_parser import PDFContent, PDFParseError, parse_pdf

__all__ = ["PDFContent", "PDFParseError", "parse_pdf"]
