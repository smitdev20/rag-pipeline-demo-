"""Integration tests for components working together as a system.

No mocks for core functionality - tests real interactions.

Coverage:
    - API endpoints with real HTTP requests
    - PDF parsing with actual sample documents
    - Agent responses with live LLM calls (when configured)
    - Full chat workflow from upload to response

Uses real sample files in tests/data/. External services may be required.
Slower than unit tests but provides higher confidence.
Requires environment variables for API keys.
"""
