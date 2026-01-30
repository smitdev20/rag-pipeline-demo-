"""Unit tests for individual components in isolation.

Ensures fast execution with minimal dependencies.

Coverage:
    - models/: Pydantic validation and serialization
    - parsing/: Text extraction and chunking logic
    - agent/: Agent configuration and prompt handling

Uses mocks for external services when needed. Follows single responsibility
per test function. Leverages pytest-check for multiple assertions per test.
"""
