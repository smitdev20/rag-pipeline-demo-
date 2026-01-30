"""Pytest fixtures and shared test configuration.

Provides reusable fixtures for unit and integration tests.

Fixtures:
    - test_data_dir: Path to sample files directory
    - async_client: HTTPX client for API testing
    - sample_pdf_path: Path to test PDF file
    - mock_session_id: Consistent session ID for tests

Implements async fixtures with proper cleanup, scoped appropriately for performance.
"""

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.api import app


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory.

    Returns:
        Absolute path to tests/data/ directory.
    """
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_pdf_path(test_data_dir: Path) -> Path:
    """Return path to sample PDF for testing.

    Args:
        test_data_dir: Base directory for test data.

    Returns:
        Path to sample.pdf test file.
    """
    return test_data_dir / "sample.pdf"


@pytest.fixture
def mock_session_id() -> str:
    """Generate consistent session ID for testing.

    Returns:
        Predictable session ID for test assertions.
    """
    return "test-session-12345"


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient]:
    """Create async HTTP client for API testing.

    Yields:
        Configured AsyncClient for making test requests.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
