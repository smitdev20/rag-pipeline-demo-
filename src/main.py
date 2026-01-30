"""Main application entry point.

Runs FastAPI with NiceGUI mounted on the same server.
Environment variables are loaded from .env file.

For separate servers (development/scaling), run independently:
    uvicorn src.api.app:app --port 8000
    python -m src.ui.chat_page
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables before any other imports
load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run FastAPI with NiceGUI mounted."""
    import uvicorn
    from nicegui import ui

    from src.api.app import create_app
    from src.ui.chat_page import chat_page  # noqa: F401 - Registers the page

    app = create_app()

    ui.run_with(
        app,
        title="RAG Assistant",
        favicon="🤖",
        storage_secret=os.getenv("NICEGUI_STORAGE_SECRET", "rag-chatbot-secret"),
    )

    logger.info("Starting server on http://localhost:%s", os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
