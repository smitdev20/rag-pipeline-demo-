"""Main application entry point.

Runs FastAPI (port 8000) with NiceGUI mounted for the chat interface.
Environment variables are loaded from .env file.
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables before any other imports that might need them
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_integrated() -> None:
    """Run FastAPI with NiceGUI mounted on the same server.

    FastAPI handles API routes, NiceGUI handles the UI.
    Both accessible on port 8000.
    """
    import uvicorn
    from nicegui import ui

    from src.api.app import create_app
    from src.ui.chat_page import chat_page  # noqa: F401 - Registers the page

    app = create_app()

    # Mount NiceGUI onto FastAPI
    ui.run_with(
        app,
        title="RAG Assistant",
        favicon="🤖",
        storage_secret=os.getenv("NICEGUI_STORAGE_SECRET", "rag-chatbot-secret"),
    )

    logger.info("Starting integrated server on http://localhost:8000")
    logger.info("API docs available at http://localhost:8000/docs")
    logger.info("Chat UI available at http://localhost:8000/")

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


def run_separate() -> None:
    """Run FastAPI and NiceGUI as separate servers.

    FastAPI on port 8000, NiceGUI on port 8080.
    Useful for development or when you need separate scaling.
    """
    import asyncio
    import subprocess

    async def run_servers() -> None:
        logger.info("Starting FastAPI on http://localhost:8000")
        logger.info("Starting NiceGUI on http://localhost:8080")

        fastapi_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api.app:app",
                "--host",
                os.getenv("HOST", "0.0.0.0"),
                "--port",
                "8000",
                "--reload",
            ]
        )

        nicegui_proc = subprocess.Popen(
            [sys.executable, "-c", "from src.ui.chat_page import main; main()"]
        )

        try:
            while True:
                await asyncio.sleep(1)
                if fastapi_proc.poll() is not None or nicegui_proc.poll() is not None:
                    break
        except KeyboardInterrupt:
            logger.info("Shutting down servers...")
        finally:
            fastapi_proc.terminate()
            nicegui_proc.terminate()
            fastapi_proc.wait()
            nicegui_proc.wait()

    asyncio.run(run_servers())


def main() -> None:
    """Application entry point.

    Set RUN_MODE=separate to run FastAPI and NiceGUI on different ports.
    Default is integrated mode (both on port 8000).
    """
    mode = os.getenv("RUN_MODE", "integrated").lower()

    logger.info(f"Starting RAG Chatbot in {mode} mode")

    if mode == "separate":
        run_separate()
    else:
        run_integrated()


if __name__ == "__main__":
    main()
