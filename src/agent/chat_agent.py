"""Agno agent integration for RAG chatbot with FastAPI.

Provides the main agent logic and session management for the chatbot, including:

- LanceDB: Local vector database for knowledge retrieval (RAG).
- OpenAIChat & OpenAIEmbedder: OpenAI-powered chat and embedding using API settings.
- SQLite (via Agno's SqliteDb): For persistent storage of session/chat history.
- Singleton service: Ensures agent/model instances are efficiently shared across requests.
- Pydantic: All configuration and I/O models.
"""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb

from src.agent.config import AgentConfig, get_agent_config

logger = logging.getLogger(__name__)

# Data directories
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_SESSIONS_DB = _DATA_DIR / "sessions.db"
_KNOWLEDGE_DIR = _DATA_DIR / "knowledge"


class AgentService:
    """Service for managing the Agno chat agent.

    Provides:
    - Persistent SQLite storage for session history
    - LanceDB knowledge base for RAG document retrieval
    - Singleton lifecycle management
    - Clean streaming interface for SSE endpoints
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the agent service."""
        self._config = config or get_agent_config()
        self._storage = self._create_storage()
        self._knowledge = self._create_knowledge()
        self._agent = self._create_agent()

    def _create_storage(self) -> SqliteDb:
        """Create SQLite storage for session persistence."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        return SqliteDb(
            db_file=str(_SESSIONS_DB),
            session_table="chat_sessions",
        )

    def _create_embedder(self) -> OpenAIEmbedder:
        """Create embedder using configured API settings (same base_url as chat)."""
        return OpenAIEmbedder(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
        )

    def _create_knowledge(self) -> Knowledge:
        """Create LanceDB-backed knowledge base for RAG."""
        _KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        vector_db = LanceDb(
            uri=str(_KNOWLEDGE_DIR),
            table_name="documents",
            embedder=self._create_embedder(),
        )

        knowledge = Knowledge(
            vector_db=vector_db,
            max_results=5,  # Retrieve enough chunks for RAG context
        )
        return knowledge

    def _create_agent(self) -> Agent:
        """Create the Agno agent instance."""
        model = OpenAIChat(
            id=self._config.model_name,
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        return Agent(
            model=model,
            db=self._storage,
            knowledge=self._knowledge,
            description="A general-purpose document-grounded RAG assistant.",
            instructions=[
                # --- Retrieval & grounding ---
                "You must search the knowledge base before answering every question.",
                "Answer strictly and only using the retrieved document context.",
                "Do not use prior knowledge, assumptions, or training data.",

                # --- Missing or insufficient context ---
                (
                    "If the retrieved document context does not explicitly contain the answer, "
                    "response with briefly explain the mismatch using the document's wording",
                    " or structure, without adding any new information."
                ),

                # --- Incorrect or misleading questions ---
                (
                    "If a question is based on an incorrect assumption or "
                    "contradicts the document, explain the discrepancy using "
                    "the document content instead of answering directly"
                ),

                # --- Structure & classification discipline ---
                (
                    "When grouping, categorizing, or listing items, use only the exact structure, "
                    "terminology, and categories defined in the document. "
                    "Do not introduce new or inferred groupings."
                ),

                # --- Metadata guard ---
                (
                    "Do not answer metadata questions (such as author, publisher, version, date, "
                    "ownership, or responsibility) unless explicitly stated in the document."
                ),

                # --- Evidence & citation ---
                (
                    "Every factual answer must reference where the information "
                    "appears in the document (such as a section, heading, clause, "
                    "page, or paragraph)."
                ),
                "Include a short direct quote from the document when possible.",

                # --- Confidence control ---
                (
                    "If you are not confident that the answer is fully supported "
                    "by the retrieved document context, do not answer and state "
                    "that the information is not available."
                ),
                # --- Output style ---
                "Use the document's wording and terminology.",
                "Be precise, factual, and concise.",
            ],
            add_history_to_context=True,
            num_history_messages=20,
            search_knowledge=True,
            markdown=True,
        )
    async def _remove_existing_document(self, name: str) -> bool:
        """Remove existing document chunks by name to prevent duplicates."""
        try:
            vector_db = self._knowledge.vector_db
            if not vector_db or not hasattr(vector_db, "db"):
                return False

            table_name = getattr(vector_db, "table_name", "documents")
            if table_name not in vector_db.db.table_names():
                return False

            table = vector_db.db.open_table(table_name)
            table.delete(f'name = "{name}"')
            logger.info(f"Removed existing document chunks: {name}")
            return True

        except Exception as e:
            logger.warning(f"Could not remove existing document {name}: {e}")
            return False

    async def add_document(
        self,
        content: str,
        name: str,
        metadata: dict[str, str | None] | None = None,
    ) -> None:
        """Add a document to the knowledge base.

        If a document with the same name exists, it is replaced.
        """
        if not content.strip():
            logger.warning(f"Skipping empty document: {name}")
            return

        # Remove existing document to prevent duplicates
        await self._remove_existing_document(name)

        doc_metadata: dict[str, str] = {}
        if metadata:
            doc_metadata.update({k: v for k, v in metadata.items() if v is not None})

        # Add content to knowledge base (already searchable after add_content_async)
        await self._knowledge.add_content_async(
            name=name,
            text_content=content,
            metadata=doc_metadata if doc_metadata else None,
        )

        logger.info(f"Added document to knowledge base: {name}")

    async def stream_response(
        self,
        message: str,
        session_id: str,
    ) -> AsyncGenerator[str]:
        """Stream response chunks. Agno built-in: Agent knowledge + search_knowledge=True."""
        try:
            response_stream = self._agent.arun(
                message,
                session_id=session_id,
                stream=True,
            )

            async for chunk in response_stream:
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

        except Exception as e:
            yield f"\n\n[Error: {e}]"


# Singleton instance
_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """Get or create the global agent service."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
