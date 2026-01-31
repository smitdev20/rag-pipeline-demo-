"""Agno agent service with streaming support and RAG knowledge base.

Core module for the chatbot's intelligence and conversation handling.

Architecture Decisions:

1. **SQLite Storage** - Agno's Agent has no default persistence. SQLite provides
   conversation continuity across server restarts with zero infrastructure.

2. **Singleton Pattern** - Agent initialization is expensive (model loading,
   storage connection). Singleton ensures reuse across all requests.

3. **Service Wrapper** - Decouples API from Agno's interface for maintainability.

4. **LanceDB Knowledge Base** - Zero-config vector database for RAG. Stores
   vectors locally without external dependencies.
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
        """Create embedder using configured API settings."""
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

        knowledge = Knowledge(vector_db=vector_db)
        # Agno Knowledge has no load() method; vector_db is queried directly at runtime.
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
            description="A helpful RAG chatbot assistant with document access.",
            instructions=[
                "Provide helpful and accurate responses.",
                "Answer strictly using the retrieved document context.",
                (
                    "If the answer is not in the documents, say: "
                    "'This information is not present in the provided document.'"
                ),
                (
                    "When grouping or categorizing content, use only the exact "
                    "categories defined in the document. Do not introduce new ones."
                ),
                (
                    "Do not answer metadata questions (author, publisher, date) "
                    "unless explicitly stated in the document."
                ),
                "Cite specific information from documents when answering.",
                "Include short direct quotes from documents to support answers.",
                "Be concise yet thorough.",
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
        """Stream response chunks for a message."""
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
