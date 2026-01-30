"""Agno agent service with streaming support and RAG knowledge base.

Core module for the chatbot's intelligence and conversation handling.

Architecture Decisions (why we layer on top of Agno's built-ins):

1. **SQLite Storage** - Agno's Agent has no default persistence. Without explicit
   storage, session_id is ignored and every request is stateless. SQLite gives us:
   - Conversation continuity across server restarts
   - Multi-turn context without client-side history management
   - Zero infrastructure (single file, no external DB required for dev)

2. **Singleton Pattern** - Agent initialization is expensive (model loading, storage
   connection). The singleton ensures we reuse the same agent instance across all
   requests rather than recreating it per-request.

3. **Service Wrapper** - Decouples our API from Agno's interface. If Agno's API
   changes (as it did with add_history_to_messages -> add_history_to_context),
   we only fix one place. Also lets us add custom error handling and logging.

4. **Explicit num_history_messages** - Agno's default may include too much/little
   context. We set 20 messages (~10 turns) to balance context quality vs token cost.

5. **Streaming Generator** - Agno returns raw chunks with metadata. We extract just
   the content string, providing a clean interface for the SSE endpoint.

6. **LanceDB Knowledge Base** - Zero-config vector database for RAG. Like SQLite for
   sessions, LanceDB stores vectors locally without external dependencies. Uploaded
   PDFs are chunked, embedded, and made searchable for context injection.
"""

import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb

from src.agent.config import AgentConfig, get_agent_config

logger = logging.getLogger(__name__)

# Store sessions and knowledge in project data directory
_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_SESSIONS_DB = _DATA_DIR / "sessions.db"
_KNOWLEDGE_DIR = _DATA_DIR / "knowledge"


class AgentService:
    """Service for managing the Agno chat agent.

    Wraps Agno's Agent with:
    - Persistent SQLite storage for session history
    - LanceDB knowledge base for RAG document retrieval
    - Singleton lifecycle management
    - Clean streaming interface for SSE endpoints
    - Centralized error handling
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the agent service.

        Args:
            config: Optional agent configuration.
                    Loads from environment if not provided.
        """
        self._config = config or get_agent_config()
        self._storage = self._create_storage()
        self._knowledge = self._create_knowledge()
        self._agent = self._create_agent()

    def _create_storage(self) -> SqliteDb:
        """Create SQLite storage for session persistence.

        Why SQLite over in-memory:
        - Survives server restarts (critical for dev with --reload)
        - Allows inspecting conversation history for debugging
        - Zero-config, no external dependencies

        Returns:
            Configured SqliteDb instance.
        """
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        return SqliteDb(
            db_file=str(_SESSIONS_DB),
            session_table="chat_sessions",
        )

    def _create_knowledge(self) -> Knowledge:
        """Create LanceDB-backed knowledge base for RAG.

        Why LanceDB:
        - Zero-config local vector database (like SQLite for vectors)
        - No external dependencies or services required
        - Semantic vector search for document retrieval
        - Survives server restarts

        Returns:
            Configured Knowledge instance.
        """
        _KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        vector_db = LanceDb(
            uri=str(_KNOWLEDGE_DIR),
            table_name="documents",
        )

        return Knowledge(vector_db=vector_db)

    def _create_agent(self) -> Agent:
        """Create the Agno agent instance.

        Returns:
            Configured Agent with OpenAI model, SQLite storage, and knowledge base.
        """
        model = OpenAIChat(
            id=self._config.model_name,
            api_key=self._config.openai_api_key,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        return Agent(
            model=model,
            db=self._storage,
            knowledge=self._knowledge,
            description="A helpful RAG chatbot assistant with access to uploaded documents.",
            instructions=[
                "Provide helpful and accurate responses.",
                "When relevant documents are available, reference them in your answers.",
                "Cite specific information from documents when answering questions.",
                "Be concise yet thorough.",
            ],
            # History config: include last 20 messages (~10 conversation turns)
            # in context. Balances continuity vs token cost.
            add_history_to_context=True,
            num_history_messages=20,
            # Enable agentic RAG - agent decides when to search knowledge base
            search_knowledge=True,
            # Output as markdown for rich formatting in UI
            markdown=True,
        )

    async def add_document(
        self,
        content: str,
        name: str,
        metadata: dict[str, str | None] | None = None,
    ) -> None:
        """Add a document to the knowledge base.

        Adds text content to the knowledge base for RAG retrieval.

        Args:
            content: The text content of the document.
            name: Document name/identifier (e.g., filename).
            metadata: Optional metadata (author, title, etc.).
        """
        if not content.strip():
            logger.warning(f"Skipping empty document: {name}")
            return

        doc_metadata: dict[str, str] = {}
        if metadata:
            doc_metadata.update({k: v for k, v in metadata.items() if v is not None})

        # Add content to knowledge base using Agno's API
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
        """Stream response chunks for a message.

        Yields response tokens as they arrive.
        Agno automatically maintains conversation history per session.

        Args:
            message: The user's message.
            session_id: Session identifier for history tracking.

        Yields:
            Response text chunks as they arrive.
        """
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

    async def get_response(
        self,
        message: str,
        session_id: str,
    ) -> str:
        """Get complete response for a message.

        Non-streaming alternative for simpler use cases.

        Args:
            message: The user's message.
            session_id: Session identifier for history tracking.

        Returns:
            Complete response text.
        """
        try:
            response = await self._agent.arun(
                message,
                session_id=session_id,
            )
            return response.content or ""

        except Exception as e:
            return f"[Error: {e}]"


# Module-level singleton instance
_agent_service: AgentService | None = None


def get_agent_service() -> AgentService:
    """Get or create the global agent service.

    Uses singleton pattern for resource efficiency.

    Returns:
        The AgentService instance.
    """
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
