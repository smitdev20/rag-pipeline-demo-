"""Agno agent service with streaming support.

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
"""

from collections.abc import AsyncGenerator
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat

from src.agent.config import AgentConfig, get_agent_config

# Store sessions in project data directory
_SESSIONS_DB = Path(__file__).parent.parent.parent / "data" / "sessions.db"


class AgentService:
    """Service for managing the Agno chat agent.

    Wraps Agno's Agent with:
    - Persistent SQLite storage for session history
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
        _SESSIONS_DB.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDb(
            db_file=str(_SESSIONS_DB),
            session_table="chat_sessions",
        )

    def _create_agent(self) -> Agent:
        """Create the Agno agent instance.

        Returns:
            Configured Agent with OpenAI model and SQLite storage.
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
            description="A helpful RAG chatbot assistant.",
            instructions=[
                "Provide helpful and accurate responses.",
                "Reference document context when available.",
                "Be concise yet thorough.",
            ],
            # History config: include last 20 messages (~10 conversation turns)
            # in context. Balances continuity vs token cost.
            add_history_to_context=True,
            num_history_messages=20,
            # Output as markdown for rich formatting in UI
            markdown=True,
        )

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
