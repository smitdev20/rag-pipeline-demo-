"""Agno agent service with streaming support.

Core module for the chatbot's intelligence and conversation handling.
"""

from collections.abc import AsyncGenerator

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from src.agent.config import AgentConfig, get_agent_config


class AgentService:
    """Service for managing the Agno chat agent.

    Handles streaming responses and session management.
    Maintains conversation history per session via Agno.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the agent service.

        Args:
            config: Optional agent configuration.
                    Loads from environment if not provided.
        """
        self._config = config or get_agent_config()
        self._agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the Agno agent instance.

        Returns:
            Configured Agent with OpenAI model.
        """
        model = OpenAIChat(
            id=self._config.model_name,
            api_key=self._config.openai_api_key,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        return Agent(
            model=model,
            description="A helpful RAG chatbot assistant.",
            instructions=[
                "Provide helpful and accurate responses.",
                "Reference document context when available.",
                "Be concise yet thorough.",
            ],
            add_history_to_messages=True,
            num_history_runs=10,
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
            response_stream = await self._agent.arun(
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
