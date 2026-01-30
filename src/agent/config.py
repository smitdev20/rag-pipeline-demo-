"""Agent configuration with environment variable loading.

Pydantic-based configuration for the Agno chat agent.
Supports OpenAI and OpenAI-compatible APIs via custom base URL.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()


class AgentConfig(BaseModel):
    """Configuration for the Agno chat agent.

    Supports OpenAI and any OpenAI-compatible API via LLM_BASE_URL.

    Attributes:
        api_key: API key for model access.
        base_url: API base URL (None for OpenAI default).
        model_name: Model identifier to use.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative).
        max_tokens: Maximum tokens in generated response.
    """

    api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        description="API key for LLM provider",
    )
    base_url: str | None = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or None,
        description="API base URL (None for OpenAI default)",
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"),
        description="Model to use",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for response generation",
    )
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=128000,
        description="Maximum tokens in generated response",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is provided and non-empty."""
        if not v or not v.strip():
            raise ValueError(
                "API key required. Set LLM_API_KEY or OPENAI_API_KEY in .env"
            )
        return v.strip()


def get_agent_config() -> AgentConfig:
    """Create agent configuration from environment.

    Returns:
        Configured AgentConfig instance.

    Raises:
        ValueError: If no API key is set.
    """
    return AgentConfig()
