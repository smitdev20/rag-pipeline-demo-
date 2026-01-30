"""Agent configuration with environment variable loading.

Pydantic-based configuration for the Agno chat agent.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load environment variables from .env file
load_dotenv()


class AgentConfig(BaseModel):
    """Configuration for the Agno chat agent.

    Attributes:
        openai_api_key: OpenAI API key for model access.
        model_name: OpenAI model identifier to use.
        temperature: Sampling temperature (0.0 = deterministic, 2.0 = creative).
        max_tokens: Maximum tokens in generated response.
    """

    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use",
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

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is provided and non-empty."""
        if not v or not v.strip():
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in .env file or environment."
            )
        return v.strip()


def get_agent_config() -> AgentConfig:
    """Create agent configuration from environment.

    Returns:
        Configured AgentConfig instance.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
    """
    return AgentConfig()
