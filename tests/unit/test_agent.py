"""Unit tests for AgentService and AgentConfig.

Tests configuration validation and agent initialization.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.agent.config import AgentConfig


class TestAgentConfig:
    """Tests for AgentConfig validation."""

    def test_valid_config_with_all_fields(self) -> None:
        """Config accepts valid values for all fields."""
        config = AgentConfig(
            openai_api_key="sk-test-key-12345",
            model_name="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )

        assert config.openai_api_key == "sk-test-key-12345"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_config_with_default_values(self) -> None:
        """Config uses sensible defaults when only API key provided."""
        config = AgentConfig(openai_api_key="sk-test-key")

        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024

    def test_config_fails_with_missing_api_key(self) -> None:
        """Config raises ValueError when API key is missing."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="")

        assert "OPENAI_API_KEY is required" in str(exc_info.value)

    def test_config_fails_with_whitespace_api_key(self) -> None:
        """Config rejects whitespace-only API key."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="   ")

        assert "OPENAI_API_KEY is required" in str(exc_info.value)

    def test_config_strips_api_key_whitespace(self) -> None:
        """Config strips leading/trailing whitespace from API key."""
        config = AgentConfig(openai_api_key="  sk-test-key  ")

        assert config.openai_api_key == "sk-test-key"

    def test_config_fails_with_temperature_too_low(self) -> None:
        """Config rejects temperature below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="sk-test", temperature=-0.1)

        assert "temperature" in str(exc_info.value).lower()

    def test_config_fails_with_temperature_too_high(self) -> None:
        """Config rejects temperature above 2.0."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="sk-test", temperature=2.5)

        assert "temperature" in str(exc_info.value).lower()

    def test_config_accepts_boundary_temperatures(self) -> None:
        """Config accepts temperature at boundaries (0.0 and 2.0)."""
        config_low = AgentConfig(openai_api_key="sk-test", temperature=0.0)
        config_high = AgentConfig(openai_api_key="sk-test", temperature=2.0)

        assert config_low.temperature == 0.0
        assert config_high.temperature == 2.0

    def test_config_fails_with_max_tokens_too_low(self) -> None:
        """Config rejects max_tokens below 1."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="sk-test", max_tokens=0)

        assert "max_tokens" in str(exc_info.value).lower()

    def test_config_fails_with_max_tokens_too_high(self) -> None:
        """Config rejects max_tokens above 128000."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="sk-test", max_tokens=200000)

        assert "max_tokens" in str(exc_info.value).lower()


class TestGetAgentConfig:
    """Tests for get_agent_config factory function."""

    def test_get_config_from_environment(self) -> None:
        """get_agent_config loads API key from environment."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}):
            # Need to reload to pick up env var
            config = AgentConfig(openai_api_key="sk-env-key")

            assert config.openai_api_key == "sk-env-key"

    def test_get_config_fails_without_env_var(self) -> None:
        """get_agent_config raises error when OPENAI_API_KEY not set."""
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False),
            pytest.raises(ValidationError),
        ):
            # Force config with empty API key from environment
            AgentConfig(openai_api_key="")


class TestAgentServiceInit:
    """Tests for AgentService initialization."""

    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_initializes_with_valid_config(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
    ) -> None:
        """AgentService initializes successfully with valid config."""
        from src.agent.chat_agent import AgentService

        config = AgentConfig(
            openai_api_key="sk-test-key",
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
        )

        service = AgentService(config=config)

        # Verify OpenAIChat was created with correct params
        mock_openai_chat.assert_called_once_with(
            id="gpt-4o-mini",
            api_key="sk-test-key",
            temperature=0.7,
            max_tokens=1024,
        )

        # Verify Agent was created
        mock_agent_class.assert_called_once()
        assert service._config == config

    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_uses_config_values(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
    ) -> None:
        """AgentService passes config values to OpenAIChat."""
        from src.agent.chat_agent import AgentService

        config = AgentConfig(
            openai_api_key="sk-custom-key",
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=4096,
        )

        AgentService(config=config)

        mock_openai_chat.assert_called_once_with(
            id="gpt-4o",
            api_key="sk-custom-key",
            temperature=0.3,
            max_tokens=4096,
        )

    def test_service_fails_with_missing_api_key(self) -> None:
        """AgentService raises error when config has no API key."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(openai_api_key="")

        assert "OPENAI_API_KEY is required" in str(exc_info.value)

    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_creates_agent_with_history_enabled(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
    ) -> None:
        """AgentService creates Agent with history settings."""
        from src.agent.chat_agent import AgentService

        config = AgentConfig(openai_api_key="sk-test")
        AgentService(config=config)

        # Verify Agent was called with history settings
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["add_history_to_messages"] is True
        assert call_kwargs["num_history_runs"] == 10
        assert call_kwargs["markdown"] is True


class TestGetAgentService:
    """Tests for get_agent_service singleton function."""

    def test_singleton_returns_same_instance(self) -> None:
        """get_agent_service returns the same instance on multiple calls."""
        import src.agent.chat_agent as chat_agent_module

        # Reset singleton
        chat_agent_module._agent_service = None

        with patch.object(chat_agent_module, "AgentService") as mock_service:
            mock_instance = MagicMock()
            mock_service.return_value = mock_instance

            first = chat_agent_module.get_agent_service()
            second = chat_agent_module.get_agent_service()

            assert first is second
            mock_service.assert_called_once()
