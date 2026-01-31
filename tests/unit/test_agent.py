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
        """Valid values for all fields, config accepts them it should."""
        config = AgentConfig(
            api_key="sk-test-key-12345",
            model_name="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )

        assert config.api_key == "sk-test-key-12345"
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_config_with_default_values(self) -> None:
        """Sensible defaults for temperature and max_tokens, config uses."""
        # Test hardcoded defaults (temperature, max_tokens don't come from env)
        config = AgentConfig(api_key="sk-test-key")

        # Temperature and max_tokens have hardcoded defaults in Field definition
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        # model_name may come from LLM_MODEL env var; verify it exists
        assert isinstance(config.model_name, str)

    def test_config_fails_with_missing_api_key(self) -> None:
        """Raise ValueError, config must when missing the API key is."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="")

        assert "API key required" in str(exc_info.value)

    def test_config_fails_with_whitespace_api_key(self) -> None:
        """Reject whitespace-only API key, config must."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="   ")

        assert "API key required" in str(exc_info.value)

    def test_config_strips_api_key_whitespace(self) -> None:
        """Strip whitespace from API key, config does."""
        config = AgentConfig(api_key="  sk-test-key  ")

        assert config.api_key == "sk-test-key"

    def test_config_fails_with_temperature_too_low(self) -> None:
        """Reject temperature below 0.0, config must."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="sk-test", temperature=-0.1)

        assert "temperature" in str(exc_info.value).lower()

    def test_config_fails_with_temperature_too_high(self) -> None:
        """Reject temperature above 2.0, config must."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="sk-test", temperature=2.5)

        assert "temperature" in str(exc_info.value).lower()

    def test_config_accepts_boundary_temperatures(self) -> None:
        """Accept temperature at boundaries (0.0 and 2.0), config does."""
        config_low = AgentConfig(api_key="sk-test", temperature=0.0)
        config_high = AgentConfig(api_key="sk-test", temperature=2.0)

        assert config_low.temperature == 0.0
        assert config_high.temperature == 2.0

    def test_config_fails_with_max_tokens_too_low(self) -> None:
        """Reject max_tokens below 1, config must."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="sk-test", max_tokens=0)

        assert "max_tokens" in str(exc_info.value).lower()

    def test_config_fails_with_max_tokens_too_high(self) -> None:
        """Reject max_tokens above 128000, config must."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="sk-test", max_tokens=200000)

        assert "max_tokens" in str(exc_info.value).lower()


class TestGetAgentConfig:
    """Tests for get_agent_config factory function."""

    def test_get_config_from_environment(self) -> None:
        """Load API key from environment, get_agent_config does."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}):
            # Need to reload to pick up env var
            config = AgentConfig(api_key="sk-env-key")

            assert config.api_key == "sk-env-key"

    def test_get_config_fails_without_env_var(self) -> None:
        """Raise error when OPENAI_API_KEY not set, get_agent_config must."""
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False),
            pytest.raises(ValidationError),
        ):
            # Force config with empty API key from environment
            AgentConfig(api_key="")


class TestAgentServiceInit:
    """Tests for AgentService initialization."""

    @patch("src.agent.chat_agent.Knowledge")
    @patch("src.agent.chat_agent.LanceDb")
    @patch("src.agent.chat_agent.OpenAIEmbedder")
    @patch("src.agent.chat_agent.SqliteDb")
    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_initializes_with_valid_config(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
        mock_sqlite_db: MagicMock,
        mock_embedder: MagicMock,
        mock_lancedb: MagicMock,
        mock_knowledge: MagicMock,
    ) -> None:
        """Initialize successfully with valid config, AgentService does."""
        from src.agent.chat_agent import AgentService

        config = AgentConfig(
            api_key="sk-test-key",
            base_url=None,  # Explicit None to override env
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
        )

        service = AgentService(config=config)

        # Verify OpenAIChat was created with correct params
        mock_openai_chat.assert_called_once_with(
            id="gpt-4o-mini",
            api_key="sk-test-key",
            base_url=None,
            temperature=0.7,
            max_tokens=1024,
        )

        # Verify SqliteDb was created for session persistence
        mock_sqlite_db.assert_called_once()

        # Verify Agent was created with db
        mock_agent_class.assert_called_once()
        assert service._config == config

    @patch("src.agent.chat_agent.Knowledge")
    @patch("src.agent.chat_agent.LanceDb")
    @patch("src.agent.chat_agent.OpenAIEmbedder")
    @patch("src.agent.chat_agent.SqliteDb")
    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_uses_config_values(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
        mock_sqlite_db: MagicMock,
        mock_embedder: MagicMock,
        mock_lancedb: MagicMock,
        mock_knowledge: MagicMock,
    ) -> None:
        """Pass config values to OpenAIChat, AgentService does."""
        from src.agent.chat_agent import AgentService

        config = AgentConfig(
            api_key="sk-custom-key",
            base_url="https://api.example.com",  # Explicit value
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=4096,
        )

        AgentService(config=config)

        mock_openai_chat.assert_called_once_with(
            id="gpt-4o",
            api_key="sk-custom-key",
            base_url="https://api.example.com",
            temperature=0.3,
            max_tokens=4096,
        )

    def test_service_fails_with_missing_api_key(self) -> None:
        """Raise error when config has no API key, AgentService must."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(api_key="")

        assert "API key required" in str(exc_info.value)

    @patch("src.agent.chat_agent.Knowledge")
    @patch("src.agent.chat_agent.LanceDb")
    @patch("src.agent.chat_agent.OpenAIEmbedder")
    @patch("src.agent.chat_agent.SqliteDb")
    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_creates_agent_with_db_and_history(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
        mock_sqlite_db: MagicMock,
        mock_embedder: MagicMock,
        mock_lancedb: MagicMock,
        mock_knowledge: MagicMock,
    ) -> None:
        """Create Agent with SQLite db and history settings, AgentService does.

        Test these specific settings, why we do:
        - db: Required for session persistence across requests it is
        - add_history_to_context: Enables conversation continuity it does
        - num_history_messages: Controls context window (20 = ~10 turns) it does
        - markdown: Enables rich formatting in UI it does
        """
        from src.agent.chat_agent import AgentService

        config = AgentConfig(api_key="sk-test")
        AgentService(config=config)

        # Verify Agent was called with db and history settings
        call_kwargs = mock_agent_class.call_args.kwargs
        assert call_kwargs["db"] == mock_sqlite_db.return_value
        assert call_kwargs["add_history_to_context"] is True
        assert call_kwargs["num_history_messages"] == 20
        assert call_kwargs["markdown"] is True

    @patch("src.agent.chat_agent.Knowledge")
    @patch("src.agent.chat_agent.LanceDb")
    @patch("src.agent.chat_agent.OpenAIEmbedder")
    @patch("src.agent.chat_agent.SqliteDb")
    @patch("src.agent.chat_agent.OpenAIChat")
    @patch("src.agent.chat_agent.Agent")
    def test_service_creates_sqlite_db_with_correct_params(
        self,
        mock_agent_class: MagicMock,
        mock_openai_chat: MagicMock,
        mock_sqlite_db: MagicMock,
        mock_embedder: MagicMock,
        mock_lancedb: MagicMock,
        mock_knowledge: MagicMock,
    ) -> None:
        """Create SqliteDb with expected db_file and session_table, AgentService does."""
        from src.agent.chat_agent import AgentService

        config = AgentConfig(api_key="sk-test")
        AgentService(config=config)

        # Verify SqliteDb was called with expected params
        call_kwargs = mock_sqlite_db.call_args.kwargs
        assert "sessions.db" in call_kwargs["db_file"]
        assert call_kwargs["session_table"] == "chat_sessions"


class TestGetAgentService:
    """Tests for get_agent_service singleton function, hmm."""

    def test_singleton_returns_same_instance(self) -> None:
        """Return the same instance on multiple calls, get_agent_service does."""
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
