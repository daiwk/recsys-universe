"""
Tests for config module.
"""
import os
import pytest

# Set environment before importing
os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["OPENAI_BASE_URL"] = "http://test.local/v1"
os.environ["RECSYS_DEBUG"] = "true"

from config import AppConfig, LLMConfig, get_config


class TestLLMConfig:
    """Tests for LLMConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.api_key == "test_key"  # From env
        assert config.base_url == "http://test.local/v1"  # From env
        assert config.model == "Qwen/Qwen3-1.7B"
        assert config.temperature == 0.3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LLMConfig(
            api_key="custom_key",
            base_url="http://custom.local/v1",
            model="custom/model",
            temperature=0.5
        )
        assert config.api_key == "custom_key"
        assert config.base_url == "http://custom.local/v1"
        assert config.model == "custom/model"
        assert config.temperature == 0.5


class TestAppConfig:
    """Tests for AppConfig class."""

    def test_default_values(self):
        """Test default application configuration."""
        config = AppConfig()
        assert config.debug is True  # From env RECSYS_DEBUG
        assert config.max_steps == 10
        assert config.max_planner_steps == 8
        assert config.content_top_k == 15
        assert config.collab_top_k == 30
        assert config.merge_top_k == 40
        assert config.final_top_k == 5

    def test_nested_llm_config(self):
        """Test nested LLM configuration."""
        config = get_config()
        assert isinstance(config.llm, LLMConfig)
        assert config.llm.model == "Qwen/Qwen3-1.7B"

    def test_get_config_singleton(self):
        """Test that get_config returns consistent config."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
