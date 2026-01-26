"""
Configuration management for the movie recommendation system.
Supports environment variables and Pydantic-style validation.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    api_key: Optional[str] = None  # Use None as sentinel
    base_url: Optional[str] = None
    model: str = "Qwen/Qwen3-1.7B"
    temperature: float = 0.3
    max_tokens: Optional[int] = None

    def __post_init__(self):
        # Override with environment variables only if not explicitly set
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")


@dataclass
class AppConfig:
    """Application configuration settings."""
    # Debug mode
    debug: bool = False

    # Recommendation settings
    max_steps: int = 10
    max_planner_steps: int = 8
    content_top_k: int = 15
    collab_top_k: int = 30
    merge_top_k: int = 40
    final_top_k: int = 5

    # Data settings
    movielens_url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    local_zip: str = "ml-1m.zip"

    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self):
        # Override debug mode from environment
        debug_env = os.environ.get("RECSYS_DEBUG", "")
        if debug_env.lower() in ("true", "1", "yes"):
            self.debug = True


def get_config() -> AppConfig:
    """
    Get the application configuration.
    Returns a cached singleton instance.

    Returns:
        AppConfig instance with all settings
    """
    if not hasattr(get_config, "_instance"):
        get_config._instance = AppConfig()
    return get_config._instance


# Global config instance (lazy initialization)
config = None  # Will be initialized on first access
