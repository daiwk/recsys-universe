"""
Base skill class for the movie recommendation system.
"""
import os
import logging
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from functools import lru_cache

from openai import OpenAI

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseSkill(ABC):
    """
    Base class for all skills in the movie recommendation system.
    Each skill represents a specific capability similar to Claude Skills.
    """

    # Class-level LLM client cache (shared across all skill instances)
    _client_cache: Dict[str, OpenAI] = {}

    def __init__(self, model: str = None):
        """
        Initialize the skill with LLM client.

        Args:
            model: The model name to use for LLM calls. Defaults to config setting.
        """
        config = get_config()
        self.model = model or config.llm.model
        self.debug = config.debug
        self._client = self._get_client()

    def _get_client(self) -> OpenAI:
        """
        Get or create a cached LLM client.

        Returns:
            OpenAI client instance
        """
        cache_key = f"{self.model}:{get_config().llm.base_url}"

        if cache_key not in BaseSkill._client_cache:
            config = get_config()
            # Validate required configuration
            if not config.llm.api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is not set. "
                    "Please configure your LLM credentials."
                )
            if not config.llm.base_url:
                raise ValueError(
                    "OPENAI_BASE_URL environment variable is not set. "
                    "Please configure your LLM endpoint."
                )

            BaseSkill._client_cache[cache_key] = OpenAI(
                api_key=config.llm.api_key,
                base_url=config.llm.base_url,
            )
            logger.debug(f"Created new LLM client for model={self.model}")

        return BaseSkill._client_cache[cache_key]

    @property
    def client(self) -> OpenAI:
        """Get the LLM client (read-only property for safety)."""
        return self._client

    def debug_log(self, tag: str, msg: str):
        """Debug print wrapper with logging support."""
        if self.debug:
            logger.debug(f"[{tag}] {msg}")

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        tag: str = "LLM",
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Wrapper for LLM calls, shared by all skills.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            tag: Tag for logging
            temperature: Override temperature setting
            max_tokens: Override max tokens setting

        Returns:
            LLM response content

        Raises:
            RuntimeError: If LLM call fails
        """
        config = get_config()
        temp = temperature if temperature is not None else config.llm.temperature
        tokens = max_tokens if max_tokens is not None else config.llm.max_tokens

        self.debug_log(tag, f"Calling LLM, model={self.model}")
        self.debug_log(tag, f"System prompt length: {len(system_prompt)}")
        self.debug_log(tag, f"User prompt length: {len(user_prompt)}")

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temp,
            }
            if tokens is not None:
                kwargs["max_tokens"] = tokens

            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""

            self.debug_log(tag, f"LLM response length: {len(content)}")
            return content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}") from e

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the skill with the given state.

        Args:
            state: Current state of the recommendation process

        Returns:
            Updated state after skill execution
        """
        pass