"""
Base skill class for the movie recommendation system
"""
import os
import json
from typing import Any, Dict, List
from abc import ABC, abstractmethod

from openai import OpenAI


class BaseSkill(ABC):
    """
    Base class for all skills in the movie recommendation system.
    Each skill represents a specific capability similar to Claude Skills.
    """
    
    def __init__(self, model: str = "qwen3:1.7b"):
        """
        Initialize the skill with LLM client.
        
        Args:
            model: The model name to use for LLM calls
        """
        self.model = model
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )
        self.DEBUG = True

    def debug_log(self, tag: str, msg: str):
        """Simple debug print wrapper."""
        if self.DEBUG:
            print(f"[DEBUG][{tag}] {msg}")

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        tag: str = "LLM",
    ) -> str:
        """
        Wrapper for LLM calls, shared by all skills.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            tag: Tag for debugging
            
        Returns:
            LLM response content
        """
        self.debug_log(tag, f"调用 LLM，model={self.model}")
        self.debug_log(tag, f"System prompt：{system_prompt}")
        self.debug_log(tag, f"User prompt：{user_prompt}")

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = resp.choices[0].message.content or ""
        self.debug_log(tag, f"LLM 返回：{content}")
        return content

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