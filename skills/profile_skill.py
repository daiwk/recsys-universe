"""
Profile skill for generating user profiles in the movie recommendation system.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from .data_utils import get_user_history

logger = logging.getLogger(__name__)


class ProfileSkill(BaseSkill):
    """
    Skill 1: User Profile Generation
    - Uses get_user_history to retrieve user's movie history
    - Generates Chinese user profile using LLM
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the profile generation skill.

        Args:
            state: Current state containing user_id

        Returns:
            Updated state with user_profile and user_history

        Raises:
            KeyError: If user_id is missing from state
            ValueError: If user_id is invalid
        """
        user_id = state.get("user_id")

        if user_id is None:
            raise KeyError("user_id is required in state for ProfileSkill")

        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        logger.info(f"Generating profile for user_id={user_id}")

        history = get_user_history(user_id, n=10)

        hist_text_lines = [
            f"- {h['title']} (类型: {h['genres']}, 用户评分: {h['rating']})"
            for h in history
        ]
        hist_text = "\n".join(hist_text_lines) if hist_text_lines else "这个用户目前没有任何观影历史记录。"

        system_prompt = (
            "你是一个电影推荐系统的分析师。\n"
            "给定一个用户过去观看并评分过的电影列表，请用 3-5 条要点，总结这个用户的观影偏好。\n"
            "请重点描述：偏好的题材类型、年代、风格（轻松/烧脑/黑暗/温情等），以及你观察到的模式。\n"
            "输出使用中文。"
        )
        user_prompt = (
            "下面是该用户评分最高的一些电影列表，每行包含电影名称、类型和评分：\n"
            f"{hist_text}\n\n"
            "请根据以上信息，用 3-5 条要点，帮我总结这个用户的电影喜好画像。"
        )

        logger.debug(f"User {user_id} has {len(history)} history items")
        profile = self.call_llm(system_prompt, user_prompt, tag="PROFILE_SKILL")

        return {
            "user_history": history,
            "user_profile": profile,
        }
