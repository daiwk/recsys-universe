"""
Collaborative filtering skill for the movie recommendation system.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from .data_utils import get_collab_candidates_by_genre
from config import get_config

logger = logging.getLogger(__name__)


class CollabSkill(BaseSkill):
    """
    Skill 3: Simple "collaborative filtering" (actually genre + global rating heuristic)
    - Calls get_collab_candidates_by_genre tool
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the collaborative filtering skill.

        Args:
            state: Current state containing user_id

        Returns:
            Updated state with collab_candidates
        """
        user_id = state.get("user_id")

        if user_id is None:
            raise KeyError("user_id is required in state for CollabSkill")

        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        config = get_config()
        logger.info(f"Generating collaborative candidates for user_id={user_id}")

        cands = get_collab_candidates_by_genre(user_id, k=config.collab_top_k)
        logger.debug(f"User {user_id} got {len(cands)} collaborative candidates")

        return {"collab_candidates": cands}
