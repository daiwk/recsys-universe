"""
Collaborative filtering / Ranking skill for the recommendation system.
Supports both Legacy (CF heuristic) and Industrial (DNN Ranking) modes.
"""
import os
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from .data_utils import get_collab_candidates_by_genre
from serving.rank_service import RankService

logger = logging.getLogger(__name__)


class CollabSkill(BaseSkill):
    """
    Skill 3: Collaborative Filtering / Ranking
    - Legacy mode: Uses genre-based heuristic (get_collab_candidates_by_genre)
    - Industrial mode: Uses DNN ranking model for CTR prediction
    """

    def __init__(self, model: str = None):
        """
        Initialize collaborative filtering skill.

        Args:
            model: LLM model name (for potential explanation generation)
        """
        super().__init__(model)
        self.rank_service = None
        self._is_industrial = os.environ.get("RECSYS_ARCHITECTURE", "industrial") == "industrial"

    def _get_rank_service(self) -> RankService:
        """Get or create rank service (Industrial mode)."""
        if self.rank_service is None:
            from config import get_config
            config = get_config()
            self.rank_service = RankService(config)
        return self.rank_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute collaborative filtering / ranking skill.

        Args:
            state: Current state containing user_id and content_candidates

        Returns:
            Updated state with collab_candidates
        """
        if self._is_industrial:
            return self._execute_industrial(state)
        else:
            return self._execute_legacy(state)

    def _execute_industrial(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DNN ranking (Industrial mode)."""
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        content_candidates = state.get("content_candidates", [])

        logger.info(f"CollabSkill [Industrial]: ranking for user_id={user_id}, {len(content_candidates)} candidates")

        if user_id is None:
            raise ValueError("user_id is required for CollabSkill")

        if not content_candidates:
            return {"collab_candidates": []}

        service = self._get_rank_service()

        # Prepare candidates for ranking
        candidates_for_ranking = []
        for item in content_candidates:
            candidates_for_ranking.append({
                "item_id": item.get("movie_id"),
                "recall_score": item.get("recall_score", 0.5),
                "title": item.get("title", ""),
                "genres": item.get("genres", []),
            })

        # Perform ranking
        top_k = config.collab_top_k
        ranked_items = service.rank_with_recall(user_id, candidates_for_ranking, top_k=top_k)

        ranked_candidates = []
        for item in ranked_items:
            ranked_candidates.append({
                "movie_id": item.get("item_id"),
                "ctr_score": item.get("ctr_score", 0.5),
                "recall_score": item.get("recall_score", 0.5),
                "title": item.get("title", ""),
                "genres": item.get("genres", []),
            })

        logger.info(f"CollabSkill [Industrial]: ranked {len(ranked_candidates)} candidates")
        return {"collab_candidates": ranked_candidates}

    def _execute_legacy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute genre-based collaborative filtering (Legacy mode)."""
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")

        if user_id is None:
            raise KeyError("user_id is required in state for CollabSkill")

        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        logger.info(f"CollabSkill [Legacy]: CF heuristic for user_id={user_id}")

        cands = get_collab_candidates_by_genre(user_id, k=config.collab_top_k)
        logger.debug(f"User {user_id} got {len(cands)} collaborative candidates")

        return {"collab_candidates": cands}


class RankingSkill(BaseSkill):
    """
    Skill 3 Alternative: Direct Ranking (Industrial mode only)
    - Ranks items directly without intermediate steps
    """

    def __init__(self, model: str = None):
        super().__init__(model)
        self.rank_service = None

    def _get_rank_service(self) -> RankService:
        if self.rank_service is None:
            from config import get_config
            config = get_config()
            self.rank_service = RankService(config)
        return self.rank_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute direct ranking."""
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        candidates = state.get("content_candidates", state.get("recall_candidates", []))

        if not candidates or user_id is None:
            return {"collab_candidates": [], "ranked": True}

        service = self._get_rank_service()
        ranked = service.rank_with_recall(user_id, candidates, top_k=config.collab_top_k)

        return {
            "collab_candidates": [
                {
                    "movie_id": item.get("item_id"),
                    "ctr_score": item.get("ctr_score", 0.5),
                    "recall_score": item.get("recall_score", 0.5),
                    "title": item.get("title", ""),
                    "genres": item.get("genres", []),
                }
                for item in ranked
            ],
            "ranked": True,
        }
