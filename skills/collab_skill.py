"""
Ranking skill for industrial recommendation system.
Uses DNN ranking model for CTR prediction and re-ranking.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from serving.rank_service import RankService

logger = logging.getLogger(__name__)


class CollabSkill(BaseSkill):
    """
    Skill 3: Industrial Ranking (was simple collaborative filtering)
    - Uses DNN ranking model for CTR prediction
    - Re-ranks candidates from content/cold-start recall
    - Returns ranked candidates with CTR scores
    """

    def __init__(self, model: str = None):
        """
        Initialize ranking skill.

        Args:
            model: LLM model name (for potential explanation generation)
        """
        super().__init__(model)
        self.rank_service = None

    def _get_rank_service(self) -> RankService:
        """Get or create rank service."""
        if self.rank_service is None:
            from config import get_config
            config = get_config()
            self.rank_service = RankService(config)
        return self.rank_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ranking skill.

        Args:
            state: Current state containing user_id and content_candidates

        Returns:
            Updated state with collab_candidates (ranked candidates)
        """
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        content_candidates = state.get("content_candidates", [])

        logger.info(f"CollabSkill: ranking for user_id={user_id}, {len(content_candidates)} candidates")

        if user_id is None:
            raise ValueError("user_id is required for CollabSkill")

        # If no content candidates, return empty
        if not content_candidates:
            logger.warning(f"No content candidates for user {user_id}")
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

        # Build response with CTR scores
        ranked_candidates = []
        for item in ranked_items:
            ranked_candidates.append({
                "movie_id": item.get("item_id"),
                "ctr_score": item.get("ctr_score", 0.5),
                "recall_score": item.get("recall_score", 0.5),
                "title": item.get("title", ""),
                "genres": item.get("genres", []),
            })

        logger.info(f"CollabSkill: ranked {len(ranked_candidates)} candidates")

        return {"collab_candidates": ranked_candidates}


class RankingSkill(BaseSkill):
    """
    Skill 3 Alternative: Direct Ranking
    - Ranks items directly without intermediate steps
    - Returns fully ranked candidates
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
        """
        Execute direct ranking.

        Args:
            state: Current state with candidates to rank

        Returns:
            Updated state with ranked candidates
        """
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
