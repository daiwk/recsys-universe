"""
Ranking skill for industrial recommendation system.
Uses DNN ranking model for CTR prediction.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from serving.rank_service import RankService

logger = logging.getLogger(__name__)


class RankingSkill(BaseSkill):
    """
    Skill 2: DNN Ranking
    - Uses RankingModel for CTR prediction
    - Takes recall candidates and returns ranked items
    """

    def __init__(self, model: str = None):
        """
        Initialize ranking skill.

        Args:
            model: LLM model name (for potential LLM augmentation)
        """
        super().__init__(model)
        self.rank_service = None

    def _get_rank_service(self) -> RankService:
        """Get or create rank service."""
        if self.rank_service is None:
            self.rank_service = RankService()
        return self.rank_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ranking skill.

        Args:
            state: Current state containing user_id and recall_candidates

        Returns:
            Updated state with ranked_recommendations
        """
        user_id = state.get("user_id")
        candidates = state.get("recall_candidates", [])

        if user_id is None:
            raise ValueError("user_id is required for RankingSkill")

        service = self._get_rank_service()

        logger.info(f"RankingSkill: ranking {len(candidates)} candidates for user {user_id}")

        if not candidates:
            return {
                "ranked_recommendations": [],
                "ranking_warning": "No recall candidates to rank"
            }

        # Perform ranking
        ranked = service.rank_with_recall(user_id, candidates)

        logger.info(f"RankingSkill: ranked {len(ranked)} items")

        return {
            "ranked_recommendations": ranked,
            "top_ctr": [r["ctr_score"] for r in ranked[:5]] if ranked else [],
        }
