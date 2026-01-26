"""
Vector recall skill for industrial recommendation system.
Uses Two-Tower model + FAISS for vector-based retrieval.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from serving.recall_service import RecallService

logger = logging.getLogger(__name__)


class VectorRecallSkill(BaseSkill):
    """
    Skill 1: Vector-based Recall
    - Uses Two-Tower model for embedding generation
    - Uses FAISS for vector search
    - Returns candidate items with recall scores
    """

    def __init__(self, model: str = None):
        """
        Initialize vector recall skill.

        Args:
            model: LLM model name (for potential LLM augmentation)
        """
        super().__init__(model)
        self.recall_service = None

    def _get_recall_service(self) -> RecallService:
        """Get or create recall service."""
        if self.recall_service is None:
            self.recall_service = RecallService()
            self.recall_service.initialize()
        return self.recall_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vector recall skill.

        Args:
            state: Current state containing user_id and query

        Returns:
            Updated state with recall_candidates
        """
        user_id = state.get("user_id")

        if user_id is None:
            raise ValueError("user_id is required for VectorRecallSkill")

        service = self._get_recall_service()

        logger.info(f"VectorRecallSkill: recalling for user {user_id}")

        # Perform recall
        item_ids, scores = service.recall(user_id, top_k=100)

        # Build candidates
        candidates = []
        for item_id, score in zip(item_ids, scores):
            # Get item details from feature store
            item_features = service.item_features.store.get_item_features(item_id)
            candidates.append({
                "item_id": item_id,
                "recall_score": float(score),
                "title": item_features.get("basic", {}).get("title", ""),
                "genres": item_features.get("basic", {}).get("genres", []),
            })

        logger.info(f"VectorRecallSkill: found {len(candidates)} candidates")

        return {
            "recall_candidates": candidates,
            "recall_scores": scores[:5] if scores else [],
        }


class ColdStartRecallSkill(BaseSkill):
    """
    Skill: Cold Start Recall
    - Handles new users with no history
    - Uses popular items and demographic similarity
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute cold start recall skill.

        Args:
            state: Current state

        Returns:
            Updated state with popular candidates
        """
        from config import get_config
        config = get_config()

        logger.info("ColdStartRecallSkill: handling cold start user")

        # Get popular items
        popular_count = config.recall.recall_top_k
        popular_items = list(range(1, popular_count + 1))

        candidates = []
        for item_id in popular_items:
            candidates.append({
                "item_id": item_id,
                "recall_score": 0.5,  # Default score
                "title": f"Popular Item {item_id}",
                "genres": [],
                "is_cold_start": True,
            })

        return {
            "recall_candidates": candidates,
            "recall_scores": [0.5] * 5,
            "is_cold_start": True,
        }
