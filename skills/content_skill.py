"""
Content skill for vector-based movie retrieval in the recommendation system.
Uses Two-Tower model + Milvus for industrial-grade recall.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from serving.recall_service import RecallService

logger = logging.getLogger(__name__)


class ContentSkill(BaseSkill):
    """
    Skill 2: Vector-based Content Retrieval (Industrial)
    - Uses Two-Tower model for embedding generation
    - Uses Milvus for vector search
    - Returns candidate items with recall scores
    """

    def __init__(self, model: str = None):
        """
        Initialize content skill with vector recall.

        Args:
            model: LLM model name (for potential query understanding)
        """
        super().__init__(model)
        self.recall_service = None

    def _get_recall_service(self) -> RecallService:
        """Get or create recall service."""
        if self.recall_service is None:
            from config import get_config
            config = get_config()
            self.recall_service = RecallService(config)
            self.recall_service.initialize()
        return self.recall_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vector-based content retrieval skill.

        Args:
            state: Current state containing user_id

        Returns:
            Updated state with content_candidates (recall results)
        """
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        query = state.get("query", "") or ""

        logger.info(f"ContentSkill: vector recall for user_id={user_id}, query='{query[:50]}...'")

        if user_id is None:
            raise ValueError("user_id is required for ContentSkill")

        service = self._get_recall_service()

        # Perform vector recall
        top_k = config.content_top_k
        item_ids, scores = service.recall(user_id, top_k=top_k)

        # Build candidates with item details
        candidates = []
        for item_id, score in zip(item_ids, scores):
            item_basic = service.item_features.store.get_basic_features(item_id)
            candidates.append({
                "movie_id": item_id,
                "recall_score": float(score),
                "title": item_basic.get("title", f"Item {item_id}"),
                "genres": item_basic.get("genres", []),
            })

        logger.info(f"ContentSkill: found {len(candidates)} candidates via vector recall")

        return {"content_candidates": candidates}


class ColdStartContentSkill(BaseSkill):
    """
    Skill: Cold Start Content Retrieval
    - Handles new users with no history
    - Returns popular items as fallback
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute cold start content retrieval.

        Args:
            state: Current state

        Returns:
            Updated state with popular content candidates
        """
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        logger.info(f"ColdStartContentSkill: handling cold start for user_id={user_id}")

        # Get popular items from recall service
        service = None
        try:
            if hasattr(self, '_get_recall_service'):
                service = self._get_recall_service()
        except Exception:
            pass

        if service:
            popular_items = service._get_popular_items(config.content_top_k)
        else:
            popular_items = list(range(1, config.content_top_k + 1))

        candidates = []
        for rank, item_id in enumerate(popular_items, 1):
            candidates.append({
                "movie_id": item_id,
                "recall_score": 0.5,
                "title": f"Popular Movie {item_id}",
                "genres": [],
            })

        logger.info(f"ColdStartContentSkill: {len(candidates)} popular items")

        return {
            "content_candidates": candidates,
            "is_cold_start": True,
        }
