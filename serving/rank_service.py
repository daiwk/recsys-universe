"""
Ranking service for industrial recommendation system.
Handles CTR prediction using ranking model.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import get_config
from features.base import create_feature_store
from features.user_features import UserFeatures
from features.item_features import ItemFeatures
from features.cross_features import CrossFeatures
from models.ranking_model import RankingModel

logger = logging.getLogger(__name__)


class RankService:
    """
    Ranking service for CTR prediction.
    """

    def __init__(self, config=None, feature_store=None, ranking_model=None):
        """
        Initialize ranking service.

        Args:
            config: AppConfig (uses global if not provided)
            feature_store: Optional shared feature store
            ranking_model: Optional shared ranking model
        """
        self.config = config or get_config()

        # Initialize components (use provided or create new)
        self.feature_store = feature_store if feature_store is not None else create_feature_store(self.config)
        self.user_features = UserFeatures(self.feature_store)
        self.item_features = ItemFeatures(self.feature_store)
        self.cross_features = CrossFeatures(self.user_features, self.item_features)

        # Initialize ranking model (use provided or create new)
        self.ranking_model = ranking_model if ranking_model is not None else RankingModel(self.config)

        logger.info("RankService initialized")

    def rank(
        self,
        user_id: int,
        item_ids: List[int],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rank items for a user.

        Args:
            user_id: User ID
            item_ids: List of item IDs to rank
            top_k: Number of items to return

        Returns:
            List of ranked items with scores
        """
        if top_k is None:
            top_k = self.config.rank.rank_top_k

        if not item_ids:
            return []

        logger.info(f"Ranking {len(item_ids)} items for user {user_id}")

        # Get rankings
        ranked_ids, scores = self.ranking_model.get_top_k(
            user_id, item_ids, self.cross_features, top_k
        )

        # Build result
        results = []
        for item_id, score in zip(ranked_ids, scores):
            item_features = self.item_features.store.get_item_features(item_id)
            results.append({
                "item_id": item_id,
                "ctr_score": float(score),
                "title": item_features.get("basic", {}).get("title", ""),
                "genres": item_features.get("basic", {}).get("genres", []),
            })

        return results

    def predict_ctr(self, user_id: int, item_id: int) -> float:
        """
        Predict CTR for a single user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            CTR prediction
        """
        # Get features
        features = self.cross_features.get_ranking_features(user_id, item_id)

        # Predict
        score = self.ranking_model.predict(features)

        return float(score)

    def batch_predict_ctr(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> List[float]:
        """
        Predict CTR for multiple items.

        Args:
            user_id: User ID
            item_ids: List of item IDs

        Returns:
            List of CTR predictions
        """
        if not item_ids:
            return []

        # Get features for all items
        features = self.cross_features.batch_get_ranking_features(user_id, item_ids)

        # Batch predict
        scores = self.ranking_model.batch_predict(features)

        return scores.tolist()

    def rank_with_recall(
        self,
        user_id: int,
        recall_candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rank recall candidates.

        Args:
            user_id: User ID
            recall_candidates: List of candidate dicts with item_id
            top_k: Number of items to return

        Returns:
            List of ranked items
        """
        if top_k is None:
            top_k = self.config.rank.rank_top_k

        if not recall_candidates:
            return []

        # Extract item IDs
        item_ids = [c["item_id"] for c in recall_candidates]

        # Rank
        ranked = self.rank(user_id, item_ids, top_k)

        # Merge with recall scores
        recall_scores = {c["item_id"]: c.get("score", 0.0) for c in recall_candidates}

        for item in ranked:
            item["recall_score"] = recall_scores.get(item["item_id"], 0.0)

        return ranked

    def train(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Train ranking model on a batch.

        Args:
            features: Feature matrix
            labels: Labels

        Returns:
            Loss value
        """
        loss = self.ranking_model.train_step(features, labels)
        return loss

    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Dict with health status
        """
        return {
            "ranking_model": "loaded",
            "feature_store": "connected",
            "input_dim": self.ranking_model.get_feature_dim(),
        }


class RankingSkill:
    """
    Skills-compatible interface for ranking.
    """

    def __init__(self):
        self.service = RankService()

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute ranking skill.

        Args:
            state: Current state with user_id and recall_candidates

        Returns:
            Updated state with ranked_recommendations
        """
        user_id = state.get("user_id")
        candidates = state.get("recall_candidates", [])

        if user_id is None:
            raise ValueError("user_id is required for RankingSkill")

        if not candidates:
            return {"ranked_recommendations": []}

        # Perform ranking
        ranked = self.service.rank_with_recall(user_id, candidates)

        return {
            "ranked_recommendations": ranked,
            "top_ctr": [r["ctr_score"] for r in ranked[:5]] if ranked else [],
        }
