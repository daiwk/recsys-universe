"""
Cross features for ranking model in industrial recommendation system.
Handles user-item interaction features.
"""
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from .user_features import UserFeatures
from .item_features import ItemFeatures
from config import get_config

logger = logging.getLogger(__name__)


class CrossFeatures:
    """
    Cross feature processor.
    Computes user-item interaction features for ranking model.
    """

    def __init__(
        self,
        user_features: UserFeatures,
        item_features: ItemFeatures
    ):
        """
        Initialize cross features processor.

        Args:
            user_features: User features processor
            item_features: Item features processor
        """
        self.user_features = user_features
        self.item_features = item_features
        self.config = get_config()

    def get_user_item_features(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """
        Get all features for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Dict with user, item, and cross features
        """
        user_basic = self.user_features.store.get_basic_features(user_id)
        user_behavior = self.user_features.store.get_behavior_features(user_id)
        item_basic = self.item_features.store.get_basic_features(item_id)
        item_stats = self.item_features.store.get_statistics(item_id)

        return {
            'user': {
                'basic': user_basic,
                'behavior': user_behavior
            },
            'item': {
                'basic': item_basic,
                'statistics': item_stats
            },
            'cross': self._compute_cross_features(user_id, item_id)
        }

    def _compute_cross_features(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """
        Compute cross features between user and item.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Dict of cross features
        """
        user_genres = set(
            self.user_features.store.get_history_genres(user_id)
        )
        item_genres = set(
            self.item_features.store.get_genres(item_id)
        )

        # Genre overlap
        genre_overlap = len(user_genres & item_genres)
        genre_jaccard = (
            genre_overlap / len(user_genres | item_genres)
            if user_genres or item_genres else 0
        )

        # User has interacted with item
        interacted_items = self.user_features.store.get_interaction_history(user_id)
        is_interacted = int(item_id in interacted_items)

        # Popularity match
        item_popularity = self.item_features.get_popularity_score(item_id)
        user_activity = self.user_features.get_feature_vector(user_id)
        user_activity_level = user_activity[2] if len(user_activity) > 2 else 0.5

        # Rating match (if user rated similar items highly)
        # Simplified: assume neutral
        rating_match = 0.5

        return {
            'genre_overlap': genre_overlap,
            'genre_jaccard': genre_jaccard,
            'is_interacted': is_interacted,
            'item_popularity': item_popularity,
            'user_activity_level': user_activity_level,
            'popularity_activity_ratio': (
                item_popularity / (user_activity_level + 0.01)
            ),
            'rating_match': rating_match,
            # Interaction recency (simplified)
            'recency_score': 0.5
        }

    def get_ranking_features(
        self,
        user_id: int,
        item_id: int
    ) -> np.ndarray:
        """
        Get dense feature vector for ranking model.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Feature vector for ranking
        """
        # User features
        user_vec = self.user_features.get_feature_vector(user_id)

        # Item features
        item_vec = self.item_features.get_feature_vector(item_id)

        # Cross features
        cross = self._compute_cross_features(user_id, item_id)

        # Combine all features
        cross_vec = np.array([
            cross['genre_overlap'] / 10.0,  # Normalize
            cross['genre_jaccard'],
            cross['is_interacted'],
            cross['item_popularity'],
            cross['user_activity_level'],
            cross['popularity_activity_ratio'],
            cross['rating_match'],
            cross['recency_score']
        ], dtype=np.float32)

        # Concatenate user + item + cross features
        return np.concatenate([user_vec, item_vec, cross_vec])

    def batch_get_ranking_features(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> np.ndarray:
        """
        Get ranking features for one user and multiple items.

        Args:
            user_id: User ID
            item_ids: List of item IDs

        Returns:
            2D array of shape (len(item_ids), feature_dim)
        """
        return np.array([
            self.get_ranking_features(user_id, item_id)
            for item_id in item_ids
        ], dtype=np.float32)

    def compute_similarity(
        self,
        user_id: int,
        item_id: int,
        method: str = 'cosine'
    ) -> float:
        """
        Compute similarity between user and item.

        Args:
            user_id: User ID
            item_id: Item ID
            method: Similarity method ('cosine', 'dot', 'euclidean')

        Returns:
            Similarity score
        """
        user_emb = self.user_features.get_user_embedding(user_id)
        item_emb = self.item_features.get_item_embedding(item_id)

        if user_emb is None or item_emb is None:
            return 0.0

        if method == 'cosine':
            # Cosine similarity
            dot = np.dot(user_emb, item_emb)
            norm = np.linalg.norm(user_emb) * np.linalg.norm(item_emb)
            return dot / (norm + 1e-8)

        elif method == 'dot':
            # Dot product
            return float(np.dot(user_emb, item_emb))

        elif method == 'euclidean':
            # Negative Euclidean distance (higher is better)
            dist = np.linalg.norm(user_emb - item_emb)
            return 1.0 / (dist + 1.0)

        return 0.0

    def get_diversity_score(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> float:
        """
        Compute diversity score for a list of items.

        Args:
            user_id: User ID
            item_ids: List of item IDs

        Returns:
            Diversity score (higher = more diverse)
        """
        if len(item_ids) <= 1:
            return 1.0

        # Get genres for all items
        all_genres = []
        for item_id in item_ids:
            genres = self.item_features.store.get_genres(item_id)
            all_genres.extend(genres)

        # Count unique genres
        unique_genres = len(set(all_genres))
        total_genres = len(all_genres)

        # Jaccard diversity
        if total_genres == 0:
            return 1.0

        return unique_genres / total_genres
