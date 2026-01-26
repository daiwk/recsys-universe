"""
User feature processing for industrial recommendation system.
"""
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from .base import FeatureHasher, RedisFeatureStore, UserFeatureStore
from config import get_config

logger = logging.getLogger(__name__)


class UserFeatures:
    """
    User feature processor.
    Handles user ID embedding, behavior features, and feature vectors.
    """

    def __init__(
        self,
        feature_store: RedisFeatureStore,
        hasher: Optional[FeatureHasher] = None
    ):
        """
        Initialize user features processor.

        Args:
            feature_store: Redis feature store
            hasher: Feature hasher for ID hashing
        """
        self.store = UserFeatureStore(feature_store)
        self.config = get_config()
        self.hasher = hasher or FeatureHasher(
            num_buckets=self.config.model.two_tower.num_hash_buckets
        )

    def get_user_id_hashes(self, user_id: int) -> List[int]:
        """
        Get hashed user ID for embedding lookup.

        Args:
            user_id: Raw user ID

        Returns:
            List of hash bucket indices
        """
        return [self.hasher.hash(user_id)]

    def get_behavior_hashes(self, user_id: int) -> List[int]:
        """
        Get hashed behavior features (genres, categories).

        Args:
            user_id: User ID

        Returns:
            List of hash bucket indices
        """
        genres = self.store.get_history_genres(user_id)
        # Hash each genre
        return [self.hasher.hash(g) for g in genres]

    def get_all_hashes(self, user_id: int) -> Dict[str, List[int]]:
        """
        Get all feature hashes for a user.

        Args:
            user_id: User ID

        Returns:
            Dict with 'id' and 'behavior' hash lists
        """
        return {
            'id': self.get_user_id_hashes(user_id),
            'behavior': self.get_behavior_hashes(user_id),
            'all': self.get_user_id_hashes(user_id) + self.get_behavior_hashes(user_id)
        }

    def get_feature_vector(self, user_id: int) -> np.ndarray:
        """
        Get dense feature vector for a user.
        Used as input to user tower.

        Args:
            user_id: User ID

        Returns:
            Feature vector as numpy array
        """
        basic = self.store.get_basic_features(user_id)
        behavior = self.store.get_behavior_features(user_id)

        # Build feature vector
        features = []

        # Numerical features
        features.append(basic.get('age', 25) / 100.0)  # Normalized
        features.append(1.0 if basic.get('gender') == 'M' else 0.0)

        # Behavior features
        features.append(min(behavior.get('total_views', 0) / 10000.0, 1.0))
        features.append(min(behavior.get('total_ratings', 0) / 1000.0, 1.0))
        features.append(min(behavior.get('avg_session_length', 0) / 3600.0, 1.0))

        # Recency features (days since last activity)
        last_login = behavior.get('last_login')
        if last_login:
            from datetime import datetime
            try:
                last_dt = datetime.fromisoformat(last_login)
                days_since = (datetime.now() - last_dt).days
                features.append(min(days_since / 30.0, 1.0))
            except:
                features.append(0.5)
        else:
            features.append(1.0)  # Unknown = cold start

        return np.array(features, dtype=np.float32)

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user embedding from cache or compute.

        Args:
            user_id: User ID

        Returns:
            User embedding vector or None
        """
        # Try cache first
        cached = self.store.get_embedding(user_id)
        if cached:
            return np.array(cached, dtype=np.float32)
        return None

    def set_user_embedding(self, user_id: int, embedding: np.ndarray) -> bool:
        """
        Cache user embedding.

        Args:
            user_id: User ID
            embedding: User embedding vector

        Returns:
            True if successful
        """
        return self.store.set_user_embedding(user_id, embedding.tolist())

    def get_interacted_items(self, user_id: int) -> List[int]:
        """
        Get list of items user has interacted with.

        Args:
            user_id: User ID

        Returns:
            List of item IDs
        """
        return self.store.get_interaction_history(user_id)

    def is_cold_start(self, user_id: int) -> bool:
        """
        Check if user is a cold start user.

        Args:
            user_id: User ID

        Returns:
            True if cold start
        """
        history = self.store.get_behavior_features(user_id)
        return len(history.get('viewed_items', [])) == 0

    def get_cold_start_features(self) -> np.ndarray:
        """
        Get default features for cold start users.

        Returns:
            Default feature vector
        """
        return self.get_feature_vector(-1)  # Uses defaults


class UserFeatureBuilder:
    """Builder for creating user feature records."""

    def __init__(self, user_id: int):
        """
        Initialize feature builder.

        Args:
            user_id: User ID
        """
        self.user_id = user_id
        self._basic: Dict[str, Any] = {}
        self._behavior: Dict[str, Any] = {}

    def set_basic(
        self,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        city: Optional[str] = None,
        **kwargs
    ) -> 'UserFeatureBuilder':
        """Set basic user features."""
        self._basic.update({
            'user_id': self.user_id,
            'age': age,
            'gender': gender,
            'city': city,
            **kwargs
        })
        return self

    def set_behavior(
        self,
        total_views: int = 0,
        total_ratings: int = 0,
        preferred_genres: List[str] = None,
        viewed_items: List[int] = None,
        last_login: str = None,
        avg_session_length: float = 0.0,
        **kwargs
    ) -> 'UserFeatureBuilder':
        """Set behavior features."""
        self._behavior.update({
            'total_views': total_views,
            'total_ratings': total_ratings,
            'preferred_genres': preferred_genres or [],
            'viewed_items': viewed_items or [],
            'last_login': last_login,
            'avg_session_length': avg_session_length,
            **kwargs
        })
        return self

    def build(self) -> Dict[str, Any]:
        """Build the complete feature record."""
        from datetime import datetime
        return {
            'basic': self._basic,
            'behavior': self._behavior,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
