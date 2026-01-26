"""
Item feature processing for industrial recommendation system.
"""
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from .base import FeatureHasher, RedisFeatureStore, ItemFeatureStore
from config import get_config

logger = logging.getLogger(__name__)


class ItemFeatures:
    """
    Item feature processor.
    Handles item ID embedding, content features, and feature vectors.
    """

    def __init__(
        self,
        feature_store: RedisFeatureStore,
        hasher: Optional[FeatureHasher] = None
    ):
        """
        Initialize item features processor.

        Args:
            feature_store: Redis feature store
            hasher: Feature hasher for ID hashing
        """
        self.store = ItemFeatureStore(feature_store)
        self.config = get_config()
        self.hasher = hasher or FeatureHasher(
            num_buckets=self.config.model.two_tower.num_hash_buckets
        )

    def get_item_id_hashes(self, item_id: int) -> List[int]:
        """
        Get hashed item ID for embedding lookup.

        Args:
            item_id: Raw item ID

        Returns:
            List of hash bucket indices
        """
        return [self.hasher.hash(item_id)]

    def get_genre_hashes(self, item_id: int) -> List[int]:
        """
        Get hashed genre features for embedding lookup.

        Args:
            item_id: Item ID

        Returns:
            List of hash bucket indices
        """
        genres = self.store.get_genres(item_id)
        return [self.hasher.hash(g) for g in genres]

    def get_all_hashes(self, item_id: int) -> Dict[str, List[int]]:
        """
        Get all feature hashes for an item.

        Args:
            item_id: Item ID

        Returns:
            Dict with 'id' and 'genre' hash lists
        """
        return {
            'id': self.get_item_id_hashes(item_id),
            'genre': self.get_genre_hashes(item_id),
            'all': self.get_item_id_hashes(item_id) + self.get_genre_hashes(item_id)
        }

    def get_feature_vector(self, item_id: int) -> np.ndarray:
        """
        Get dense feature vector for an item.
        Used as input to item tower.

        Args:
            item_id: Item ID

        Returns:
            Feature vector as numpy array
        """
        basic = self.store.get_basic_features(item_id)
        stats = self.store.get_statistics(item_id)

        # Build feature vector
        features = []

        # Release year (normalized to [0, 1])
        release_year = basic.get('release_year', 2000)
        features.append(max(0, min((release_year - 1900) / 124.0, 1.0)))

        # Statistics features
        views = stats.get('views', 0)
        features.append(min(views / 1000000.0, 1.0))  # Normalize by 1M

        avg_rating = stats.get('avg_rating', 3.0)
        features.append((avg_rating - 1.0) / 4.0)  # Normalize [1, 5] to [0, 1]

        ctr = stats.get('ctr', 0.05)
        features.append(min(ctr / 0.3, 1.0))  # Normalize by max CTR

        # Genre count (popularity indicator)
        genres = basic.get('genres', [])
        features.append(len(genres) / 10.0)  # Normalize

        # Freshness (days since release)
        import time
        if release_year > 1900:
            release_timestamp = time.mktime(
                (release_year, 1, 1, 0, 0, 0, 0, 0, 0)
            )
            days_since_release = (time.time() - release_timestamp) / 86400
            features.append(min(days_since_release / 36500.0, 1.0))  # Max 100 years
        else:
            features.append(1.0)

        return np.array(features, dtype=np.float32)

    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """
        Get item embedding from cache or compute.

        Args:
            item_id: Item ID

        Returns:
            Item embedding vector or None
        """
        # Try cache first
        cached = self.store.get_embedding(item_id)
        if cached:
            return np.array(cached, dtype=np.float32)
        return None

    def set_item_embedding(self, item_id: int, embedding: np.ndarray) -> bool:
        """
        Cache item embedding.

        Args:
            item_id: Item ID
            embedding: Item embedding vector

        Returns:
            True if successful
        """
        return self.store.set_item_embedding(item_id, embedding.tolist())

    def get_genres(self, item_id: int) -> List[str]:
        """
        Get genres for an item.

        Args:
            item_id: Item ID

        Returns:
            List of genre strings
        """
        return self.store.get_genres(item_id)

    def is_popular(self, item_id: int, threshold: int = 10000) -> bool:
        """
        Check if item is popular based on views.

        Args:
            item_id: Item ID
            threshold: View threshold for popularity

        Returns:
            True if popular
        """
        stats = self.store.get_statistics(item_id)
        return stats.get('views', 0) >= threshold

    def get_popularity_score(self, item_id: int) -> float:
        """
        Get normalized popularity score.

        Args:
            item_id: Item ID

        Returns:
            Popularity score in [0, 1]
        """
        stats = self.store.get_statistics(item_id)
        views = stats.get('views', 0)
        return min(views / 100000.0, 1.0)  # Normalize by 100K


class ItemFeatureBuilder:
    """Builder for creating item feature records."""

    def __init__(self, item_id: int):
        """
        Initialize feature builder.

        Args:
            item_id: Item ID
        """
        self.item_id = item_id
        self._basic: Dict[str, Any] = {}
        self._statistics: Dict[str, Any] = {}

    def set_basic(
        self,
        title: str = "",
        genres: List[str] = None,
        release_year: Optional[int] = None,
        **kwargs
    ) -> 'ItemFeatureBuilder':
        """Set basic item features."""
        self._basic.update({
            'item_id': self.item_id,
            'title': title,
            'genres': genres or [],
            'release_year': release_year,
            **kwargs
        })
        return self

    def set_statistics(
        self,
        views: int = 0,
        avg_rating: float = 3.0,
        ctr: float = 0.05,
        conversion_rate: float = 0.02,
        **kwargs
    ) -> 'ItemFeatureBuilder':
        """Set statistics features."""
        self._statistics.update({
            'views': views,
            'avg_rating': avg_rating,
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            **kwargs
        })
        return self

    def build(self) -> Dict[str, Any]:
        """Build the complete feature record."""
        from datetime import datetime
        return {
            'basic': self._basic,
            'statistics': self._statistics,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
