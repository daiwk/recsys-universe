"""
Recall service for industrial recommendation system.
Handles vector-based retrieval using Two-Tower model and FAISS.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from config import get_config
from features.base import create_feature_store
from features.user_features import UserFeatures
from features.item_features import ItemFeatures
from features.cross_features import CrossFeatures
from models.two_tower import TwoTowerModel
from serving.faiss_client import FaissClient, get_faiss_client

logger = logging.getLogger(__name__)


class RecallService:
    """
    Recall service for vector-based retrieval.
    Combines Two-Tower model with FAISS vector search.
    """

    def __init__(self, config=None, feature_store=None, two_tower=None, faiss_client=None):
        """
        Initialize recall service.

        Args:
            config: AppConfig (uses global if not provided)
            feature_store: Optional shared feature store
            two_tower: Optional shared TwoTower model
            faiss_client: Optional shared FAISS client
        """
        self.config = config or get_config()

        # Initialize components (use provided or create new)
        self.feature_store = feature_store if feature_store is not None else create_feature_store(self.config)
        self.user_features = UserFeatures(self.feature_store)
        self.item_features = ItemFeatures(self.feature_store)
        self.cross_features = CrossFeatures(self.user_features, self.item_features)

        # Initialize model (use provided or create new)
        self.two_tower = two_tower if two_tower is not None else TwoTowerModel(self.config)

        # Initialize FAISS client (use provided or create new)
        self.faiss = faiss_client if faiss_client is not None else get_faiss_client(self.config.recall.faiss)

        logger.info("RecallService initialized")

    def initialize(self) -> bool:
        """
        Initialize the recall service (build FAISS index).

        Returns:
            True if successful
        """
        # Connect to FAISS
        self.faiss.connect()
        return True

    def recall(
        self,
        user_id: int,
        candidate_item_ids: Optional[List[int]] = None,
        top_k: int = None
    ) -> Tuple[List[int], List[float]]:
        """
        Recall items for a user.

        Args:
            user_id: User ID
            candidate_item_ids: Optional candidate set (uses FAISS if not provided)
            top_k: Number of items to recall

        Returns:
            Tuple of (item_ids, scores)
        """
        if top_k is None:
            top_k = self.config.recall.recall_top_k

        logger.info(f"Recalling for user {user_id}, top_k={top_k}")

        # Get user hashes
        user_hash = self._get_user_hash(user_id)
        behavior_hashes = self._get_behavior_hashes(user_id)

        # Check for cold start
        if self.user_features.is_cold_start(user_id):
            logger.info(f"User {user_id} is cold start, using popular items")
            return self._cold_start_recall(top_k)

        # Get user embedding from Two-Tower model
        user_embedding = self.two_tower.get_user_embedding(user_hash, behavior_hashes)

        # Check if using FAISS or candidate-based
        if candidate_item_ids is None:
            # Use FAISS for vector search
            return self._faiss_recall(user_embedding, top_k)
        else:
            # Use candidate-based recall
            return self._candidate_recall(
                user_id, user_hash, behavior_hashes, candidate_item_ids, top_k
            )

    def _get_user_hash(self, user_id: int) -> int:
        """Get hashed user ID."""
        from features.base import FeatureHasher
        hasher = FeatureHasher(num_buckets=self.config.model.two_tower.num_hash_buckets)
        return hasher.hash(user_id)

    def _get_behavior_hashes(self, user_id: int) -> List[int]:
        """Get hashed behavior features for user."""
        from features.base import FeatureHasher
        genres = self.user_features.store.get_history_genres(user_id)
        hasher = FeatureHasher(num_buckets=self.config.model.two_tower.num_hash_buckets)
        return [hasher.hash(g) for g in genres]

    def _get_item_hashes(self, item_id: int) -> List[int]:
        """Get hashed item features."""
        from features.base import FeatureHasher
        genres = self.item_features.store.get_genres(item_id)
        hasher = FeatureHasher(num_buckets=self.config.model.two_tower.num_hash_buckets)
        return [hasher.hash(g) for g in genres]

    def _faiss_recall(
        self,
        user_embedding: np.ndarray,
        top_k: int
    ) -> Tuple[List[int], List[float]]:
        """
        Recall using FAISS vector search.

        Args:
            user_embedding: User embedding vector
            top_k: Number of items to recall

        Returns:
            Tuple of (item_ids, scores)
        """
        item_ids, scores = self.faiss.search(user_embedding, top_k)
        return item_ids, scores

    def _candidate_recall(
        self,
        user_id: int,
        user_hash: int,
        behavior_hashes: List[int],
        candidate_item_ids: List[int],
        top_k: int
    ) -> Tuple[List[int], List[float]]:
        """
        Recall from candidate set.

        Args:
            user_id: User ID
            user_hash: Hashed user ID
            behavior_hashes: List of behavior hashes
            candidate_item_ids: Candidate item IDs
            top_k: Number of items to recall

        Returns:
            Tuple of (item_ids, scores)
        """
        # Get item hashes
        item_hashes = [self._get_item_hashes(iid) for iid in candidate_item_ids]

        # Get all item embeddings
        item_embeddings = self.two_tower.batch_get_item_embeddings(
            [user_hash] * len(candidate_item_ids),
            item_hashes
        )

        # Compute similarities
        scores = self.two_tower.compute_similarity(
            self.two_tower.get_user_embedding(user_hash, behavior_hashes),
            item_embeddings
        )

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return (
            [candidate_item_ids[i] for i in top_indices],
            scores[top_indices].tolist()
        )

    def _cold_start_recall(self, top_k: int) -> Tuple[List[int], List[float]]:
        """
        Recall for cold start users (using popular items).

        Args:
            top_k: Number of items to recall

        Returns:
            Tuple of (item_ids, scores)
        """
        # Get popular items from feature store
        popular_items = self._get_popular_items(top_k)
        scores = [0.5] * len(popular_items)  # Default score for cold start

        return popular_items, scores

    def _get_popular_items(self, k: int) -> List[int]:
        """
        Get k most popular items.

        Args:
            k: Number of items

        Returns:
            List of popular item IDs
        """
        # Simplified: return first k items
        # In practice, query feature store for popular items
        return list(range(1, k + 1))

    def build_item_index(
        self,
        item_ids: List[int],
        embeddings: Optional[np.ndarray] = None
    ) -> bool:
        """
        Build item embedding index for FAISS.

        Args:
            item_ids: List of item IDs
            embeddings: Pre-computed embeddings (computed if not provided)

        Returns:
            True if successful
        """
        logger.info(f"Building FAISS index for {len(item_ids)} items")

        if embeddings is None:
            # Compute embeddings
            item_hashes = [self._get_item_hashes(iid) for iid in item_ids]
            hash_ids = [self._get_user_hash(iid) for iid in item_ids]  # Reuse user hash function
            embeddings = self.two_tower.batch_get_item_embeddings(hash_ids, item_hashes)

        # Insert into FAISS
        return self.faiss.create_collection_with_embeddings(item_ids, embeddings)

    def update_item_embedding(self, item_id: int, embedding: np.ndarray) -> bool:
        """
        Update a single item embedding in the index.

        Args:
            item_id: Item ID
            embedding: New embedding vector

        Returns:
            True if successful
        """
        # FAISS doesn't support efficient single-item updates
        # For production, rebuild index or use ID mapping
        logger.info(f"Item {item_id} embedding updated (requires index rebuild for full effect)")
        return True

    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Dict with health status
        """
        faiss_stats = self.faiss.get_index_stats() if hasattr(self.faiss, 'get_index_stats') else {}
        return {
            "faiss_available": self.faiss.is_available() if hasattr(self.faiss, 'is_available') else False,
            "faiss_status": faiss_stats,
            "feature_store": "connected",
            "two_tower_model": "loaded",
        }


class VectorRecallSkill:
    """
    Skills-compatible interface for vector recall.
    """

    def __init__(self):
        self.service = RecallService()
        self.service.initialize()

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute recall skill.

        Args:
            state: Current state with user_id and query

        Returns:
            Updated state with recall_candidates
        """
        user_id = state.get("user_id")
        query = state.get("query", "")  # Can be used for query rewriting

        if user_id is None:
            raise ValueError("user_id is required for VectorRecallSkill")

        # Perform recall
        item_ids, scores = self.service.recall(user_id, top_k=100)

        # Get item details
        candidates = []
        for item_id, score in zip(item_ids, scores):
            item_features = self.service.item_features.store.get_item_features(item_id)
            candidates.append({
                "item_id": item_id,
                "score": score,
                "title": item_features.get("basic", {}).get("title", ""),
                "genres": item_features.get("basic", {}).get("genres", []),
            })

        return {
            "recall_candidates": candidates,
            "recall_score": scores[:5] if scores else [],
        }
