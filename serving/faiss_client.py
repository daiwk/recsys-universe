"""
FAISS-based vector search client for industrial recommendation system.

FAISS (Facebook AI Similarity Search) is a library for efficient
similarity search and clustering of dense vectors.

Installation:
    pip install faiss-cpu  # CPU-only version
    # or
    pip install faiss-gpu  # GPU support (requires CUDA)

Note: FAISS requires the index to be rebuilt if data changes.
For dynamic updates, consider using FAISS with ID mapping.
"""
import logging
import os
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class FaissClient:
    """
    FAISS-based vector search client.

    Supports multiple index types:
    - IndexFlatIP: Inner product (use for normalized vectors)
    - IndexFlatL2: L2 distance
    - IndexIVFFlat: Inverted file with flat storage (faster search)
    - IndexIVFPQ: IVF with Product Quantization (memory efficient for large datasets)
    """

    def __init__(self, config=None):
        """
        Initialize FAISS client.

        Args:
            config: FaissConfig with index parameters
        """
        self.config = config
        self.index = None
        self.id_map = {}  # Map FAISS internal indices to item IDs
        self.reverse_map = {}  # Map item IDs to FAISS internal indices
        self._is_built = False

        # Try to import faiss
        try:
            import faiss
            self.faiss = faiss
            self.faiss_available = True
        except ImportError:
            logger.warning("faiss not installed. Install with: pip install faiss-cpu")
            self.faiss = None
            self.faiss_available = False

    def connect(self) -> bool:
        """Connect/initialize the FAISS index."""
        if not self.faiss_available:
            logger.warning("FAISS not available, using mock mode")
            return False

        try:
            if self._is_built:
                logger.info("FAISS index already built")
                return True

            # Create index based on config
            dim = self.config.dim if self.config else 32
            metric = getattr(self.config, 'metric_type', 'COSINE').upper()

            if metric == 'COSINE':
                # Use inner product for cosine similarity with normalized vectors
                metric_type = self.faiss.METRIC_INNER_PRODUCT
            else:
                metric_type = self.faiss.METRIC_L2

            nlist = getattr(self.config, 'nlist', 100) if self.config else 100

            # Use IVF index for better performance with large datasets
            # For small datasets (<10000), use flat index
            if nlist > 0 and nlist < 100:
                # Flat index (exact search, good for small datasets)
                self.index = self.faiss.IndexFlatIP(dim) if metric == 'COSINE' else self.faiss.IndexFlatL2(dim)
                logger.info(f"Using IndexFlat{'(IP)' if metric == 'COSINE' else 'L2'} (exact search)")
            else:
                # IVF index (approximate search, better for large datasets)
                quantizer = self.faiss.IndexFlatIP(dim) if metric == 'COSINE' else self.faiss.IndexFlatL2(dim)
                self.index = self.faiss.IndexIVFFlat(quantizer, dim, min(nlist, 100), metric_type)
                logger.info(f"Using IndexIVFFlat (approximate search, nlist={min(nlist, 100)})")

            self._is_built = True
            logger.info("FAISS index initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            return False

    def is_available(self) -> bool:
        """Check if FAISS is available."""
        return self.faiss_available and self.faiss is not None

    def create_collection_with_embeddings(
        self,
        item_ids: List[int],
        embeddings: List[List[float]] or np.ndarray
    ) -> bool:
        """
        Create collection and add item embeddings.

        Args:
            item_ids: List of item IDs
            embeddings: List or np.ndarray of item embeddings

        Returns:
            True if successful
        """
        if not self.is_available():
            logger.warning("FAISS not available, using mock implementation")
            return False

        try:
            self.connect()

            # Convert to numpy array
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings, dtype=np.float32)

            # Normalize for cosine similarity (if using IP)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm[norm == 0] = 1  # Avoid division by zero
            embeddings_normalized = embeddings / norm

            # Train the index (for IVF indexes)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                # Use a subset for training
                train_size = min(10000, len(embeddings_normalized))
                self.index.train(embeddings_normalized[:train_size])
                logger.info("FAISS index trained")

            # Add embeddings to index
            self.index.add(embeddings_normalized)

            # Build ID mappings
            self.id_map = {i: item_ids[i] for i in range(len(item_ids))}
            self.reverse_map = {item_ids[i]: i for i in range(len(item_ids))}

            logger.info(f"Added {len(item_ids)} embeddings to FAISS index")
            return True

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return False

    def search(
        self,
        query_embedding: List[float] or np.ndarray,
        top_k: int = 10,
        collection_name: str = None
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar items.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            collection_name: Ignored (FAISS uses single index)

        Returns:
            Tuple of (item_ids, distances/scores)
        """
        if not self.is_available():
            # Mock implementation
            return self._mock_search(top_k)

        try:
            # Convert to numpy array
            if isinstance(query_embedding, list):
                query = np.array([query_embedding], dtype=np.float32)
            else:
                query = query_embedding.reshape(1, -1)

            # Normalize for cosine similarity
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # Search
            distances, indices = self.index.search(query, top_k)

            # Map FAISS indices to item IDs
            item_ids = []
            scores = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.id_map):
                    item_ids.append(self.id_map[idx])
                    # Convert distance to similarity score
                    if hasattr(self.index, 'metric_type'):
                        if self.index.metric_type == self.faiss.METRIC_INNER_PRODUCT:
                            scores.append(float(dist))  # Already similarity score
                        else:
                            scores.append(float(1.0 / (1.0 + dist)))  # Convert distance to score
                    else:
                        scores.append(float(dist))
                else:
                    # Invalid index, skip
                    pass

            return item_ids, scores

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._mock_search(top_k)

    def batch_search(
        self,
        query_embeddings: List[List[float]] or np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Search for multiple queries.

        Args:
            query_embeddings: List of query embeddings
            top_k: Number of results per query

        Returns:
            List of (item_ids, scores) tuples for each query
        """
        if not self.is_available():
            return [self._mock_search(top_k) for _ in query_embeddings]

        try:
            if isinstance(query_embeddings, list):
                queries = np.array(query_embeddings, dtype=np.float32)
            else:
                queries = query_embeddings

            # Normalize
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            norms[norms == 0] = 1
            queries_normalized = queries / norms

            # Batch search
            distances, indices = self.index.search(queries_normalized, top_k)

            results = []
            for i in range(len(queries)):
                item_ids = []
                scores = []
                for dist, idx in zip(distances[i], indices[i]):
                    if idx >= 0 and idx < len(self.id_map):
                        item_ids.append(self.id_map[idx])
                        scores.append(float(dist))
                results.append((item_ids, scores))

            return results

        except Exception as e:
            logger.error(f"FAISS batch search failed: {e}")
            return [self._mock_search(top_k) for _ in query_embeddings]

    def _mock_search(self, top_k: int) -> Tuple[List[int], List[float]]:
        """Mock search implementation for testing."""
        import random
        item_ids = list(range(1, min(top_k, 10) + 1))
        scores = [1.0 - i * 0.1 for i in range(len(item_ids))]
        return item_ids, scores

    def delete_collection(self, collection_name: str = None) -> bool:
        """Delete the current index."""
        try:
            self.index.reset()
            self.id_map = {}
            self.reverse_map = {}
            self._is_built = False
            logger.info("FAISS index deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete FAISS index: {e}")
            return False

    def get_index_stats(self) -> dict:
        """Get index statistics."""
        if not self.is_available():
            return {"status": "not_available"}

        return {
            "is_built": self._is_built,
            "total_items": self.index.ntotal if self.index else 0,
            "dimension": self.index.d if self.index else 0,
            "index_type": type(self.index).__name__ if self.index else None,
        }


def create_faiss_index(
    dim: int = 32,
    nlist: int = 100,
    metric: str = "COSINE"
):
    """
    Factory function to create a FAISS index.

    Args:
        dim: Vector dimension
        nlist: Number of clusters (0 for flat index)
        metric: 'COSINE' or 'L2'

    Returns:
        Configured FaissClient instance
    """
    class _Config:
        dim = dim
        nlist = nlist
        metric_type = metric

    return FaissClient(_Config())


# Mock FAISS implementation for when faiss is not installed
class MockFaissClient:
    """Mock FAISS client for testing without faiss installed."""

    def __init__(self, config=None):
        self._items = {}
        self._embeddings = {}

    def connect(self) -> bool:
        return True

    def is_available(self) -> bool:
        return False

    def create_collection_with_embeddings(
        self,
        item_ids: List[int],
        embeddings: List[List[float]]
    ) -> bool:
        for i, item_id in enumerate(item_ids):
            self._items[item_id] = embeddings[i] if i < len(embeddings) else None
        logger.info(f"Mock FAISS: stored {len(item_ids)} items")
        return True

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10
    ) -> Tuple[List[int], List[float]]:
        # Simple mock: return top_k items with decreasing scores
        item_ids = list(self._items.keys())[:min(top_k, len(self._items))]
        scores = [1.0 - i * 0.1 for i in range(len(item_ids))]
        return item_ids, scores

    def batch_search(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 10
    ) -> List[Tuple[List[int], List[float]]]:
        return [self.search(q, top_k) for q in query_embeddings]

    def delete_collection(self) -> bool:
        self._items = {}
        return True


def get_faiss_client(config=None) -> FaissClient:
    """
    Get FAISS client, falling back to mock if not available.

    Args:
        config: FaissConfig

    Returns:
        FaissClient or MockFaissClient
    """
    try:
        import faiss
        return FaissClient(config)
    except ImportError:
        logger.warning("FAISS not installed, using mock client")
        return MockFaissClient(config)
