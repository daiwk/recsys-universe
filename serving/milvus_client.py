"""
Milvus client for vector search in industrial recommendation system.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus client wrapper for vector search operations.
    Handles collection management, indexing, and search.
    """

    def __init__(self, config):
        """
        Initialize Milvus client.

        Args:
            config: MilvusConfig
        """
        self.config = config
        self.client = None
        self._connected = False

        # Try to import pymilvus
        try:
            from pymilvus import (
                connections, Collection, CollectionSchema,
                FieldSchema, DataType, utility, IndexParams
            )
            self.pymilvus_available = True
            self.Collection = Collection
            self.CollectionSchema = CollectionSchema
            self.FieldSchema = FieldSchema
            self.DataType = DataType
            self.utility = utility
            self.IndexParams = IndexParams
        except ImportError:
            self.pymilvus_available = False
            logger.warning("pymilvus not installed, using mock implementation")

    def connect(self) -> bool:
        """
        Connect to Milvus server.

        Returns:
            True if connected successfully
        """
        if self._connected:
            return True

        if not self.pymilvus_available:
            logger.warning("pymilvus not available, using mock")
            self._connected = True
            return True

        try:
            from pymilvus import connections
            connections.connect(
                host=self.config.host,
                port=str(self.config.port)
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Milvus server."""
        if self._connected and self.pymilvus_available:
            try:
                from pymilvus import connections
                connections.disconnect("default")
            except:
                pass
        self._connected = False

    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None
    ) -> bool:
        """
        Create collection for item embeddings.

        Args:
            collection_name: Collection name (uses config if not provided)
            dim: Embedding dimension (uses config if not provided)

        Returns:
            True if successful
        """
        if not self.pymilvus_available:
            logger.info(f"[Mock] Creating collection: {collection_name}")
            return True

        name = collection_name or self.config.collection_name
        dimension = dim or self.config.dim

        # Define schema
        fields = [
            FieldSchema(name="item_id", dtype=self.DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=self.DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="genre_hashes", dtype=self.DataType.INT64, max_length=100),
        ]
        schema = self.CollectionSchema(fields=fields, description="Item embeddings for recall")

        # Create collection
        try:
            if self.utility.has_collection(name):
                logger.info(f"Collection {name} already exists")
                return True

            self.Collection(name, schema)
            logger.info(f"Created collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def create_index(
        self,
        collection_name: Optional[str] = None,
        index_type: Optional[str] = None,
        metric_type: Optional[str] = None,
        nlist: Optional[int] = None
    ) -> bool:
        """
        Create index for the collection.

        Args:
            collection_name: Collection name
            index_type: Index type (IVF_PQ, HNSW, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
            nlist: Number of clusters

        Returns:
            True if successful
        """
        if not self.pymilvus_available:
            logger.info(f"[Mock] Creating index for collection")
            return True

        name = collection_name or self.config.collection_name
        idx_type = index_type or self.config.index_type
        mtype = metric_type or self.config.metric_type
        nlist_val = nlist or self.config.nlist

        try:
            collection = self.Collection(name)
            index_params = self.IndexParams()
            index_params.add_metric_type(mtype)
            index_params.add_index_type(idx_type)
            index_params.add_params("nlist", str(nlist_val))

            # Index type specific params
            if idx_type == "IVF_PQ":
                index_params.add_params("nprobe", str(self.config.nprobe))
            elif idx_type == "HNSW":
                index_params.add_params("M", "16")
                index_params.add_params("efConstruction", "200")

            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info(f"Created index for collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def load_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Load collection into memory for search.

        Args:
            collection_name: Collection name

        Returns:
            True if successful
        """
        if not self.pymilvus_available:
            return True

        name = collection_name or self.config.collection_name

        try:
            collection = self.Collection(name)
            collection.load()
            logger.info(f"Loaded collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            return False

    def insert_item_embeddings(
        self,
        item_ids: List[int],
        embeddings: np.ndarray,
        genre_hashes: Optional[List[List[int]]] = None,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Insert item embeddings into Milvus.

        Args:
            item_ids: List of item IDs
            embeddings: Embedding matrix [num_items, dim]
            genre_hashes: List of genre hash lists
            collection_name: Collection name

        Returns:
            True if successful
        """
        if not self.pymilvus_available:
            logger.info(f"[Mock] Inserted {len(item_ids)} item embeddings")
            return True

        name = collection_name or self.config.collection_name

        try:
            collection = self.Collection(name)

            # Prepare data
            data = [
                item_ids,  # primary keys
                embeddings.tolist(),  # embeddings
                genre_hashes or [[] for _ in item_ids]  # genre hashes
            ]

            collection.insert(data)
            logger.info(f"Inserted {len(item_ids)} item embeddings")
            return True
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            return False

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
        collection_name: Optional[str] = None,
        nprobe: Optional[int] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Search for similar items.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            collection_name: Collection name
            nprobe: Number of probes (overrides config)

        Returns:
            Tuple of (item_ids, distances)
        """
        if not self.pymilvus_available:
            # Mock implementation
            mock_ids = list(range(min(top_k, 10)))
            mock_scores = [1.0 - i * 0.1 for i in range(len(mock_ids))]
            return mock_ids, mock_scores

        name = collection_name or self.config.collection_name
        nprobe_val = nprobe or self.config.nprobe

        try:
            collection = self.Collection(name)

            # Search parameters
            search_params = {
                "nprobe": nprobe_val,
                "metric_type": self.config.metric_type
            }

            # Execute search
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=None
            )

            # Parse results
            if results and len(results) > 0:
                hits = results[0]
                item_ids = [hit.id for hit in hits]
                distances = [hit.score for hit in hits]
                return item_ids, distances

            return [], []

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], []

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
        collection_name: Optional[str] = None
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Batch search for multiple queries.

        Args:
            query_embeddings: Query embedding matrix [batch_size, dim]
            top_k: Number of results per query
            collection_name: Collection name

        Returns:
            List of search results for each query
        """
        if not self.pymilvus_available:
            # Mock implementation
            return [
                (list(range(min(top_k, 10)), [1.0 - i * 0.1 for i in range(min(top_k, 10))])
                for _ in range(len(query_embeddings))
            ]

        results = []
        for emb in query_embeddings:
            ids, dists = self.search(emb, top_k, collection_name)
            results.append((ids, dists))

        return results

    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Collection name

        Returns:
            True if successful
        """
        if not self.pymilvus_available:
            return True

        name = collection_name or self.config.collection_name

        try:
            if self.utility.has_collection(name):
                self.Collection(name).drop()
                logger.info(f"Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get collection statistics.

        Args:
            collection_name: Collection name

        Returns:
            Dict with collection stats
        """
        if not self.pymilvus_available:
            return {"status": "mock", "count": 0}

        name = collection_name or self.config.collection_name

        try:
            if self.utility.has_collection(name):
                collection = self.Collection(name)
                return {
                    "status": "loaded",
                    "count": collection.num_entities,
                    "schema": collection.schema
                }
            return {"status": "not_found"}
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}

    def is_healthy(self) -> bool:
        """
        Check if Milvus is healthy.

        Returns:
            True if healthy
        """
        if not self.pymilvus_available:
            return True

        try:
            from pymilvus import connections
            connections.connect(host=self.config.host, port=str(self.config.port))
            return True
        except:
            return False
