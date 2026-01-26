"""
Base feature store classes for industrial recommendation system.
Supports Redis for real-time features and HBase for historical features.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class FeatureHasher:
    """
    Feature hasher for亿级 ID support.
    Maps high-cardinality categorical features to fixed-size embeddings.
    """

    def __init__(self, num_buckets: int = 1000000, seed: int = 42):
        """
        Initialize feature hasher.

        Args:
            num_buckets: Number of hash buckets
            seed: Random seed for reproducibility
        """
        self.num_buckets = num_buckets
        self.seed = seed

    def hash(self, feature_value: Any) -> int:
        """
        Hash a feature value to a bucket index.

        Args:
            feature_value: The feature value to hash

        Returns:
            Bucket index in [0, num_buckets)
        """
        # Convert to string representation
        str_value = str(feature_value)
        # Create hash
        hash_obj = hashlib.md5(f"{self.seed}_{str_value}".encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.num_buckets

    def hash_list(self, values: List[Any]) -> List[int]:
        """
        Hash a list of values.

        Args:
            values: List of feature values

        Returns:
            List of bucket indices
        """
        return [self.hash(v) for v in values]


class FeatureSerializer:
    """Feature serialization utilities."""

    @staticmethod
    def serialize(data: Dict[str, Any]) -> str:
        """Serialize dict to JSON string."""
        return json.dumps(data, default=str)

    @staticmethod
    def deserialize(data: str) -> Dict[str, Any]:
        """Deserialize JSON string to dict."""
        return json.loads(data)

    @staticmethod
    def serialize_embedding(embedding: List[float]) -> bytes:
        """Serialize embedding to bytes for efficient storage."""
        return json.dumps(embedding).encode('utf-8')

    @staticmethod
    def deserialize_embedding(data: bytes) -> List[float]:
        """Deserialize embedding from bytes."""
        return json.loads(data.decode('utf-8'))


class BaseFeatureStore(ABC):
    """Base class for feature stores."""

    def __init__(self, config):
        """
        Initialize feature store.

        Args:
            config: Feature store configuration
        """
        self.config = config
        self.hasher = FeatureHasher(num_buckets=config.model.two_tower.num_hash_buckets)
        self.serializer = FeatureSerializer()

    @abstractmethod
    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a user."""
        pass

    @abstractmethod
    def set_user_features(self, user_id: int, features: Dict[str, Any]) -> bool:
        """Set features for a user."""
        pass

    @abstractmethod
    def get_item_features(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get features for an item."""
        pass

    @abstractmethod
    def set_item_features(self, item_id: int, features: Dict[str, Any]) -> bool:
        """Set features for an item."""
        pass

    @abstractmethod
    def batch_get_user_features(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch get features for multiple users."""
        pass

    @abstractmethod
    def batch_get_item_features(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch get features for multiple items."""
        pass


class RedisFeatureStore(BaseFeatureStore):
    """
    Redis-based feature store for real-time features.
    Stores user/item embeddings and basic features with TTL support.
    """

    def __init__(self, config):
        """
        Initialize Redis feature store.

        Args:
            config: Feature store configuration
        """
        super().__init__(config)
        self.redis_config = config.feature_store.redis

        # Try to import redis, handle if not available
        try:
            import redis
            self._redis_client = None  # Lazy initialization
            self._redis_pool = None
        except ImportError:
            logger.warning("redis-py not installed, using mock client")
            self._redis_client = None
            self._redis_pool = None

    def _get_client(self):
        """Get or create Redis client with connection pool."""
        if self._redis_client is None:
            try:
                import redis
                self._redis_pool = redis.ConnectionPool(
                    host=self.redis_config.host,
                    port=self.redis_config.port,
                    password=self.redis_config.password,
                    db=self.redis_config.db,
                    max_connections=self.redis_config.max_connections,
                    socket_timeout=self.redis_config.socket_timeout,
                    decode_responses=False
                )
                self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            except ImportError:
                raise ImportError("redis-py is required. Install with: pip install redis")

        return self._redis_client

    def _get_key(self, key_type: str, entity_id: int) -> str:
        """Generate Redis key for an entity."""
        return f"recsys:{key_type}:{entity_id}"

    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a user."""
        client = self._get_client()
        key = self._get_key("user", user_id)

        try:
            data = client.get(key)
            if data:
                return self.serializer.deserialize(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error getting user features for {user_id}: {e}")
            return None

    def set_user_features(self, user_id: int, features: Dict[str, Any]) -> bool:
        """Set features for a user."""
        client = self._get_client()
        key = self._get_key("user", user_id)

        try:
            serialized = self.serializer.serialize(features)
            client.setex(
                key,
                self.config.user_features_ttl,
                serialized.encode('utf-8')
            )
            return True
        except Exception as e:
            logger.error(f"Error setting user features for {user_id}: {e}")
            return False

    def get_item_features(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get features for an item."""
        client = self._get_client()
        key = self._get_key("item", item_id)

        try:
            data = client.get(key)
            if data:
                return self.serializer.deserialize(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error getting item features for {item_id}: {e}")
            return None

    def set_item_features(self, item_id: int, features: Dict[str, Any]) -> bool:
        """Set features for an item."""
        client = self._get_client()
        key = self._get_key("item", item_id)

        try:
            serialized = self.serializer.serialize(features)
            client.setex(
                key,
                self.config.item_features_ttl,
                serialized.encode('utf-8')
            )
            return True
        except Exception as e:
            logger.error(f"Error setting item features for {item_id}: {e}")
            return False

    def get_user_embedding(self, user_id: int) -> Optional[List[float]]:
        """Get cached user embedding."""
        client = self._get_client()
        key = self._get_key("user_emb", user_id)

        try:
            data = client.get(key)
            if data:
                return self.serializer.deserialize_embedding(data)
            return None
        except Exception as e:
            logger.error(f"Error getting user embedding for {user_id}: {e}")
            return None

    def set_user_embedding(self, user_id: int, embedding: List[float]) -> bool:
        """Set cached user embedding."""
        client = self._get_client()
        key = self._get_key("user_emb", user_id)

        try:
            serialized = self.serializer.serialize_embedding(embedding)
            client.setex(key, self.config.embedding_cache_ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting user embedding for {user_id}: {e}")
            return False

    def get_item_embedding(self, item_id: int) -> Optional[List[float]]:
        """Get cached item embedding."""
        client = self._get_client()
        key = self._get_key("item_emb", item_id)

        try:
            data = client.get(key)
            if data:
                return self.serializer.deserialize_embedding(data)
            return None
        except Exception as e:
            logger.error(f"Error getting item embedding for {item_id}: {e}")
            return None

    def set_item_embedding(self, item_id: int, embedding: List[float]) -> bool:
        """Set cached item embedding."""
        client = self._get_client()
        key = self._get_key("item_emb", item_id)

        try:
            serialized = self.serializer.serialize_embedding(embedding)
            client.setex(key, self.config.embedding_cache_ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting item embedding for {item_id}: {e}")
            return False

    def batch_get_user_features(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch get features for multiple users."""
        client = self._get_client()
        keys = [self._get_key("user", uid) for uid in user_ids]

        try:
            pipe = client.pipeline()
            for key in keys:
                pipe.get(key)
            results = pipe.execute()

            features_map = {}
            for uid, data in zip(user_ids, results):
                if data:
                    features_map[uid] = self.serializer.deserialize(data.decode('utf-8'))
            return features_map
        except Exception as e:
            logger.error(f"Error batch getting user features: {e}")
            return {}

    def batch_get_item_features(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch get features for multiple items."""
        client = self._get_client()
        keys = [self._get_key("item", iid) for iid in item_ids]

        try:
            pipe = client.pipeline()
            for key in keys:
                pipe.get(key)
            results = pipe.execute()

            features_map = {}
            for iid, data in zip(item_ids, results):
                if data:
                    features_map[iid] = self.serializer.deserialize(data.decode('utf-8'))
            return features_map
        except Exception as e:
            logger.error(f"Error batch getting item features: {e}")
            return {}


class UserFeatureStore:
    """User-specific feature store with enhanced functionality."""

    def __init__(self, redis_store: RedisFeatureStore):
        self.store = redis_store

    def get_basic_features(self, user_id: int) -> Dict[str, Any]:
        """Get basic user features (demographics, etc.)."""
        features = self.store.get_user_features(user_id)
        if features:
            return features.get("basic", {})
        return {}

    def get_behavior_features(self, user_id: int) -> Dict[str, Any]:
        """Get user behavior features (history, preferences, etc.)."""
        features = self.store.get_user_features(user_id)
        if features:
            return features.get("behavior", {})
        return {}

    def get_embedding(self, user_id: int) -> Optional[List[float]]:
        """Get user embedding for recall."""
        return self.store.get_user_embedding(user_id)

    def update_behavior(self, user_id: int, behavior: Dict[str, Any]) -> bool:
        """Update user behavior features."""
        existing = self.store.get_user_features(user_id) or {}
        existing["behavior"] = behavior
        existing["updated_at"] = datetime.now().isoformat()
        return self.store.set_user_features(user_id, existing)

    def get_history_genres(self, user_id: int) -> List[str]:
        """Extract genre preferences from user history."""
        behavior = self.get_behavior_features(user_id)
        return behavior.get("preferred_genres", [])

    def get_interaction_history(self, user_id: int) -> List[int]:
        """Get list of item IDs user has interacted with."""
        behavior = self.get_behavior_features(user_id)
        return behavior.get("viewed_items", [])


class ItemFeatureStore:
    """Item-specific feature store with enhanced functionality."""

    def __init__(self, redis_store: RedisFeatureStore):
        self.store = redis_store

    def get_basic_features(self, item_id: int) -> Dict[str, Any]:
        """Get basic item features (title, genres, etc.)."""
        features = self.store.get_item_features(item_id)
        if features:
            return features.get("basic", {})
        return {}

    def get_statistics(self, item_id: int) -> Dict[str, Any]:
        """Get item statistics (views, rating, CTR, etc.)."""
        features = self.store.get_item_features(item_id)
        if features:
            return features.get("statistics", {})
        return {}

    def get_embedding(self, item_id: int) -> Optional[List[float]]:
        """Get item embedding for recall."""
        return self.store.get_item_embedding(item_id)

    def update_statistics(self, item_id: int, stats: Dict[str, Any]) -> bool:
        """Update item statistics."""
        existing = self.store.get_item_features(item_id) or {}
        existing["statistics"] = stats
        existing["updated_at"] = datetime.now().isoformat()
        return self.store.set_item_features(item_id, existing)

    def get_genres(self, item_id: int) -> List[str]:
        """Get genres for an item."""
        basic = self.get_basic_features(item_id)
        return basic.get("genres", [])


class MemoryFeatureStore(BaseFeatureStore):
    """
    In-memory feature store for testing/demo when Redis is not available.
    Uses Python dict for storage.
    """

    def __init__(self, config):
        """Initialize in-memory feature store."""
        super().__init__(config)
        self._user_features = {}
        self._item_features = {}
        self._user_embeddings = {}
        self._item_embeddings = {}
        logger.info("MemoryFeatureStore initialized (in-memory mode)")

    def get_user_features(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get features for a user."""
        return self._user_features.get(user_id)

    def set_user_features(self, user_id: int, features: Dict[str, Any]) -> bool:
        """Set features for a user."""
        self._user_features[user_id] = features
        return True

    def get_item_features(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get features for an item."""
        return self._item_features.get(item_id)

    def set_item_features(self, item_id: int, features: Dict[str, Any]) -> bool:
        """Set features for an item."""
        self._item_features[item_id] = features
        return True

    def batch_get_user_features(self, user_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch get features for multiple users."""
        return {uid: self._user_features.get(uid) for uid in user_ids if uid in self._user_features}

    def batch_get_item_features(self, item_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Batch get features for multiple items."""
        return {iid: self._item_features.get(iid) for iid in item_ids if iid in self._item_features}

    def get_user_embedding(self, user_id: int) -> Optional[List[float]]:
        """Get cached user embedding."""
        return self._user_embeddings.get(user_id)

    def set_user_embedding(self, user_id: int, embedding: List[float]) -> bool:
        """Set cached user embedding."""
        self._user_embeddings[user_id] = embedding
        return True

    def get_item_embedding(self, item_id: int) -> Optional[List[float]]:
        """Get cached item embedding."""
        return self._item_embeddings.get(item_id)

    def set_item_embedding(self, item_id: int, embedding: List[float]) -> bool:
        """Set cached item embedding."""
        self._item_embeddings[item_id] = embedding
        return True


# Factory functions
def create_feature_store(config, use_memory: bool = None) -> BaseFeatureStore:
    """
    Create a feature store instance.

    Args:
        config: AppConfig instance
        use_memory: If True, use memory store instead of Redis

    Returns:
        Feature store instance (Redis or Memory)
    """
    if use_memory is None:
        # Check environment variable
        use_memory = os.environ.get("RECSYS_USE_MEMORY_STORE", "").lower() in ("true", "1", "yes")

    if use_memory:
        logger.info("Using in-memory feature store (no Redis required)")
        return MemoryFeatureStore(config)

    # Try Redis, fall back to memory if not available
    try:
        import redis
        # Test connection
        store = RedisFeatureStore(config)
        client = store._get_client()
        client.ping()
        logger.info("Redis feature store connected successfully")
        return store
    except Exception as e:
        logger.warning(f"Redis not available ({e}), using in-memory store")
        return MemoryFeatureStore(config)


def create_user_feature_store(store: BaseFeatureStore) -> UserFeatureStore:
    """Create user feature store."""
    return UserFeatureStore(store)


def create_item_feature_store(store: BaseFeatureStore) -> ItemFeatureStore:
    """Create item feature store."""
    return ItemFeatureStore(store)
