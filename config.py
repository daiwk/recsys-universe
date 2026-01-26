"""
Configuration management for the movie recommendation system.
Supports environment variables and Pydantic-style validation.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ================ LLM Configuration ================

@dataclass
class LLMConfig:
    """LLM configuration settings."""
    api_key: Optional[str] = None  # Use None as sentinel
    base_url: Optional[str] = None
    model: str = "Qwen/Qwen3-1.7B"
    temperature: float = 0.3
    max_tokens: Optional[int] = None

    def __post_init__(self):
        # Override with environment variables only if not explicitly set
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")


# ================ Feature Store Configuration ================

@dataclass
class RedisConfig:
    """Redis configuration for feature storage."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50
    socket_timeout: int = 5
    connection_pool_size: int = 100

    def __post_init__(self):
        self.host = os.environ.get("REDIS_HOST", self.host)
        self.port = int(os.environ.get("REDIS_PORT", str(self.port)))
        self.password = os.environ.get("REDIS_PASSWORD", self.password)


@dataclass
class HBaseConfig:
    """HBase configuration for historical features."""
    host: str = "localhost"
    port: int = 9090
    table_prefix: str = "recsys_"
    timeout: int = 30

    def __post_init__(self):
        self.host = os.environ.get("HBASE_HOST", self.host)
        self.port = int(os.environ.get("HBASE_PORT", str(self.port)))


@dataclass
class FeatureStoreConfig:
    """Feature store configuration."""
    redis: RedisConfig = field(default_factory=RedisConfig)
    hbase: HBaseConfig = field(default_factory=HBaseConfig)
    user_features_ttl: int = 86400  # 24 hours
    item_features_ttl: int = 604800  # 7 days
    embedding_cache_ttl: int = 3600  # 1 hour


# ================ Model Configuration ================

@dataclass
class TwoTowerConfig:
    """Two-tower model configuration."""
    # Embedding settings
    user_id_embedding_dim: int = 64
    item_id_embedding_dim: int = 64
    genre_embedding_dim: int = 32
    user_embedding_dim: int = 32
    item_embedding_dim: int = 32

    # Hash bucket for亿级 ID support
    num_hash_buckets: int = 1000000

    # User tower
    user_tower_layers: List[int] = field(default_factory=list)
    user_dropout: float = 0.1

    # Item tower
    item_tower_layers: List[int] = field(default_factory=list)
    item_dropout: float = 0.1

    # Training
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 10
    margin: float = 0.1  # For triplet loss

    def __post_init__(self):
        # Set tower layers after embedding_dim is known
        self.user_tower_layers = [128, 64, self.user_embedding_dim]
        self.item_tower_layers = [128, 64, self.item_embedding_dim]


@dataclass
class RankingConfig:
    """Ranking model configuration."""
    # Feature dimensions
    user_feature_dim: int = 64
    item_feature_dim: int = 64
    cross_feature_dim: int = 128

    # Model layers
    dnn_layers: List[int] = field(default_factory=lambda: [256, 128, 64, 1])
    activation: str = "relu"
    dropout: float = 0.2

    # Training
    learning_rate: float = 0.001
    batch_size: int = 512
    epochs: int = 5

    # Output
    output_activation: str = "sigmoid"


@dataclass
class ModelConfig:
    """Model configuration."""
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)


# ================ Vector Search Configuration ================

@dataclass
class FaissConfig:
    """FAISS vector search configuration."""
    dim: int = 32  # Embedding dimension
    nlist: int = 0  # Number of clusters (0 = use flat index, >0 = use IVF index)
    metric_type: str = "COSINE"  # COSINE or L2

    def __post_init__(self):
        self.nlist = int(os.environ.get("FAISS_NLIST", str(self.nlist)))


@dataclass
class RecallConfig:
    """Recall service configuration."""
    faiss: FaissConfig = field(default_factory=FaissConfig)
    recall_top_k: int = 100
    approximate_ratio: float = 0.99  # Approximate vs exact search
    batch_size: int = 64


@dataclass
class RankConfig:
    """Ranking service configuration."""
    rank_top_k: int = 10
    batch_size: int = 32


# ================ Online Learning Configuration ================

@dataclass
class OnlineLearningConfig:
    """Online learning configuration."""
    enabled: bool = False
    stream_buffer_size: int = 1000
    update_frequency: int = 100  # Steps between updates
    learning_rate: float = 0.01
    decay: float = 0.95

    # Kafka settings (optional)
    kafka_bootstrap_servers: Optional[str] = None
    kafka_topic: str = "user_behavior"
    consumer_group: str = "recsys_learner"


# ================ Main Application Configuration ================

@dataclass
class AppConfig:
    """Application configuration settings."""

    # Debug mode
    debug: bool = False

    # Legacy recommendation settings (for backward compatibility)
    max_steps: int = 10
    max_planner_steps: int = 8
    content_top_k: int = 15
    collab_top_k: int = 30
    merge_top_k: int = 40
    final_top_k: int = 5

    # Data settings
    movielens_url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    local_zip: str = "ml-1m.zip"

    # LLM settings
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Industrial recommendation settings
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    rank: RankConfig = field(default_factory=RankConfig)
    online_learning: OnlineLearningConfig = field(default_factory=OnlineLearningConfig)

    # Architecture mode: "legacy" (TF-IDF) or "industrial" (Two-Tower + Ranking)
    architecture_mode: str = "legacy"

    def __post_init__(self):
        # Override debug mode from environment
        debug_env = os.environ.get("RECSYS_DEBUG", "")
        if debug_env.lower() in ("true", "1", "yes"):
            self.debug = True

        # Override architecture mode
        mode = os.environ.get("RECSYS_ARCHITECTURE", "")
        if mode in ("legacy", "industrial"):
            self.architecture_mode = mode


def get_config() -> AppConfig:
    """
    Get the application configuration.
    Returns a cached singleton instance.

    Returns:
        AppConfig instance with all settings
    """
    if not hasattr(get_config, "_instance"):
        get_config._instance = AppConfig()
    return get_config._instance


# Global config instance (lazy initialization)
config = None  # Will be initialized on first access
