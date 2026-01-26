"""
Two-Tower model for industrial recommendation system.
Implements user tower and item tower for vector retrieval.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch, fallback to numpy
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not installed, using numpy implementation")


class EmbeddingLayer(nn.Module if HAS_TORCH else object):
    """
    Embedding layer for亿级 ID features.
    Uses hash bucketing to handle large ID spaces efficiently.
    """

    def __init__(
        self,
        num_buckets: int,
        embedding_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize embedding layer.

        Args:
            num_buckets: Number of hash buckets
            embedding_dim: Dimension of each embedding
            num_layers: Number of stacked embeddings
            dropout: Dropout rate
        """
        super().__init__()

        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if HAS_TORCH:
            # Create learnable embeddings
            self.embeddings = nn.ModuleList([
                nn.Embedding(num_buckets, embedding_dim)
                for _ in range(num_layers)
            ])
            self.dropout = nn.Dropout(dropout)
        else:
            # Numpy fallback - use random vectors
            np.random.seed(42)
            self.embeddings = [
                np.random.randn(num_buckets, embedding_dim).astype(np.float32) * 0.01
                for _ in range(num_layers)
            ]

    def forward(self, indices: List[int]) -> np.ndarray:
        """
        Forward pass for numpy implementation.

        Args:
            indices: List of bucket indices

        Returns:
            Sum of embeddings from all layers
        """
        if HAS_TORCH:
            raise NotImplementedError("Use torch tensor input for PyTorch")

        # Sum embeddings from all layers
        result = np.zeros(self.embedding_dim, dtype=np.float32)
        for emb in self.embeddings:
            for idx in indices:
                result += emb[idx]
        return result / (len(indices) + 1e-8)

    def forward_torch(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch implementation.

        Args:
            indices: Tensor of bucket indices [batch_size]

        Returns:
            Embedding tensor [batch_size, embedding_dim]
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        result = self.embeddings[0](indices)
        for emb in self.embeddings[1:]:
            result = result + emb(indices)
        return self.dropout(result) / 2.0

    def get_embedding(self, indices: List[int]) -> np.ndarray:
        """Get embedding for numpy usage."""
        return self.forward(indices)


class DNNLayer(nn.Module if HAS_TORCH else object):
    """
    Deep Neural Network layer for tower networks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Initialize DNN layer.

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_dims = hidden_dims
        self.input_dim = input_dim

        if HAS_TORCH:
            # Build PyTorch layers
            dims = [input_dim] + hidden_dims
            self.layers = nn.ModuleList()
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.activation = activation
            self.dropout = nn.Dropout(dropout)
        else:
            # Numpy fallback
            np.random.seed(42)
            self.weights = []
            self.biases = []
            dims = [input_dim] + hidden_dims
            for i in range(len(dims) - 1):
                w = np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * 0.01
                b = np.zeros(dims[i + 1], dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for numpy implementation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if HAS_TORCH:
            raise NotImplementedError("Use torch tensor input for PyTorch")

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.matmul(x, w) + b
            if i < len(self.weights) - 1:  # No activation on output
                if self.activation == 'relu':
                    x = np.maximum(0, x)
                elif self.activation == 'tanh':
                    x = np.tanh(x)
                elif self.activation == 'sigmoid':
                    x = 1 / (1 + np.exp(-x))
        return x

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PyTorch implementation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    x = F.relu(x)
                elif self.activation == 'tanh':
                    x = torch.tanh(x)
                elif self.activation == 'sigmoid':
                    x = torch.sigmoid(x)
                x = self.dropout(x)
        return x


class UserTower(nn.Module if HAS_TORCH else object):
    """
    User tower of the Two-Tower model.
    Generates user embedding from user features.
    """

    def __init__(self, config):
        """
        Initialize user tower.

        Args:
            config: TwoTowerConfig
        """
        super().__init__()
        two_tower = config.model.two_tower

        self.embedding_dim = two_tower.user_embedding_dim
        self.hasher_num_buckets = two_tower.num_hash_buckets

        if HAS_TORCH:
            # ID embedding
            self.id_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.user_id_embedding_dim,
                num_layers=1,
                dropout=two_tower.user_dropout
            )

            # Behavior embedding
            self.behavior_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.genre_embedding_dim,
                num_layers=1,
                dropout=two_tower.user_dropout
            )

            # DNN layers
            self.dnn = DNNLayer(
                input_dim=two_tower.user_id_embedding_dim + two_tower.genre_embedding_dim,
                hidden_dims=two_tower.user_tower_layers,
                activation='relu',
                dropout=two_tower.user_dropout
            )
        else:
            # Numpy fallback
            self.id_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.user_id_embedding_dim,
                num_layers=1
            )
            self.behavior_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.genre_embedding_dim,
                num_layers=1
            )
            self.dnn = DNNLayer(
                input_dim=two_tower.user_id_embedding_dim + two_tower.genre_embedding_dim,
                hidden_dims=two_tower.user_tower_layers,
                activation='relu',
                dropout=two_tower.user_dropout
            )

    def forward(
        self,
        user_id_hash: int,
        behavior_hashes: List[int]
    ) -> np.ndarray:
        """
        Forward pass for numpy implementation.

        Args:
            user_id_hash: Hashed user ID
            behavior_hashes: List of hashed behavior features

        Returns:
            User embedding vector
        """
        # Get ID embedding
        id_emb = self.id_embedding.get_embedding([user_id_hash])

        # Get behavior embedding
        behavior_emb = self.behavior_embedding.get_embedding(behavior_hashes)

        # Concatenate
        combined = np.concatenate([id_emb, behavior_emb])

        # Pass through DNN
        return self.dnn.forward(combined)

    def forward_torch(
        self,
        user_id_hash: torch.Tensor,
        behavior_hashes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for PyTorch implementation.

        Args:
            user_id_hash: Tensor of user ID hashes [batch_size]
            behavior_hashes: Tensor of behavior hashes [batch_size, max_behaviors]

        Returns:
            User embedding tensor [batch_size, embedding_dim]
        """
        id_emb = self.id_embedding.forward_torch(user_id_hash)
        behavior_emb = self.behavior_embedding.forward_torch(behavior_hashes)

        # Mean pooling for behavior
        behavior_pooled = behavior_emb.mean(dim=1)

        # Concatenate
        combined = torch.cat([id_emb, behavior_pooled], dim=1)

        return self.dnn.forward_torch(combined)

    def get_embedding(self, user_id_hash: int, behavior_hashes: List[int]) -> np.ndarray:
        """Get user embedding (numpy interface)."""
        return self.forward(user_id_hash, behavior_hashes)


class ItemTower(nn.Module if HAS_TORCH else object):
    """
    Item tower of the Two-Tower model.
    Generates item embedding from item features.
    """

    def __init__(self, config):
        """
        Initialize item tower.

        Args:
            config: TwoTowerConfig
        """
        super().__init__()
        two_tower = config.model.two_tower

        self.embedding_dim = two_tower.item_embedding_dim

        if HAS_TORCH:
            # ID embedding
            self.id_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.item_id_embedding_dim,
                num_layers=1,
                dropout=two_tower.item_dropout
            )

            # Genre embedding
            self.genre_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.genre_embedding_dim,
                num_layers=1,
                dropout=two_tower.item_dropout
            )

            # DNN layers
            self.dnn = DNNLayer(
                input_dim=two_tower.item_id_embedding_dim + two_tower.genre_embedding_dim,
                hidden_dims=two_tower.item_tower_layers,
                activation='relu',
                dropout=two_tower.item_dropout
            )
        else:
            # Numpy fallback
            self.id_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.item_id_embedding_dim,
                num_layers=1
            )
            self.genre_embedding = EmbeddingLayer(
                num_buckets=two_tower.num_hash_buckets,
                embedding_dim=two_tower.genre_embedding_dim,
                num_layers=1
            )
            self.dnn = DNNLayer(
                input_dim=two_tower.item_id_embedding_dim + two_tower.genre_embedding_dim,
                hidden_dims=two_tower.item_tower_layers,
                activation='relu',
                dropout=two_tower.item_dropout
            )

    def forward(
        self,
        item_id_hash: int,
        genre_hashes: List[int]
    ) -> np.ndarray:
        """
        Forward pass for numpy implementation.

        Args:
            item_id_hash: Hashed item ID
            genre_hashes: List of hashed genre features

        Returns:
            Item embedding vector
        """
        # Get ID embedding
        id_emb = self.id_embedding.get_embedding([item_id_hash])

        # Get genre embedding
        genre_emb = self.genre_embedding.get_embedding(genre_hashes)

        # Concatenate
        combined = np.concatenate([id_emb, genre_emb])

        # Pass through DNN
        return self.dnn.forward(combined)

    def forward_torch(
        self,
        item_id_hash: torch.Tensor,
        genre_hashes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for PyTorch implementation.

        Args:
            item_id_hash: Tensor of item ID hashes [batch_size]
            genre_hashes: Tensor of genre hashes [batch_size, max_genres]

        Returns:
            Item embedding tensor [batch_size, embedding_dim]
        """
        id_emb = self.id_embedding.forward_torch(item_id_hash)
        genre_emb = self.genre_embedding.forward_torch(genre_hashes)

        # Mean pooling for genres
        genre_pooled = genre_emb.mean(dim=1)

        # Concatenate
        combined = torch.cat([id_emb, genre_pooled], dim=1)

        return self.dnn.forward_torch(combined)

    def get_embedding(self, item_id_hash: int, genre_hashes: List[int]) -> np.ndarray:
        """Get item embedding (numpy interface)."""
        return self.forward(item_id_hash, genre_hashes)


class TwoTowerModel:
    """
    Complete Two-Tower model for retrieval.
    """

    def __init__(self, config):
        """
        Initialize Two-Tower model.

        Args:
            config: AppConfig
        """
        self.config = config
        self.user_tower = UserTower(config)
        self.item_tower = ItemTower(config)

        logger.info(
            f"TwoTowerModel initialized with "
            f"user_dim={config.model.two_tower.user_embedding_dim}, "
            f"item_dim={config.model.two_tower.item_embedding_dim}"
        )

    def get_user_embedding(
        self,
        user_id_hash: int,
        behavior_hashes: List[int]
    ) -> np.ndarray:
        """
        Get user embedding for recall.

        Args:
            user_id_hash: Hashed user ID
            behavior_hashes: List of hashed behavior features

        Returns:
            User embedding vector
        """
        return self.user_tower.get_embedding(user_id_hash, behavior_hashes)

    def get_item_embedding(
        self,
        item_id_hash: int,
        genre_hashes: List[int]
    ) -> np.ndarray:
        """
        Get item embedding for indexing.

        Args:
            item_id_hash: Hashed item ID
            genre_hashes: List of hashed genre features

        Returns:
            Item embedding vector
        """
        return self.item_tower.get_embedding(item_id_hash, genre_hashes)

    def compute_similarity(
        self,
        user_embedding: np.ndarray,
        item_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between user and items.

        Args:
            user_embedding: User embedding [dim]
            item_embeddings: Item embeddings [num_items, dim]

        Returns:
            Similarity scores [num_items]
        """
        # Cosine similarity
        user_norm = user_embedding / (np.linalg.norm(user_embedding) + 1e-8)
        item_norms = item_embeddings / (
            np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8
        )
        return np.dot(item_norms, user_norm)

    def batch_get_user_embeddings(
        self,
        user_id_hashes: List[int],
        behavior_hashes_list: List[List[int]]
    ) -> np.ndarray:
        """
        Batch get user embeddings.

        Args:
            user_id_hashes: List of hashed user IDs
            behavior_hashes_list: List of behavior hash lists

        Returns:
            User embeddings [batch_size, dim]
        """
        return np.array([
            self.get_user_embedding(uid, bh)
            for uid, bh in zip(user_id_hashes, behavior_hashes_list)
        ])

    def batch_get_item_embeddings(
        self,
        item_id_hashes: List[int],
        genre_hashes_list: List[List[int]]
    ) -> np.ndarray:
        """
        Batch get item embeddings.

        Args:
            item_id_hashes: List of hashed item IDs
            genre_hashes_list: List of genre hash lists

        Returns:
            Item embeddings [batch_size, dim]
        """
        return np.array([
            self.get_item_embedding(iid, gh)
            for iid, gh in zip(item_id_hashes, genre_hashes_list)
        ])

    def recall(
        self,
        user_id_hash: int,
        behavior_hashes: List[int],
        item_id_hashes: List[int],
        genre_hashes_list: List[List[int]],
        top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recall items for a user.

        Args:
            user_id_hash: Hashed user ID
            behavior_hashes: List of hashed behavior features
            item_id_hashes: List of candidate item hashes
            genre_hashes_list: List of genre hash lists for items
            top_k: Number of items to recall

        Returns:
            Tuple of (top_k_item_indices, top_k_scores)
        """
        # Get user embedding
        user_emb = self.get_user_embedding(user_id_hash, behavior_hashes)

        # Get all item embeddings
        item_embs = self.batch_get_item_embeddings(item_id_hashes, genre_hashes_list)

        # Compute similarities
        scores = self.compute_similarity(user_emb, item_embs)

        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return top_indices, scores[top_indices]

    def save(self, path: str) -> None:
        """Save model weights (placeholder for PyTorch)."""
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights (placeholder for PyTorch)."""
        logger.info(f"Model loaded from {path}")
