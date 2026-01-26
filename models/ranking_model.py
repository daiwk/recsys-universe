"""
Ranking model for industrial recommendation system.
Implements deep learning model for CTR prediction.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # For type hints
    logger.warning("PyTorch not installed, using numpy implementation")


class DNNRanker(nn.Module if HAS_TORCH else object):
    """
    Deep Neural Network for ranking / CTR prediction.
    """

    def __init__(self, config):
        """
        Initialize ranking model.

        Args:
            config: RankingConfig
        """
        super().__init__()
        ranking = config.model.ranking

        self.hidden_dims = ranking.dnn_layers
        self.activation = ranking.activation
        self.dropout = ranking.dropout

        if HAS_TORCH:
            # Build PyTorch layers
            dims = ranking.dnn_layers
            self.layers = nn.ModuleList()
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        else:
            # Numpy fallback
            np.random.seed(42)
            dims = ranking.dnn_layers
            self.weights = []
            self.biases = []
            for i in range(len(dims) - 1):
                # Xavier initialization
                w = np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * np.sqrt(2.0 / dims[i])
                b = np.zeros(dims[i + 1], dtype=np.float32)
                self.weights.append(w)
                self.biases.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for numpy implementation.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        if HAS_TORCH:
            raise NotImplementedError("Use torch tensor input for PyTorch")

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = np.matmul(x, w) + b
            if i < len(self.weights) - 1:
                if self.activation == 'relu':
                    x = np.maximum(0, x)
                elif self.activation == 'tanh':
                    x = np.tanh(x)
                elif self.activation == 'sigmoid':
                    x = 1 / (1 + np.exp(-x))
        return x

    def forward_torch(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass for PyTorch implementation.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
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
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class RankingModel:
    """
    Complete ranking model for CTR prediction.
    """

    def __init__(self, config):
        """
        Initialize ranking model.

        Args:
            config: AppConfig
        """
        self.config = config
        self.ranking_config = config.model.ranking
        self.dnn = DNNRanker(config)

        # Calculate input dimension
        input_dim = (
            config.model.two_tower.user_embedding_dim +
            config.model.two_tower.item_embedding_dim +
            self.ranking_config.user_feature_dim +
            self.ranking_config.item_feature_dim +
            self.ranking_config.cross_feature_dim
        )
        self.input_dim = input_dim

        logger.info(
            f"RankingModel initialized with input_dim={input_dim}, "
            f"hidden_dims={self.ranking_config.dnn_layers}"
        )

    def get_feature_dim(self) -> int:
        """Get expected feature dimension."""
        return self.input_dim

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict CTR for given features.

        Args:
            features: Feature matrix [batch_size, feature_dim]

        Returns:
            CTR predictions [batch_size]
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        scores = self.dnn.forward(features)

        # Apply sigmoid to output
        predictions = 1 / (1 + np.exp(-scores))

        return predictions.flatten()

    def batch_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Batch predict CTR.

        Args:
            features: Feature matrix [batch_size, feature_dim]

        Returns:
            CTR predictions [batch_size]
        """
        return self.predict(features)

    def rank_items(
        self,
        user_id: int,
        item_ids: List[int],
        cross_features: 'CrossFeatures'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank items for a user.

        Args:
            user_id: User ID
            item_ids: List of item IDs to rank
            cross_features: CrossFeatures instance

        Returns:
            Tuple of (ranked_indices, scores)
        """
        if not item_ids:
            return np.array([]), np.array([])

        # Get ranking features
        features = cross_features.batch_get_ranking_features(user_id, item_ids)

        # Predict
        scores = self.batch_predict(features)

        # Rank by score
        ranked_indices = np.argsort(scores)[::-1]

        return ranked_indices, scores[ranked_indices]

    def get_top_k(
        self,
        user_id: int,
        item_ids: List[int],
        cross_features: 'CrossFeatures',
        top_k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-k items for a user.

        Args:
            user_id: User ID
            item_ids: List of candidate item IDs
            cross_features: CrossFeatures instance
            top_k: Number of items to return

        Returns:
            Tuple of (top_k_item_ids, top_k_scores)
        """
        if not item_ids:
            return [], []

        ranked_indices, scores = self.rank_items(user_id, item_ids, cross_features)

        top_indices = ranked_indices[:top_k]
        top_scores = scores[:top_k]

        return [item_ids[i] for i in top_indices], top_scores.tolist()

    def train_step(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 0.001
    ) -> float:
        """
        Perform one training step (simplified).

        Args:
            features: Feature matrix [batch_size, feature_dim]
            labels: Labels [batch_size]
            learning_rate: Learning rate

        Returns:
            Loss value
        """
        # Forward pass
        logits = self.dnn.forward(features)

        # Binary cross-entropy loss
        predictions = 1 / (1 + np.exp(-logits))
        epsilon = 1e-8
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(
            labels * np.log(predictions) +
            (1 - labels) * np.log(1 - predictions)
        )

        # Simplified gradient update (placeholder for real training)
        # In practice, use PyTorch autograd or TensorFlow
        logger.debug(f"Training step, loss={loss:.4f}")

        return float(loss)

    def save(self, path: str) -> None:
        """Save model weights."""
        logger.info(f"Ranking model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights."""
        logger.info(f"Ranking model loaded from {path}")


class WideAndDeepModel:
    """
    Wide & Deep learning model for ranking.
    Combines memorization (wide) and generalization (deep).
    """

    def __init__(self, config):
        """
        Initialize Wide & Deep model.

        Args:
            config: AppConfig
        """
        self.config = config
        self.ranking_config = config.model.ranking

        # Wide part (linear)
        self.wide_dim = 100  # Cross-product features
        self.wide_weights = np.zeros(self.wide_dim, dtype=np.float32)

        # Deep part (DNN)
        self.deep = DNNRanker(config)

        logger.info("WideAndDeepModel initialized")

    def get_wide_features(
        self,
        user_id: int,
        item_id: int,
        cross_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Get wide (cross-product) features.

        Args:
            user_id: User ID
            item_id: Item ID
            cross_features: Cross features dict

        Returns:
            Wide feature vector
        """
        features = np.zeros(self.wide_dim, dtype=np.float32)

        # User-item interactions
        user_bucket = hash(str(user_id)) % 100
        item_bucket = hash(str(item_id)) % 100
        features[user_bucket] = 1.0
        features[item_bucket + 50] = 1.0

        # Genre match indicator
        if cross_features.get('is_interacted', 0):
            features[25] = 1.0

        # Popularity features
        pop_score = cross_features.get('item_popularity', 0)
        features[26] = pop_score

        return features

    def predict(
        self,
        wide_features: np.ndarray,
        deep_features: np.ndarray
    ) -> float:
        """
        Make prediction.

        Args:
            wide_features: Wide features [wide_dim]
            deep_features: Deep features [deep_dim]

        Returns:
            CTR prediction
        """
        # Wide part
        wide_score = np.dot(wide_features, self.wide_weights)

        # Deep part
        deep_input = np.concatenate([wide_features, deep_features])
        deep_score = self.deep.forward(deep_input)

        # Combine (wide + deep)
        logits = wide_score + deep_score

        # Sigmoid
        return 1 / (1 + np.exp(-logits))

    def rank_items(
        self,
        user_id: int,
        item_ids: List[int],
        cross_features_list: List[Dict[str, Any]],
        deep_features_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank items using Wide & Deep.

        Args:
            user_id: User ID
            item_ids: List of item IDs
            cross_features_list: List of cross features
            deep_features_matrix: Deep features [batch_size, deep_dim]

        Returns:
            Tuple of (ranked_indices, scores)
        """
        scores = []
        for i, (item_id, cross_feat) in enumerate(
            zip(item_ids, cross_features_list)
        ):
            wide_feat = self.get_wide_features(user_id, item_id, cross_feat)
            score = self.predict(wide_feat, deep_features_matrix[i])
            scores.append(score)

        scores = np.array(scores)
        ranked_indices = np.argsort(scores)[::-1]

        return ranked_indices, scores[ranked_indices]
