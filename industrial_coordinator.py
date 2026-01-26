"""
Industrial skills coordinator for movie recommendation system.
Implements Two-Tower recall + Ranking pipeline with Skills-style orchestration.
"""
import logging
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from config import get_config
from features.base import create_feature_store
from features.user_features import UserFeatures
from features.item_features import ItemFeatures
from features.cross_features import CrossFeatures
from models.two_tower import TwoTowerModel
from models.ranking_model import RankingModel
from serving.recall_service import RecallService
from serving.rank_service import RankService
from serving.faiss_client import get_faiss_client

logger = logging.getLogger(__name__)


class IndustrialState(TypedDict, total=False):
    """
    State passed between skills in the industrial recommendation process.
    """
    user_id: int
    query: str

    # Recall results
    recall_candidates: List[Dict[str, Any]]
    recall_scores: List[float]

    # Rank results
    ranked_recommendations: List[Dict[str, Any]]
    ctr_scores: List[float]

    # Feature store results
    user_embedding: List[float]
    item_embeddings: List[List[float]]

    # Cold start flag
    is_cold_start: bool


class IndustrialSkillsCoordinator:
    """
    Industrial-grade coordinator that manages Two-Tower recall + Ranking pipeline.
    Provides vector-based retrieval and CTR prediction.
    """

    def __init__(self, model: str = None, feature_store=None, two_tower=None, ranking_model=None, faiss_client=None):
        """
        Initialize the industrial coordinator.

        Args:
            model: LLM model name (for LLM-based explanation if needed)
            feature_store: Optional pre-initialized feature store (for sharing state)
            two_tower: Optional pre-initialized TwoTower model
            ranking_model: Optional pre-initialized Ranking model
            faiss_client: Optional pre-initialized FAISS client
        """
        self.config = get_config()
        self.model = model or self.config.llm.model

        # Initialize feature store (use provided or create new)
        self.feature_store = feature_store if feature_store is not None else create_feature_store(self.config)
        self.user_features = UserFeatures(self.feature_store)
        self.item_features = ItemFeatures(self.feature_store)
        self.cross_features = CrossFeatures(
            self.user_features,
            self.item_features
        )

        # Initialize models (use provided or create new)
        self.two_tower = two_tower if two_tower is not None else TwoTowerModel(self.config)
        self.ranking_model = ranking_model if ranking_model is not None else RankingModel(self.config)

        # Initialize FAISS client (use provided or create new)
        self.faiss_client = faiss_client if faiss_client is not None else get_faiss_client(self.config.recall.faiss)

        # Initialize services with shared components
        self.recall_service = RecallService(
            self.config,
            feature_store=self.feature_store,
            two_tower=self.two_tower,
            faiss_client=self.faiss_client
        )
        self.rank_service = RankService(
            self.config,
            feature_store=self.feature_store,
            ranking_model=self.ranking_model
        )

        logger.info("IndustrialSkillsCoordinator initialized")

    def _validate_input(self, user_id: int, query: str = None) -> None:
        """Validate input parameters."""
        if not isinstance(user_id, int):
            raise TypeError(f"user_id must be an integer, got {type(user_id).__name__}")
        if user_id <= 0:
            raise ValueError(f"user_id must be positive, got {user_id}")

    def run_recommendation(
        self,
        user_id: int,
        query: str = None,
        industrial: bool = None
    ) -> List[Dict[str, Any]]:
        """
        Main method to run the recommendation process.

        Args:
            user_id: ID of the user to recommend movies for
            query: Natural language query describing preferences (optional)
            industrial: Whether to use industrial pipeline (defaults to config)

        Returns:
            List of recommended movies with reasons and scores
        """
        self._validate_input(user_id, query)

        if industrial is None:
            industrial = self.config.architecture_mode == "industrial"

        logger.info(
            f"Starting recommendation for user_id={user_id}, "
            f"industrial={industrial}"
        )

        if industrial:
            return self._run_industrial_pipeline(user_id)
        else:
            return self._run_legacy_pipeline(user_id, query)

    def _run_industrial_pipeline(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Run the industrial pipeline: Recall -> Rank -> Format.

        Args:
            user_id: User ID

        Returns:
            List of recommended movies
        """
        # Step 1: Check cold start
        is_cold_start = self.user_features.is_cold_start(user_id)
        logger.info(f"User {user_id} cold_start={is_cold_start}")

        if is_cold_start:
            return self._handle_cold_start(user_id)

        # Step 2: Vector Recall
        logger.info("Step 1: Vector Recall")
        recall_item_ids, recall_scores = self.recall_service.recall(user_id)

        # Build recall candidates
        recall_candidates = []
        for item_id, score in zip(recall_item_ids, recall_scores):
            item_basic = self.item_features.store.get_basic_features(item_id)
            recall_candidates.append({
                "item_id": item_id,
                "recall_score": float(score),
                "title": item_basic.get("title", f"Item {item_id}"),
                "genres": item_basic.get("genres", []),
            })

        logger.info(f"Recall: found {len(recall_candidates)} candidates")

        # Step 3: Ranking
        logger.info("Step 2: Ranking")
        ranked_items = self.rank_service.rank_with_recall(
            user_id,
            recall_candidates,
            top_k=10
        )

        logger.info(f"Ranked: {len(ranked_items)} items")

        # Step 4: Format results
        recommendations = []
        for rank, item in enumerate(ranked_items, 1):
            recommendations.append({
                "movie_id": item["item_id"],
                "title": item["title"],
                "genres": item["genres"],
                "reason": f"推荐指数: {item['ctr_score']:.2%}",
                "rank": rank,
                "recall_score": item.get("recall_score", 0),
                "ctr_score": item.get("ctr_score", 0),
            })

        logger.info(f"Recommendation complete: {len(recommendations)} movies")
        return recommendations

    def _handle_cold_start(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Handle cold start user with popular items.

        Args:
            user_id: User ID

        Returns:
            List of popular item recommendations
        """
        logger.info(f"Handling cold start for user {user_id}")

        # Get popular items
        popular_count = 10
        popular_items = self.recall_service._get_popular_items(popular_count)

        recommendations = []
        for rank, item_id in enumerate(popular_items, 1):
            item_basic = self.item_features.store.get_basic_features(item_id)
            recommendations.append({
                "movie_id": item_id,
                "title": item_basic.get("title", f"Popular Item {item_id}"),
                "genres": item_basic.get("genres", []),
                "reason": f"热门推荐 - 适合新用户",
                "rank": rank,
                "recall_score": 0.5,
                "ctr_score": 0.5,
                "is_cold_start": True,
            })

        return recommendations

    def _run_legacy_pipeline(
        self,
        user_id: int,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Run the legacy TF-IDF based pipeline.

        Args:
            user_id: User ID
            query: Query string

        Returns:
            List of recommended movies (requires LLM)
        """
        # Import legacy components
        from skills_coordinator import SkillsCoordinator as LegacyCoordinator

        logger.info("Using legacy TF-IDF pipeline")
        coordinator = LegacyCoordinator(model=self.model)
        return coordinator.run_recommendation(user_id, query)

    def build_index(self, item_ids: List[int]) -> bool:
        """
        Build the FAISS index for items.

        Args:
            item_ids: List of item IDs to index

        Returns:
            True if successful
        """
        logger.info(f"Building index for {len(item_ids)} items")
        return self.recall_service.build_item_index(item_ids)

    def health_check(self) -> Dict[str, Any]:
        """
        Check service health.

        Returns:
            Dict with health status
        """
        return {
            "config": {
                "architecture_mode": self.config.architecture_mode,
                "recall_top_k": self.config.recall.recall_top_k,
                "rank_top_k": self.config.rank.rank_top_k,
            },
            "recall_service": self.recall_service.health_check(),
            "rank_service": self.rank_service.health_check(),
            "models_loaded": True,
        }


def run_industrial_recommendation(
    user_id: int,
    query: str = None,
    industrial: bool = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to run industrial recommendation.

    Args:
        user_id: User ID
        query: Optional query (for legacy pipeline)
        industrial: Use industrial pipeline (defaults to config)

    Returns:
        List of recommended movies
    """
    coordinator = IndustrialSkillsCoordinator()
    return coordinator.run_recommendation(user_id, query, industrial)


def demo_run(
    user_id: int = 1,
    query: str = "我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道",
):
    """
    Demo function with industrial pipeline.
    """
    print(f"[DEMO] Starting demo_run, user_id={user_id}, query={query!r}")

    config = get_config()
    coordinator = IndustrialSkillsCoordinator()

    # Use industrial pipeline
    recs = coordinator.run_recommendation(user_id, query, industrial=True)

    print("\n================ 工业级推荐结果 ================")
    for r in recs:
        print(
            f"{r['rank']}. {r['title']}  "
            f"[{r['genres']}]  (movie_id={r['movie_id']})\n"
            f"   召回分: {r['recall_score']:.3f} | CTR预测: {r['ctr_score']:.2%}\n"
            f"   推荐理由：{r['reason']}\n"
        )


if __name__ == "__main__":
    # Run demo with example user
    demo_run(user_id=123)
