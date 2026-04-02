"""
Run a public MovieLens smoke demo through the industrial pipeline.

This script is designed for two scenarios:
1. zero-network smoke test using the checked-in public subset under
   `data/ml-latest-small-public-subset/`
2. locally downloaded public MovieLens CSV prepared by
   `scripts/prepare_public_movielens_csv.py`

The demo intentionally uses:
- in-memory feature store
- Two-Tower embedding generation
- FAISS client if installed, otherwise mock FAISS fallback
- industrial coordinator end-to-end recommendation path

Usage:
    RECSYS_USE_MEMORY_STORE=true python scripts/run_public_movielens_smoke.py

    RECSYS_USE_MEMORY_STORE=true python scripts/run_public_movielens_smoke.py \
      --data-path ./data/ml-latest-small-public-subset \
      --user-id 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import get_config
from features.base import MemoryFeatureStore
from features.user_features import UserFeatures
from features.item_features import ItemFeatures
from models.two_tower import TwoTowerModel
from serving.faiss_client import get_faiss_client
from scripts.ingest_data import (
    load_movielens_csv,
    ingest_to_redis,
    generate_item_embeddings,
    build_user_embeddings,
)
from industrial_coordinator import IndustrialSkillsCoordinator


def run_demo(data_path: Path, user_id: int) -> None:
    config = get_config()

    feature_store = MemoryFeatureStore(config)
    user_features = UserFeatures(feature_store)
    item_features = ItemFeatures(feature_store)

    data = load_movielens_csv(str(data_path))
    ingest_to_redis(data, feature_store, user_features, item_features)

    two_tower = TwoTowerModel(config)
    faiss = get_faiss_client(config.recall.faiss)
    generate_item_embeddings(data, two_tower, item_features, faiss)
    build_user_embeddings(data, two_tower, user_features)

    coordinator = IndustrialSkillsCoordinator(
        feature_store=feature_store,
        two_tower=two_tower,
        ranking_model=None,
        faiss_client=faiss,
    )

    print("health_check=")
    print(coordinator.health_check())
    print()

    recs = coordinator.run_recommendation(user_id, industrial=True)

    print(f"recommendations for user_id={user_id}:")
    for rec in recs[:10]:
        print(
            f"{rec['rank']}. {rec['title']} | genres={rec['genres']} | "
            f"recall={rec['recall_score']:.4f} | ctr={rec['ctr_score']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        default="data/ml-latest-small-public-subset",
        help="Prepared CSV dataset directory",
    )
    parser.add_argument("--user-id", type=int, default=1)
    args = parser.parse_args()

    run_demo(Path(args.data_path), args.user_id)


if __name__ == "__main__":
    main()
