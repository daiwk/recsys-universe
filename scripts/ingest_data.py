"""
Data ingestion script for MovieLens-1M dataset.
Imports data into Redis and FAISS for the industrial recommendation system.

Usage:
    python scripts/ingest_data.py                    # Use Redis + FAISS
    python scripts/ingest_data.py --memory           # Use memory store only
    python scripts/ingest_data.py --rebuild-index    # Rebuild FAISS index only

Environment variables:
    RECSYS_USE_MEMORY_STORE=true  # Use memory store instead of Redis
    REDIS_HOST=localhost         # Redis host
    REDIS_PORT=6379              # Redis port
    FAISS_NLIST=0                # FAISS IVF index nlist (0=Flat)
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from features.base import create_feature_store, MemoryFeatureStore
from features.user_features import UserFeatures
from features.item_features import ItemFeatures
from models.two_tower import TwoTowerModel
from serving.faiss_client import FaissClient, get_faiss_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_movielens_data(data_path: str = None) -> Dict[str, Any]:
    """
    Load MovieLens-1M dataset.

    Returns:
        Dict with 'users', 'items', 'ratings' keys
    """
    if data_path is None:
        data_path = os.environ.get("MOVIELENS_PATH", "ml-1m")

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"MovieLens data not found at {data_path}. "
            "Please download MovieLens-1M and extract it to this directory."
        )

    # Load users
    users_file = data_path / "users.dat"
    users = {}
    if users_file.exists():
        with open(users_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 4:
                    user_id = int(parts[0])
                    users[user_id] = {
                        "user_id": user_id,
                        "gender": parts[1],
                        "age": int(parts[2]),
                        "occupation": parts[3],
                        "zip_code": parts[4] if len(parts) > 4 else "",
                    }
        logger.info(f"Loaded {len(users)} users")

    # Load movies
    items_file = data_path / "movies.dat"
    items = {}
    if items_file.exists():
        with open(items_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 3:
                    item_id = int(parts[0])
                    title = parts[1]
                    genres = parts[2].split('|') if parts[2] else []
                    items[item_id] = {
                        "item_id": item_id,
                        "title": title,
                        "genres": genres,
                        "release_year": int(title[-5:-1]) if title[-5:-1].isdigit() else 1999,
                    }
        logger.info(f"Loaded {len(items)} items")

    # Load ratings
    ratings_file = data_path / "ratings.dat"
    ratings = []
    if ratings_file.exists():
        with open(ratings_file, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 4:
                    ratings.append({
                        "user_id": int(parts[0]),
                        "item_id": int(parts[1]),
                        "rating": float(parts[2]),
                        "timestamp": int(parts[3]),
                    })
        logger.info(f"Loaded {len(ratings)} ratings")

    return {"users": users, "items": items, "ratings": ratings}


def load_movielens_csv(data_path: str = None) -> Dict[str, Any]:
    """
    Load MovieLens-1M dataset from CSV files (alternative format).

    Returns:
        Dict with 'users', 'items', 'ratings' keys
    """
    import pandas as pd

    if data_path is None:
        data_path = os.environ.get("MOVIELENS_PATH", "ml-1m")

    data_path = Path(data_path)

    users = {}
    items = {}
    ratings = []

    # Load users
    users_file = data_path / "users.csv"
    if users_file.exists():
        df = pd.read_csv(users_file)
        for _, row in df.iterrows():
            users[row['user_id']] = {
                "user_id": row['user_id'],
                "gender": row.get('gender', 'M'),
                "age": row.get('age', 25),
                "occupation": row.get('occupation', 'other'),
                "zip_code": row.get('zip_code', ''),
            }
        logger.info(f"Loaded {len(users)} users")

    # Load movies
    items_file = data_path / "movies.csv"
    if items_file.exists():
        df = pd.read_csv(items_file)
        for _, row in df.iterrows():
            items[row['movie_id']] = {
                "item_id": row['movie_id'],
                "title": row.get('title', f"Movie {row['movie_id']}"),
                "genres": row.get('genres', '').split('|') if row.get('genres') else [],
                "release_year": row.get('year', 1999),
            }
        logger.info(f"Loaded {len(items)} items")

    # Load ratings
    ratings_file = data_path / "ratings.csv"
    if ratings_file.exists():
        df = pd.read_csv(ratings_file)
        for _, row in df.iterrows():
            ratings.append({
                "user_id": row['user_id'],
                "item_id": row['movie_id'],
                "rating": float(row.get('rating', 3.0)),
                "timestamp": row.get('timestamp', 0),
            })
        logger.info(f"Loaded {len(ratings)} ratings")

    return {"users": users, "items": items, "ratings": ratings}


def ingest_to_redis(
    data: Dict[str, Any],
    feature_store,
    user_features: UserFeatures,
    item_features: ItemFeatures
):
    """Ingest data into Redis."""
    logger.info("Ingesting data to Redis...")

    # Ingest items first (needed for recall)
    item_stats = {}
    for item_id, item in data['items'].items():
        # Set basic features
        item_features.store.set_item_features(item_id, {
            "basic": item,
            "statistics": {
                "views": 0,
                "avg_rating": 0,
                "ctr": 0.5,
            }
        })

    # Count item statistics from ratings
    item_view_count = {}
    item_rating_sum = {}
    for rating in data['ratings']:
        item_id = rating['item_id']
        item_view_count[item_id] = item_view_count.get(item_id, 0) + 1
        item_rating_sum[item_id] = item_rating_sum.get(item_id, 0) + rating['rating']

    # Update item statistics
    for item_id in data['items']:
        views = item_view_count.get(item_id, 0)
        rating_sum = item_rating_sum.get(item_id, 0)
        avg_rating = rating_sum / views if views > 0 else 0
        item_features.update_statistics(item_id, {
            "views": views,
            "avg_rating": round(avg_rating, 2),
            "ctr": min(0.9, 0.1 + views * 0.001),  # Simple CTR estimate
        })

    logger.info(f"Ingested {len(data['items'])} items to Redis")

    # Ingest users
    user_history = {}
    for rating in data['ratings']:
        user_id = rating['user_id']
        if user_id not in user_history:
            user_history[user_id] = {
                "viewed_items": [],
                "preferred_genres": set(),
                "total_views": 0,
            }
        user_history[user_id]["viewed_items"].append(rating['item_id'])
        user_history[user_id]["total_views"] += 1

        # Get genre preferences
        item = data['items'].get(rating['item_id'])
        if item:
            user_history[user_id]["preferred_genres"].update(item['genres'])

    for user_id, user in data['users'].items():
        history = user_history.get(user_id, {})
        preferred_genres = list(history.get("preferred_genres", set()))

        # Set user features
        feature_store.set_user_features(user_id, {
            "basic": user,
            "behavior": {
                "last_login": "2026-01-26T10:00:00",
                "total_views": history.get("total_views", 0),
                "preferred_genres": preferred_genres,
                "viewed_items": history.get("viewed_items", []),
            }
        })

    logger.info(f"Ingested {len(data['users'])} users to Redis")


def generate_item_embeddings(
    data: Dict[str, Any],
    two_tower: TwoTowerModel,
    item_features: ItemFeatures,
    faiss: FaissClient
):
    """Generate and store item embeddings, build FAISS index."""
    logger.info("Generating item embeddings...")

    item_ids = list(data['items'].keys())
    logger.info(f"Processing {len(item_ids)} items")

    # Get genre list for hashing
    all_genres = set()
    for item in data['items'].values():
        all_genres.update(item.get('genres', []))
    genre_list = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    # Generate embeddings in batches
    batch_size = 64
    embeddings = []
    valid_item_ids = []

    for i in range(0, len(item_ids), batch_size):
        batch_item_ids = item_ids[i:i + batch_size]

        for item_id in batch_item_ids:
            item = data['items'][getattr(item_id, 'item_id', item_id)]
            genres = item.get('genres', [])

            # Hash item ID
            item_id_hash = item_id % 1000000

            # Hash genres
            genre_hashes = [genre_to_idx.get(g, 0) for g in genres]

            # Generate embedding
            embedding = two_tower.get_item_embedding(item_id_hash, genre_hashes)
            embeddings.append(embedding)

            # Store embedding in feature store
            item_features.store.set_item_embedding(item_id, embedding.tolist())

            valid_item_ids.append(item_id)

        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {min(i + batch_size, len(item_ids))}/{len(item_ids)} items")

    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")

    # Build FAISS index
    logger.info("Building FAISS index...")
    faiss.build_index(valid_item_ids, embeddings)
    logger.info(f"FAISS index built with {len(valid_item_ids)} items")

    return valid_item_ids, embeddings


def build_user_embeddings(
    data: Dict[str, Any],
    two_tower: TwoTowerModel,
    user_features: UserFeatures
):
    """Pre-compute user embeddings."""
    logger.info("Generating user embeddings...")

    user_ids = list(data['users'].keys())
    genre_list = sorted(set(g for item in data['items'].values() for g in item.get('genres', [])))
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    for user_id in user_ids:
        user = data['users'][user_id]

        # Get user history
        viewed_items = [r['item_id'] for r in data['ratings'] if r['user_id'] == user_id]
        preferred_genres = []
        for item_id in viewed_items[:100]:  # Limit to recent 100 items
            item = data['items'].get(item_id)
            if item:
                preferred_genres.extend(item.get('genres', []))
        preferred_genres = list(set(preferred_genres))[:5]  # Top 5 genres

        # Hash user ID
        user_id_hash = user_id % 1000000

        # Hash genres
        genre_hashes = [genre_to_idx.get(g, 0) for g in preferred_genres]

        # Generate embedding
        embedding = two_tower.get_user_embedding(user_id_hash, genre_hashes)

        # Store in feature store
        user_features.store.set_user_embedding(user_id, embedding.tolist())

    logger.info(f"Generated embeddings for {len(user_ids)} users")


def main():
    parser = argparse.ArgumentParser(description="Ingest MovieLens data into Redis and FAISS")
    parser.add_argument("--data-path", type=str, help="Path to MovieLens data directory")
    parser.add_argument("--memory", action="store_true", help="Use memory store instead of Redis")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild FAISS index only")
    parser.add_argument("--skip-redis", action="store_true", help="Skip Redis ingestion")
    args = parser.parse_args()

    # Check for memory store mode
    use_memory = args.memory or os.environ.get("RECSYS_USE_MEMORY_STORE", "").lower() in ("true", "1", "yes")

    if use_memory:
        logger.info("Running in memory store mode (no Redis/FAISS required)")
        # Only run if we have the data
        data_path = args.data_path or os.environ.get("MOVIELENS_PATH", "ml-1m")
        if not Path(data_path).exists():
            logger.warning(f"Data path {data_path} not found. Please set MOVIELENS_PATH or provide --data-path")
            return

    # Load data
    try:
        data = load_movielens_data(args.data_path)
    except FileNotFoundError:
        try:
            data = load_movielens_csv(args.data_path)
        except FileNotFoundError:
            logger.error("Could not load MovieLens data. Please ensure the data exists.")
            sys.exit(1)

    if not data['items']:
        logger.error("No items loaded. Please check your MovieLens data format.")
        sys.exit(1)

    # Initialize components
    config = get_config()

    if use_memory:
        feature_store = MemoryFeatureStore(config)
    else:
        feature_store = create_feature_store(config)

    user_features = UserFeatures(feature_store)
    item_features = ItemFeatures(feature_store)

    if args.rebuild_index:
        # Only rebuild FAISS index
        logger.info("Rebuilding FAISS index only...")
        two_tower = TwoTowerModel(config)

        faiss = get_faiss_client(config.recall.faiss)
        generate_item_embeddings(data, two_tower, item_features, faiss)
        logger.info("Index rebuild complete!")
        return

    # Ingest to Redis
    if not args.skip_redis and not use_memory:
        ingest_to_redis(data, feature_store, user_features, item_features)

    # Generate item embeddings and build FAISS index
    if not use_memory:
        logger.info("Initializing Two-Tower model...")
        two_tower = TwoTowerModel(config)

        logger.info("Initializing FAISS...")
        faiss = get_faiss_client(config.recall.faiss)

        generate_item_embeddings(data, two_tower, item_features, faiss)

        # Pre-compute user embeddings
        build_user_embeddings(data, two_tower, user_features)
    else:
        logger.info("Skipping FAISS (memory mode)")

    logger.info("=" * 50)
    logger.info("Data ingestion complete!")
    logger.info(f"  - Users: {len(data['users'])}")
    logger.info(f"  - Items: {len(data['items'])}")
    logger.info(f"  - Ratings: {len(data['ratings'])}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
