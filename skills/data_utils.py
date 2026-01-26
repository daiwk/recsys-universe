"""
Data loading and utility functions for the movie recommendation system.
"""
import os
import io
import zipfile
import logging
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config

logger = logging.getLogger(__name__)

# Constants
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
LOCAL_ZIP = "ml-1m.zip"

# Module-level cache (lazy initialization)
_movies_df: Optional[pd.DataFrame] = None
_ratings_df: Optional[pd.DataFrame] = None
_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_movie_tfidf_matrix = None


def download_movielens_if_needed() -> str:
    """
    Download MovieLens 1M dataset if local file doesn't exist.

    Returns:
        Path to the downloaded zip file

    Raises:
        RuntimeError: If download fails
    """
    if os.path.exists(LOCAL_ZIP):
        logger.info("Local MovieLens zip file already exists, skipping download")
        return LOCAL_ZIP

    config = get_config()
    logger.info(f"Downloading MovieLens 1M dataset from {config.movielens_url}")

    try:
        response = requests.get(config.movielens_url, timeout=120)
        response.raise_for_status()

        with open(LOCAL_ZIP, "wb") as f:
            f.write(response.content)

        logger.info("MovieLens 1M dataset downloaded successfully")
        return LOCAL_ZIP

    except requests.RequestException as e:
        logger.error(f"Failed to download MovieLens dataset: {e}")
        raise RuntimeError(f"Failed to download MovieLens dataset: {e}") from e


def load_movielens(force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies.dat and ratings.dat from ml-1m.zip.

    Args:
        force_reload: If True, reload data even if cached

    Returns:
        Tuple of (movies_df, ratings_df)

    Raises:
        RuntimeError: If data loading fails
    """
    global _movies_df, _ratings_df

    if not force_reload and _movies_df is not None and _ratings_df is not None:
        logger.debug("Returning cached MovieLens data")
        return _movies_df, _ratings_df

    try:
        zip_path = download_movielens_if_needed()

        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("ml-1m/movies.dat") as f:
                movies = pd.read_csv(
                    io.TextIOWrapper(f, encoding="latin-1"),
                    sep="::",
                    header=None,
                    names=["movie_id", "title", "genres"],
                    engine="python",
                    on_bad_lines="skip",
                )
            with zf.open("ml-1m/ratings.dat") as f:
                ratings = pd.read_csv(
                    io.TextIOWrapper(f, encoding="latin-1"),
                    sep="::",
                    header=None,
                    names=["user_id", "movie_id", "rating", "timestamp"],
                    engine="python",
                    on_bad_lines="skip",
                )

        # Type conversion with error handling
        movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce").astype("Int64")
        ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce").astype("Int64")
        ratings["movie_id"] = pd.to_numeric(ratings["movie_id"], errors="coerce").astype("Int64")
        ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

        # Remove any rows with NaN values after conversion
        movies = movies.dropna(subset=["movie_id"])
        ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])

        _movies_df, _ratings_df = movies, ratings
        logger.info(f"Loaded {len(movies)} movies and {len(ratings)} ratings")

        return movies, ratings

    except Exception as e:
        logger.error(f"Failed to load MovieLens data: {e}")
        raise RuntimeError(f"Failed to load MovieLens data: {e}") from e


def build_movie_tfidf(force_rebuild: bool = False) -> Tuple[TfidfVectorizer, Any]:
    """
    Build a simple TF-IDF index using title + genres as content-side RAG retrieval tool.

    Args:
        force_rebuild: If True, rebuild index even if cached

    Returns:
        Tuple of (vectorizer, tfidf_matrix)
    """
    global _tfidf_vectorizer, _movie_tfidf_matrix

    if not force_rebuild and _tfidf_vectorizer is not None and _movie_tfidf_matrix is not None:
        logger.debug("Returning cached TF-IDF index")
        return _tfidf_vectorizer, _movie_tfidf_matrix

    movies, _ = load_movielens()
    logger.info("Building TF-IDF index...")

    # Prepare documents for TF-IDF
    docs = (movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)).tolist()

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(docs)

    _tfidf_vectorizer = vectorizer
    _movie_tfidf_matrix = matrix

    logger.info(f"TF-IDF index built: {len(docs)} movies, {matrix.shape[1]} features")
    return vectorizer, matrix


@lru_cache(maxsize=1)
def rag_search_movies(query: str, k: int = 20) -> Tuple[List[Dict[str, Any]], int]:
    """
    RAG tool: Given a text query, return the most similar k movies.

    Note: Using lru_cache for performance. For dynamic results,
    use the non-cached version or clear the cache.

    Args:
        query: Search query
        k: Number of results to return

    Returns:
        Tuple list, total matches of (results found)
    """
    global _movies_df, _tfidf_vectorizer, _movie_tfidf_matrix

    if _movies_df is None or _tfidf_vectorizer is None:
        build_movie_tfidf()

    movies = _movies_df
    vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(vec, _movie_tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:k]

    results = []
    for idx in top_idx:
        movie_id = int(movies.iloc[idx]["movie_id"])
        results.append({
            "movie_id": movie_id,
            "title": movies.iloc[idx]["title"],
            "genres": movies.iloc[idx]["genres"],
            "score": float(sims[idx]),
        })

    logger.debug(f"TF-IDF search for '{query}' returned {len(results)} results")
    return results, len(sims)


def get_user_history(user_id: int, n: int = 10) -> List[Dict[str, Any]]:
    """
    Tool: Return movies rated by a user (top n by rating).

    Args:
        user_id: User ID to query
        n: Number of results to return

    Returns:
        List of movie records with rating info
    """
    movies, ratings = load_movielens()

    user_ratings = (
        ratings[ratings["user_id"] == user_id]
        .sort_values("rating", ascending=False)
        .head(n)
    )

    merged = user_ratings.merge(movies, on="movie_id", how="left")
    results = merged[["movie_id", "title", "genres", "rating"]].to_dict("records")

    logger.debug(f"User {user_id} has {len(results)} history items")
    return results


def get_collab_candidates_by_genre(user_id: int, k: int = 30) -> List[Dict[str, Any]]:
    """
    Non-strict "collaborative filtering style" tool (simple heuristic):
    - Look at high-frequency genres in user's history
    - Pick highly-rated movies in those genres that haven't been seen

    Args:
        user_id: User ID to query
        k: Number of candidates to return

    Returns:
        List of candidate movie records
    """
    from collections import Counter

    movies, ratings = load_movielens()
    config = get_config()

    hist = get_user_history(user_id, n=50)

    if not hist:
        logger.debug(f"User {user_id} has no history, returning global top movies")
        mean_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
        merged = mean_rating.merge(movies, on="movie_id", how="left")
        merged = merged.sort_values("rating", ascending=False).head(k)
        return merged[["movie_id", "title", "genres", "rating"]].to_dict("records")

    # Count genres from user history
    genre_counter = Counter()
    for h in hist:
        for g in str(h["genres"]).split("|"):
            g = g.strip()
            if g:
                genre_counter[g] += 1

    if not genre_counter:
        top_genres = []
    else:
        top_genres = [g for g, _ in genre_counter.most_common(3)]

    logger.debug(f"User {user_id} preferred genres: {top_genres}")

    mean_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
    merged = mean_rating.merge(movies, on="movie_id", how="left")
    watched_ids = {h["movie_id"] for h in hist}

    # Filter for preferred genres and unwatched movies
    def has_pref_genre(row) -> bool:
        gs = str(row["genres"]).split("|")
        return any(g.strip() in gs for g in top_genres)

    cand = merged[merged.apply(has_pref_genre, axis=1)]
    cand = cand[~cand["movie_id"].isin(watched_ids)]
    cand = cand.sort_values("rating", ascending=False).head(k)

    logger.debug(f"User {user_id} got {len(cand)} collaborative candidates")
    return cand[["movie_id", "title", "genres", "rating"]].to_dict("records")


def clear_cache() -> None:
    """
    Clear all cached data. Useful for testing or memory management.
    """
    global _movies_df, _ratings_df, _tfidf_vectorizer, _movie_tfidf_matrix

    _movies_df = None
    _ratings_df = None
    _tfidf_vectorizer = None
    _movie_tfidf_matrix = None

    # Clear the lru_cache
    rag_search_movies.cache_clear()

    logger.info("Data cache cleared")


def is_data_loaded() -> bool:
    """
    Check if data has been loaded.

    Returns:
        True if data is loaded, False otherwise
    """
    return _movies_df is not None and _ratings_df is not None


def is_tfidf_loaded() -> bool:
    """
    Check if TF-IDF index has been built.

    Returns:
        True if TF-IDF is loaded, False otherwise
    """
    return _tfidf_vectorizer is not None and _movie_tfidf_matrix is not None
