"""
Data loading and utility functions for the movie recommendation system
"""
import os
import io
import zipfile
import json
from typing import Any, Dict, List

import requests
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
LOCAL_ZIP = "ml-1m.zip"

_movies_df: pd.DataFrame | None = None
_ratings_df: pd.DataFrame | None = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_movie_tfidf_matrix = None


def download_movielens_if_needed() -> str:
    """Download MovieLens 1M dataset if local file doesn't exist."""
    if not os.path.exists(LOCAL_ZIP):
        print("[DEBUG][DATA] 开始下载 MovieLens 1M 数据集...")
        r = requests.get(MOVIELENS_URL, timeout=60)
        r.raise_for_status()
        with open(LOCAL_ZIP, "wb") as f:
            f.write(r.content)
        print("[DEBUG][DATA] MovieLens 1M 下载完成。")
    else:
        print("[DEBUG][DATA] 检测到本地已存在 MovieLens 1M 压缩包，跳过下载。")
    return LOCAL_ZIP


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load movies.dat and ratings.dat from ml-1m.zip
    
    movies.dat: MovieID::Title::Genres
    ratings.dat: UserID::MovieID::Rating::Timestamp
    """
    global _movies_df, _ratings_df
    if _movies_df is not None and _ratings_df is not None:
        return _movies_df, _ratings_df

    zip_path = download_movielens_if_needed()
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                sep="::",
                header=None,
                names=["movie_id", "title", "genres"],
                engine="python",
            )
        with zf.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                sep="::",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                engine="python",
            )
    movies["movie_id"] = movies["movie_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    _movies_df, _ratings_df = movies, ratings
    print(f"[DEBUG][DATA] 电影数={len(movies)}, 打分数={len(ratings)}")
    return movies, ratings


def build_movie_tfidf():
    """
    Build a simple TF-IDF index using title + genres as content-side RAG retrieval tool.
    """
    global _tfidf_vectorizer, _movie_tfidf_matrix
    movies, _ = load_movielens()
    if _tfidf_vectorizer is not None:
        return

    print("[DEBUG][TFIDF] 开始构建电影 TF-IDF 索引...")
    docs = (movies["title"] + " " + movies["genres"].str.replace("|", " ")).tolist()
    vectorizer = TfidfVectorizer(max_features=5000)
    mat = vectorizer.fit_transform(docs)
    _tfidf_vectorizer = vectorizer
    _movie_tfidf_matrix = mat
    print(f"[DEBUG][TFIDF] TF-IDF 索引构建完成，电影数={len(docs)}, 维度={mat.shape[1]}")


def rag_search_movies(query: str, k: int = 20) -> List[Dict[str, Any]]:
    """
    RAG tool: Given a text query, return the most similar k movies.
    """
    global _movies_df, _tfidf_vectorizer, _movie_tfidf_matrix
    if _movies_df is None or _tfidf_vectorizer is None:
        build_movie_tfidf()

    movies = _movies_df
    vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(vec, _movie_tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    sub = movies.iloc[top_idx][["movie_id", "title", "genres"]].copy()
    sub["score"] = sims[top_idx]
    print(f"[DEBUG][RAG] 基于查询 {query!r} 检索到 {len(sub)} 条电影候选")
    return sub.to_dict("records")


def get_user_history(user_id: int, n: int = 10) -> List[Dict[str, Any]]:
    """
    Tool: Return movies rated by a user (top n by rating).
    """
    movies, ratings = load_movielens()
    user_ratings = (
        ratings[ratings["user_id"] == user_id]
        .sort_values("rating", ascending=False)
        .head(n)
    )
    merged = user_ratings.merge(movies, on="movie_id", how="left")
    print(f"[DEBUG][USER_HIST] user_id={user_id} 历史记录条数={len(merged)}")
    return merged[["movie_id", "title", "genres", "rating"]].to_dict("records")


def get_collab_candidates_by_genre(user_id: int, k: int = 30) -> List[Dict[str, Any]]:
    """
    Non-strict "collaborative filtering style" tool (simple heuristic):
    - Look at high-frequency genres in user's history
    - Pick highly-rated movies in those genres that haven't been seen
    """
    movies, ratings = load_movielens()

    hist = get_user_history(user_id, n=50)
    if not hist:
        print(f"[DEBUG][COLLAB] user_id={user_id} 没有历史，直接返回全局 top 电影")
        mean_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
        merged = mean_rating.merge(movies, on="movie_id", how="left")
        merged = merged.sort_values("rating", ascending=False).head(k)
        return merged[["movie_id", "title", "genres", "rating"]].to_dict("records")

    from collections import Counter

    genre_counter = Counter()
    for h in hist:
        for g in str(h["genres"]).split("|"):
            genre_counter[g] += 1

    if not genre_counter:
        top_genres = []
    else:
        top_genres = [g for g, _ in genre_counter.most_common(3)]

    print(
        f"[DEBUG][COLLAB]",
        f"user_id={user_id} 偏好的前 3 个类型={top_genres}",
    )

    mean_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
    merged = mean_rating.merge(movies, on="movie_id", how="left")

    watched_ids = {h["movie_id"] for h in hist}

    def has_pref_genre(row) -> bool:
        gs = str(row["genres"]).split("|")
        return any(g in gs for g in top_genres)

    cand = merged[merged.apply(has_pref_genre, axis=1)]
    cand = cand[~cand["movie_id"].isin(watched_ids)]
    cand = cand.sort_values("rating", ascending=False).head(k)

    print(
        f"[DEBUG][COLLAB]",
        f"user_id={user_id} 在偏好类型中候选数={len(cand)}, 返回前 {k} 条",
    )

    return cand[["movie_id", "title", "genres", "rating"]].to_dict("records")