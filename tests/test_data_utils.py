"""
Tests for data_utils module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment before importing
os.environ["OPENAI_API_KEY"] = "test_key"

from skills.data_utils import (
    load_movielens,
    build_movie_tfidf,
    get_user_history,
    get_collab_candidates_by_genre,
    clear_cache,
    is_data_loaded,
    is_tfidf_loaded,
)


class TestDataUtils:
    """Tests for data_utils functions."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_cache()

    def test_clear_cache(self):
        """Test cache clearing."""
        # Initially cache is empty
        clear_cache()  # Should not raise
        assert not is_data_loaded()
        assert not is_tfidf_loaded()

    @patch('skills.data_utils.requests.get')
    def test_download_movielens_if_needed_skips_existing(self, mock_get):
        """Test that download is skipped if file exists."""
        from skills.data_utils import download_movielens_if_needed, LOCAL_ZIP

        # Create a dummy file
        with open(LOCAL_ZIP, 'w') as f:
            f.write("dummy")

        try:
            result = download_movielens_if_needed()
            assert result == LOCAL_ZIP
            mock_get.assert_not_called()
        finally:
            if os.path.exists(LOCAL_ZIP):
                os.remove(LOCAL_ZIP)

    def test_load_movielens_returns_tuple(self):
        """Test that load_movielens returns tuple of DataFrames."""
        movies, ratings = load_movielens()
        assert movies is not None
        assert ratings is not None
        assert len(movies) > 0
        assert len(ratings) > 0

    def test_load_movielens_caches_result(self):
        """Test that load_movielens returns cached result."""
        movies1, _ = load_movielens()
        movies2, _ = load_movielens()
        assert movies1 is movies2  # Same object (cached)

    def test_build_movie_tfidf_returns_tuple(self):
        """Test that build_movie_tfidf returns vectorizer and matrix."""
        vectorizer, matrix = build_movie_tfidf()
        assert vectorizer is not None
        assert matrix is not None
        assert matrix.shape[0] > 0  # Should have rows
        assert matrix.shape[1] > 0  # Should have features

    def test_build_movie_tfidf_caches_result(self):
        """Test that build_movie_tfidf caches result."""
        vec1, mat1 = build_movie_tfidf()
        vec2, mat2 = build_movie_tfidf()
        assert vec1 is vec2  # Same object (cached)
        assert mat1 is mat2

    def test_get_user_history_returns_list(self):
        """Test that get_user_history returns list of dicts."""
        history = get_user_history(user_id=1, n=10)
        assert isinstance(history, list)
        # Each item should have these keys
        if len(history) > 0:
            assert "movie_id" in history[0]
            assert "title" in history[0]
            assert "genres" in history[0]
            assert "rating" in history[0]

    def test_get_user_history_respects_n(self):
        """Test that get_user_history respects the n parameter."""
        history_5 = get_user_history(user_id=1, n=5)
        history_10 = get_user_history(user_id=1, n=10)
        assert len(history_5) <= 5
        assert len(history_10) <= 10

    def test_get_collab_candidates_by_genre_returns_list(self):
        """Test that get_collab_candidates_by_genre returns list."""
        candidates = get_collab_candidates_by_genre(user_id=1, k=10)
        assert isinstance(candidates, list)
        # Each item should have these keys
        if len(candidates) > 0:
            assert "movie_id" in candidates[0]
            assert "title" in candidates[0]
            assert "genres" in candidates[0]

    def test_get_collab_candidates_by_genre_respects_k(self):
        """Test that get_collab_candidates_by_genre respects k parameter."""
        candidates = get_collab_candidates_by_genre(user_id=1, k=5)
        assert len(candidates) <= 5

    def test_is_data_loaded(self):
        """Test is_data_loaded function."""
        assert not is_data_loaded()  # Before loading
        load_movielens()
        assert is_data_loaded()  # After loading

    def test_is_tfidf_loaded(self):
        """Test is_tfidf_loaded function."""
        assert not is_tfidf_loaded()  # Before building
        build_movie_tfidf()
        assert is_tfidf_loaded()  # After building
