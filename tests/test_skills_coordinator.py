"""
Tests for skills_coordinator module.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Set environment before importing
os.environ["OPENAI_API_KEY"] = "test_key"

from skills_coordinator import SkillsCoordinator, RecState


class TestSkillsCoordinator:
    """Tests for SkillsCoordinator class."""

    def test_init(self):
        """Test coordinator initialization."""
        coordinator = SkillsCoordinator()
        assert coordinator.model is not None
        assert coordinator.registry is not None

    def test_valid_skills(self):
        """Test that VALID_SKILLS is properly defined."""
        assert "profile_skill" in SkillsCoordinator.VALID_SKILLS
        assert "content_skill" in SkillsCoordinator.VALID_SKILLS
        assert "collab_skill" in SkillsCoordinator.VALID_SKILLS
        assert "merge_skill" in SkillsCoordinator.VALID_SKILLS
        assert "final_skill" in SkillsCoordinator.VALID_SKILLS
        assert "planner_skill" in SkillsCoordinator.VALID_SKILLS

    def test_validate_input_valid(self):
        """Test input validation with valid input."""
        coordinator = SkillsCoordinator()
        # Should not raise
        coordinator._validate_input(user_id=1, query="test query")

    def test_validate_input_invalid_user_id_type(self):
        """Test input validation with invalid user_id type."""
        coordinator = SkillsCoordinator()
        with pytest.raises(TypeError):
            coordinator._validate_input(user_id="1", query="test")

    def test_validate_input_invalid_user_id_value(self):
        """Test input validation with invalid user_id value."""
        coordinator = SkillsCoordinator()
        with pytest.raises(ValueError):
            coordinator._validate_input(user_id=0, query="test")
        with pytest.raises(ValueError):
            coordinator._validate_input(user_id=-1, query="test")

    def test_validate_input_invalid_query_type(self):
        """Test input validation with invalid query type."""
        coordinator = SkillsCoordinator()
        with pytest.raises(TypeError):
            coordinator._validate_input(user_id=1, query=123)

    def test_validate_input_empty_query(self):
        """Test input validation with empty query."""
        coordinator = SkillsCoordinator()
        with pytest.raises(ValueError):
            coordinator._validate_input(user_id=1, query="   ")

    @patch('skills_coordinator.load_movielens')
    @patch('skills_coordinator.build_movie_tfidf')
    def test_run_recommendation_calls_data_loading(self, mock_tfidf, mock_load):
        """Test that run_recommendation loads data."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_tfidf.return_value = (MagicMock(), MagicMock())

        coordinator = SkillsCoordinator()
        # This will fail without a real LLM, but we can verify data loading is called
        try:
            coordinator.run_recommendation(user_id=1, query="test")
        except Exception:
            pass  # Expected to fail without real LLM

        mock_load.assert_called_once()
        mock_tfidf.assert_called_once()

    @patch('skills_coordinator.load_movielens')
    @patch('skills_coordinator.build_movie_tfidf')
    def test_execute_skill_unknown_skill(self, mock_tfidf, mock_load):
        """Test that _execute_skill handles unknown skills."""
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_tfidf.return_value = (MagicMock(), MagicMock())

        coordinator = SkillsCoordinator()
        state: RecState = {"user_id": 1, "query": "test"}

        # Should not raise, just return state unchanged
        result = coordinator._execute_skill("unknown_skill", state)
        assert result == state


class TestRecState:
    """Tests for RecState TypedDict."""

    def test_rec_state_creation(self):
        """Test RecState can be created with valid fields."""
        state: RecState = {
            "user_id": 1,
            "query": "test query",
            "user_history": [],
            "user_profile": "test profile",
            "content_candidates": [],
            "collab_candidates": [],
            "merged_candidates": [],
            "final_recommendations": [],
            "next_skill": "profile_skill",
            "planner_reason": "test reason",
            "step_count": 1,
        }
        assert state["user_id"] == 1
        assert state["step_count"] == 1

    def test_rec_state_partial_creation(self):
        """Test RecState can be created with partial fields."""
        state: RecState = {
            "user_id": 1,
            "query": "test",
        }
        assert state["user_id"] == 1
        assert "user_history" not in state  # Optional field
