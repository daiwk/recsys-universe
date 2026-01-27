"""
Skills coordinator for movie recommendation system.
Implements Claude-style skills coordination while maintaining original API compatibility.
"""
import logging
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from skills import (
    ProfileSkill,
    ContentSkill,
    CollabSkill,
    MergeSkill,
    FinalSkill,
    PlannerSkill,
    VectorRecallSkill,
    RankingSkill,
    SkillRegistry
)
from skills.data_utils import load_movielens, build_movie_tfidf
from config import get_config

logger = logging.getLogger(__name__)


class RecState(TypedDict, total=False):
    """
    State passed between skills in the recommendation process.
    Each skill only reads/writes the fields it cares about.
    """
    user_id: int
    query: str

    user_history: List[Dict[str, Any]]
    user_profile: str

    content_candidates: List[Dict[str, Any]]
    collab_candidates: List[Dict[str, Any]]

    merged_candidates: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]

    # Planning related fields (Claude Skills style)
    next_skill: str
    planner_reason: str
    step_count: int


class SkillsCoordinator:
    """
    Coordinator that manages Claude-style skills orchestration
    while maintaining compatibility with the original API.
    """

    # Valid skills that can be called
    VALID_SKILLS = frozenset({
        "profile_skill",
        "content_skill",
        "collab_skill",
        "merge_skill",
        "final_skill",
        "planner_skill",
        "vector_recall_skill",
        "ranking_skill",
    })

    # Mapping of skill names to their class names for dynamic dispatch
    _SKILL_DISPATCH = None

    @classmethod
    def _get_skill_dispatch(cls) -> Dict[str, str]:
        """Get the skill dispatch mapping (lazy initialization)."""
        if cls._SKILL_DISPATCH is None:
            cls._SKILL_DISPATCH = {
                "profile_skill": "ProfileSkill",
                "content_skill": "ContentSkill",
                "collab_skill": "CollabSkill",
                "merge_skill": "MergeSkill",
                "final_skill": "FinalSkill",
                "planner_skill": "PlannerSkill",
            }
        return cls._SKILL_DISPATCH

    def __init__(self, model: str = None):
        """
        Initialize the coordinator.

        Args:
            model: LLM model name. Defaults to config setting.
        """
        self.config = get_config()
        self.model = model or self.config.llm.model
        self.registry = SkillRegistry()

        # Register all skills
        self.registry.register("profile_skill", ProfileSkill)
        self.registry.register("content_skill", ContentSkill)
        self.registry.register("collab_skill", CollabSkill)
        self.registry.register("merge_skill", MergeSkill)
        self.registry.register("final_skill", FinalSkill)
        self.registry.register("planner_skill", PlannerSkill)
        self.registry.register("vector_recall_skill", VectorRecallSkill)
        self.registry.register("ranking_skill", RankingSkill)

        logger.info(f"SkillsCoordinator initialized with model={self.model}")

    def _validate_input(self, user_id: int, query: str) -> None:
        """
        Validate input parameters.

        Args:
            user_id: User ID to validate
            query: Query string to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(user_id, int):
            raise TypeError(f"user_id must be an integer, got {type(user_id).__name__}")
        if user_id <= 0:
            raise ValueError(f"user_id must be positive, got {user_id}")
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        if len(query.strip()) == 0:
            raise ValueError("query cannot be empty")

    def _execute_skill(self, skill_name: str, state: RecState) -> RecState:
        """
        Execute a skill dynamically (replaces if-elif chain).

        Args:
            skill_name: Name of the skill to execute
            state: Current state

        Returns:
            Updated state after skill execution
        """
        if skill_name not in self.VALID_SKILLS:
            logger.warning(f"Unknown skill '{skill_name}', skipping")
            return state

        logger.info(f"Executing skill: {skill_name}")

        skill_result = self.registry.execute_skill(
            skill_name,
            state,
            model=self.model
        )
        state.update(skill_result)
        return state

    def run_recommendation(self, user_id: int, query: str) -> List[Dict[str, Any]]:
        """
        Main method to run the recommendation process using skills coordination.

        Args:
            user_id: ID of the user to recommend movies for
            query: Natural language query describing preferences

        Returns:
            List of recommended movies with reasons

        Raises:
            ValueError: If input validation fails
            RuntimeError: If recommendation process fails
        """
        # Validate input
        self._validate_input(user_id, query)

        logger.info(f"Starting recommendation for user_id={user_id}, query='{query[:50]}...'")

        # Load data if needed
        load_movielens()
        build_movie_tfidf()

        # Initial state
        state: RecState = {
            "user_id": user_id,
            "query": query.strip(),
            "step_count": 0
        }

        # Run the coordination loop with dynamic dispatch
        max_steps = self.config.max_steps
        step = 0
        last_skill = None

        while step < max_steps:
            try:
                # Execute planner to decide next skill
                planner_result = self.registry.execute_skill(
                    "planner_skill",
                    state,
                    model=self.model
                )
                state.update(planner_result)

                # Get the next skill to execute
                next_skill = state.get("next_skill", "final_skill")

                # Validate next skill
                if next_skill not in self.VALID_SKILLS:
                    logger.warning(f"Invalid next_skill '{next_skill}', defaulting to final_skill")
                    next_skill = "final_skill"

                logger.info(f"Step {step + 1}: Executing skill '{next_skill}'")

                # Dynamic dispatch instead of if-elif chain
                state = self._execute_skill(next_skill, state)

                # Check if we should terminate
                if next_skill == "final_skill":
                    logger.info("Reached final_skill, ending recommendation process")
                    break

                last_skill = next_skill
                step += 1

            except Exception as e:
                logger.error(f"Error in step {step + 1}: {e}")
                if step >= self.config.max_planner_steps - 1:
                    logger.warning("Max steps reached, forcing final_skill")
                    state = self._execute_skill("final_skill", state)
                    break
                raise RuntimeError(f"Recommendation failed at step {step + 1}: {e}") from e

        # Return final recommendations
        recommendations = state.get("final_recommendations", [])
        logger.info(f"Recommendation complete, {len(recommendations)} movies recommended")
        return recommendations


def demo_run(
    user_id: int = 1,
    query: str = "我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道",
):
    """
    Demo function that maintains compatibility with the original API.
    """
    print(f"[DEMO] Starting demo_run, user_id={user_id}, query={query!r}")

    config = get_config()
    coordinator = SkillsCoordinator(model=config.llm.model)
    recs = coordinator.run_recommendation(user_id, query)

    print("\n================ 最终推荐结果 ================")
    for i, r in enumerate(recs, 1):
        print(
            f"{i}. {r['title']}  "
            f"[{r['genres']}]  (movie_id={r['movie_id']})\n"
            f"   推荐理由：{r['reason']}\n"
        )


if __name__ == "__main__":
    # Run demo with example user
    demo_run(user_id=123)
