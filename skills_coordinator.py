"""
Skills coordinator for movie recommendation system
Implements Claude-style skills coordination while maintaining original API compatibility
"""
from typing import Any, Dict, List
from typing_extensions import TypedDict
from skills import (
    ProfileSkill,
    ContentSkill,
    CollabSkill,
    MergeSkill,
    FinalSkill,
    PlannerSkill,
    SkillRegistry
)
from skills.data_utils import load_movielens, build_movie_tfidf


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
    
    def __init__(self, model: str = "qwen3:1.7b"):
        self.model = model
        self.registry = SkillRegistry()
        
        # Register all skills
        self.registry.register("profile_skill", ProfileSkill)
        self.registry.register("content_skill", ContentSkill)
        self.registry.register("collab_skill", CollabSkill)
        self.registry.register("merge_skill", MergeSkill)
        self.registry.register("final_skill", FinalSkill)
        self.registry.register("planner_skill", PlannerSkill)
    
    def run_recommendation(self, user_id: int, query: str) -> List[Dict[str, Any]]:
        """
        Main method to run the recommendation process using skills coordination.
        Maintains the same API as the original implementation.
        
        Args:
            user_id: ID of the user to recommend movies for
            query: Natural language query describing preferences
            
        Returns:
            List of recommended movies with reasons
        """
        # Load data if needed
        load_movielens()
        build_movie_tfidf()
        
        # Initial state
        state: RecState = {
            "user_id": user_id,
            "query": query,
            "step_count": 0
        }
        
        # Run the coordination loop
        max_steps = 10  # Prevent infinite loops
        step = 0
        
        while step < max_steps:
            # Execute planner to decide next skill
            planner_result = self.registry.execute_skill(
                "planner_skill", 
                state, 
                model=self.model
            )
            state.update(planner_result)
            
            # Get the next skill to execute
            next_skill = state.get("next_skill", "final_skill")
            
            print(f"[COORDINATOR] Step {step + 1}: Executing skill '{next_skill}'")
            
            # Execute the chosen skill
            if next_skill == "profile_skill":
                skill_result = self.registry.execute_skill(
                    "profile_skill", 
                    state, 
                    model=self.model
                )
                state.update(skill_result)
                
            elif next_skill == "content_skill":
                skill_result = self.registry.execute_skill(
                    "content_skill", 
                    state, 
                    model=self.model
                )
                state.update(skill_result)
                
            elif next_skill == "collab_skill":
                skill_result = self.registry.execute_skill(
                    "collab_skill", 
                    state, 
                    model=self.model
                )
                state.update(skill_result)
                
            elif next_skill == "merge_skill":
                skill_result = self.registry.execute_skill(
                    "merge_skill", 
                    state, 
                    model=self.model
                )
                state.update(skill_result)
                
            elif next_skill == "final_skill":
                skill_result = self.registry.execute_skill(
                    "final_skill", 
                    state, 
                    model=self.model
                )
                state.update(skill_result)
                break  # End the loop after final skill
                
            else:
                print(f"[COORDINATOR] Unknown skill '{next_skill}', ending process")
                break
                
            step += 1
        
        # Return final recommendations
        return state.get("final_recommendations", [])


def demo_run(
    user_id: int = 1,
    query: str = "我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道",
):
    """
    Demo function that maintains compatibility with the original API
    """
    print(f"[DEMO] Starting demo_run, user_id={user_id}, query={query!r}")
    
    coordinator = SkillsCoordinator(model="qwen3:1.7b")
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