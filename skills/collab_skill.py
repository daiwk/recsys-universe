"""
Collaborative filtering skill for the movie recommendation system
"""
from typing import Any, Dict, List
from .base_skill import BaseSkill
from .data_utils import get_collab_candidates_by_genre


class CollabSkill(BaseSkill):
    """
    Skill 3: Simple "collaborative filtering" (actually genre + global rating heuristic)
    - Calls get_collab_candidates_by_genre tool
    """
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the collaborative filtering skill.
        
        Args:
            state: Current state containing user_id
            
        Returns:
            Updated state with collab_candidates
        """
        user_id = state["user_id"]
        cands = get_collab_candidates_by_genre(user_id, k=30)
        self.debug_log(
            "COLLAB_SKILL",
            f"user_id={user_id} 协同候选数={len(cands)}",
        )
        return {"collab_candidates": cands}