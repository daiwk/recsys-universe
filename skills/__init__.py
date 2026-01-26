"""
Skills framework for movie recommendation system

Skills automatically adapt to the architecture mode set by RECSYS_ARCHITECTURE:
- Legacy: ContentSkill uses TF-IDF, CollabSkill uses CF heuristic
- Industrial: ContentSkill uses Vector Recall, CollabSkill uses DNN Ranking
"""
from .base_skill import BaseSkill
from .profile_skill import ProfileSkill
from .content_skill import ContentSkill
from .collab_skill import CollabSkill
from .merge_skill import MergeSkill
from .final_skill import FinalSkill
from .planner_skill import PlannerSkill
from .skill_registry import SkillRegistry

__all__ = [
    'BaseSkill',
    'ProfileSkill',
    'ContentSkill',
    'CollabSkill',
    'MergeSkill',
    'FinalSkill',
    'PlannerSkill',
    'SkillRegistry'
]
