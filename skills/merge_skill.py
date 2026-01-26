"""
Merge skill for combining candidate sets in the movie recommendation system.
"""
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from config import get_config

logger = logging.getLogger(__name__)


class MergeSkill(BaseSkill):
    """
    Skill 4: Merge candidate sets (non-LLM)
    - Merges content_candidates and collab_candidates
    - Simple sort by "has rating + rating size" and truncate
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the merge skill.

        Args:
            state: Current state containing content_candidates and collab_candidates

        Returns:
            Updated state with merged_candidates
        """
        content = state.get("content_candidates", []) or []
        collab = state.get("collab_candidates", []) or []
        config = get_config()

        logger.info(f"MergeSkill: merging {len(content)} content + {len(collab)} collab candidates")

        merged: Dict[int, Dict[str, Any]] = {}

        # Content candidates
        for rec in content:
            mid = int(rec["movie_id"])
            r = dict(rec)
            r["source"] = r.get("source", "") + "|content"
            merged[mid] = r

        # Collaborative candidates
        for rec in collab:
            mid = int(rec["movie_id"])
            if mid in merged:
                merged[mid]["source"] = merged[mid].get("source", "") + "|collab"
                if "rating" in rec and "rating" not in merged[mid]:
                    merged[mid]["rating"] = rec["rating"]
            else:
                r = dict(rec)
                r["source"] = r.get("source", "") + "|collab"
                merged[mid] = r

        merged_list = list(merged.values())

        def sort_key(x):
            return (0 if "rating" in x else 1, -float(x.get("rating", 0.0)))

        merged_list.sort(key=sort_key)
        logger.debug(f"After deduplication: {len(merged_list)} candidates")

        # Truncate to configured limit
        max_candidates = config.merge_top_k
        merged_list = merged_list[:max_candidates]
        logger.info(f"MergeSkill: returning top {len(merged_list)} merged candidates")

        return {"merged_candidates": merged_list}
