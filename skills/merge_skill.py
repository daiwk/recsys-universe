"""
Merge skill for combining candidate sets in the movie recommendation system
"""
from typing import Any, Dict, List
from .base_skill import BaseSkill


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

        self.debug_log(
            "MERGE_SKILL",
            f"准备合并候选：content={len(content)}, collab={len(collab)}",
        )

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
        self.debug_log(
            "MERGE_SKILL",
            f"候选去重后共 {len(merged_list)} 条，示例前 3 条={[(m['title'], m.get('rating')) for m in merged_list[:3]]}",
        )

        # Truncate to first 40 to control prompt length
        merged_list = merged_list[:40]
        self.debug_log("MERGE_SKILL", f"截断后保留前 {len(merged_list)} 条候选用于最终决策")

        return {"merged_candidates": merged_list}