"""
Content skill for semantic movie search in the recommendation system
"""
import json
from typing import Any, Dict, List
from .base_skill import BaseSkill
from .data_utils import rag_search_movies


class ContentSkill(BaseSkill):
    """
    Skill 2: Content / Semantic Retrieval
    - Uses user query + user_profile to generate search queries (in English)
    - Uses rag_search_movies() to find candidate movies
    """
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the content retrieval skill.
        
        Args:
            state: Current state containing query and user_profile
            
        Returns:
            Updated state with content_candidates
        """
        query = state.get("query", "") or ""
        user_profile = state.get("user_profile", "") or ""

        system_prompt = (
            "你是一个电影搜索查询改写代理。\n"
            "已知用户的一段中文画像描述，以及用户当前的中文需求描述（query），\n"
            "请帮我生成 2-3 个简短的英文搜索短语，用于在电影标题和类型字段上做 TF-IDF / BM25 检索。\n"
            "例如可以是：\"dark sci-fi thriller\", \"romantic comedy\", \"cyberpunk action\" 等。\n"
            "要求：\n"
            "1. 尽量结合用户画像和当前 query 中提到的偏好。\n"
            "2. 返回必须是 JSON 数组格式，例如：[\"dark sci-fi thriller\", \"time travel movies\" ]，不要包含额外解释。"
        )

        user_payload = {
            "user_profile": user_profile,
            "user_query": query,
        }
        user_prompt = (
            "下面是用户的中文画像和当前需求：\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\n请按照要求输出 JSON 数组格式的搜索短语。"
        )

        self.debug_log("CONTENT_SKILL", f"开始生成搜索 query，用户 query={query!r}")

        raw = self.call_llm(system_prompt, user_prompt, tag="CONTENT_SKILL")

        # Try to parse JSON list
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            json_str = raw[start : end + 1]
            queries = json.loads(json_str)
            if not isinstance(queries, list):
                raise ValueError("解析结果不是列表")
            self.debug_log("CONTENT_SKILL", f"解析出的搜索 query 列表={queries}")
        except Exception as e:
            self.debug_log("CONTENT_SKILL", f"解析搜索 query 失败，raw={raw[:200]!r}, error={e}")
            queries = [query] if query else ["popular movies"]

        all_cands: Dict[int, Dict[str, Any]] = {}
        for q in queries:
            recs = rag_search_movies(q, k=15)
            self.debug_log(
                "CONTENT_SKILL",
                f"基于搜索短语 {q!r} 检索到候选数={len(recs)}",
            )
            for rec in recs:
                mid = int(rec["movie_id"])
                all_cands[mid] = rec

        self.debug_log(
            "CONTENT_SKILL",
            f"内容检索总去重候选数={len(all_cands)}（合并所有搜索短语）",
        )

        return {"content_candidates": list(all_cands.values())}