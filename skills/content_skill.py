"""
Content skill for semantic movie search in the recommendation system.
"""
import json
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from .data_utils import rag_search_movies
from config import get_config

logger = logging.getLogger(__name__)


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
        config = get_config()

        logger.info(f"ContentSkill: processing query='{query[:50]}...'")

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

        raw = self.call_llm(system_prompt, user_prompt, tag="CONTENT_SKILL")

        # Try to parse JSON list with better error handling
        queries = self._parse_queries(raw, query)

        all_cands: Dict[int, Dict[str, Any]] = {}
        for q in queries:
            results, _ = rag_search_movies(q, k=config.content_top_k)
            logger.debug(f"Query '{q}' returned {len(results)} results")
            for rec in results:
                mid = int(rec["movie_id"])
                all_cands[mid] = rec

        logger.info(f"ContentSkill: {len(all_cands)} unique candidates after merging")

        return {"content_candidates": list(all_cands.values())}

    def _parse_queries(self, raw: str, fallback: str) -> List[str]:
        """
        Parse JSON array from LLM response.

        Args:
            raw: Raw LLM response
            fallback: Fallback query if parsing fails

        Returns:
            List of query strings
        """
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                json_str = raw[start : end + 1]
                queries = json.loads(json_str)
                if isinstance(queries, list) and queries:
                    logger.debug(f"Parsed queries: {queries}")
                    return queries
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse queries: {e}")

        logger.warning(f"Using fallback query: {fallback}")
        return [fallback] if fallback else ["popular movies"]
