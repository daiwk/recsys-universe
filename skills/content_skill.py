"""
Content skill for the recommendation system.
Supports both Legacy (TF-IDF) and Industrial (Vector Recall) modes.
"""
import json
import os
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from .data_utils import rag_search_movies
from serving.recall_service import RecallService

logger = logging.getLogger(__name__)


class ContentSkill(BaseSkill):
    """
    Skill 2: Content Retrieval
    - Legacy mode: Uses TF-IDF / BM25 for content matching
    - Industrial mode: Uses Two-Tower + Milvus for vector recall
    """

    def __init__(self, model: str = None):
        """
        Initialize content skill.

        Args:
            model: LLM model name (for query rewriting in Legacy mode)
        """
        super().__init__(model)
        self.recall_service = None
        self._is_industrial = os.environ.get("RECSYS_ARCHITECTURE", "industrial") == "industrial"

    def _get_recall_service(self) -> RecallService:
        """Get or create recall service (Industrial mode)."""
        if self.recall_service is None:
            from config import get_config
            config = get_config()
            self.recall_service = RecallService(config)
            self.recall_service.initialize()
        return self.recall_service

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute content retrieval skill.

        Args:
            state: Current state containing user_id and query

        Returns:
            Updated state with content_candidates
        """
        if self._is_industrial:
            return self._execute_industrial(state)
        else:
            return self._execute_legacy(state)

    def _execute_industrial(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector-based content retrieval (Industrial mode)."""
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        query = state.get("query", "") or ""

        logger.info(f"ContentSkill [Industrial]: vector recall for user_id={user_id}")

        if user_id is None:
            raise ValueError("user_id is required for ContentSkill")

        service = self._get_recall_service()
        top_k = config.content_top_k
        item_ids, scores = service.recall(user_id, top_k=top_k)

        # Build candidates
        candidates = []
        for item_id, score in zip(item_ids, scores):
            item_basic = service.item_features.store.get_basic_features(item_id)
            candidates.append({
                "movie_id": item_id,
                "recall_score": float(score),
                "title": item_basic.get("title", f"Item {item_id}"),
                "genres": item_basic.get("genres", []),
            })

        logger.info(f"ContentSkill [Industrial]: {len(candidates)} candidates")
        return {"content_candidates": candidates}

    def _execute_legacy(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute TF-IDF content retrieval (Legacy mode)."""
        import json as json_lib

        from config import get_config
        config = get_config()

        query = state.get("query", "") or ""
        user_profile = state.get("user_profile", "") or ""

        logger.info(f"ContentSkill [Legacy]: TF-IDF for query='{query[:50]}...'")

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
            + json_lib.dumps(user_payload, ensure_ascii=False, indent=2)
            + "\n\n请按照要求输出 JSON 数组格式的搜索短语。"
        )

        raw = self.call_llm(system_prompt, user_prompt, tag="CONTENT_SKILL")
        queries = self._parse_queries(raw, query)

        all_cands: Dict[int, Dict[str, Any]] = {}
        for q in queries:
            results, _ = rag_search_movies(q, k=config.content_top_k)
            logger.debug(f"Query '{q}' returned {len(results)} results")
            for rec in results:
                mid = int(rec["movie_id"])
                all_cands[mid] = rec

        logger.info(f"ContentSkill [Legacy]: {len(all_cands)} unique candidates")
        return {"content_candidates": list(all_cands.values())}

    def _parse_queries(self, raw: str, fallback: str) -> List[str]:
        """Parse JSON array from LLM response (Legacy mode)."""
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                json_str = raw[start : end + 1]
                queries = json.loads(json_str)
                if isinstance(queries, list) and queries:
                    return queries
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse queries: {e}")

        return [fallback] if fallback else ["popular movies"]


class ColdStartContentSkill(BaseSkill):
    """
    Skill: Cold Start Content Retrieval
    - Handles new users with no history
    - Returns popular items as fallback
    """

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cold start content retrieval."""
        from config import get_config
        config = get_config()

        user_id = state.get("user_id")
        logger.info(f"ColdStartContentSkill: handling cold start for user_id={user_id}")

        # Try to get popular items from recall service
        popular_items = list(range(1, config.content_top_k + 1))

        candidates = []
        for rank, item_id in enumerate(popular_items, 1):
            candidates.append({
                "movie_id": item_id,
                "recall_score": 0.5,
                "title": f"Popular Movie {item_id}",
                "genres": [],
            })

        return {
            "content_candidates": candidates,
            "is_cold_start": True,
        }
