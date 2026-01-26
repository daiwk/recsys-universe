"""
Final recommendation skill for the movie recommendation system.
"""
import json
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from .data_utils import load_movielens
from config import get_config

logger = logging.getLogger(__name__)


class FinalSkill(BaseSkill):
    """
    Skill 5: Main decision LLM skill (Chinese)
    - Input: user profile + merged candidate movies
    - Output: Top-K recommendations (with Chinese reasons) and parse JSON
    """

    # Valid next skills that can be returned
    VALID_SKILLS = frozenset({
        "profile_skill",
        "content_skill",
        "collab_skill",
        "merge_skill",
        "final_skill"
    })

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the final recommendation skill.

        Args:
            state: Current state containing user_profile and merged_candidates

        Returns:
            Updated state with final_recommendations
        """
        user_profile = state.get("user_profile", "") or ""
        query = state.get("query", "") or ""
        merged = state.get("merged_candidates", []) or []

        logger.info(f"FinalSkill: generating recommendations from {len(merged)} candidates")

        # Construct candidate list for LLM (simplified fields)
        for_prompt = []
        for rec in merged:
            for_prompt.append({
                "movie_id": int(rec["movie_id"]),
                "title": rec["title"],
                "genres": rec["genres"],
                "source": rec.get("source", ""),
                "rating_hint": rec.get("rating", None),
            })

        system_prompt = (
            "你是一个专业的电影推荐引擎，输出语言为中文。\n"
            "现在你会收到：\n"
            "1）一段用户的中文画像描述；\n"
            "2）用户当前的中文需求描述；\n"
            "3）一批候选电影，每个候选包含 movie_id、title、genres、来源 source 和一个可选的 rating_hint。\n\n"
            "你的任务：\n"
            "1. 从这些候选中选出最适合该用户的 5 部电影，既要相关性高，又要注意题材/风格上的一定多样性。\n"
            "2. 为每个推荐给出一两句话的中文解释说明，解释要和用户画像、电影特点对应上。\n"
            "3. 返回结果必须是一个 JSON 对象，格式严格如下：\n"
            "{\n"
            '  "recommendations": [\n'
            "    {\n"
            "      \"movie_id\": 整数,\n"
            "      \"reason\": \"简短的中文推荐理由\"\n"
            "    },\n"
            "    ... 一共 5 个对象\n"
            "  ]\n"
            "}\n"
            "不要额外输出其它自然语言说明，只输出 JSON。"
        )

        user_payload = {
            "user_profile": user_profile,
            "explicit_query": query,
            "candidates": for_prompt,
        }

        user_prompt = (
            "下面是用户画像、当前需求和候选电影列表，请根据说明返回 JSON：\n"
            + json.dumps(user_payload, ensure_ascii=False, indent=2)
        )

        logger.debug(f"FinalSkill: calling LLM with {len(merged)} candidates")
        raw = self.call_llm(system_prompt, user_prompt, tag="FINAL_SKILL")

        # Parse JSON with better error handling
        recs = self._parse_recommendations(raw)

        movies, _ = load_movielens()
        movie_map = movies.set_index("movie_id")[["title", "genres"]].to_dict("index")

        final_recs = []
        for r in recs:
            try:
                mid = int(r.get("movie_id"))
            except (ValueError, TypeError):
                logger.warning(f"Invalid movie_id: {r.get('movie_id')}")
                continue

            info = movie_map.get(mid, {})
            final_recs.append({
                "movie_id": mid,
                "title": info.get("title", ""),
                "genres": info.get("genres", ""),
                "reason": r.get("reason", ""),
            })

        logger.info(f"FinalSkill: generated {len(final_recs)} recommendations")
        return {"final_recommendations": final_recs}

    def _parse_recommendations(self, raw: str) -> List[Dict[str, Any]]:
        """
        Parse recommendations from LLM response.

        Args:
            raw: Raw LLM response

        Returns:
            List of recommendation dictionaries
        """
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                json_str = raw[start : end + 1]
                obj = json.loads(json_str)
                recs = obj.get("recommendations", [])
                if isinstance(recs, list) and recs:
                    logger.debug(f"Parsed {len(recs)} recommendations")
                    return recs
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse recommendations: {e}")

        logger.warning("No valid recommendations found, returning empty list")
        return []
