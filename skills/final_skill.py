"""
Final recommendation skill for the movie recommendation system
"""
import json
from typing import Any, Dict, List
from .base_skill import BaseSkill
from .data_utils import load_movielens


class FinalSkill(BaseSkill):
    """
    Skill 5: Main decision LLM skill (Chinese)
    - Input: user profile + merged candidate movies
    - Output: Top-K recommendations (with Chinese reasons) and parse JSON
    """
    
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

        # Construct candidate list for LLM (simplified fields)
        for_prompt = []
        for rec in merged:
            for_prompt.append(
                {
                    "movie_id": int(rec["movie_id"]),
                    "title": rec["title"],
                    "genres": rec["genres"],
                    "source": rec.get("source", ""),
                    "rating_hint": rec.get("rating", None),
                }
            )

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

        self.debug_log(
            "FINAL_SKILL",
            f"调用最终 LLM 进行推荐决策，候选数={len(merged)}, user_query={query!r}",
        )

        raw = self.call_llm(system_prompt, user_prompt, tag="FINAL_SKILL")

        # Try to parse JSON object
        recs: List[Dict[str, Any]] = []
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            json_str = raw[start : end + 1]
            obj = json.loads(json_str)
            recs = obj.get("recommendations", [])
            if not isinstance(recs, list):
                raise ValueError("recommendations 字段不是列表")
            self.debug_log("FINAL_SKILL", f"解析到的推荐条数={len(recs)}")
        except Exception as e:
            self.debug_log("FINAL_SKILL", f"解析最终 JSON 失败，raw={raw[:300]!r}, error={e}")
            recs = []

        movies, _ = load_movielens()
        movie_map = movies.set_index("movie_id")[["title", "genres"]].to_dict("index")

        final_recs = []
        for r in recs:
            try:
                mid = int(r.get("movie_id"))
            except Exception:
                continue
            info = movie_map.get(mid, {})
            final_recs.append(
                {
                    "movie_id": mid,
                    "title": info.get("title", ""),
                    "genres": info.get("genres", ""),
                    "reason": r.get("reason", ""),
                }
            )

        self.debug_log(
            "FINAL_SKILL",
            f"最终组装出的推荐结果条数={len(final_recs)}, 示例前 3 条={[fr['title'] for fr in final_recs[:3]]}",
        )

        return {"final_recommendations": final_recs}