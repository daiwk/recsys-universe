"""
Planner skill for coordinating other skills in the movie recommendation system.
"""
import json
import logging
from typing import Any, Dict, List

from .base_skill import BaseSkill
from config import get_config

logger = logging.getLogger(__name__)


class PlannerSkill(BaseSkill):
    """
    Planner skill (similar to DeepResearch style):
    - Reads current state (what information is ready)
    - Decides which skill to call next:
        - profile_skill
        - content_skill
        - collab_skill
        - merge_skill
        - final_skill (indicates ready to finalize recommendations)
    - Uses LLM to output JSON:
        {
          "next_skill": "...",
          "reason": "..."
        }
    - Also maintains step_count (limits max steps to prevent infinite loops)
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
        Execute the planning skill.

        Args:
            state: Current state of the recommendation process

        Returns:
            Updated state with next_skill and planner_reason
        """
        user_id = state.get("user_id")
        query = state.get("query", "") or ""
        user_profile = state.get("user_profile", "")
        user_history = state.get("user_history", []) or []
        content_cands = state.get("content_candidates", []) or []
        collab_cands = state.get("collab_candidates", []) or []
        merged_cands = state.get("merged_candidates", []) or []

        config = get_config()
        step_count = int(state.get("step_count", 0)) + 1
        max_steps = config.max_planner_steps

        logger.info(f"PlannerSkill: step {step_count}, deciding next action")

        # Prevent infinite loops
        if step_count > max_steps:
            logger.warning(f"Step count {step_count} exceeds max {max_steps}, forcing final_skill")
            return {
                "next_skill": "final_skill",
                "planner_reason": f"超过最大规划步数 ({max_steps})，强制收尾。",
                "step_count": step_count,
            }

        state_summary = {
            "user_id": user_id,
            "query": query,
            "has_user_profile": bool(user_profile.strip()),
            "user_history_len": len(user_history),
            "content_candidates_len": len(content_cands),
            "collab_candidates_len": len(collab_cands),
            "merged_candidates_len": len(merged_cands),
            "step_count": step_count,
        }

        system_prompt = (
            "你现在扮演一个多智能体推荐系统的【总控调度技能】。\n"
            "系统中有以下几个子技能（工具）：\n"
            "1）profile_skill：根据用户的历史观影记录，生成一段用户画像描述（user_profile）。\n"
            "2）content_skill：在已经有 user_profile 和用户 query 的前提下，\n"
            "   生成若干英文搜索短语，并调用 TF-IDF 检索得到内容相关的候选电影（content_candidates）。\n"
            "3）collab_skill：根据用户历史记录和全局评分，在用户偏好的类型中挑选高评分但未看过的电影（collab_candidates）。\n"
            "4）merge_skill：将 content_candidates 和 collab_candidates 合并去重，并得到 merged_candidates。\n"
            "5）final_skill：在有 user_profile 和 merged_candidates 的前提下，\n"
            "   由大模型选择最终要推荐的 5 部电影并给出理由（final_recommendations）。\n\n"
            "你的任务：\n"
            "  - 根据当前状态（state_summary），决定下一步应该调用哪一个技能，\n"
            "    或者直接调用 final_skill 收尾。\n"
            "  - 一般流程建议是：优先确保有 user_profile -> 再补全候选（content + collab）-> merge -> 最后 final_skill。\n"
            "  - 但是如果已经有足够多的候选，或者步骤过多，也可以提前进入 final_skill。\n"
            "  - 你只能在如下集合中选择一个 next_skill：\n"
            "      [\"profile_skill\", \"content_skill\", \"collab_skill\", \"merge_skill\", \"final_skill\"]。\n\n"
            "输出要求（非常重要）：\n"
            "  - 只能输出一个 JSON 对象，不要带其它文本。\n"
            "  - JSON 格式如下：\n"
            "    {\n"
            "      \"next_skill\": \"profile_skill\" 或 \"content_skill\" 或 \"collab_skill\" 或 \"merge_skill\" 或 \"final_skill\",\n"
            "      \"reason\": \"用中文简要说明你做这个选择的原因\"\n"
            "    }"
        )

        user_prompt = (
            "当前系统状态摘要（state_summary）如下：\n"
            + json.dumps(state_summary, ensure_ascii=False, indent=2)
            + "\n\n请根据上述信息，输出下一步应该执行的技能以及原因。"
        )

        logger.debug(f"PlannerSkill: state_summary = {state_summary}")
        raw = self.call_llm(system_prompt, user_prompt, tag="PLANNER_SKILL")

        # Parse JSON with validation
        next_skill, reason = self._parse_planner_response(raw)

        logger.info(f"PlannerSkill: decided next_skill={next_skill}, reason={reason}")

        return {
            "next_skill": next_skill,
            "planner_reason": reason,
            "step_count": step_count,
        }

    def _parse_planner_response(self, raw: str) -> tuple[str, str]:
        """
        Parse planner response and validate next_skill.

        Args:
            raw: Raw LLM response

        Returns:
            Tuple of (next_skill, reason)
        """
        next_skill = "final_skill"
        reason = "解析失败，直接进入 final_skill。"

        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                json_str = raw[start : end + 1]
                obj = json.loads(json_str)
                cand = obj.get("next_skill", "").strip()

                if cand in self.VALID_SKILLS:
                    next_skill = cand
                else:
                    logger.warning(f"Invalid next_skill '{cand}', defaulting to final_skill")

                reason = obj.get("reason", reason)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse planner response: {e}")

        return next_skill, reason
