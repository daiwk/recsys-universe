"""
multi_agent_llm_rec_deepresearch_zh.py

多-Agent + 自动规划（类似 DeepResearch 风格）的 LLM 推荐 Demo
现在使用 Claude Skills 架构实现，保持原有 API 接口不变

- 数据集：MovieLens-1M（自动下载）
- LLM：本地 OpenAI 兼容接口，模型名默认 "qwen3:1.7b"
    需要环境变量：
        os.environ["OPENAI_API_KEY"] = ""  # 任意非空/空都行，只要客户端不校验
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:11434/v1"

- 流程：
    使用 Skills 架构替代 LangGraph，通过 PlannerSkill 协调其他技能
    保持与原 API 完全兼容

- 所有 prompt 都是中文，debug 日志比较丰富，方便观测调用路径和状态变化。
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List
from typing_extensions import TypedDict

from skills_coordinator import SkillsCoordinator, RecState

# ================
# 0. Debug 工具
# ================

DEBUG = True


def debug_log(tag: str, msg: str):
    """简单的 debug 打印封装。"""
    if DEBUG:
        print(f"[DEBUG][{tag}] {msg}")


# =========================
# 1. 主要推荐接口函数
# =========================

def run_recommendation(user_id: int, query: str) -> List[Dict[str, Any]]:
    """
    主推荐函数，保持与原API完全兼容
    现在使用Claude Skills架构实现
    
    Args:
        user_id: 用户ID
        query: 用户的推荐请求查询
        
    Returns:
        推荐电影列表，每个元素包含movie_id, title, genres, reason
    """
    debug_log("MAIN", f"开始推荐，user_id={user_id}, query={query!r}")
    
    coordinator = SkillsCoordinator(model="Qwen/Qwen3-1.7B")
    result = coordinator.run_recommendation(user_id, query)
    
    debug_log("MAIN", f"推荐完成，结果数量={len(result)}")
    return result


# =========================
# 2. 保持向后兼容的接口
# =========================

def demo_run(
    user_id: int = 1,
    query: str = "我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道",
):
    """
    示例运行函数，保持与原API完全兼容
    """
    debug_log("DEMO", f"开始 demo_run，user_id={user_id}, query={query!r}")
    
    recs = run_recommendation(user_id, query)
    
    print("\n================ 最终推荐结果 ================")
    for i, r in enumerate(recs, 1):
        print(
            f"{i}. {r['title']}  "
            f"[{r['genres']}]  (movie_id={r['movie_id']})\n"
            f"   推荐理由：{r['reason']}\n"
        )


if __name__ == "__main__":
    # 随便挑一个 user_id + 中文 query 跑一下
    # demo_run(user_id=1)
    demo_run(user_id=123)