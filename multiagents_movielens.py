"""
Multi-Agent Movie Recommendation System

多智能体电影推荐系统，支持两种架构模式：

1. Legacy (TF-IDF): 基于 TF-IDF 内容检索的传统 Skills 架构
2. Industrial (推荐系统): 基于双塔召回 + 精排的工业级架构

- 数据集：MovieLens-1M（自动下载）
- Legacy 模式需要 LLM：本地 OpenAI 兼容接口，模型名默认 "Qwen/Qwen3-1.7B"
- Industrial 模式：双塔模型 + FAISS 向量检索

环境变量：
  - RECSYS_ARCHITECTURE: "legacy" 或 "industrial"（默认）
  - OPENAI_API_KEY: Legacy 模式需要的 API Key
  - OPENAI_BASE_URL: Legacy 模式需要的 API 地址
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_recommendation(
    user_id: int,
    query: str = None,
    industrial: bool = None
) -> List[Dict[str, Any]]:
    """
    主推荐函数，根据配置自动选择架构模式。

    Args:
        user_id: 用户ID
        query: 用户的推荐请求查询（Legacy 模式必需）
        industrial: 是否使用工业级架构（默认根据配置）

    Returns:
        推荐电影列表，每个元素包含 movie_id, title, genres, reason
    """
    config = get_config()

    # Determine architecture mode
    if industrial is None:
        industrial = config.architecture_mode == "industrial"

    logger.info(f"开始推荐，user_id={user_id}, industrial={industrial}")

    if industrial:
        # Industrial pipeline: use Skills framework with industrial recall/rank
        from skills_coordinator import SkillsCoordinator
        coordinator = SkillsCoordinator(model=config.llm.model)
        return coordinator.run_recommendation(user_id, query)
    else:
        # Legacy pipeline (TF-IDF + LLM)
        from skills_coordinator import SkillsCoordinator
        coordinator = SkillsCoordinator(model=config.llm.model)
        return coordinator.run_recommendation(user_id, query)


def run_industrial_recommendation(user_id: int) -> List[Dict[str, Any]]:
    """
    工业级推荐接口（不需要 LLM）。

    Args:
        user_id: 用户ID

    Returns:
        推荐电影列表，包含召回分和 CTR 预测分
    """
    from industrial_coordinator import run_industrial_recommendation
    return run_industrial_recommendation(user_id)


def demo_run(
    user_id: int = 1,
    query: str = "我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道",
    industrial: bool = None
):
    """
    示例运行函数，自动根据配置选择架构。
    """
    config = get_config()

    if industrial is None:
        industrial = config.architecture_mode == "industrial"

    print(f"[DEMO] 开始 demo_run，user_id={user_id}, industrial={industrial}")
    print(f"[DEMO] 架构模式: {'工业级 (双塔+精排)' if industrial else '传统 (TF-IDF+LLM)'}")

    recs = run_recommendation(user_id, query, industrial=industrial)

    print("\n" + "=" * 50)
    if industrial:
        print("推荐结果 (工业级)")
        print("=" * 50)
        for r in recs:
            print(
                f"{r['rank']}. {r['title']}\n"
                f"   类型: [{r['genres']}]\n"
                f"   召回分: {r['recall_score']:.3f} | CTR预测: {r['ctr_score']:.2%}\n"
                f"   推荐理由：{r['reason']}\n"
            )
    else:
        print("推荐结果 (传统)")
        print("=" * 50)
        for i, r in enumerate(recs, 1):
            print(
                f"{i}. {r['title']}  "
                f"[{r['genres']}]  (movie_id={r['movie_id']})\n"
                f"   推荐理由：{r['reason']}\n"
            )


def demo_industrial_only():
    """
    纯工业级演示（不需要 LLM）。
    """
    print("[DEMO] 运行纯工业级推荐流程...")

    from industrial_coordinator import IndustrialSkillsCoordinator
    from scripts.ingest_data import (
        load_movielens_data,
        ingest_to_redis,
        generate_item_embeddings,
        build_user_embeddings,
    )
    from config import get_config
    from features.base import MemoryFeatureStore
    from features.user_features import UserFeatures
    from features.item_features import ItemFeatures
    from models.two_tower import TwoTowerModel
    from serving.faiss_client import get_faiss_client

    # 检查是否需要导入数据
    print("[DEMO] 检查数据状态...")

    config = get_config()
    feature_store = MemoryFeatureStore(config)
    user_features = UserFeatures(feature_store)
    item_features = ItemFeatures(feature_store)

    # 初始化模型和 FAISS 客户端（后面会根据需要使用）
    two_tower = None
    faiss = None

    # 检查是否有数据
    sample_user = user_features.store.get_user_features(1)
    if sample_user is None:
        print("[DEMO] 数据未导入，开始导入...")
        try:
            data = load_movielens_data()
            print(f"[DEMO] 加载数据: {len(data['users'])} 用户, {len(data['items'])} 物品, {len(data['ratings'])} 评分")

            # 导入数据
            ingest_to_redis(data, feature_store, user_features, item_features)
            print("[DEMO] 数据导入完成")

            # 生成 embeddings
            print("[DEMO] 生成 embeddings...")
            two_tower = TwoTowerModel(config)
            faiss = get_faiss_client(config.recall.faiss)
            generate_item_embeddings(data, two_tower, item_features, faiss)
            build_user_embeddings(data, two_tower, user_features)
            print("[DEMO] embeddings 生成完成")
        except FileNotFoundError:
            print("[DEMO] 错误: 未找到 MovieLens 数据。请先运行: python scripts/ingest_data.py --memory")
            return
    else:
        print("[DEMO] 数据已存在，跳过导入")
        # 数据已存在时，也需要初始化模型和 FAISS
        two_tower = TwoTowerModel(config)
        faiss = get_faiss_client(config.recall.faiss)

    # 现在创建 coordinator，传入共享的组件
    print("\n[DEMO] 初始化推荐服务...")
    coordinator = IndustrialSkillsCoordinator(
        feature_store=feature_store,
        two_tower=two_tower,
        ranking_model=None,
        faiss_client=faiss
    )

    # 检查健康状态
    health = coordinator.health_check()
    print(f"服务状态: {health}")

    # 运行推荐
    user_id = 123
    recs = coordinator.run_recommendation(user_id, industrial=True)

    print("\n" + "=" * 50)
    print(f"用户 {user_id} 的推荐结果")
    print("=" * 50)
    for r in recs:
        print(
            f"{r['rank']}. {r['title']}\n"
            f"   类型: [{r['genres']}]\n"
            f"   召回分: {r['recall_score']:.3f} | CTR预测: {r['ctr_score']:.2%}\n"
        )


if __name__ == "__main__":
    import sys

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--industrial":
            # Force industrial mode
            demo_run(industrial=True)
        elif sys.argv[1] == "--legacy":
            # Force legacy mode
            demo_run(industrial=False)
        elif sys.argv[1] == "--demo-industrial":
            # Demo industrial only
            demo_industrial_only()
        else:
            print("Usage: python multiagents_movielens.py [--industrial|--legacy|--demo-industrial]")
            sys.exit(1)
    else:
        # Auto mode based on config
        demo_run()
