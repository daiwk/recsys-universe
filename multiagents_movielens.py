"""
multi_agent_llm_rec_deepresearch_zh.py

多-Agent + 自动规划（类似 DeepResearch 风格）的 LLM 推荐 Demo

- 数据集：MovieLens-1M（自动下载）
- LLM：本地 OpenAI 兼容接口，模型名默认 "qwen3:1.7b"
    需要环境变量：
        os.environ["OPENAI_API_KEY"] = ""  # 任意非空/空都行，只要客户端不校验
        os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:11434/v1"

- 流程：
    START -> planner_agent
    planner_agent 根据当前 state 决定下一步调用哪个 agent：
        - profile_agent      （生成用户画像）
        - content_agent      （内容侧检索候选）
        - collab_agent       （协同侧候选）
        - merge_agent        （候选合并）
        - final_llm_agent    （最终 Top-K 推荐）
    除 final_llm_agent 外，其余 agent 跑完都回到 planner_agent 继续规划；
    planner 有 step_count 上限，避免死循环。

- 所有 prompt 都是中文，debug 日志比较丰富，方便观测调用路径和状态变化。
"""

from __future__ import annotations

import os
import io
import zipfile
import json
from typing import Any, Dict, List
from typing_extensions import TypedDict

import requests
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

from langgraph.graph import StateGraph, START, END

# ================
# 0. Debug 工具
# ================

DEBUG = True


def debug_log(tag: str, msg: str):
    """简单的 debug 打印封装。"""
    if DEBUG:
        print(f"[DEBUG][{tag}] {msg}")


# ================
# 1. LLM Client
# ================

# 建议外部设置：
#   export OPENAI_API_KEY=""   # 可以为空字符串
#   export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1"),
)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "qwen3:1.7b",
    tag: str = "LLM",
) -> str:
    """
    封装 LLM 调用，所有 agent 共用。
    - system_prompt / user_prompt 都用中文说明；
    - model 默认 qwen3:1.7b（可按需改成你的模型名）。
    """
    debug_log(tag, f"调用 LLM，model={model}")
    # debug_log(tag, f"System prompt 前 200 字：{system_prompt[:200]}")
    # debug_log(tag, f"User prompt 前 200 字：{user_prompt[:200]}")
    debug_log(tag, f"System prompt：{system_prompt}")
    debug_log(tag, f"User prompt：{user_prompt}")


    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    content = resp.choices[0].message.content or ""
    # debug_log(tag, f"LLM 返回前 300 字：{content[:300]}")
    debug_log(tag, f"LLM 返回：{content}")
    return content


# ============================
# 2. 数据加载 & 简单 RAG 工具
# ============================

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
LOCAL_ZIP = "ml-1m.zip"

_movies_df: pd.DataFrame | None = None
_ratings_df: pd.DataFrame | None = None
_tfidf_vectorizer: TfidfVectorizer | None = None
_movie_tfidf_matrix = None


def download_movielens_if_needed() -> str:
    """如本地不存在 ml-1m.zip 则自动下载。"""
    if not os.path.exists(LOCAL_ZIP):
        debug_log("DATA", "开始下载 MovieLens 1M 数据集...")
        r = requests.get(MOVIELENS_URL, timeout=60)
        r.raise_for_status()
        with open(LOCAL_ZIP, "wb") as f:
            f.write(r.content)
        debug_log("DATA", "MovieLens 1M 下载完成。")
    else:
        debug_log("DATA", "检测到本地已存在 MovieLens 1M 压缩包，跳过下载。")
    return LOCAL_ZIP


def load_movielens() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    解析 ml-1m.zip 中的 movies.dat / ratings.dat

    movies.dat: MovieID::Title::Genres
    ratings.dat: UserID::MovieID::Rating::Timestamp
    """
    global _movies_df, _ratings_df
    if _movies_df is not None and _ratings_df is not None:
        return _movies_df, _ratings_df

    zip_path = download_movielens_if_needed()
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                sep="::",
                header=None,
                names=["movie_id", "title", "genres"],
                engine="python",
            )
        with zf.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(
                io.TextIOWrapper(f, encoding="latin-1"),
                sep="::",
                header=None,
                names=["user_id", "movie_id", "rating", "timestamp"],
                engine="python",
            )
    movies["movie_id"] = movies["movie_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    _movies_df, _ratings_df = movies, ratings
    debug_log("DATA", f"电影数={len(movies)}, 打分数={len(ratings)}")
    return movies, ratings


def build_movie_tfidf():
    """
    用 title + genres 构建一个简单 TF-IDF 索引，作为内容侧 RAG 检索工具。
    """
    global _tfidf_vectorizer, _movie_tfidf_matrix
    movies, _ = load_movielens()
    if _tfidf_vectorizer is not None:
        return

    debug_log("TFIDF", "开始构建电影 TF-IDF 索引...")
    docs = (movies["title"] + " " + movies["genres"].str.replace("|", " ")).tolist()
    vectorizer = TfidfVectorizer(max_features=5000)
    mat = vectorizer.fit_transform(docs)
    _tfidf_vectorizer = vectorizer
    _movie_tfidf_matrix = mat
    debug_log("TFIDF", f"TF-IDF 索引构建完成，电影数={len(docs)}, 维度={mat.shape[1]}")


def rag_search_movies(query: str, k: int = 20) -> List[Dict[str, Any]]:
    """
    RAG 工具：给定文本 query，返回最相似的 k 部电影。
    """
    global _movies_df, _tfidf_vectorizer, _movie_tfidf_matrix
    if _movies_df is None or _tfidf_vectorizer is None:
        build_movie_tfidf()

    movies = _movies_df
    vec = _tfidf_vectorizer.transform([query])
    sims = cosine_similarity(vec, _movie_tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    sub = movies.iloc[top_idx][["movie_id", "title", "genres"]].copy()
    sub["score"] = sims[top_idx]
    debug_log("RAG", f"基于查询 {query!r} 检索到 {len(sub)} 条电影候选")
    return sub.to_dict("records")


def get_user_history(user_id: int, n: int = 10) -> List[Dict[str, Any]]:
    """
    工具：返回某个用户打过分的电影（按评分从高到低取前 n 条）。
    """
    movies, ratings = load_movielens()
    user_ratings = (
        ratings[ratings["user_id"] == user_id]
        .sort_values("rating", ascending=False)
        .head(n)
    )
    merged = user_ratings.merge(movies, on="movie_id", how="left")
    debug_log("USER_HIST", f"user_id={user_id} 历史记录条数={len(merged)}")
    return merged[["movie_id", "title", "genres", "rating"]].to_dict("records")


def get_collab_candidates_by_genre(user_id: int, k: int = 30) -> List[Dict[str, Any]]:
    """
    非严格的“协同过滤风格”工具（极简启发式）：
    - 看这个用户历史里的高频 genres
    - 在这些 genres 里挑出全局评分较高且没看过的电影
    """
    movies, ratings = load_movielens()

    hist = get_user_history(user_id, n=50)
    if not hist:
        debug_log("COLLAB", f"user_id={user_id} 没有历史，直接返回全局 top 电影")
        mean_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
        merged = mean_rating.merge(movies, on="movie_id", how="left")
        merged = merged.sort_values("rating", ascending=False).head(k)
        return merged[["movie_id", "title", "genres", "rating"]].to_dict("records")

    from collections import Counter

    genre_counter = Counter()
    for h in hist:
        for g in str(h["genres"]).split("|"):
            genre_counter[g] += 1

    if not genre_counter:
        top_genres = []
    else:
        top_genres = [g for g, _ in genre_counter.most_common(3)]

    debug_log(
        "COLLAB",
        f"user_id={user_id} 偏好的前 3 个类型={top_genres}",
    )

    mean_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
    merged = mean_rating.merge(movies, on="movie_id", how="left")

    watched_ids = {h["movie_id"] for h in hist}

    def has_pref_genre(row) -> bool:
        gs = str(row["genres"]).split("|")
        return any(g in gs for g in top_genres)

    cand = merged[merged.apply(has_pref_genre, axis=1)]
    cand = cand[~cand["movie_id"].isin(watched_ids)]
    cand = cand.sort_values("rating", ascending=False).head(k)

    debug_log(
        "COLLAB",
        f"user_id={user_id} 在偏好类型中候选数={len(cand)}, 返回前 {k} 条",
    )

    return cand[["movie_id", "title", "genres", "rating"]].to_dict("records")


# =====================
# 3. LangGraph 的 State
# =====================

class RecState(TypedDict, total=False):
    """
    Graph 在节点间传递的状态。
    每个 agent（node）只读写自己关心的字段。
    """
    user_id: int
    query: str

    user_history: List[Dict[str, Any]]
    user_profile: str

    content_candidates: List[Dict[str, Any]]
    collab_candidates: List[Dict[str, Any]]

    merged_candidates: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]

    # 规划相关字段（DeepResearch 风格）
    next_agent: str
    planner_reason: str
    step_count: int


# ==========================
# 4. 各个“Worker Agent”节点
# ==========================

def profile_agent(state: RecState) -> RecState:
    """
    Agent 1：用户画像 Agent
    - 调用 get_user_history
    - 用 LLM 总结出中文用户画像 user_profile
    """
    user_id = state["user_id"]
    history = get_user_history(user_id, n=10)

    hist_text_lines = [
        f"- {h['title']} (类型: {h['genres']}, 用户评分: {h['rating']})"
        for h in history
    ]
    hist_text = "\n".join(hist_text_lines) if hist_text_lines else "这个用户目前没有任何观影历史记录。"

    system_prompt = (
        "你是一个电影推荐系统的分析师。\n"
        "给定一个用户过去观看并评分过的电影列表，请用 3-5 条要点，总结这个用户的观影偏好。\n"
        "请重点描述：偏好的题材类型、年代、风格（轻松/烧脑/黑暗/温情等），以及你观察到的模式。\n"
        "输出使用中文。"
    )
    user_prompt = (
        "下面是该用户评分最高的一些电影列表，每行包含电影名称、类型和评分：\n"
        f"{hist_text}\n\n"
        "请根据以上信息，用 3-5 条要点，帮我总结这个用户的电影喜好画像。"
    )

    debug_log("PROFILE_AGENT", f"开始生成用户画像，user_id={user_id}, 历史条数={len(history)}")

    profile = call_llm(system_prompt, user_prompt, tag="PROFILE_AGENT")

    return {
        "user_history": history,
        "user_profile": profile,
    }


def content_agent(state: RecState) -> RecState:
    """
    Agent 2：内容 / 语义检索 Agent
    - 利用用户 query + user_profile，让 LLM 生成 2~3 个搜索 query（英文短语）
    - 再用 rag_search_movies() 找到候选电影
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

    debug_log("CONTENT_AGENT", f"开始生成搜索 query，用户 query={query!r}")

    raw = call_llm(system_prompt, user_prompt, tag="CONTENT_AGENT")

    # 尝试解析 JSON list
    try:
        start = raw.find("[")
        end = raw.rfind("]")
        json_str = raw[start : end + 1]
        queries = json.loads(json_str)
        if not isinstance(queries, list):
            raise ValueError("解析结果不是列表")
        debug_log("CONTENT_AGENT", f"解析出的搜索 query 列表={queries}")
    except Exception as e:
        debug_log("CONTENT_AGENT", f"解析搜索 query 失败，raw={raw[:200]!r}, error={e}")
        queries = [query] if query else ["popular movies"]

    all_cands: Dict[int, Dict[str, Any]] = {}
    for q in queries:
        recs = rag_search_movies(q, k=15)
        debug_log(
            "CONTENT_AGENT",
            f"基于搜索短语 {q!r} 检索到候选数={len(recs)}",
        )
        for rec in recs:
            mid = int(rec["movie_id"])
            all_cands[mid] = rec

    debug_log(
        "CONTENT_AGENT",
        f"内容检索总去重候选数={len(all_cands)}（合并所有搜索短语）",
    )

    return {"content_candidates": list(all_cands.values())}


def collab_agent(state: RecState) -> RecState:
    """
    Agent 3：简单的“协同过滤” Agent（实际上是 genre + 全局评分的启发式）
    - 调用 get_collab_candidates_by_genre 工具
    """
    user_id = state["user_id"]
    cands = get_collab_candidates_by_genre(user_id, k=30)
    debug_log(
        "COLLAB_AGENT",
        f"user_id={user_id} 协同候选数={len(cands)}",
    )
    return {"collab_candidates": cands}


def merge_agent(state: RecState) -> RecState:
    """
    Agent 4：合并候选集合（非 LLM）
    - 把 content_candidates 和 collab_candidates 合并去重
    - 简单按“是否有 rating + rating 大小”做排序并截断
    """
    content = state.get("content_candidates", []) or []
    collab = state.get("collab_candidates", []) or []

    debug_log(
        "MERGE_AGENT",
        f"准备合并候选：content={len(content)}, collab={len(collab)}",
    )

    merged: Dict[int, Dict[str, Any]] = {}

    # 内容候选
    for rec in content:
        mid = int(rec["movie_id"])
        r = dict(rec)
        r["source"] = r.get("source", "") + "|content"
        merged[mid] = r

    # 协同候选
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
    debug_log(
        "MERGE_AGENT",
        f"候选去重后共 {len(merged_list)} 条，示例前 3 条={[(m['title'], m.get('rating')) for m in merged_list[:3]]}",
    )

    # 为了控制后面 prompt 长度，这里截断前 40 条
    merged_list = merged_list[:40]
    debug_log("MERGE_AGENT", f"截断后保留前 {len(merged_list)} 条候选用于最终决策")

    return {"merged_candidates": merged_list}


def final_llm_agent(state: RecState) -> RecState:
    """
    Agent 5：主决策 LLM Agent（中文）
    - 输入：用户画像 + 合并后的候选电影
    - 输出：Top-K 推荐结果（包含中文理由），并解析 JSON
    """
    user_profile = state.get("user_profile", "") or ""
    query = state.get("query", "") or ""
    merged = state.get("merged_candidates", []) or []

    # 构造给 LLM 看的候选列表（简化字段）
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
        '  \"recommendations\": [\n'
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

    debug_log(
        "FINAL_AGENT",
        f"调用最终 LLM 进行推荐决策，候选数={len(merged)}, user_query={query!r}",
    )

    raw = call_llm(system_prompt, user_prompt, tag="FINAL_AGENT")

    # 尝试解析 JSON 对象
    recs: List[Dict[str, Any]] = []
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        json_str = raw[start : end + 1]
        obj = json.loads(json_str)
        recs = obj.get("recommendations", [])
        if not isinstance(recs, list):
            raise ValueError("recommendations 字段不是列表")
        debug_log("FINAL_AGENT", f"解析到的推荐条数={len(recs)}")
    except Exception as e:
        debug_log("FINAL_AGENT", f"解析最终 JSON 失败，raw={raw[:300]!r}, error={e}")
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

    debug_log(
        "FINAL_AGENT",
        f"最终组装出的推荐结果条数={len(final_recs)}, 示例前 3 条={[fr['title'] for fr in final_recs[:3]]}",
    )

    return {"final_recommendations": final_recs}


# ==========================
# 5. Planner Agent + 路由
# ==========================

def planner_agent(state: RecState) -> RecState:
    """
    DeepResearch 风格的“总控 / 规划 Agent”。

    - 读取当前状态（有哪些信息已经准备好了）
    - 决定下一步要调用哪个 Agent：
        - profile_agent
        - content_agent
        - collab_agent
        - merge_agent
        - final_llm_agent  （表示准备收尾出最终推荐）

    - 使用 LLM 输出 JSON：
        {
          "next_agent": "...",
          "reason": "..."
        }

    - 同时维护 step_count（限制最多循环 N 步）
    """
    user_id = state.get("user_id")
    query = state.get("query", "") or ""
    user_profile = state.get("user_profile", "")
    user_history = state.get("user_history", []) or []
    content_cands = state.get("content_candidates", []) or []
    collab_cands = state.get("collab_candidates", []) or []
    merged_cands = state.get("merged_candidates", []) or []

    step_count = int(state.get("step_count", 0)) + 1

    # 防止死循环
    MAX_STEPS = 8
    if step_count > MAX_STEPS:
        debug_log(
            "PLANNER",
            f"step_count={step_count} 超过上限 {MAX_STEPS}，强制切换到 final_llm_agent",
        )
        return {
            "next_agent": "final_llm_agent",
            "planner_reason": "超过最大规划步数，强制收尾。",
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
        "你现在扮演一个多智能体推荐系统的【总控调度 Agent】。\n"
        "系统中有以下几个子 Agent（工具）：\n"
        "1）profile_agent：根据用户的历史观影记录，生成一段用户画像描述（user_profile）。\n"
        "2）content_agent：在已经有 user_profile 和用户 query 的前提下，\n"
        "   生成若干英文搜索短语，并调用 TF-IDF 检索得到内容相关的候选电影（content_candidates）。\n"
        "3）collab_agent：根据用户历史记录和全局评分，在用户偏好的类型中挑选高评分但未看过的电影（collab_candidates）。\n"
        "4）merge_agent：将 content_candidates 和 collab_candidates 合并去重，并得到 merged_candidates。\n"
        "5）final_llm_agent：在有 user_profile 和 merged_candidates 的前提下，\n"
        "   由大模型选择最终要推荐的 5 部电影并给出理由（final_recommendations）。\n\n"
        "你的任务：\n"
        "  - 根据当前状态（state_summary），决定下一步应该调用哪一个 Agent，\n"
        "    或者直接调用 final_llm_agent 收尾。\n"
        "  - 一般流程建议是：优先确保有 user_profile -> 再补全候选（content + collab）-> merge -> 最后 final_llm_agent。\n"
        "  - 但是如果已经有足够多的候选，或者步骤过多，也可以提前进入 final_llm_agent。\n"
        "  - 你只能在如下集合中选择一个 next_agent：\n"
        "      [\"profile_agent\", \"content_agent\", \"collab_agent\", \"merge_agent\", \"final_llm_agent\"]。\n\n"
        "输出要求（非常重要）：\n"
        "  - 只能输出一个 JSON 对象，不要带其它文本。\n"
        "  - JSON 格式如下：\n"
        "    {\n"
        "      \"next_agent\": \"profile_agent\" 或 \"content_agent\" 或 \"collab_agent\" 或 \"merge_agent\" 或 \"final_llm_agent\",\n"
        "      \"reason\": \"用中文简要说明你做这个选择的原因\"\n"
        "    }"
    )

    user_prompt = (
        "当前系统状态摘要（state_summary）如下：\n"
        + json.dumps(state_summary, ensure_ascii=False, indent=2)
        + "\n\n请根据上述信息，输出下一步应该执行的 Agent 以及原因。"
    )

    debug_log("PLANNER", f"开始规划下一步，state_summary={state_summary}")

    raw = call_llm(system_prompt, user_prompt, tag="PLANNER")

    # 解析 JSON
    next_agent = "final_llm_agent"
    reason = "解析失败，直接进入 final_llm_agent。"

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        json_str = raw[start : end + 1]
        obj = json.loads(json_str)
        cand = obj.get("next_agent", "").strip()
        if cand in {
            "profile_agent",
            "content_agent",
            "collab_agent",
            "merge_agent",
            "final_llm_agent",
        }:
            next_agent = cand
        else:
            debug_log(
                "PLANNER",
                f"解析到的 next_agent={cand!r} 不在允许列表中，将 fallback 到 final_llm_agent",
            )
        reason = obj.get("reason", reason)
    except Exception as e:
        debug_log("PLANNER", f"解析 planner JSON 失败，raw={raw[:300]!r}, error={e}")

    debug_log(
        "PLANNER",
        f"决策结果：next_agent={next_agent}, reason={reason}, step_count={step_count}",
    )

    return {
        "next_agent": next_agent,
        "planner_reason": reason,
        "step_count": step_count,
    }


def planner_router(state: RecState) -> str:
    """
    LangGraph 的条件路由函数：
    - 读取 state["next_agent"]，返回对应的 path key
    - 如果缺失或异常，就 fallback 到 final_llm_agent
    """
    na = state.get("next_agent", "") or ""
    if na not in {
        "profile_agent",
        "content_agent",
        "collab_agent",
        "merge_agent",
        "final_llm_agent",
    }:
        debug_log(
            "ROUTER",
            f"state.next_agent={na!r} 非法，fallback 到 final_llm_agent",
        )
        return "final_llm_agent"

    debug_log("ROUTER", f"路由到 next_agent={na}")
    return na


# ==========================
# 6. LangGraph 图结构搭建（带自动规划）
# ==========================

def build_graph():
    """
    DeepResearch 风格多 Agent 图结构：

    START
      -> planner_agent
          ├─(next=profile_agent)─> profile_agent  ─┐
          ├─(next=content_agent)─> content_agent  ─┤
          ├─(next=collab_agent)─> collab_agent   ─┤
          ├─(next=merge_agent)───> merge_agent    ─┤
          └─(next=final_llm_agent)───────────────> final_llm_agent -> END

    profile / content / collab / merge 跑完之后全部回到 planner_agent 再规划下一步，
    直到 planner 决定 next_agent = final_llm_agent。
    """
    graph_builder = StateGraph(RecState)

    # 注册节点
    graph_builder.add_node("planner_agent", planner_agent)

    graph_builder.add_node("profile_agent", profile_agent)
    graph_builder.add_node("content_agent", content_agent)
    graph_builder.add_node("collab_agent", collab_agent)
    graph_builder.add_node("merge_agent", merge_agent)
    graph_builder.add_node("final_llm_agent", final_llm_agent)

    # 起点：直接进入 planner
    graph_builder.add_edge(START, "planner_agent")

    # 条件边：由 planner 决定下一个 Agent
    graph_builder.add_conditional_edges(
        "planner_agent",
        planner_router,
        {
            "profile_agent": "profile_agent",
            "content_agent": "content_agent",
            "collab_agent": "collab_agent",
            "merge_agent": "merge_agent",
            "final_llm_agent": "final_llm_agent",
        },
    )

    # worker 节点执行完之后回到 planner，继续规划
    graph_builder.add_edge("profile_agent", "planner_agent")
    graph_builder.add_edge("content_agent", "planner_agent")
    graph_builder.add_edge("collab_agent", "planner_agent")
    graph_builder.add_edge("merge_agent", "planner_agent")

    # 最终决策节点走向 END，不再回 planner
    graph_builder.add_edge("final_llm_agent", END)

    graph = graph_builder.compile()
    debug_log("GRAPH", "LangGraph（带自动规划）编译完成。")
    return graph


# ==========================
# 7. 示例运行
# ==========================

def demo_run(
    user_id: int = 1,
    query: str = "我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道",
):
    debug_log("DEMO", f"开始 demo_run，user_id={user_id}, query={query!r}")

    load_movielens()
    build_movie_tfidf()

    graph = build_graph()

    init_state: RecState = {
        "user_id": user_id,
        "query": query,
    }

    final_state: RecState = graph.invoke(init_state)
    recs = final_state.get("final_recommendations", [])

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
