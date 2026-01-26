# recsys-universe

基于 Claude-style Skills 架构的多智能体电影推荐系统 Demo。

**核心特性**: 保留 Skills 编排框架，将传统 TF-IDF/协同过滤替换为工业级双塔召回 + 精排。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      Skills 架构流程                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌───────┐       │
│  │ Profile │───>│ Content  │───>│  Collab │───>│ Final │       │
│  │ Skill   │    │ Skill    │    │  Skill  │    │ Skill │       │
│  └─────────┘    └──────────┘    └─────────┘    └───────┘       │
│                                                                 │
│  Legacy模式: TF-IDF召回 → 协同过滤 → LLM生成                      │
│  Industrial模式: 向量召回(双塔+FAISS) → 精排(DNN+CTR)           │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 可选依赖
pip install torch          # GPU 加速
pip install faiss-cpu      # FAISS 向量检索 (纯Python，无需外部服务)
pip install redis          # Redis 特征存储 (需要 redis-server)
```

### 2. 下载 MovieLens 数据集

```bash
# 下载 MovieLens-1M (约 24MB)
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip

# 解压到项目根目录
unzip ml-1m.zip -d .
# 解压后会得到 ml-1m/ 目录，包含:
#   - movies.dat    (电影信息)
#   - ratings.dat   (评分数据)
#   - users.dat     (用户信息)
```

### 3. 数据导入

根据运行环境选择以下方式之一：

#### 方式 A: 完整服务 (Redis + FAISS)

需要先启动 Redis 服务：

```bash
# 启动 Redis (默认端口 6379)
redis-server

# FAISS 是纯 Python 库，无需外部服务
# 导入数据到 Redis + FAISS
export MOVIELENS_PATH=./ml-1m
python scripts/ingest_data.py
```

#### 方式 B: 内存模式 (无需外部服务)

```bash
# 使用内存存储模式，不需要 Redis/FAISS
export MOVIELENS_PATH=./ml-1m
python scripts/ingest_data.py --memory
```

### 4. 运行推荐系统

#### Industrial 模式 (推荐)

**使用已导入的数据:**
```bash
python multiagents_movielens.py --industrial
```

**纯演示模式 (无需数据导入):**
```bash
# 内存模式下直接运行演示
RECSYS_USE_MEMORY_STORE=true python multiagents_movielens.py --demo-industrial
```

#### Legacy 模式

需要配置 LLM (如本地 Qwen 模型):

```bash
export RECSYS_ARCHITECTURE=legacy
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://localhost:8000/v1"

python multiagents_movielens.py --legacy
```

### 5. API 服务

```bash
# 启动 API 服务
python serving/api_server.py --port 8080
```

支持 API:
- `GET /health` - 健康检查
- `POST /recall` - 向量召回
- `POST /rank` - 排序
- `POST /recommend` - 召回 + 排序

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RECSYS_ARCHITECTURE` | `legacy` | 架构模式: `legacy` 或 `industrial` |
| `RECSYS_USE_MEMORY_STORE` | `false` | 是否使用内存存储 (无需 Redis) |
| `RECSYS_DEBUG` | `false` | 调试模式 |
| `MOVIELENS_PATH` | `ml-1m` | MovieLens 数据路径 |
| `OPENAI_API_KEY` | - | Legacy 模式需要的 API Key |
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | LLM 服务地址 |
| `REDIS_HOST` | `localhost` | Redis 地址 |
| `REDIS_PORT` | `6379` | Redis 端口 |
| `FAISS_NLIST` | `0` | FAISS IVF 索引聚类数 (0=Flat) |

## 项目结构

```
recsys-universe/
├── config.py                    # 统一配置管理
├── multiagents_movielens.py     # 主入口
├── skills_coordinator.py        # Skills 协调器
│
├── skills/                      # Skills 框架
│   ├── base_skill.py            # BaseSkill 基类
│   ├── profile_skill.py         # 用户画像技能
│   ├── content_skill.py         # Content检索 (TF-IDF/向量双模式)
│   ├── collab_skill.py          # Collab排序 (CF/精排双模式)
│   ├── merge_skill.py           # 候选合并技能
│   ├── final_skill.py           # 最终推荐技能
│   ├── planner_skill.py         # 规划调度技能
│   └── skill_registry.py        # Skill 注册表
│
├── models/                      # 推荐模型
│   ├── two_tower.py             # 双塔召回模型
│   └── ranking_model.py         # 精排模型
│
├── features/                    # 特征工程
│   ├── base.py                  # 特征存储 (Redis/Memory)
│   ├── user_features.py         # 用户特征
│   ├── item_features.py         # 物品特征
│   └── cross_features.py        # 交叉特征
│
├── serving/                     # 在线服务
│   ├── faiss_client.py          # FAISS 客户端
│   ├── recall_service.py        # 召回服务
│   ├── rank_service.py          # 排序服务
│   └── api_server.py            # HTTP API
│
├── scripts/                     # 脚本工具
│   └── ingest_data.py           # 数据导入脚本
│
└── training/                    # 在线学习
    └── online_learner.py        # 在线学习模块
```

## 脚本使用说明

### ingest_data.py - 数据导入

```bash
# 导入到 Redis + FAISS
python scripts/ingest_data.py

# 使用内存存储模式
python scripts/ingest_data.py --memory

# 仅重建 FAISS 索引
python scripts/ingest_data.py --rebuild-index

# 指定数据路径
python scripts/ingest_data.py --data-path /path/to/ml-1m
```

## 两种模式对比

| 特性 | Legacy | Industrial |
|------|--------|------------|
| 召回方式 | TF-IDF 文本匹配 | 向量相似度检索 |
| 排序 | 协同过滤启发式 | DNN CTR 预测 |
| 外部依赖 | LLM 服务 | Redis + FAISS |
| 推荐理由 | LLM 生成 | CTR 分数 |
| ID 规模 | 万级 | 亿级 (Hash 桶) |
| 推理延迟 | 高 (LLM 延迟) | 低 (< 50ms) |
| 冷启动 | 差 | 一般 |

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_config.py -v
```

## 常见问题

**Q: 内存模式下运行报错?**
A: 确保设置了 `RECSYS_USE_MEMORY_STORE=true`

**Q: Industrial 模式找不到数据?**
A: 需要先运行 `python scripts/ingest_data.py` 导入数据

**Q: Legacy 模式需要什么?**
A: 需要配置有效的 LLM 服务 (OpenAI 兼容接口)

**Q: 如何切换模式?**
A: 设置环境变量 `export RECSYS_ARCHITECTURE=industrial` 或直接使用 `--industrial/--legacy` 参数
