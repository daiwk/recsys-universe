# recsys-universe

基于 Claude-style Skills 架构的多智能体电影推荐系统 Demo。

**核心特性**: 保留 Skills 编排框架，将传统 TF-IDF/协同过滤替换为工业级双塔召回 + 精排。

## Skills 架构演进

```
┌─────────────────────────────────────────────────────────────────┐
│                      Skills 架构流程                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌───────┐       │
│  │ Profile │───>│ Content  │───>│  Collab │───>│ Final │       │
│  │ Skill   │    │ Skill    │    │  Skill  │    │ Skill │       │
│  └─────────┘    └──────────┘    └─────────┘    └───────┘       │
│       │             │               │               │          │
│       v             v               v               v          │
│   用户画像      TF-IDF召回      协同过滤        结果生成       │
│                  ↓               ↓                            │
│              向量召回         精排排序                         │
│            (双塔+Milvus)    (DNN+CTR)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**两种模式**:
- **Legacy**: Content=TF-IDF, Collab=协同过滤 (需要 LLM)
- **Industrial**: Content=向量召回, Collab=精排排序 (无需 LLM)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 可选依赖
pip install torch          # GPU 加速模型训练
pip install pymilvus       # Milvus 向量检索
pip install redis          # Redis 特征存储
```

### 2. 配置环境变量

```bash
# 工业级模式配置
export RECSYS_ARCHITECTURE=industrial

# Legacy 模式需要 LLM
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export RECSYS_DEBUG="false"

# 内存存储模式 (无需 Redis/Milvus)
export RECSYS_USE_MEMORY_STORE=true

# Redis 配置
export REDIS_HOST=localhost
export REDIS_PORT=6379

# Milvus 配置
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
```

### 3. 数据导入 (Industrial 模式)

在使用 Industrial 模式之前，需要先将 MovieLens 数据导入 Redis 和 Milvus：

```bash
# 设置数据路径 (解压后的 ml-1m 目录)
export MOVIELENS_PATH=./ml-1m

# 导入数据到 Redis + Milvus
python scripts/ingest_data.py

# 或者使用内存存储模式 (无需 Redis/Milvus)
python scripts/ingest_data.py --memory

# 仅重建 Milvus 索引
python scripts/ingest_data.py --rebuild-index
```

### 4. 运行

**Legacy 模式 (TF-IDF + LLM):**
```bash
python multiagents_movielens.py --legacy
```

**Industrial 模式 (双塔 + 精排):**
```bash
python multiagents_movielens.py --industrial
```

**纯工业级演示 (内存模式):**
```bash
RECSYS_USE_MEMORY_STORE=true python multiagents_movielens.py --demo-industrial
```

### 5. API 服务

```bash
python serving/api_server.py --port 8080
```

支持 API:
- `GET /health` - 健康检查
- `POST /recall` - 向量召回
- `POST /rank` - 排序
- `POST /recommend` - 召回 + 排序

## 项目结构

```
recsys-universe/
├── config.py                    # 统一配置管理
├── multiagents_movielens.py     # 主入口（支持双架构）
├── industrial_coordinator.py    # 工业级协调器（新）
├── skills_coordinator.py        # 传统 Skills 协调器
│
├── skills/                      # 传统 Skills
│   ├── base_skill.py            # BaseSkill 基类
│   ├── skill_registry.py        # Skill 注册表
│   ├── planner_skill.py         # 规划调度技能
│   ├── profile_skill.py         # 用户画像技能
│   ├── content_skill.py         # TF-IDF 检索
│   ├── collab_skill.py          # 协同过滤技能
│   ├── merge_skill.py           # 候选合并技能
│   ├── final_skill.py           # 最终推荐技能
│   └── vector_recall_skill.py   # 向量召回（新）
│
├── models/                      # 推荐模型
│   ├── two_tower.py             # 双塔召回模型
│   └── ranking_model.py         # 精排模型
│
├── features/                    # 特征工程
│   ├── base.py                  # 特征存储（Redis）
│   ├── user_features.py         # 用户特征
│   ├── item_features.py         # 物品特征
│   └── cross_features.py        # 交叉特征
│
├── serving/                     # 在线服务
│   ├── milvus_client.py         # Milvus 客户端
│   ├── recall_service.py        # 召回服务
│   ├── rank_service.py          # 排序服务
│   └── api_server.py            # HTTP API
│
├── training/                    # 在线学习
│   ├── online_learner.py        # 在线学习模块
│   └── streaming.py             # 流式处理
│
└── tests/                       # 单元测试
```

## 配置说明

所有配置通过 `config.py` 管理：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `architecture_mode` | `RECSYS_ARCHITECTURE` | `industrial` | 架构模式 |
| `debug` | `RECSYS_DEBUG` | `false` | 调试模式 |
| `recall.recall_top_k` | - | `100` | 召回数量 |
| `rank.rank_top_k` | - | `10` | 排序数量 |
| `model.two_tower.user_embedding_dim` | - | `32` | 用户向量维度 |
| `model.two_tower.item_embedding_dim` | - | `32` | 物品向量维度 |
| `model.two_tower.num_hash_buckets` | - | `1000000` | Hash bucket 数 |
| `milvus.host` | `MILVUS_HOST` | `localhost` | Milvus 地址 |
| `redis.host` | `REDIS_HOST` | `localhost` | Redis 地址 |

## 工业级架构详解

### 双塔召回 (Two-Tower Retrieval)

```
User Tower:                    Item Tower:
┌─────────────────┐           ┌─────────────────┐
│ ID Features     │           │ ID Features     │
│ - user_id (Hash)│           │ - item_id (Hash)│
│ - genres        │           │ - genres        │
├─────────────────┤           ├─────────────────┤
│ Embedding Layer │           │ Embedding Layer │
│ (亿级ID → 64维) │           │ (亿级ID → 64维) │
├─────────────────┤           ├─────────────────┤
│ DNN Layers      │           │ DNN Layers      │
│ (128→64→32)     │           │ (128→64→32)     │
├─────────────────┤           ├─────────────────┐
│ User Vector     │           │ Item Vector     │
│ (32维)          │           │ (32维)          │
└─────────────────┘           └─────────────────┘
```

### 精排模型 (Ranking Model)

输入特征:
- 用户特征 (User Tower Output)
- 物品特征 (Item Tower Output)
- 交叉特征 (user-item interactions)

模型结构:
- DNN (256→128→64→1)
- 输出: CTR 预测

### 向量检索 (Milvus)

- 索引类型: IVF_PQ (支持亿级向量)
- 距离度量: COSINE
- 离线: Item Embeddings → Milvus
- 在线: User Embedding → Milvus Search → Top-100 Items

### 在线学习

- 实时事件流处理
- 特征增量更新
- 模型在线微调

## 测试

```bash
pytest tests/ -v
```

## 与原系统对比

| 特性 | Legacy (TF-IDF) | Industrial (双塔+精排) |
|------|-----------------|----------------------|
| 召回方式 | TF-IDF 文本匹配 | 向量相似度检索 |
| 排序 | LLM 生成 | DNN CTR 预测 |
| ID 规模 | 万级 | 亿级 |
| 推理延迟 | 高 (LLM) | 低 |
| 在线学习 | 不支持 | 支持 |
| 冷启动 | 差 | 一般 |

## 更新日志

### v3.0.0 (工业级架构)

- 新增双塔召回模型 (`models/two_tower.py`)
- 新增精排模型 (`models/ranking_model.py`)
- 新增特征存储层 (`features/`) - Redis 实时特征
- 新增 Milvus 集成 (`serving/milvus_client.py`)
- 新增召回服务 (`serving/recall_service.py`)
- 新增排序服务 (`serving/rank_service.py`)
- 新增 API 服务 (`serving/api_server.py`)
- 新增在线学习模块 (`training/online_learner.py`)
- 新增工业级协调器 (`industrial_coordinator.py`)
- 支持双架构切换 (Legacy / Industrial)

### v2.0.0

- 新增统一配置管理
- 重构 Skills 协调器
- 完善单元测试
