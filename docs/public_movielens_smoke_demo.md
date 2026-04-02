# Public MovieLens smoke demo

这份文档说明如何用一个**公开数据集**把 `recsys-universe` 的 industrial pipeline 跑通，并提供一个**零网络 smoke demo** 用于快速验证全链路。

---

## 1. 选择的数据集

这里选用的是 **MovieLens latest small**。

选择原因：
- 公开可获取
- 体量很小，适合 demo / smoke test
- 自带 `movies.csv` 与 `ratings.csv`
- 与仓库现有 `load_movielens_csv()` 的 CSV ingestion 路径高度兼容

需要注意：
- 官方 `ml-latest-small` **没有 `users.csv` demographics 表**
- 当前仓库的 industrial CSV demo 在有一个最小 `users.csv` 时更容易跑完整链路
- 因此这里额外提供了一个**适配脚本**，用于从 `ratings.csv` 中抽取唯一用户，生成默认画像字段

---

## 2. 你可以怎么跑

### 方式 A：直接跑仓库里自带的零网络 smoke demo

这个仓库新增了一个公开可核验的最小子集：

```text
data/ml-latest-small-public-subset/
```

直接运行：

```bash
export RECSYS_USE_MEMORY_STORE=true
python scripts/run_public_movielens_smoke.py
```

或者显式指定：

```bash
RECSYS_USE_MEMORY_STORE=true \
python scripts/run_public_movielens_smoke.py \
  --data-path ./data/ml-latest-small-public-subset \
  --user-id 1
```

这个流程会完成：

1. 读取 CSV 数据
2. 写入 in-memory feature store
3. 生成 item embeddings
4. 生成 user embeddings
5. 构建 FAISS（若未安装则自动回退到 mock FAISS）
6. 走 `IndustrialSkillsCoordinator` 的 recall + rank + recommend

---

### 方式 B：使用你自己下载的完整公开 `ml-latest-small`

假设你已经手动下载并解压了官方数据集，目录里至少有：

```text
ml-latest-small/
├── movies.csv
├── ratings.csv
├── tags.csv
└── links.csv
```

先执行适配脚本：

```bash
python scripts/prepare_public_movielens_csv.py \
  --src /path/to/ml-latest-small \
  --dst /path/to/ml-latest-small-prepared
```

执行完后，输出目录会变成：

```text
ml-latest-small-prepared/
├── movies.csv
├── ratings.csv
└── users.csv
```

然后运行 smoke demo：

```bash
export RECSYS_USE_MEMORY_STORE=true
python scripts/run_public_movielens_smoke.py \
  --data-path /path/to/ml-latest-small-prepared \
  --user-id 1
```

---

## 3. 推荐的运行环境

最小环境：

```bash
pip install pandas numpy
```

如果想启用真实 FAISS：

```bash
pip install faiss-cpu
```

如果没有安装 FAISS，也可以运行；脚本会自动回退到 mock FAISS 路径，只是召回阶段不是真实向量索引检索。

---

## 4. 我本地跑通时验证了什么

我本地验证的是这条链路：

```text
CSV -> ingest_data.load_movielens_csv()
    -> ingest_to_redis() [memory mode]
    -> generate_item_embeddings()
    -> build_user_embeddings()
    -> IndustrialSkillsCoordinator.run_recommendation()
    -> recall + rank + formatted recommendations
```

本地 smoke run 的关键状态：

- feature store: in-memory 正常
- two-tower: 正常生成 embedding
- FAISS: 环境里未安装真实 faiss 时，已走 mock fallback
- ranking: 正常输出 CTR score
- final recommendation: 正常返回 top-K 结果

也就是说，这次跑通的是**完整推荐流程骨架**，不是只跑到 ingestion 或只跑到 recall。

---

## 5. 预期输出长什么样

运行后你会先看到一段 health check，然后看到推荐结果，例如：

```text
health_check=
{...}

recommendations for user_id=1:
1. Escape from New York (1981) | genres=['Action', 'Adventure', 'Sci-Fi', 'Thriller'] | recall=0.7000 | ctr=0.7721
2. Deer Hunter, The (1978) | genres=['Drama', 'War'] | recall=0.5000 | ctr=0.7325
...
```

具体分数会因为：
- 是否安装了真实 FAISS
- 运行环境中的数值库版本
- 你使用的是仓库内子集还是完整 public dataset

而略有差异。

---

## 6. 这个 smoke demo 的定位

它的目标不是做 benchmark，也不是证明模型效果好坏，而是：

1. 给一个**公开数据集入口**
2. 把 repo 的 industrial path **真正跑起来**
3. 提供一个**容易复现**的最小验证链路
4. 让后续接完整公开数据集、Redis、真实 FAISS、甚至线上服务时，有一个稳定起点

---

## 7. 关键 caveat

### caveat 1：`users.csv` 是适配层

MovieLens latest small 官方没有 demographics 表，所以这里自动生成一个默认 `users.csv`，只为兼容仓库当前 CSV ingestion 路径。

### caveat 2：子集 smoke demo 不等于正式实验

仓库里的 `data/ml-latest-small-public-subset/` 只是一个**公开可核验的最小子集**，用于验证流程可执行，不应拿来做离线指标对比。

### caveat 3：mock FAISS 只用于 smoke test

如果环境里没装 `faiss-cpu`，脚本仍能跑，但 recall 不是严格意义上的真实 FAISS ANN 检索。要验证真实向量检索，请安装 `faiss-cpu`。

---

## 8. 推荐的下一步

如果你想把这条路径继续做实，可以按这个顺序推进：

1. 用完整 `ml-latest-small` 跑一遍
2. 安装 `faiss-cpu`，验证真实索引路径
3. 把 `README.md` 里的 quick start 补一个指向这份文档的链接
4. 再考虑接 Redis / API server / 更大公开数据集
