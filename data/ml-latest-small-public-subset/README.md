# MovieLens latest small public subset

这个目录放的是一个**公开可核验的最小 smoke-demo 子集**，用于在没有 Redis、没有 LLM、甚至不下载完整官方压缩包的情况下，把 `recsys-universe` 的 industrial pipeline 走通。

## 对应公开数据集

- 数据集：MovieLens latest small
- 官方说明：`ml-latest-small`
- 公开描述：包含 `ratings.csv`、`movies.csv`、`tags.csv`、`links.csv`，约 100,836 条评分、9,742 部电影、610 个用户

## 这个子集里有什么

- `movies.csv`：挑了少量公开可见、可核验的电影行
- `ratings.csv`：挑了少量公开可见、可核验的评分行
- `users.csv`：为了适配当前仓库的 `load_movielens_csv()`，补了一份最小用户画像表

## 为什么需要 `users.csv`

MovieLens latest small 官方并不提供 demographics 表；但本仓库当前的 industrial demo 在 CSV 模式下如果有 `users.csv`，会更顺畅地走完整条 pipeline。

所以这里的 `users.csv` 是一个**适配层文件**：
- user_id 来自 `ratings.csv` 中出现的用户
- 其余字段使用默认值，仅用于跑通 demo

## 适用范围

这不是正式 benchmark，也不是完整训练集，只用于：

1. 验证 ingestion 能否跑通
2. 验证 in-memory feature store 能否工作
3. 验证 two-tower embedding / FAISS(or mock) recall / rank / recommend 全链路是否可执行

如果你要接完整公开数据集，请用仓库里的脚本：

- `scripts/prepare_public_movielens_csv.py`
- `scripts/run_public_movielens_smoke.py`
