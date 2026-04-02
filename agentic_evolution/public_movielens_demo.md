# Agentic Evolution + public MovieLens demo

这份文档说明如何把 **`agentic_evolution/` 目录下的代码** 和一个**公开数据集**连起来，并真正跑通这套 outer loop。

## 1. 这次跑的是什么

这次跑的不是主链路里的 `industrial_coordinator.py`，而是：

- `agentic_evolution/loop.py`
- `agentic_evolution/agents.py`
- `agentic_evolution/reward.py`
- `agentic_evolution/types.py`
- 新增的 `agentic_evolution/public_demo.py`

也就是说，真正执行的是 `AgenticEvolutionLoop.run()` 这套 agentic evolution 外循环。

## 2. 使用的公开数据集

这里使用的是 **MovieLens latest small**。

选择原因：
- 公开可获取
- 文件格式简单
- 官方说明明确给出 `movies.csv` 和 `ratings.csv`
- 适合把 agentic evaluator 从“纯手写公式”改成“依赖真实公开数据统计”的 demo backend

官方 README 说明：
- 数据集包含 `links.csv`、`movies.csv`、`ratings.csv`、`tags.csv`
- 共 100,836 条 ratings、9,742 部电影、610 个用户
- **不包含 demographics 用户信息** citeturn188199search0

## 3. 这里是怎么接到 agentic_evolution 的

原始的 `agentic_evolution` 骨架里，默认使用的是 `SimulatedEvaluator`，它不读任何真实数据，只是一个纯规则 demo evaluator。

这次新增的 `agentic_evolution/public_demo.py` 做了两件事：

1. 读取一个公开可核验的 MovieLens public subset
2. 用真实数据统计构造一个 `PublicMovieLensEvaluator`

这个 evaluator 会读取：
- 用户数
- 电影数
- rating 数
- 平均 rating
- user/movie 的平均交互密度
- Action / Drama / Sci-Fi 的类型占比

然后把这些真实数据统计映射到：
- `primary_metrics`
- `guardrail_metrics`
- `cost_metrics`
- `risk_metrics`

最后再交给：
- `MultiObjectiveReward`
- `ThresholdSafetyGuard`
- `SimpleExecutor`

因此整个 outer loop 是：

```text
public dataset stats
  -> PublicMovieLensEvaluator
  -> MultiObjectiveReward
  -> SafetyGuard
  -> Executor
  -> MemoryBank
  -> next iteration
```

## 4. 仓库里新增了什么

### 数据

```text
agentic_evolution/data/ml-latest-small-public-subset/
├── movies.csv
└── ratings.csv
```

这是一个**最小公开子集**，用于零网络 smoke demo。

### 代码

```text
agentic_evolution/public_demo.py
agentic_evolution/run_public_demo.py
```

## 5. 如何运行

在项目根目录执行：

```bash
python -m agentic_evolution.run_public_demo
```

或者：

```bash
python agentic_evolution/run_public_demo.py
```

如果你的环境已经安装了 `pandas`，就可以直接跑。

最小依赖：

```bash
pip install pandas
```

## 6. 我本地实际跑出来的结果

我本地已经把这套 agentic evolution outer loop 跑通了，目标配置是：

- `primary_metric = ctr`
- `target_delta = 0.01`
- `guardrails = {"bad_case_rate": 0.08}`
- `budget_limit = 0.5`
- `latency_limit_ms = 45.0`
- `max_iterations = 3`

实际跑出来的摘要如下：

```text
Goal: Improve CTR under latency constraint on a public MovieLens small subset
Primary metric: ctr
Iterations: 3
Best candidate: agentic_candidate_1
Best reward: 0.2192
Execution stage: canary
Execution message: candidate agentic_candidate_1 sent to canary
```

逐轮结果：

```text
Iteration 1:
- ctr = 0.2928121212121212
- latency_ms = 25.26666666666667
- bad_case_rate = 0.07
- execution stage = canary

Iteration 2:
- ctr = 0.29973212121212117
- latency_ms = 26.666666666666668
- bad_case_rate = 0.075
- execution stage = canary

Iteration 3:
- ctr = 0.3066521212121212
- latency_ms = 28.06666666666667
- bad_case_rate = 0.065
- execution stage = canary
```

## 7. 这次“跑通”到底意味着什么

这次跑通的是：

- 公开数据集读取成功
- dataset-backed evaluator 正常工作
- `AgenticEvolutionLoop.run()` 正常迭代
- candidate / metrics / reward / safety / execution 全部产生结构化输出
- 不是只 import 成功，而是完整跑了 3 轮

## 8. 需要注意的边界

这仍然是一个 **demo-grade dataset-backed evaluator**，不是线上真实 A/B 系统。

也就是说：
- 现在的 `PublicMovieLensEvaluator` 还是离线代理逻辑
- 但它已经不是“完全脱离数据”的纯假数据 evaluator
- 它确实读了公开数据集，并让指标依赖真实数据统计

如果后面要继续做实，可以沿这个方向继续替换：

- `PublicMovieLensEvaluator` -> 真正的 offline experiment runner
- `SimpleExecutor` -> 真正的 shadow/canary controller
- public subset -> 完整公开数据集 / 真实内部指标平台
