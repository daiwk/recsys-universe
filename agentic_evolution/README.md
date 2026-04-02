# Agentic Evolution Framework

这是一个放在 `recsys-universe` 里的最小可执行框架，用来把“agentic recommender / AutoModel / evolutionary recsys”这类论文里偏概念化的部分，补成一个**能跑、能扩展、能接工业平台**的闭环。

## 目标

它不直接替代现有的 `industrial_coordinator.py`，而是提供一层 **策略演化 / 实验编排 / 奖励聚合 / 安全约束** 的外循环：

```text
目标定义
  -> Planner 拆解实验方向
  -> CandidateGenerator 生成候选策略
  -> Evaluator 跑离线评估
  -> RewardAggregator 聚合多目标奖励
  -> SafetyGuard 做上线前约束检查
  -> Executor 执行 shadow / canary / promote
  -> MemoryBank 记录经验，进入下一轮
```

## 适合接到哪里

这个目录适合接你现有项目中的这些组件：

- `industrial_coordinator.py`: 作为实际推荐链路入口
- `training/`: 作为训练任务提交入口
- `serving/`: 作为 shadow / canary / promote 的发布入口
- `features/`: 作为特征/画像/统计输入来源
- A/B 平台 / 实验平台 / Hive 指标表：作为 `Evaluator` 与 `Executor` 的真实后端

## 目录结构

```text
agentic_evolution/
├── __init__.py
├── types.py          # 状态、候选、结果、奖励等核心数据结构
├── interfaces.py     # Planner / Evaluator / Executor / Memory 等接口定义
├── reward.py         # 多目标奖励聚合 + 硬约束校验
├── agents.py         # 默认的 planner/generator/evaluator/safety/executor 实现
├── loop.py           # 主循环 AgenticEvolutionLoop
└── demo.py           # 可直接运行的 demo
```

## 设计原则

### 1. 外循环与内循环分离

- 内循环：你的精排、召回、训练、serving
- 外循环：agent 决定“接下来改什么、试什么、怎么上线”

### 2. 多目标不是一句话，而是显式奖励

默认奖励支持：

- `primary`: 主目标，例如 CTR / GMV / order_rate
- `guardrail`: 守护目标，例如 latency / crash_rate / bad_case_rate
- `cost`: 训练或 serving 成本
- `risk`: 发布风险

### 3. 先离线，再 shadow，再 canary，再 promote

框架默认把工业安全路径做成状态机，而不是一句“上线实验”。

### 4. 所有 agent 的输出都是结构化对象

避免“全靠 prompt 文本拼接”，便于之后接真实平台。

## 最小使用方式

```python
from agentic_evolution.demo import build_demo_loop

loop = build_demo_loop()
summary = loop.run(
    goal="在不显著增加延迟的前提下，提高首页推荐 CTR",
    max_iterations=3,
)

print(summary.pretty())
```

## 一个真实工业落地方向

你后面可以按这个思路替换 demo 组件：

### Planner
输入：
- 当前模型版本
- 最近 N 次实验表现
- 目标与约束

输出：
- 下一轮搜索方向，例如：
  - 扩大召回候选数
  - 引入新 cross feature
  - 调整精排 loss 权重
  - 打开/关闭某个 re-rank stage

### CandidateGenerator
输出结构化候选，例如：
- 模型超参
- 特征开关
- 训练配置
- serving 开关

### Evaluator
对接你真实平台后可以做：
- 提交训练任务
- 读取离线评估结果
- 读取 replay / counterfactual 评估
- 读取线上 shadow 指标

### Executor
对接真实发布平台后可以做：
- 创建 shadow 实验
- 创建 canary 实验
- 达标后 promote
- 不达标则 rollback

## 和两篇论文的关系

这套骨架专门补它们缺失的四块：

1. 候选策略的结构化表示
2. 多目标奖励的显式计算
3. 上线安全约束与阶段迁移
4. 经验记忆与下一轮搜索

所以它不是“再写一篇概念论文”，而是把论文里那套 agentic 叙事，压成一套代码骨架。
