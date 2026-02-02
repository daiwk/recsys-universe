# Agent Swarm - PARL

该目录提供论文中 Agent Swarm 的 PARL (Parallel Agent Reinforcement Learning) 阶段的最小实现，
并预留了与 **verl** 框架的并行执行器集成接口。

## 结构

- `parl.py`: PARL 核心逻辑（并行采样、优势估计、指标聚合）。
- `__init__.py`: 对外导出主要类型与训练器。

## 关键接口

- `SwarmAgent`: 包装策略与价值函数。
- `SwarmEnvironment`: 环境协议（reset/step）。
- `PARLTrainer`: 负责并行 rollout、swarm 优势估计与单步训练。
- `SwarmBatch`: 聚合后的 swarm 级数据（共享优势、聚合奖励/价值）。
- `RewardComponents`: PARL 奖励的三项分解（并行化奖励/完成奖励/性能奖励）。
- `PARLRewardConfig`: 管理 λ1/λ2 并支持退火到 0。
- `CriticalStepTracker`: 跟踪 critical steps（主代理 + 最长子代理路径）。
- `build_verl_backend`: 使用 verl 的 `WorkerPool` 作为并行后端。

## 与 verl 集成

默认后端使用 `concurrent.futures`。训练更新函数建议同时使用 swarm 优势与本地优势：

```python
def update_fn(agent, trajectory, swarm_advantages, local_advantages):
    # swarm_advantages: 共享优势（用于策略更新）
    # local_advantages: 单智能体优势（用于价值更新或辅助损失）
    ...
```

如果已安装 `verl`，可使用：

```python
from agent_swarm.parl import PARLTrainer, PARLConfig, build_verl_backend

trainer = PARLTrainer(
    config=PARLConfig(num_parallel_envs=8),
    parallel_backend=build_verl_backend(),
)
```

该接口保持轻量，便于直接替换为 verl 的并行执行机制。

## PARL 奖励与 critical steps

PARL 奖励按报告中的形式拆分：

```
r_PARL(x, y) = λ1 * r_parallel + λ2 * r_finish + r_perf(x, y)
```

其中 `λ1/λ2` 会随训练退火到 0。`CriticalStepTracker` 用于计算并行执行的 critical steps：

```
CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))
```

使用示例：

```python
from agent_swarm.parl import RewardComponents

components = RewardComponents(
    parallel_reward=1.0,
    finish_reward=0.5,
    performance_reward=2.0,
)
parl_reward = trainer.compute_parl_reward(components)
```
