## 相关论文

这个目录的直接背景是下面两篇 2026 年 3 月的论文，它们都在讨论推荐系统里的 agent 化、自动化演进与系统级协同，但整体上更偏**框架与愿景**，不是那种已经把推荐算法完整落到工业线上、并给出明确 A/B 收益的工业论文。

### 1. Agentic Recommender Systems / Agentic Recsys Vision Paper
- arXiv: `2603.26100`
- 链接：<https://arxiv.org/pdf/2603.26100>

简要理解：
- 这篇更像一篇 **position / vision paper**
- 它想表达的是：推荐系统不该只看成固定的召回-排序流水线，而可以看成一个由多个 agent 组成、持续自我演化的决策系统
- 论文会讨论 decision layer、evolution layer、infrastructure layer 这类抽象分层，也会提到多目标优化、内外奖励、agent 协同等概念
- 但它并没有真正给出一个可直接复现的工业实现框架，也没有提供很硬的线上落地结果

对这个目录的启发：
- 我们借用了它“推荐系统需要一个外循环去持续搜索、评估、演化策略”的视角
- 但把它进一步压成了代码里的 `Planner / CandidateGenerator / Evaluator / SafetyGuard / Executor / MemoryBank`

### 2. AutoModel / Automated Recommender Evolution Paper
- arXiv: `2603.26085`
- 链接：<https://arxiv.org/pdf/2603.26085>

简要理解：
- 这篇比上面更进一步，尝试把 agentic recommender 组织成一个更像系统平台的结构
- 它会把能力拆成类似 AutoFeature、AutoTrain、AutoPerf 等模块，并强调自动实验、自动训练、自动评估、自动部署这样的闭环
- 它的 case study 更像是“自动化完成论文复现 / 模型试验流程”的 workflow showcase
- 但依然不是严格意义上的工业推荐落地论文：线上收益、延迟、成本、安全灰度这些细节都不够硬

对这个目录的启发：
- 我们借用了它“自动化实验编排”的思路
- 但补上了更显式的结构化候选、多目标 reward、上线安全阶段、经验记忆等代码接口

### 建议插入 README 的位置

建议把这一节放在 `agentic_evolution/README.md` 的开头简介之后、`## 目标` 之前。
