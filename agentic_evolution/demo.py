from __future__ import annotations

from .agents import (
    HeuristicPlanner,
    InMemoryBank,
    SimpleExecutor,
    SimulatedEvaluator,
    TemplateCandidateGenerator,
    ThresholdSafetyGuard,
)
from .loop import AgenticEvolutionLoop
from .reward import MultiObjectiveReward, RewardWeights
from .types import GoalSpec


def build_demo_loop() -> AgenticEvolutionLoop:
    reward_model = MultiObjectiveReward(
        RewardWeights(primary=1.0, guardrail=2.0, cost=0.3, risk=0.8)
    )
    return AgenticEvolutionLoop(
        planner=HeuristicPlanner(),
        candidate_generator=TemplateCandidateGenerator(),
        evaluator=SimulatedEvaluator(),
        reward_model=reward_model,
        safety_guard=ThresholdSafetyGuard(reward_model),
        executor=SimpleExecutor(),
        memory_bank=InMemoryBank(),
    )


def main() -> None:
    loop = build_demo_loop()
    summary = loop.run(
        GoalSpec(
            goal="在不显著增加延迟的前提下，提高首页推荐 CTR",
            primary_metric="ctr",
            target_delta=0.01,
            guardrails={
                "bad_case_rate": 0.08,
            },
            budget_limit=0.5,
            latency_limit_ms=45.0,
        ),
        max_iterations=3,
    )

    print(summary.pretty())
    print("\nDetailed records:\n")
    for record in summary.records:
        print(f"[Iteration {record.iteration}] {record.candidate.name}")
        print(f"  plan      : {record.plan}")
        print(f"  primary   : {record.metrics.primary_metrics}")
        print(f"  guardrail : {record.metrics.guardrail_metrics}")
        print(f"  cost      : {record.metrics.cost_metrics}")
        print(f"  risk      : {record.metrics.risk_metrics}")
        print(f"  reward    : {record.reward.total_reward:.4f}")
        print(f"  safety    : approved={record.safety.approved}, stage={record.safety.target_stage.value}")
        print(f"  execution : {record.execution.message}")
        if record.reward.hard_constraint_violations:
            print(f"  violations: {record.reward.hard_constraint_violations}")
        print()


if __name__ == "__main__":
    main()
