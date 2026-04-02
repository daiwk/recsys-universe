from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .types import EvaluationMetrics, GoalSpec, RewardBreakdown


@dataclass
class RewardWeights:
    primary: float = 1.0
    guardrail: float = 1.0
    cost: float = 0.2
    risk: float = 0.5


class MultiObjectiveReward:
    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    def compute(self, metrics: EvaluationMetrics, goal: GoalSpec) -> RewardBreakdown:
        primary_value = metrics.primary_metrics.get(goal.primary_metric, 0.0)
        direction = 1.0 if goal.higher_is_better else -1.0
        primary_reward = direction * (primary_value - goal.target_delta)

        violations: List[str] = []
        guardrail_penalty = 0.0
        for metric_name, threshold in goal.guardrails.items():
            value = metrics.guardrail_metrics.get(metric_name)
            if value is None:
                continue
            # 默认 guardrail 都按照“越小越好”的工业约束处理
            overflow = max(0.0, value - threshold)
            guardrail_penalty += overflow
            if overflow > 0:
                violations.append(
                    f"guardrail:{metric_name}={value:.4f} exceeds {threshold:.4f}"
                )

        cost_penalty = sum(float(v) for v in metrics.cost_metrics.values())
        risk_penalty = sum(float(v) for v in metrics.risk_metrics.values())

        if goal.budget_limit is not None:
            cost_value = metrics.cost_metrics.get("budget", 0.0)
            if cost_value > goal.budget_limit:
                violations.append(
                    f"budget={cost_value:.4f} exceeds {goal.budget_limit:.4f}"
                )

        if goal.latency_limit_ms is not None:
            latency = metrics.guardrail_metrics.get("latency_ms", 0.0)
            if latency > goal.latency_limit_ms:
                violations.append(
                    f"latency_ms={latency:.4f} exceeds {goal.latency_limit_ms:.4f}"
                )

        total = (
            self.weights.primary * primary_reward
            - self.weights.guardrail * guardrail_penalty
            - self.weights.cost * cost_penalty
            - self.weights.risk * risk_penalty
        )

        return RewardBreakdown(
            total_reward=total,
            primary_reward=primary_reward,
            guardrail_penalty=guardrail_penalty,
            cost_penalty=cost_penalty,
            risk_penalty=risk_penalty,
            hard_constraint_violations=violations,
            extra={
                "primary_value": primary_value,
                "weights": {
                    "primary": self.weights.primary,
                    "guardrail": self.weights.guardrail,
                    "cost": self.weights.cost,
                    "risk": self.weights.risk,
                },
            },
        )
