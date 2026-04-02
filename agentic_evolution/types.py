from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutionStage(str, Enum):
    OFFLINE = "offline"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REJECT = "reject"


@dataclass
class GoalSpec:
    goal: str
    primary_metric: str = "ctr"
    higher_is_better: bool = True
    target_delta: float = 0.0
    guardrails: Dict[str, float] = field(default_factory=dict)
    budget_limit: Optional[float] = None
    latency_limit_ms: Optional[float] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentCandidate:
    candidate_id: str
    name: str
    description: str
    changes: Dict[str, Any]
    expected_impact: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    parent_candidate_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    primary_metrics: Dict[str, float]
    guardrail_metrics: Dict[str, float] = field(default_factory=dict)
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: float = 0.0) -> float:
        if key in self.primary_metrics:
            return self.primary_metrics[key]
        if key in self.guardrail_metrics:
            return self.guardrail_metrics[key]
        if key in self.cost_metrics:
            return self.cost_metrics[key]
        if key in self.risk_metrics:
            return self.risk_metrics[key]
        return default


@dataclass
class RewardBreakdown:
    total_reward: float
    primary_reward: float
    guardrail_penalty: float
    cost_penalty: float
    risk_penalty: float
    hard_constraint_violations: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyDecision:
    approved: bool
    target_stage: ExecutionStage
    reasons: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)


@dataclass
class ExecutionDecision:
    stage: ExecutionStage
    executed: bool
    message: str
    rollout_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    plan: str
    candidate: ExperimentCandidate
    metrics: EvaluationMetrics
    reward: RewardBreakdown
    safety: SafetyDecision
    execution: ExecutionDecision


@dataclass
class LoopSummary:
    goal: GoalSpec
    records: List[IterationRecord] = field(default_factory=list)

    @property
    def best_record(self) -> Optional[IterationRecord]:
        if not self.records:
            return None
        return max(self.records, key=lambda r: r.reward.total_reward)

    def pretty(self) -> str:
        lines = [
            f"Goal: {self.goal.goal}",
            f"Primary metric: {self.goal.primary_metric}",
            f"Iterations: {len(self.records)}",
        ]
        best = self.best_record
        if best is not None:
            lines.extend([
                f"Best candidate: {best.candidate.name}",
                f"Best reward: {best.reward.total_reward:.4f}",
                f"Execution stage: {best.execution.stage.value}",
                f"Execution message: {best.execution.message}",
            ])
        return "\n".join(lines)
