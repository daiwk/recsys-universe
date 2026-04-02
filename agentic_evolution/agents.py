from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .interfaces import CandidateGenerator, Evaluator, Executor, MemoryBank, Planner, SafetyGuard
from .reward import MultiObjectiveReward
from .types import (
    EvaluationMetrics,
    ExecutionDecision,
    ExecutionStage,
    ExperimentCandidate,
    GoalSpec,
    IterationRecord,
    SafetyDecision,
)


class InMemoryBank(MemoryBank):
    def __init__(self):
        self._records: List[IterationRecord] = []

    def add(self, record: IterationRecord) -> None:
        self._records.append(record)

    def history(self) -> List[IterationRecord]:
        return list(self._records)


class HeuristicPlanner(Planner):
    def plan(self, goal: GoalSpec, history: Sequence[IterationRecord]) -> str:
        if not history:
            return (
                f"bootstrap search for goal='{goal.goal}', first try safer candidate with limited serving risk"
            )

        best = max(history, key=lambda r: r.reward.total_reward)
        if best.reward.hard_constraint_violations:
            return "repair previous best candidate by reducing latency/cost/risk before further rollout"
        if best.execution.stage in {ExecutionStage.SHADOW, ExecutionStage.CANARY}:
            return "expand rollout gradually while preserving current winning direction"
        return "search a nearby variant around the current best candidate"


class TemplateCandidateGenerator(CandidateGenerator):
    def generate(
        self,
        plan: str,
        goal: GoalSpec,
        history: Sequence[IterationRecord],
    ) -> ExperimentCandidate:
        idx = len(history) + 1
        base_k = 100 + idx * 20
        ranker_hidden = 64 + 16 * idx
        serving_gate = max(0.05, 0.20 - idx * 0.02)
        return ExperimentCandidate(
            candidate_id=f"cand_{idx}",
            name=f"agentic_candidate_{idx}",
            description=f"Iteration {idx} candidate derived from plan: {plan}",
            changes={
                "recall_top_k": base_k,
                "ranker_hidden_dim": ranker_hidden,
                "feature_flags": {
                    "enable_cross_feature_bundle": idx % 2 == 1,
                    "enable_rerank_distillation": idx >= 2,
                },
                "rollout_policy": {
                    "max_canary_ratio": min(0.5, 0.1 * idx),
                    "safe_gate": serving_gate,
                },
            },
            expected_impact={
                goal.primary_metric: 0.01 * idx,
                "latency_ms": 5.0 * idx,
            },
            rationale="Heuristic candidate that jointly tweaks retrieval breadth, ranker capacity, and rollout policy.",
            parent_candidate_id=history[-1].candidate.candidate_id if history else None,
            metadata={"plan": plan},
        )


class SimulatedEvaluator(Evaluator):
    """A deterministic evaluator for demo and local wiring tests.

    后续你可以把这里替换成：
    - 提交训练任务
    - 拉取离线验证集结果
    - 拉取 replay / shadow / canary 指标
    """

    def evaluate(self, candidate: ExperimentCandidate, goal: GoalSpec) -> EvaluationMetrics:
        recall_top_k = float(candidate.changes.get("recall_top_k", 100))
        ranker_hidden_dim = float(candidate.changes.get("ranker_hidden_dim", 64))
        rerank_distill = bool(
            candidate.changes.get("feature_flags", {}).get("enable_rerank_distillation", False)
        )

        primary_gain = 0.00015 * recall_top_k + 0.0004 * ranker_hidden_dim
        if rerank_distill:
            primary_gain += 0.01

        latency_ms = 25.0 + 0.05 * recall_top_k + 0.08 * ranker_hidden_dim
        training_budget = 0.2 + 0.002 * ranker_hidden_dim
        rollout_risk = 0.03 if rerank_distill else 0.01
        bad_case_rate = max(0.0, 0.06 - 0.005 * int(rerank_distill))

        return EvaluationMetrics(
            primary_metrics={goal.primary_metric: primary_gain},
            guardrail_metrics={
                "latency_ms": latency_ms,
                "bad_case_rate": bad_case_rate,
            },
            cost_metrics={
                "budget": training_budget,
                "gpu_day": training_budget * 2.5,
            },
            risk_metrics={
                "rollout_risk": rollout_risk,
            },
            debug={
                "candidate_name": candidate.name,
                "candidate_changes": candidate.changes,
            },
        )


class ThresholdSafetyGuard(SafetyGuard):
    def __init__(self, reward_model: MultiObjectiveReward):
        self.reward_model = reward_model

    def review(
        self,
        candidate: ExperimentCandidate,
        metrics: EvaluationMetrics,
        goal: GoalSpec,
    ) -> SafetyDecision:
        reward = self.reward_model.compute(metrics, goal)
        if reward.hard_constraint_violations:
            return SafetyDecision(
                approved=False,
                target_stage=ExecutionStage.REJECT,
                reasons=["hard constraints violated"],
                blocked_by=reward.hard_constraint_violations,
            )

        primary = metrics.primary_metrics.get(goal.primary_metric, 0.0)
        if primary <= goal.target_delta:
            return SafetyDecision(
                approved=False,
                target_stage=ExecutionStage.REJECT,
                reasons=["primary metric does not beat target delta"],
                blocked_by=[f"{goal.primary_metric}={primary:.4f} <= target_delta={goal.target_delta:.4f}"],
            )

        latency = metrics.guardrail_metrics.get("latency_ms", 0.0)
        if latency < (goal.latency_limit_ms or float("inf")) * 0.8:
            stage = ExecutionStage.CANARY
        else:
            stage = ExecutionStage.SHADOW

        return SafetyDecision(
            approved=True,
            target_stage=stage,
            reasons=["candidate passes current offline safety checks"],
            blocked_by=[],
        )


class SimpleExecutor(Executor):
    def execute(
        self,
        candidate: ExperimentCandidate,
        safety: SafetyDecision,
        metrics: EvaluationMetrics,
        goal: GoalSpec,
    ) -> ExecutionDecision:
        if not safety.approved:
            return ExecutionDecision(
                stage=ExecutionStage.REJECT,
                executed=False,
                message="candidate rejected before rollout",
            )

        rollout_ratio = 0.0
        if safety.target_stage == ExecutionStage.SHADOW:
            rollout_ratio = 0.0
        elif safety.target_stage == ExecutionStage.CANARY:
            rollout_ratio = min(
                0.1,
                candidate.changes.get("rollout_policy", {}).get("max_canary_ratio", 0.1),
            )

        return ExecutionDecision(
            stage=safety.target_stage,
            executed=True,
            message=f"candidate {candidate.name} sent to {safety.target_stage.value}",
            rollout_ratio=rollout_ratio,
            metadata={
                "primary_metric": metrics.primary_metrics.get(goal.primary_metric, 0.0),
            },
        )
