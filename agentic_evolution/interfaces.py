from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from .types import (
    EvaluationMetrics,
    ExecutionDecision,
    ExperimentCandidate,
    GoalSpec,
    IterationRecord,
    SafetyDecision,
)


class Planner(ABC):
    @abstractmethod
    def plan(self, goal: GoalSpec, history: Sequence[IterationRecord]) -> str:
        raise NotImplementedError


class CandidateGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        plan: str,
        goal: GoalSpec,
        history: Sequence[IterationRecord],
    ) -> ExperimentCandidate:
        raise NotImplementedError


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, candidate: ExperimentCandidate, goal: GoalSpec) -> EvaluationMetrics:
        raise NotImplementedError


class SafetyGuard(ABC):
    @abstractmethod
    def review(
        self,
        candidate: ExperimentCandidate,
        metrics: EvaluationMetrics,
        goal: GoalSpec,
    ) -> SafetyDecision:
        raise NotImplementedError


class Executor(ABC):
    @abstractmethod
    def execute(
        self,
        candidate: ExperimentCandidate,
        safety: SafetyDecision,
        metrics: EvaluationMetrics,
        goal: GoalSpec,
    ) -> ExecutionDecision:
        raise NotImplementedError


class MemoryBank(ABC):
    @abstractmethod
    def add(self, record: IterationRecord) -> None:
        raise NotImplementedError

    @abstractmethod
    def history(self) -> List[IterationRecord]:
        raise NotImplementedError
