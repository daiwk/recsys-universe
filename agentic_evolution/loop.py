from __future__ import annotations

from dataclasses import dataclass

from .interfaces import CandidateGenerator, Evaluator, Executor, MemoryBank, Planner, SafetyGuard
from .reward import MultiObjectiveReward
from .types import GoalSpec, IterationRecord, LoopSummary


class AgenticEvolutionLoop:
    def __init__(
        self,
        planner: Planner,
        candidate_generator: CandidateGenerator,
        evaluator: Evaluator,
        reward_model: MultiObjectiveReward,
        safety_guard: SafetyGuard,
        executor: Executor,
        memory_bank: MemoryBank,
    ):
        self.planner = planner
        self.candidate_generator = candidate_generator
        self.evaluator = evaluator
        self.reward_model = reward_model
        self.safety_guard = safety_guard
        self.executor = executor
        self.memory_bank = memory_bank

    def run(self, goal: str | GoalSpec, max_iterations: int = 3) -> LoopSummary:
        goal_spec = goal if isinstance(goal, GoalSpec) else GoalSpec(goal=goal)
        summary = LoopSummary(goal=goal_spec)

        for iteration in range(1, max_iterations + 1):
            history = self.memory_bank.history()
            plan = self.planner.plan(goal_spec, history)
            candidate = self.candidate_generator.generate(plan, goal_spec, history)
            metrics = self.evaluator.evaluate(candidate, goal_spec)
            reward = self.reward_model.compute(metrics, goal_spec)
            safety = self.safety_guard.review(candidate, metrics, goal_spec)
            execution = self.executor.execute(candidate, safety, metrics, goal_spec)

            record = IterationRecord(
                iteration=iteration,
                plan=plan,
                candidate=candidate,
                metrics=metrics,
                reward=reward,
                safety=safety,
                execution=execution,
            )
            self.memory_bank.add(record)
            summary.records.append(record)

        return summary
