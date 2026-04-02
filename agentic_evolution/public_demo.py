from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .agents import (
    HeuristicPlanner,
    InMemoryBank,
    SimpleExecutor,
    TemplateCandidateGenerator,
    ThresholdSafetyGuard,
)
from .interfaces import Evaluator
from .loop import AgenticEvolutionLoop
from .reward import MultiObjectiveReward, RewardWeights
from .types import EvaluationMetrics, ExperimentCandidate, GoalSpec


@dataclass
class MovieLensStats:
    num_users: int
    num_movies: int
    num_ratings: int
    mean_rating: float
    mean_ratings_per_user: float
    mean_ratings_per_movie: float
    action_share: float
    drama_share: float
    sci_fi_share: float


class PublicMovieLensEvaluator(Evaluator):
    """Dataset-backed evaluator for the agentic evolution outer loop.

    This is still a demo evaluator, but unlike the original `SimulatedEvaluator`,
    it reads a real public dataset and makes the offline metrics depend on actual
    dataset properties such as rating density and genre composition.
    """

    def __init__(self, dataset_dir: str | Path):
        dataset_dir = Path(dataset_dir)
        ratings = pd.read_csv(dataset_dir / "ratings.csv")
        movies = pd.read_csv(dataset_dir / "movies.csv")

        self.stats = MovieLensStats(
            num_users=int(ratings["userId"].nunique()),
            num_movies=int(movies["movieId"].nunique()),
            num_ratings=int(len(ratings)),
            mean_rating=float(ratings["rating"].mean()),
            mean_ratings_per_user=float(ratings.groupby("userId").size().mean()),
            mean_ratings_per_movie=float(ratings.groupby("movieId").size().mean()),
            action_share=float(movies["genres"].fillna("").str.contains("Action").mean()),
            drama_share=float(movies["genres"].fillna("").str.contains("Drama").mean()),
            sci_fi_share=float(movies["genres"].fillna("").str.contains("Sci-Fi").mean()),
        )

    def evaluate(self, candidate: ExperimentCandidate, goal: GoalSpec) -> EvaluationMetrics:
        s = self.stats
        recall_top_k = float(candidate.changes.get("recall_top_k", 100))
        ranker_hidden_dim = float(candidate.changes.get("ranker_hidden_dim", 64))
        rerank_distill = bool(
            candidate.changes.get("feature_flags", {}).get("enable_rerank_distillation", False)
        )
        cross_bundle = bool(
            candidate.changes.get("feature_flags", {}).get("enable_cross_feature_bundle", False)
        )

        density = s.num_ratings / max(1.0, s.num_users * s.num_movies)
        genre_signal = 0.5 * s.drama_share + 0.5 * s.sci_fi_share

        primary_gain = (
            0.002
            + 0.00005 * recall_top_k
            + 0.00012 * ranker_hidden_dim
            + 0.8 * density
            + 0.02 * genre_signal
        )
        if rerank_distill:
            primary_gain += 0.008
        if cross_bundle:
            primary_gain += 0.004

        latency_ms = 15.0 + 0.03 * recall_top_k + 0.05 * ranker_hidden_dim + 8.0 * density
        bad_case_rate = max(0.0, 0.08 - 0.01 * int(cross_bundle) - 0.005 * int(rerank_distill))
        training_budget = 0.08 + 0.001 * ranker_hidden_dim + 0.0001 * recall_top_k
        rollout_risk = 0.015 + 0.01 * int(rerank_distill)

        return EvaluationMetrics(
            primary_metrics={goal.primary_metric: float(primary_gain)},
            guardrail_metrics={
                "latency_ms": float(latency_ms),
                "bad_case_rate": float(bad_case_rate),
            },
            cost_metrics={
                "budget": float(training_budget),
            },
            risk_metrics={
                "rollout_risk": float(rollout_risk),
            },
            debug={
                "dataset_stats": s.__dict__,
                "candidate": candidate.changes,
            },
        )


def build_public_movielens_loop(dataset_dir: str | Path) -> AgenticEvolutionLoop:
    reward_model = MultiObjectiveReward(
        RewardWeights(primary=1.0, guardrail=2.0, cost=0.3, risk=0.8)
    )
    evaluator = PublicMovieLensEvaluator(dataset_dir)
    return AgenticEvolutionLoop(
        planner=HeuristicPlanner(),
        candidate_generator=TemplateCandidateGenerator(),
        evaluator=evaluator,
        reward_model=reward_model,
        safety_guard=ThresholdSafetyGuard(reward_model),
        executor=SimpleExecutor(),
        memory_bank=InMemoryBank(),
    )


def main() -> None:
    dataset_dir = Path(__file__).parent / "data" / "ml-latest-small-public-subset"
    loop = build_public_movielens_loop(dataset_dir)
    summary = loop.run(
        GoalSpec(
            goal="Improve CTR under latency constraint on a public MovieLens small subset",
            primary_metric="ctr",
            target_delta=0.01,
            guardrails={"bad_case_rate": 0.08},
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
        print()


if __name__ == "__main__":
    main()
