"""
Parallel Agent Reinforcement Learning (PARL) for Agent Swarm.

This module provides a minimal, extensible implementation that mirrors the
Agent Swarm paper's PARL stage:
- multiple agents roll out in parallel
- trajectories are aggregated at the swarm level
- shared (swarm) advantages are computed for policy updates
- local advantages remain available for value updates

The design is intentionally lightweight and can be swapped to Verl's
parallel executor when running in a Verl-enabled environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple
import concurrent.futures
import os
import time

import numpy as np


class SwarmEnvironment(Protocol):
    """Protocol for environments used by swarm agents."""

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        """Execute an action and return (next_obs, reward, done, info)."""


@dataclass
class Trajectory:
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    values: List[float]
    log_probs: List[float]

    def total_reward(self) -> float:
        return float(np.sum(self.rewards))


@dataclass
class SwarmBatch:
    """Container for swarm-level rollouts."""

    trajectories: List[Trajectory]
    swarm_advantages: np.ndarray
    swarm_rewards: np.ndarray
    swarm_values: np.ndarray


@dataclass
class RewardComponents:
    """Components of the PARL reward."""

    parallel_reward: float
    finish_reward: float
    performance_reward: float

    def total(self, lambda_parallel: float, lambda_finish: float) -> float:
        return (
            lambda_parallel * self.parallel_reward
            + lambda_finish * self.finish_reward
            + self.performance_reward
        )


@dataclass
class PARLRewardConfig:
    """Hyperparameters for PARL reward shaping."""

    lambda_parallel: float = 1.0
    lambda_finish: float = 1.0

    def anneal(self, factor: float) -> None:
        """Anneal auxiliary reward coefficients toward zero."""
        self.lambda_parallel = max(0.0, self.lambda_parallel * factor)
        self.lambda_finish = max(0.0, self.lambda_finish * factor)


@dataclass
class CriticalStepTracker:
    """Tracks critical steps as in the PARL report."""

    stages: List[Tuple[int, List[int]]] = field(default_factory=list)

    def add_stage(self, main_steps: int, subagent_steps: Sequence[int]) -> None:
        self.stages.append((main_steps, list(subagent_steps)))

    def total(self) -> int:
        total_steps = 0
        for main_steps, sub_steps in self.stages:
            total_steps += main_steps + (max(sub_steps) if sub_steps else 0)
        return total_steps


@dataclass
class SwarmAgent:
    """Agent wrapper with pluggable policy/value functions."""

    agent_id: str
    policy: Callable[[np.ndarray], Tuple[np.ndarray, float]]
    value_fn: Callable[[np.ndarray], float]
    frozen: bool = False

    def act(self, observation: np.ndarray) -> Tuple[np.ndarray, float, float]:
        action, log_prob = self.policy(observation)
        value = self.value_fn(observation)
        return action, log_prob, value


@dataclass
class PARLConfig:
    """Configuration for PARL training."""

    num_parallel_envs: int = 4
    rollout_horizon: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    swarm_sync_interval: int = 1
    parallel_backend: str = "concurrent"
    swarm_reward_aggregation: str = "mean"
    swarm_value_aggregation: str = "mean"
    reward_config: PARLRewardConfig = field(default_factory=PARLRewardConfig)


class ParallelBackend(Protocol):
    """Parallel backend abstraction for running rollouts."""

    def map(self, fn: Callable[..., Trajectory], tasks: Iterable[Tuple[int, SwarmAgent, SwarmEnvironment]]) -> List[Trajectory]:
        """Run fn across tasks and return trajectories."""


@dataclass
class ConcurrentBackend:
    """Default parallel backend using concurrent.futures."""

    max_workers: Optional[int] = None

    def map(self, fn: Callable[..., Trajectory], tasks: Iterable[Tuple[int, SwarmAgent, SwarmEnvironment]]) -> List[Trajectory]:
        results: List[Trajectory] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn, *task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return results


@dataclass
class PARLTrainer:
    """Trainer orchestrating Parallel Agent RL rollouts and updates."""

    config: PARLConfig
    parallel_backend: ParallelBackend = field(default_factory=ConcurrentBackend)
    critical_steps: CriticalStepTracker = field(default_factory=CriticalStepTracker)

    def _rollout(self, env_idx: int, agent: SwarmAgent, env: SwarmEnvironment) -> Trajectory:
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        dones: List[bool] = []
        values: List[float] = []
        log_probs: List[float] = []

        obs = env.reset()
        for _ in range(self.config.rollout_horizon):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            values.append(float(value))
            log_probs.append(float(log_prob))

            obs = next_obs
            if done:
                obs = env.reset()

        return Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            log_probs=log_probs,
        )

    def compute_parl_reward(self, components: RewardComponents) -> float:
        return components.total(
            lambda_parallel=self.config.reward_config.lambda_parallel,
            lambda_finish=self.config.reward_config.lambda_finish,
        )

    def collect_trajectories(
        self,
        agents: Sequence[SwarmAgent],
        envs: Sequence[SwarmEnvironment],
    ) -> List[Trajectory]:
        if len(envs) < self.config.num_parallel_envs:
            raise ValueError("Not enough environments for the configured parallelism.")
        tasks = []
        for env_idx in range(self.config.num_parallel_envs):
            agent = agents[env_idx % len(agents)]
            env = envs[env_idx]
            tasks.append((env_idx, agent, env))
        return self.parallel_backend.map(self._rollout, tasks)

    def compute_advantages(self, trajectory: Trajectory) -> np.ndarray:
        rewards = np.array(trajectory.rewards, dtype=np.float32)
        values = np.array(trajectory.values + [0.0], dtype=np.float32)
        dones = np.array(trajectory.dones, dtype=np.bool_)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.config.gamma * values[t + 1] * next_nonterminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        return advantages

    def _aggregate(self, values: np.ndarray, mode: str) -> np.ndarray:
        if mode == "mean":
            return np.mean(values, axis=0)
        if mode == "sum":
            return np.sum(values, axis=0)
        raise ValueError(f"Unsupported aggregation mode: {mode}")

    def compute_swarm_advantages(self, trajectories: Sequence[Trajectory]) -> SwarmBatch:
        if not trajectories:
            return SwarmBatch(trajectories=[], swarm_advantages=np.array([]), swarm_rewards=np.array([]), swarm_values=np.array([]))

        rewards = np.stack([np.array(traj.rewards, dtype=np.float32) for traj in trajectories], axis=0)
        values = np.stack([np.array(traj.values, dtype=np.float32) for traj in trajectories], axis=0)
        dones = np.stack([np.array(traj.dones, dtype=np.bool_) for traj in trajectories], axis=0)

        swarm_rewards = self._aggregate(rewards, self.config.swarm_reward_aggregation)
        swarm_values = self._aggregate(values, self.config.swarm_value_aggregation)
        swarm_dones = np.any(dones, axis=0)

        swarm_values_with_bootstrap = np.concatenate([swarm_values, np.zeros(1, dtype=np.float32)])
        swarm_advantages = np.zeros_like(swarm_rewards)
        last_gae = 0.0
        for t in reversed(range(len(swarm_rewards))):
            next_nonterminal = 1.0 - float(swarm_dones[t])
            delta = swarm_rewards[t] + self.config.gamma * swarm_values_with_bootstrap[t + 1] * next_nonterminal - swarm_values_with_bootstrap[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * last_gae
            swarm_advantages[t] = last_gae

        return SwarmBatch(
            trajectories=list(trajectories),
            swarm_advantages=swarm_advantages,
            swarm_rewards=swarm_rewards,
            swarm_values=swarm_values,
        )

    def aggregate_metrics(self, trajectories: Sequence[Trajectory]) -> Dict[str, float]:
        returns = [traj.total_reward() for traj in trajectories]
        mean_return = float(np.mean(returns)) if returns else 0.0
        max_return = float(np.max(returns)) if returns else 0.0
        min_return = float(np.min(returns)) if returns else 0.0
        return {
            "mean_return": mean_return,
            "max_return": max_return,
            "min_return": min_return,
            "num_trajectories": float(len(trajectories)),
            "critical_steps": float(self.critical_steps.total()),
        }

    def train_step(
        self,
        agents: Sequence[SwarmAgent],
        envs: Sequence[SwarmEnvironment],
        update_fn: Callable[[SwarmAgent, Trajectory, np.ndarray, np.ndarray], None],
    ) -> Dict[str, float]:
        start = time.time()
        trajectories = self.collect_trajectories(agents, envs)
        swarm_batch = self.compute_swarm_advantages(trajectories)
        for agent, trajectory in zip(agents, trajectories):
            local_advantages = self.compute_advantages(trajectory)
            update_fn(agent, trajectory, swarm_batch.swarm_advantages, local_advantages)
        metrics = self.aggregate_metrics(trajectories)
        metrics.update(
            {
                "swarm_return": float(np.sum(swarm_batch.swarm_rewards)) if swarm_batch.swarm_rewards.size else 0.0,
                "swarm_advantage_mean": float(np.mean(swarm_batch.swarm_advantages)) if swarm_batch.swarm_advantages.size else 0.0,
            }
        )
        metrics["step_time"] = time.time() - start
        return metrics


def build_verl_backend() -> ParallelBackend:
    """
    Construct a Verl-backed parallel executor.

    This function assumes Verl is installed. It is intentionally isolated so
    downstream projects can replace the default backend with Verl's worker pool
    implementation when available.
    """
    from verl.utils.parallel import WorkerPool

    class VerlBackend(ParallelBackend):
        def __init__(self, num_workers: int):
            self.pool = WorkerPool(num_workers=num_workers)

        def map(self, fn: Callable[..., Trajectory], tasks: Iterable[Tuple[int, SwarmAgent, SwarmEnvironment]]) -> List[Trajectory]:
            def _call(task: Tuple[int, SwarmAgent, SwarmEnvironment]) -> Trajectory:
                return fn(*task)

            return list(self.pool.map(_call, list(tasks)))

    return VerlBackend(num_workers=os.cpu_count() or 4)
