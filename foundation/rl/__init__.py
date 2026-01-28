"""
RL algorithm components for Foundation framework.

Integrates with verl (Volcano Engine Reinforcement Learning) framework:
https://github.com/volcengine/verl
"""

# Core RL trainers
from foundation.rl.ppo_trainer import PPOTrainer, PPOConfig
from foundation.rl.grpo_trainer import GRPOTrainer, GRPOConfig
from foundation.rl.reward_model import RewardModel, RewardConfig, MathRewardModel, CodeRewardModel

# verl integration
from foundation.rl.verl_integration import (
    VerlTrainerWrapper,
    VerlTrainerConfig,
    VerlDataProtoAdapter,
    VerlWorker,
    VerlWorkerGroup,
    is_verl_available,
    get_verl_version,
)

# Legacy verl adapter (kept for compatibility)
from foundation.rl.verl_adapter import VerlAdapter, VerlTrajectory, RLAlgorithm

__all__ = [
    # Core RL trainers
    "PPOTrainer",
    "PPOConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "RewardModel",
    "RewardConfig",
    "MathRewardModel",
    "CodeRewardModel",
    # verl integration
    "VerlTrainerWrapper",
    "VerlTrainerConfig",
    "VerlDataProtoAdapter",
    "VerlWorker",
    "VerlWorkerGroup",
    "is_verl_available",
    "get_verl_version",
    # Legacy
    "VerlAdapter",
    "VerlTrajectory",
    "RLAlgorithm",
]
