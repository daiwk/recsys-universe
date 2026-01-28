"""
Foundation: Distributed RL Framework for Qwen3-7b

Based on verl framework with PARL-style distributed training.
Reference: Kimi-K2.5 RL implementation patterns.
"""

__version__ = "0.1.0"

from foundation.models.qwen3_policy import Qwen3PolicyModel
from foundation.models.qwen3_critic import Qwen3CriticModel
from foundation.rl.ppo_trainer import PPOTrainer
from foundation.rl.grpo_trainer import GRPOTrainer

__all__ = [
    "Qwen3PolicyModel",
    "Qwen3CriticModel", 
    "PPOTrainer",
    "GRPOTrainer",
]
