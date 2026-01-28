"""
Rollout generator for collecting trajectories.
"""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from foundation.rl.verl_adapter import VerlTrajectory


@dataclass
class RolloutConfig:
    """Configuration for rollout generation."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_samples: int = 1


class RolloutGenerator:
    """
    Generator for collecting rollouts/trajectories from a policy.
    """
    
    def __init__(
        self,
        policy_model,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize rollout generator.
        
        Args:
            policy_model: The policy model to generate rollouts from
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.policy_model = policy_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
    def generate(
        self,
        prompts: List[str],
        num_samples: int = 1,
    ) -> List[VerlTrajectory]:
        """
        Generate rollouts for given prompts.
        
        Args:
            prompts: List of prompt strings
            num_samples: Number of samples per prompt
            
        Returns:
            List of VerlTrajectory objects
        """
        from foundation.rl.verl_adapter import VerlAdapter
        
        adapter = VerlAdapter(policy_model=self.policy_model)
        
        trajectories = adapter.generate_trajectories(
            prompts=prompts,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            num_samples=num_samples,
        )
        
        return trajectories
