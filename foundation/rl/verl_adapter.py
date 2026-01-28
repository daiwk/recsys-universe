"""
Adapter for verl (Volcano Engine Reinforcement Learning) framework.
https://github.com/volcengine/verl

verl is a flexible, efficient, and production-ready RL training framework
for LLM post-training, based on the HybridFlow paper.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class RLAlgorithm(Enum):
    """Supported RL algorithms."""
    PPO = "ppo"
    GRPO = "grpo"
    DPO = "dpo"
    IPO = "ipo"
    RPO = "rpo"
    REINFORCE = "reinforce"


@dataclass
class VerlTrajectory:
    """
    Trajectory data structure compatible with verl framework.
    
    In verl, a trajectory represents a complete episode of interaction
    between the policy and the environment.
    """
    # Input IDs (prompt + response)
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
    # Action mask (1 for generated tokens, 0 for prompt)
    action_mask: torch.Tensor
    
    # Log probabilities from old policy
    old_log_probs: torch.Tensor
    
    # Rewards
    rewards: torch.Tensor
    
    # Advantage estimates (computed by GAE or group-relative)
    advantages: Optional[torch.Tensor] = None
    
    # Return estimates
    returns: Optional[torch.Tensor] = None
    
    # Value estimates (from critic)
    values: Optional[torch.Tensor] = None
    
    # Sequence-level metadata
    prompt_lengths: Optional[List[int]] = None
    response_lengths: Optional[List[int]] = None
    
    # Extra info for debugging/analysis
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to(self, device: str):
        """Move trajectory to device."""
        result = VerlTrajectory(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            rewards=self.rewards.to(device),
            prompt_lengths=self.prompt_lengths,
            response_lengths=self.response_lengths,
            metadata=self.metadata,
        )
        
        if self.advantages is not None:
            result.advantages = self.advantages.to(device)
        if self.returns is not None:
            result.returns = self.returns.to(device)
        if self.values is not None:
            result.values = self.values.to(device)
            
        return result
    
    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]
    
    @property
    def seq_length(self) -> int:
        return self.input_ids.shape[1]


@dataclass
class VerlBatch:
    """Batch of trajectories for training."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    rewards: torch.Tensor
    
    # Optional fields
    values: Optional[torch.Tensor] = None
    
    def to(self, device: str):
        """Move batch to device."""
        batch = VerlBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            old_log_probs=self.old_log_probs.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            rewards=self.rewards.to(device),
        )
        if self.values is not None:
            batch.values = self.values.to(device)
        return batch


class VerlAdapter:
    """
    Adapter for verl framework compatibility.
    
    verl is a versatile RL framework developed by ByteDance/Volcano Engine
    that supports various RL algorithms including PPO, GRPO, DPO, etc.
    
    Key features of verl:
    1. HybridFlow architecture combining single-controller and multi-controller paradigms
    2. Efficient actor-learner separation
    3. Support for large-scale distributed training
    4. Flexible reward computation
    
    Reference: https://github.com/volcengine/verl
    """
    
    def __init__(
        self,
        policy_model,
        reference_model: Optional[Any] = None,
        critic_model: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
    ):
        """
        Initialize verl adapter.
        
        Args:
            policy_model: The policy model (actor)
            reference_model: Reference model for KL penalty (optional)
            critic_model: Critic model for value estimation (optional)
            reward_fn: Reward function
            algorithm: RL algorithm to use
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.critic_model = critic_model
        self.reward_fn = reward_fn
        self.algorithm = algorithm
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # verl-style configuration
        self.config = {
            'algorithm': algorithm.value,
            'use_critic': critic_model is not None,
            'use_reference': reference_model is not None,
        }
        
    def generate_trajectories(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_samples: int = 1,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[VerlTrajectory]:
        """
        Generate trajectories by sampling from the policy.
        
        This is the "Actor" phase in verl's actor-learner architecture.
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_samples: Number of samples per prompt
            stop_sequences: Optional stop sequences
            
        Returns:
            List of VerlTrajectory objects
        """
        tokenizer = self.policy_model.tokenizer
        
        # Tokenize prompts
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        trajectories = []
        
        # Generate for each prompt
        for i in range(len(prompts)):
            prompt_ids = prompt_inputs['input_ids'][i:i+1]
            prompt_mask = prompt_inputs['attention_mask'][i:i+1]
            prompt_length = prompt_mask.sum().item()
            
            # Generate multiple samples
            for _ in range(num_samples):
                with torch.no_grad():
                    output = self.policy_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                
                sequences = output.sequences
                log_probs = output.log_probs
                
                # Create action mask (1 for generated tokens, 0 for prompt)
                action_mask = torch.zeros_like(sequences)
                action_mask[:, prompt_length:] = 1
                
                # Create attention mask
                attention_mask = (sequences != tokenizer.pad_token_id).long()
                
                trajectory = VerlTrajectory(
                    input_ids=sequences,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    old_log_probs=log_probs,
                    rewards=torch.zeros(1),
                    prompt_lengths=[prompt_length],
                    response_lengths=[sequences.shape[1] - prompt_length],
                    metadata={
                        'prompt_idx': i,
                        'temperature': temperature,
                        'top_p': top_p,
                    }
                )
                
                trajectories.append(trajectory)
        
        return trajectories
    
    def compute_rewards(
        self,
        trajectories: List[VerlTrajectory],
        prompts: List[str],
        ground_truth: Optional[List[str]] = None,
    ) -> List[VerlTrajectory]:
        """
        Compute rewards for trajectories.
        
        In verl, rewards can come from:
        1. Rule-based reward functions
        2. Learned reward models
        3. LLM-as-judge
        
        Args:
            trajectories: List of trajectory objects
            prompts: Original prompts
            ground_truth: Optional ground truth answers
            
        Returns:
            Trajectories with rewards filled in
        """
        if self.reward_fn is None:
            return trajectories
            
        tokenizer = self.policy_model.tokenizer
        
        # Decode responses
        responses = []
        for traj in trajectories:
            prompt_len = traj.prompt_lengths[0] if traj.prompt_lengths else 0
            response_ids = traj.input_ids[0, prompt_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)
        
        # Compute rewards
        rewards = self.reward_fn(prompts * len(trajectories), responses, ground_truth)
        
        # Assign rewards to trajectories
        for i, traj in enumerate(trajectories):
            traj.rewards = rewards[i:i+1]
            traj.metadata['response'] = responses[i]
            
        return trajectories
    
    def compute_advantages(
        self,
        trajectories: List[VerlTrajectory],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_group_relative: bool = False,
        group_size: Optional[int] = None,
    ) -> List[VerlTrajectory]:
        """
        Compute advantages using GAE or group-relative method.
        
        Args:
            trajectories: List of trajectory objects
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            use_group_relative: Whether to use group-relative advantages (GRPO style)
            group_size: Group size for group-relative advantages
            
        Returns:
            Trajectories with advantages and returns
        """
        if use_group_relative:
            return self._compute_group_relative_advantages(
                trajectories, group_size or 8
            )
        
        # Standard GAE
        if self.critic_model is None:
            # Without critic, use reward as advantage
            for traj in trajectories:
                traj.advantages = traj.rewards.clone()
                traj.returns = traj.rewards.clone()
            return trajectories
            
        # With critic, compute GAE
        for traj in trajectories:
            with torch.no_grad():
                values = self.critic_model.get_values(
                    traj.input_ids,
                    traj.attention_mask,
                    use_last_token=False
                )
                
            # Simple advantage: reward - value at last token
            last_value = values[:, -1]
            advantages = traj.rewards - last_value
            
            traj.advantages = advantages
            traj.returns = traj.rewards
            traj.values = values
            
        return trajectories
    
    def _compute_group_relative_advantages(
        self,
        trajectories: List[VerlTrajectory],
        group_size: int,
    ) -> List[VerlTrajectory]:
        """
        Compute group-relative advantages (GRPO style).
        
        This is the key innovation in GRPO - advantages are computed
        relative to the group mean, eliminating the need for a critic.
        """
        num_groups = len(trajectories) // group_size
        
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = start_idx + group_size
            
            # Get rewards for this group
            group_rewards = torch.stack([
                trajectories[i].rewards for i in range(start_idx, end_idx)
            ])
            
            # Compute group statistics
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            
            # Compute group-relative advantages
            for i in range(start_idx, end_idx):
                advantage = (trajectories[i].rewards - group_mean) / group_std
                trajectories[i].advantages = advantage
                trajectories[i].returns = trajectories[i].rewards
                
        return trajectories
    
    def prepare_batch(
        self,
        trajectories: List[VerlTrajectory],
    ) -> VerlBatch:
        """
        Prepare a batch of trajectories for training.
        
        Args:
            trajectories: List of trajectory objects
            
        Returns:
            VerlBatch object
        """
        # Stack all tensors
        input_ids = torch.cat([t.input_ids for t in trajectories], dim=0)
        attention_mask = torch.cat([t.attention_mask for t in trajectories], dim=0)
        action_mask = torch.cat([t.action_mask for t in trajectories], dim=0)
        old_log_probs = torch.cat([t.old_log_probs for t in trajectories], dim=0)
        rewards = torch.cat([t.rewards for t in trajectories], dim=0)
        
        # Check for advantages and returns
        if trajectories[0].advantages is None:
            raise ValueError("Advantages not computed. Call compute_advantages first.")
        
        advantages = torch.cat([t.advantages for t in trajectories], dim=0)
        returns = torch.cat([t.returns for t in trajectories], dim=0)
        
        batch = VerlBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            rewards=rewards,
        )
        
        # Add optional fields
        if trajectories[0].values is not None:
            values = torch.cat([t.values for t in trajectories], dim=0)
            batch.values = values
            
        return batch
    
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reduction: str = "batchmean",
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            reduction: Reduction method ("batchmean", "mean", "sum", "none")
            
        Returns:
            KL divergence tensor
        """
        if self.reference_model is None:
            return torch.zeros(1, device=input_ids.device)
            
        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_probs = torch.exp(ref_log_probs)
        
        # Get policy log probs
        policy_outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        policy_logits = policy_outputs.logits[:, :-1, :]
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        
        # Compute KL divergence: KL(ref || policy)
        kl_div = torch.sum(
            ref_probs * (ref_log_probs - policy_log_probs),
            dim=-1
        )
        
        if reduction == "batchmean":
            return kl_div.mean()
        elif reduction == "mean":
            return kl_div.mean()
        elif reduction == "sum":
            return kl_div.sum()
        else:  # "none"
            return kl_div
    
    def get_metrics(self) -> Dict[str, float]:
        """Get training metrics."""
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'algorithm': self.algorithm.value,
        }
    
    def step(self):
        """Increment global step."""
        self.global_step += 1
        
    def set_epoch(self, epoch: int):
        """Set current epoch."""
        self.epoch = epoch
