"""
GRPO (Group Relative Policy Optimization) Trainer.
Implements GRPO similar to Kimi-K2.5's approach.

GRPO is a variant of PPO that uses group-relative advantages,
which eliminates the need for a separate critic model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from foundation.rl.verl_adapter import VerlAdapter, VerlTrajectory
from foundation.utils.logging_utils import get_logger


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Learning rate
    policy_lr: float = 1e-6
    
    # GRPO hyperparameters
    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    entropy_coef: float = 0.01
    
    # Group settings
    group_size: int = 8  # Number of samples per prompt
    
    # Training settings
    num_epochs: int = 2
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer.
    
    GRPO is inspired by Kimi-K2.5's RL approach and has several advantages:
    1. No critic model needed - uses group-relative advantages
    2. More stable training due to relative reward normalization
    3. Better sample efficiency
    
    Key idea: For each prompt, generate a group of responses and compute
    advantages relative to the group mean.
    """
    
    def __init__(
        self,
        policy_model,
        reference_model: Optional[nn.Module] = None,
        reward_fn: Optional[Any] = None,
        config: Optional[GRPOConfig] = None,
    ):
        self.config = config or GRPOConfig()
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_fn = reward_fn
        
        # Initialize verl adapter (no critic model)
        self.verl_adapter = VerlAdapter(
            policy_model=policy_model,
            reference_model=reference_model,
            critic_model=None,  # GRPO doesn't need critic
            reward_fn=reward_fn,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.policy_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.logger = get_logger("GRPOTrainer")
        
        # Metrics
        self.metrics = {
            'policy_loss': [],
            'kl_div': [],
            'entropy': [],
            'reward_mean': [],
            'reward_std': [],
            'group_relative_advantage': [],
        }
        
    def train(
        self,
        prompts: List[str],
        ground_truth: Optional[List[str]] = None,
        num_iterations: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train the policy using GRPO.
        
        Args:
            prompts: List of training prompts
            ground_truth: Optional ground truth answers
            num_iterations: Number of training iterations
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Starting GRPO training for {num_iterations} iterations")
        self.logger.info(f"Group size: {self.config.group_size}")
        
        for iteration in range(num_iterations):
            self.epoch = iteration
            self.verl_adapter.set_epoch(iteration)
            
            # Generate trajectories (grouped by prompt)
            self.logger.info(f"Iteration {iteration}: Generating trajectories...")
            grouped_trajectories = self._generate_grouped_trajectories(prompts)
            
            # Compute rewards for all trajectories
            self.logger.info(f"Iteration {iteration}: Computing rewards...")
            all_trajectories = []
            for prompt_idx, trajectories in enumerate(grouped_trajectories):
                prompt = prompts[prompt_idx]
                gt = ground_truth[prompt_idx] if ground_truth else None
                
                trajectories = self._compute_rewards_for_group(
                    trajectories, [prompt] * len(trajectories), 
                    [gt] * len(trajectories) if gt else None
                )
                all_trajectories.extend(trajectories)
            
            # Compute group-relative advantages
            self.logger.info(f"Iteration {iteration}: Computing group-relative advantages...")
            all_trajectories = self._compute_group_relative_advantages(
                all_trajectories, grouped_trajectories
            )
            
            # Update policy
            self.logger.info(f"Iteration {iteration}: Updating policy...")
            metrics = self._update_policy(all_trajectories)
            
            # Log metrics
            self._log_metrics(metrics, iteration)
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"grpo_checkpoint_{iteration+1}")
                
        self.logger.info("GRPO training completed")
        return self.metrics
    
    def _generate_grouped_trajectories(
        self,
        prompts: List[str]
    ) -> List[List[VerlTrajectory]]:
        """
        Generate trajectories grouped by prompt.
        
        Args:
            prompts: List of prompts
            
        Returns:
            List of trajectory groups (one group per prompt)
        """
        grouped_trajectories = []
        
        for prompt in prompts:
            # Generate group_size samples for this prompt
            trajectories = self.verl_adapter.generate_trajectories(
                prompts=[prompt],
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_samples=self.config.group_size,
            )
            grouped_trajectories.append(trajectories)
            
        return grouped_trajectories
    
    def _compute_rewards_for_group(
        self,
        trajectories: List[VerlTrajectory],
        prompts: List[str],
        ground_truth: Optional[List[str]] = None,
    ) -> List[VerlTrajectory]:
        """Compute rewards for a group of trajectories."""
        if self.reward_fn is None:
            return trajectories
            
        # Decode responses
        tokenizer = self.policy_model.tokenizer
        responses = []
        for traj in trajectories:
            prompt_len = traj.prompt_lengths[0] if traj.prompt_lengths else 0
            response_ids = traj.input_ids[0, prompt_len:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)
        
        # Compute rewards
        rewards = self.reward_fn(prompts, responses, ground_truth)
        
        # Assign rewards
        for i, traj in enumerate(trajectories):
            traj.rewards = rewards[i:i+1]
            
        return trajectories
    
    def _compute_group_relative_advantages(
        self,
        all_trajectories: List[VerlTrajectory],
        grouped_trajectories: List[List[VerlTrajectory]]
    ) -> List[VerlTrajectory]:
        """
        Compute group-relative advantages.
        
        For each group, compute the mean and std of rewards,
        then calculate advantages as (reward - mean) / std.
        """
        traj_idx = 0
        
        for group in grouped_trajectories:
            group_size = len(group)
            
            # Get rewards for this group
            group_rewards = torch.stack([
                all_trajectories[traj_idx + i].rewards 
                for i in range(group_size)
            ])
            
            # Compute group statistics
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8  # Avoid division by zero
            
            # Compute group-relative advantages
            for i in range(group_size):
                advantage = (all_trajectories[traj_idx + i].rewards - group_mean) / group_std
                all_trajectories[traj_idx + i].advantages = advantage
                all_trajectories[traj_idx + i].returns = all_trajectories[traj_idx + i].rewards
            
            traj_idx += group_size
            
        return all_trajectories
    
    def _update_policy(
        self,
        trajectories: List[VerlTrajectory]
    ) -> Dict[str, float]:
        """
        Update policy using GRPO.
        
        Similar to PPO but uses group-relative advantages.
        """
        # Prepare batch
        batch = self.verl_adapter.prepare_batch(trajectories)
        
        # Move to device
        device = next(self.policy_model.parameters()).device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Training metrics
        total_policy_loss = 0
        total_kl_div = 0
        total_entropy = 0
        num_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(self.config.num_epochs):
            # Create mini-batches
            num_samples = batch['input_ids'].shape[0]
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                mini_batch = {
                    k: v[batch_indices] for k, v in batch.items()
                }
                
                # Compute policy loss
                policy_loss, kl_div, entropy = self._compute_policy_loss(mini_batch)
                
                # Total loss
                loss = policy_loss - self.config.entropy_coef * entropy
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (num_updates + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Update parameters
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_kl_div += kl_div.item()
                total_entropy += entropy.item()
                num_updates += 1
                
                self.global_step += 1
                self.verl_adapter.step()
        
        # Average metrics
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'kl_div': total_kl_div / num_updates,
            'entropy': total_entropy / num_updates,
            'reward_mean': batch['rewards'].mean().item(),
            'reward_std': batch['rewards'].std().item(),
            'group_relative_advantage': batch['advantages'].mean().item(),
        }
        
        return metrics
    
    def _compute_policy_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GRPO policy loss.
        
        Similar to PPO but uses group-relative advantages.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        action_mask = batch['action_mask']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        
        # Get new log probs
        new_log_probs = self.policy_model.compute_log_probs(
            input_ids, attention_mask
        )
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        ) * advantages.unsqueeze(-1)
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2)
        
        # Apply action mask
        policy_loss = (policy_loss * action_mask[:, 1:]).sum() / action_mask[:, 1:].sum()
        
        # Compute KL divergence with reference model
        kl_div = torch.zeros(1, device=policy_loss.device)
        if self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            
            # Get policy logits
            policy_outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            policy_logits = policy_outputs.logits[:, :-1, :]
            policy_log_probs_all = torch.log_softmax(policy_logits, dim=-1)
            
            # KL divergence
            kl_div = torch.sum(
                torch.exp(ref_log_probs) * (ref_log_probs - policy_log_probs_all),
                dim=-1
            )
            kl_div = (kl_div * action_mask[:, 1:]).sum() / action_mask[:, 1:].sum()
            
            # Add KL penalty to loss
            policy_loss = policy_loss + self.config.kl_coef * kl_div
        
        # Compute entropy
        policy_logits = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits[:, :-1, :]
        probs = torch.softmax(policy_logits, dim=-1)
        log_probs_all = torch.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        entropy = (entropy * action_mask[:, 1:]).sum() / action_mask[:, 1:].sum()
        
        return policy_loss, kl_div, entropy
    
    def _log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log training metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
        if (iteration + 1) % self.config.log_interval == 0:
            self.logger.info(
                f"Iteration {iteration+1}: "
                f"policy_loss={metrics['policy_loss']:.4f}, "
                f"kl_div={metrics['kl_div']:.4f}, "
                f"entropy={metrics['entropy']:.4f}, "
                f"reward_mean={metrics['reward_mean']:.4f}, "
                f"group_adv={metrics['group_relative_advantage']:.4f}"
            )
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }
        torch.save(checkpoint, f"{path}.pt")
        self.logger.info(f"Checkpoint saved to {path}.pt")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(f"{path}.pt", map_location='cpu')
        self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from {path}.pt")
