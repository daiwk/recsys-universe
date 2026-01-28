"""
PPO (Proximal Policy Optimization) Trainer.
Implements PPO with KL penalty for LLM RL training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from foundation.rl.verl_adapter import VerlAdapter, VerlTrajectory
from foundation.utils.logging_utils import get_logger


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Learning rates
    policy_lr: float = 1e-6
    critic_lr: float = 1e-5
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_clip: float = 0.2
    kl_coef: float = 0.01
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training settings
    num_epochs: int = 4
    batch_size: int = 4
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: int = 1
    
    # GAE settings
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100


class PPOTrainer:
    """
    PPO Trainer for LLM RL training.
    
    Implements PPO-Clip with KL divergence penalty to prevent
    the policy from deviating too far from the reference model.
    """
    
    def __init__(
        self,
        policy_model,
        critic_model,
        reference_model: Optional[nn.Module] = None,
        reward_fn: Optional[Any] = None,
        config: Optional[PPOConfig] = None,
    ):
        self.config = config or PPOConfig()
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.reference_model = reference_model
        self.reward_fn = reward_fn
        
        # Initialize verl adapter
        self.verl_adapter = VerlAdapter(
            policy_model=policy_model,
            reference_model=reference_model,
            critic_model=critic_model,
            reward_fn=reward_fn,
        )
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.policy_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_model.parameters(),
            lr=self.config.critic_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.logger = get_logger("PPOTrainer")
        
        # Metrics
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'kl_div': [],
            'entropy': [],
            'reward_mean': [],
            'reward_std': [],
        }
        
    def train(
        self,
        prompts: List[str],
        ground_truth: Optional[List[str]] = None,
        num_iterations: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Train the policy using PPO.
        
        Args:
            prompts: List of training prompts
            ground_truth: Optional ground truth answers
            num_iterations: Number of training iterations
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Starting PPO training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            self.epoch = iteration
            self.verl_adapter.set_epoch(iteration)
            
            # Generate trajectories
            self.logger.info(f"Iteration {iteration}: Generating trajectories...")
            trajectories = self.verl_adapter.generate_trajectories(
                prompts=prompts,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_samples=self.config.num_samples,
            )
            
            # Compute rewards
            self.logger.info(f"Iteration {iteration}: Computing rewards...")
            trajectories = self.verl_adapter.compute_rewards(
                trajectories=trajectories,
                prompts=prompts,
                ground_truth=ground_truth,
            )
            
            # Compute advantages
            self.logger.info(f"Iteration {iteration}: Computing advantages...")
            trajectories = self.verl_adapter.compute_advantages(
                trajectories=trajectories,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            
            # Update policy
            self.logger.info(f"Iteration {iteration}: Updating policy...")
            metrics = self._update_policy(trajectories)
            
            # Log metrics
            self._log_metrics(metrics, iteration)
            
            # Save checkpoint
            if (iteration + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{iteration+1}")
                
        self.logger.info("PPO training completed")
        return self.metrics
    
    def _update_policy(
        self,
        trajectories: List[VerlTrajectory]
    ) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            trajectories: List of trajectory objects
            
        Returns:
            Training metrics for this update
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
        total_value_loss = 0
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
                
                # Compute value loss
                value_loss = self._compute_value_loss(mini_batch)
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.vf_coef * value_loss 
                    - self.config.entropy_coef * entropy
                )
                
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
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Update parameters
                    self.policy_optimizer.step()
                    self.critic_optimizer.step()
                    
                    # Zero gradients
                    self.policy_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl_div += kl_div.item()
                total_entropy += entropy.item()
                num_updates += 1
                
                self.global_step += 1
                self.verl_adapter.step()
        
        # Average metrics
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'kl_div': total_kl_div / num_updates,
            'entropy': total_entropy / num_updates,
            'reward_mean': batch['rewards'].mean().item(),
            'reward_std': batch['rewards'].std().item(),
        }
        
        return metrics
    
    def _compute_policy_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute PPO policy loss.
        
        Args:
            batch: Mini-batch of data
            
        Returns:
            policy_loss, kl_div, entropy
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
    
    def _compute_value_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            batch: Mini-batch of data
            
        Returns:
            value_loss
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        returns = batch['returns']
        
        # Get predicted values
        values = self.critic_model.get_values(
            input_ids, attention_mask, use_last_token=True
        )
        
        # Value loss (MSE)
        value_loss = F.mse_loss(values, returns)
        
        return value_loss
    
    def _log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log training metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
        if (iteration + 1) % self.config.log_interval == 0:
            self.logger.info(
                f"Iteration {iteration+1}: "
                f"policy_loss={metrics['policy_loss']:.4f}, "
                f"value_loss={metrics['value_loss']:.4f}, "
                f"kl_div={metrics['kl_div']:.4f}, "
                f"entropy={metrics['entropy']:.4f}, "
                f"reward_mean={metrics['reward_mean']:.4f}"
            )
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy_model.state_dict(),
            'critic_state_dict': self.critic_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
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
        self.critic_model.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.logger.info(f"Checkpoint loaded from {path}.pt")
