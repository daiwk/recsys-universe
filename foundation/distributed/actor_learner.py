"""
Actor-Learner architecture for PARL-style distributed RL.

This implements the classic IMPALA-style actor-learner architecture
where actors generate trajectories and learners update the policy.
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Callable, List, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import threading
import queue
import time

from foundation.rl.verl_adapter import VerlAdapter, VerlTrajectory
from foundation.distributed.communication import DistributedCommunicator
from foundation.utils.logging_utils import get_logger


@dataclass
class ActorConfig:
    """Configuration for Actor."""
    actor_id: int = 0
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    update_interval: int = 10
    gamma: float = 0.99
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512


@dataclass
class LearnerConfig:
    """Configuration for Learner."""
    learner_id: int = 0
    batch_size: int = 32
    learning_rate: float = 1e-6
    num_epochs: int = 4
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


class Actor:
    """
    Actor in the Actor-Learner architecture.
    
    The actor is responsible for:
    1. Generating trajectories by interacting with the environment
    2. Computing rewards for trajectories
    3. Sending trajectories to the learner
    
    This follows the IMPALA/V-trace style architecture.
    """
    
    def __init__(
        self,
        actor_id: int,
        policy_model,
        verl_adapter: VerlAdapter,
        config: ActorConfig,
        trajectory_queue: Optional[queue.Queue] = None,
        parameter_queue: Optional[queue.Queue] = None,
    ):
        self.actor_id = actor_id
        self.policy_model = policy_model
        self.verl_adapter = verl_adapter
        self.config = config
        self.trajectory_queue = trajectory_queue
        self.parameter_queue = parameter_queue
        
        self.logger = get_logger(f"Actor-{actor_id}")
        self.running = False
        self.episode_count = 0
        self.step_count = 0
        
    def run(
        self,
        prompts: List[str],
        ground_truth: Optional[List[str]] = None,
    ):
        """
        Main actor loop.
        
        Args:
            prompts: List of prompts to use for trajectory generation
            ground_truth: Optional ground truth answers
        """
        self.running = True
        self.logger.info(f"Actor {self.actor_id} started")
        
        while self.running and self.episode_count < self.config.num_episodes:
            # Check for parameter updates from learner
            self._check_parameter_updates()
            
            # Generate trajectories
            trajectories = self.verl_adapter.generate_trajectories(
                prompts=prompts,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            # Compute rewards
            trajectories = self.verl_adapter.compute_rewards(
                trajectories=trajectories,
                prompts=prompts,
                ground_truth=ground_truth,
            )
            
            # Send trajectories to learner
            if self.trajectory_queue is not None:
                for traj in trajectories:
                    self.trajectory_queue.put(traj)
            
            self.episode_count += 1
            self.step_count += len(trajectories)
            
            if self.episode_count % self.config.update_interval == 0:
                self.logger.info(
                    f"Actor {self.actor_id}: Episode {self.episode_count}, "
                    f"Steps {self.step_count}"
                )
        
        self.logger.info(f"Actor {self.actor_id} finished")
    
    def _check_parameter_updates(self):
        """Check for and apply parameter updates from learner."""
        if self.parameter_queue is None:
            return
            
        try:
            while not self.parameter_queue.empty():
                new_params = self.parameter_queue.get_nowait()
                self.policy_model.load_state_dict(new_params)
                self.logger.debug(f"Actor {self.actor_id}: Parameters updated")
        except queue.Empty:
            pass
    
    def stop(self):
        """Stop the actor."""
        self.running = False


class Learner:
    """
    Learner in the Actor-Learner architecture.
    
    The learner is responsible for:
    1. Collecting trajectories from actors
    2. Computing advantages
    3. Updating the policy using PPO/GRPO
    4. Sending updated parameters to actors
    
    This follows the IMPALA/V-trace style architecture.
    """
    
    def __init__(
        self,
        learner_id: int,
        policy_model,
        critic_model: Optional[nn.Module],
        verl_adapter: VerlAdapter,
        config: LearnerConfig,
        trajectory_queue: Optional[queue.Queue] = None,
        parameter_queues: Optional[List[queue.Queue]] = None,
    ):
        self.learner_id = learner_id
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.verl_adapter = verl_adapter
        self.config = config
        self.trajectory_queue = trajectory_queue
        self.parameter_queues = parameter_queues or []
        
        self.logger = get_logger(f"Learner-{learner_id}")
        self.running = False
        self.update_count = 0
        
        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        if self.critic_model is not None:
            self.critic_optimizer = torch.optim.AdamW(
                self.critic_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
        
        # Trajectory buffer
        self.trajectory_buffer = deque(maxlen=10000)
        
    def run(self):
        """Main learner loop."""
        self.running = True
        self.logger.info(f"Learner {self.learner_id} started")
        
        while self.running:
            # Collect trajectories from queue
            self._collect_trajectories()
            
            # Check if we have enough trajectories
            if len(self.trajectory_buffer) < self.config.batch_size:
                time.sleep(0.1)
                continue
            
            # Sample batch
            batch_trajectories = self._sample_batch()
            
            # Compute advantages
            batch_trajectories = self.verl_adapter.compute_advantages(
                batch_trajectories
            )
            
            # Update policy
            metrics = self._update_policy(batch_trajectories)
            
            # Broadcast parameters to actors
            self._broadcast_parameters()
            
            self.update_count += 1
            
            if self.update_count % 10 == 0:
                self.logger.info(
                    f"Learner {self.learner_id}: Update {self.update_count}, "
                    f"Policy Loss: {metrics.get('policy_loss', 0):.4f}"
                )
    
    def _collect_trajectories(self):
        """Collect trajectories from the queue."""
        if self.trajectory_queue is None:
            return
            
        try:
            while not self.trajectory_queue.empty():
                traj = self.trajectory_queue.get_nowait()
                self.trajectory_buffer.append(traj)
        except queue.Empty:
            pass
    
    def _sample_batch(self) -> List[VerlTrajectory]:
        """Sample a batch of trajectories."""
        import random
        batch_size = min(self.config.batch_size, len(self.trajectory_buffer))
        return random.sample(list(self.trajectory_buffer), batch_size)
    
    def _update_policy(
        self,
        trajectories: List[VerlTrajectory]
    ) -> Dict[str, float]:
        """Update policy using PPO."""
        batch = self.verl_adapter.prepare_batch(trajectories)
        
        # Move to device
        device = next(self.policy_model.parameters()).device
        batch = batch.to(device)
        
        total_policy_loss = 0
        total_value_loss = 0
        num_updates = 0
        
        for epoch in range(self.config.num_epochs):
            # Compute policy loss
            policy_loss = self._compute_policy_loss(batch)
            
            # Backward
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                self.config.max_grad_norm
            )
            self.policy_optimizer.step()
            
            # Update critic if available
            if self.critic_model is not None:
                value_loss = self._compute_value_loss(batch)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_model.parameters(),
                    self.config.max_grad_norm
                )
                self.critic_optimizer.step()
                total_value_loss += value_loss.item()
            
            total_policy_loss += policy_loss.item()
            num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates if self.critic_model else 0,
        }
    
    def _compute_policy_loss(self, batch) -> torch.Tensor:
        """Compute PPO policy loss."""
        # Get new log probs
        new_log_probs = self.policy_model.compute_log_probs(
            batch.input_ids, batch.attention_mask
        )
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - batch.old_log_probs)
        
        # Clipped surrogate objective
        advantages = batch.advantages.unsqueeze(-1)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.config.clip_epsilon,
            1 + self.config.clip_epsilon
        ) * advantages
        
        policy_loss = -torch.min(surr1, surr2)
        policy_loss = (policy_loss * batch.action_mask[:, 1:]).sum() / batch.action_mask[:, 1:].sum()
        
        return policy_loss
    
    def _compute_value_loss(self, batch) -> torch.Tensor:
        """Compute value loss."""
        if self.critic_model is None:
            return torch.tensor(0.0)
            
        values = self.critic_model.get_values(
            batch.input_ids, batch.attention_mask, use_last_token=True
        )
        
        return nn.functional.mse_loss(values, batch.returns)
    
    def _broadcast_parameters(self):
        """Broadcast updated parameters to all actors."""
        if not self.parameter_queues:
            return
            
        # Get current parameters
        params = {
            name: param.data.clone().cpu()
            for name, param in self.policy_model.named_parameters()
        }
        
        # Send to all actors
        for q in self.parameter_queues:
            try:
                q.put_nowait(params)
            except queue.Full:
                pass
    
    def stop(self):
        """Stop the learner."""
        self.running = False


class ActorLearnerSystem:
    """
    Complete Actor-Learner system.
    
    Manages multiple actors and learners in a distributed setup.
    """
    
    def __init__(
        self,
        policy_model_factory: Callable,
        critic_model_factory: Optional[Callable],
        verl_adapter_factory: Callable,
        num_actors: int = 4,
        num_learners: int = 1,
    ):
        self.policy_model_factory = policy_model_factory
        self.critic_model_factory = critic_model_factory
        self.verl_adapter_factory = verl_adapter_factory
        self.num_actors = num_actors
        self.num_learners = num_learners
        
        # Queues for communication
        self.trajectory_queue = queue.Queue(maxsize=10000)
        self.parameter_queues = [queue.Queue(maxsize=10) for _ in range(num_actors)]
        
        # Components
        self.actors: List[Actor] = []
        self.learners: List[Learner] = []
        
        self.logger = get_logger("ActorLearnerSystem")
        
    def initialize(self):
        """Initialize actors and learners."""
        self.logger.info("Initializing Actor-Learner system")
        
        # Create actors
        for i in range(self.num_actors):
            policy_model = self.policy_model_factory()
            verl_adapter = self.verl_adapter_factory(policy_model)
            
            actor = Actor(
                actor_id=i,
                policy_model=policy_model,
                verl_adapter=verl_adapter,
                config=ActorConfig(actor_id=i),
                trajectory_queue=self.trajectory_queue,
                parameter_queue=self.parameter_queues[i],
            )
            self.actors.append(actor)
        
        # Create learners
        for i in range(self.num_learners):
            policy_model = self.policy_model_factory()
            critic_model = self.critic_model_factory() if self.critic_model_factory else None
            verl_adapter = self.verl_adapter_factory(policy_model)
            
            learner = Learner(
                learner_id=i,
                policy_model=policy_model,
                critic_model=critic_model,
                verl_adapter=verl_adapter,
                config=LearnerConfig(learner_id=i),
                trajectory_queue=self.trajectory_queue,
                parameter_queues=self.parameter_queues,
            )
            self.learners.append(learner)
        
        self.logger.info(
            f"Initialized {len(self.actors)} actors and {len(self.learners)} learners"
        )
    
    def start(self, prompts: List[str], ground_truth: Optional[List[str]] = None):
        """Start the system."""
        self.logger.info("Starting Actor-Learner system")
        
        # Start learners
        learner_threads = []
        for learner in self.learners:
            thread = threading.Thread(target=learner.run)
            thread.start()
            learner_threads.append(thread)
        
        # Start actors
        actor_threads = []
        for actor in self.actors:
            thread = threading.Thread(
                target=actor.run,
                args=(prompts, ground_truth)
            )
            thread.start()
            actor_threads.append(thread)
        
        return actor_threads, learner_threads
    
    def stop(self):
        """Stop the system."""
        self.logger.info("Stopping Actor-Learner system")
        
        for actor in self.actors:
            actor.stop()
        
        for learner in self.learners:
            learner.stop()
