"""
Replay buffer implementations for RL training.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random


class ReplayBuffer:
    """
    Standard replay buffer for experience replay.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: Dict[str, Any]):
        """
        Add an experience to the buffer.
        
        Args:
            experience: Dictionary containing experience data
        """
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experience dictionaries
        """
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples experiences based on their priority (TD error).
    Higher priority experiences are sampled more frequently.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent
            beta_increment: Increment for beta annealing
            epsilon: Small constant to avoid zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, experience: Dict[str, Any], priority: float = 1.0):
        """
        Add an experience with priority.
        
        Args:
            experience: Dictionary containing experience data
            priority: Priority of the experience (higher = more important)
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max(priority, self.epsilon)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of experience dictionaries
        """
        if len(self.buffer) == 0:
            return []
            
        # Compute sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probabilities
        )
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
