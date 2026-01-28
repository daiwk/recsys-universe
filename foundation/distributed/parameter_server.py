"""
Parameter Server implementation for distributed training.

The parameter server architecture is widely used in distributed RL
for efficient parameter synchronization across multiple workers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import threading
import copy

from foundation.distributed.communication import DistributedCommunicator
from foundation.utils.logging_utils import get_logger


class ParameterServer:
    """
    Parameter Server for distributed model training.
    
    The parameter server maintains the global model parameters and
    handles parameter updates from workers. It supports:
    1. Synchronous updates
    2. Asynchronous updates
    3. Gradient aggregation
    """
    
    def __init__(
        self,
        model: nn.Module,
        update_rule: str = "sync",  # "sync" or "async"
        learning_rate: float = 0.001,
        momentum: float = 0.9,
    ):
        """
        Initialize parameter server.
        
        Args:
            model: The model to manage
            update_rule: Update rule ("sync" or "async")
            learning_rate: Learning rate for parameter updates
            momentum: Momentum for SGD
        """
        self.model = model
        self.update_rule = update_rule
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        self.logger = get_logger("ParameterServer")
        
        # Global parameters
        self.parameters = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        # For momentum
        self.velocity = {
            name: torch.zeros_like(param)
            for name, param in self.parameters.items()
        }
        
        # Gradient buffer for synchronous updates
        self.gradient_buffer: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.update_count = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def pull(self) -> Dict[str, torch.Tensor]:
        """
        Pull current parameters from the server.
        
        Returns:
            Dictionary of parameter tensors
        """
        with self.lock:
            return {
                name: param.clone()
                for name, param in self.parameters.items()
            }
    
    def push(
        self,
        gradients: Dict[str, torch.Tensor],
        worker_id: Optional[int] = None,
    ):
        """
        Push gradients to the server.
        
        Args:
            gradients: Dictionary of gradients
            worker_id: ID of the worker pushing gradients
        """
        if self.update_rule == "sync":
            self._push_sync(gradients, worker_id)
        else:
            self._push_async(gradients)
    
    def _push_sync(
        self,
        gradients: Dict[str, torch.Tensor],
        worker_id: Optional[int],
    ):
        """Synchronous update - accumulate gradients and update periodically."""
        with self.lock:
            # Accumulate gradients
            for name, grad in gradients.items():
                if name in self.gradient_buffer:
                    self.gradient_buffer[name].append(grad)
            
            # Check if we should update
            num_workers = len(set(id(g) for g in self.gradient_buffer[list(self.gradient_buffer.keys())[0]])) if self.gradient_buffer else 0
            
            # Update when we have gradients from all workers
            if num_workers >= self.expected_workers:
                self._update_parameters_sync()
    
    def _push_async(self, gradients: Dict[str, torch.Tensor]):
        """Asynchronous update - apply gradients immediately."""
        with self.lock:
            for name, grad in gradients.items():
                if name in self.parameters:
                    # Update with momentum
                    self.velocity[name] = (
                        self.momentum * self.velocity[name] - self.learning_rate * grad
                    )
                    self.parameters[name] += self.velocity[name]
            
            self.update_count += 1
    
    def _update_parameters_sync(self):
        """Update parameters using accumulated gradients."""
        for name in self.gradient_buffer:
            if self.gradient_buffer[name]:
                # Average gradients
                avg_grad = torch.stack(self.gradient_buffer[name]).mean(dim=0)
                
                # Update with momentum
                self.velocity[name] = (
                    self.momentum * self.velocity[name] - self.learning_rate * avg_grad
                )
                self.parameters[name] += self.velocity[name]
                
                # Clear buffer
                self.gradient_buffer[name].clear()
        
        self.update_count += 1
    
    def set_expected_workers(self, num_workers: int):
        """Set expected number of workers for synchronous updates."""
        self.expected_workers = num_workers
    
    def get_update_count(self) -> int:
        """Get number of parameter updates."""
        return self.update_count
    
    def sync_model(self):
        """Sync the local model with server parameters."""
        with self.lock:
            for name, param in self.model.named_parameters():
                if name in self.parameters:
                    param.data.copy_(self.parameters[name])


class ShardedParameterServer:
    """
    Sharded Parameter Server for large models.
    
    Distributes parameters across multiple parameter servers
    to handle models that don't fit in a single machine's memory.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_shards: int = 4,
    ):
        """
        Initialize sharded parameter server.
        
        Args:
            model: The model to shard
            num_shards: Number of parameter server shards
        """
        self.model = model
        self.num_shards = num_shards
        self.logger = get_logger("ShardedParameterServer")
        
        # Shard parameters
        self.shards: List[Dict[str, torch.Tensor]] = [
            {} for _ in range(num_shards)
        ]
        
        # Assign parameters to shards
        param_names = list(model.state_dict().keys())
        for i, name in enumerate(param_names):
            shard_id = i % num_shards
            self.shards[shard_id][name] = model.state_dict()[name].clone()
        
        self.logger.info(f"Model sharded into {num_shards} shards")
    
    def pull_shard(self, shard_id: int) -> Dict[str, torch.Tensor]:
        """Pull a specific shard."""
        return {
            name: param.clone()
            for name, param in self.shards[shard_id].items()
        }
    
    def push_shard(
        self,
        shard_id: int,
        gradients: Dict[str, torch.Tensor],
    ):
        """Push gradients for a specific shard."""
        for name, grad in gradients.items():
            if name in self.shards[shard_id]:
                self.shards[shard_id][name] -= 0.001 * grad
    
    def pull_all(self) -> Dict[str, torch.Tensor]:
        """Pull all parameters from all shards."""
        all_params = {}
        for shard in self.shards:
            all_params.update({
                name: param.clone()
                for name, param in shard.items()
            })
        return all_params


class FederatedParameterServer:
    """
    Federated Parameter Server for federated learning scenarios.
    
    Implements FedAvg and other federated learning algorithms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        aggregation_rule: str = "fedavg",  # "fedavg", "fedprox", "scaffold"
    ):
        """
        Initialize federated parameter server.
        
        Args:
            model: The global model
            aggregation_rule: Aggregation rule for federated learning
        """
        self.model = model
        self.aggregation_rule = aggregation_rule
        self.logger = get_logger("FederatedParameterServer")
        
        # Global parameters
        self.global_parameters = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        
        # For SCAFFOLD
        if aggregation_rule == "scaffold":
            self.control_variates = {
                name: torch.zeros_like(param)
                for name, param in self.global_parameters.items()
            }
        
        self.lock = threading.Lock()
    
    def aggregate(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ):
        """
        Aggregate updates from clients.
        
        Args:
            client_updates: List of (client_id, parameters) tuples
        """
        with self.lock:
            if self.aggregation_rule == "fedavg":
                self._fedavg(client_updates)
            elif self.aggregation_rule == "fedprox":
                self._fedprox(client_updates)
            elif self.aggregation_rule == "scaffold":
                self._scaffold(client_updates)
    
    def _fedavg(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ):
        """
        Federated Averaging (FedAvg).
        
        Reference: Communication-Efficient Learning of Deep Networks
        from Decentralized Data (McMahan et al., 2017)
        """
        total_weight = sum(weight for weight, _ in client_updates)
        
        # Initialize aggregated parameters
        aggregated = {
            name: torch.zeros_like(param)
            for name, param in self.global_parameters.items()
        }
        
        # Weighted average
        for weight, client_params in client_updates:
            for name, param in client_params.items():
                if name in aggregated:
                    aggregated[name] += (weight / total_weight) * param
        
        # Update global parameters
        self.global_parameters = aggregated
    
    def _fedprox(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ):
        """
        Federated Proximal (FedProx).
        
        Reference: Federated Optimization in Heterogeneous Networks
        (Li et al., 2020)
        """
        # Similar to FedAvg but with proximal term
        # For simplicity, we use FedAvg here
        self._fedavg(client_updates)
    
    def _scaffold(
        self,
        client_updates: List[Tuple[int, Dict[str, torch.Tensor]]],
    ):
        """
        SCAFFOLD aggregation.
        
        Reference: SCAFFOLD: Stochastic Controlled Averaging for
        Federated Learning (Karimireddy et al., 2020)
        """
        # Update global parameters with control variates
        total_weight = sum(weight for weight, _ in client_updates)
        
        for weight, client_params in client_updates:
            for name, param in client_params.items():
                if name in self.global_parameters:
                    # Update with control variate correction
                    correction = self.control_variates[name]
                    self.global_parameters[name] += (
                        (weight / total_weight) * (param - self.global_parameters[name] + correction)
                    )
    
    def pull(self) -> Dict[str, torch.Tensor]:
        """Pull global parameters."""
        with self.lock:
            return {
                name: param.clone()
                for name, param in self.global_parameters.items()
            }
    
    def get_control_variates(self) -> Dict[str, torch.Tensor]:
        """Get control variates for SCAFFOLD."""
        if self.aggregation_rule != "scaffold":
            raise ValueError("Control variates only available for SCAFFOLD")
        
        return {
            name: cv.clone()
            for name, cv in self.control_variates.items()
        }
