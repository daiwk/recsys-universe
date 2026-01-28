"""
PARL (PAddle Reinforcement Learning) style cluster management.

PARL is a flexible and high-efficient reinforcement learning framework
originally developed by Baidu. This implementation follows PARL's design
principles for distributed RL.

Key concepts:
1. @parl.remote_class - Decorator for distributed classes
2. Actor - Worker that executes tasks
3. Learner - Centralized parameter updater
4. Parameter Server - Distributed parameter storage
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Callable, List, Dict, Type, TypeVar
from dataclasses import dataclass
from collections import defaultdict
import threading
import queue

from foundation.distributed.communication import DistributedCommunicator, Backend
from foundation.utils.logging_utils import get_logger


T = TypeVar('T')


@dataclass
class PARLConfig:
    """Configuration for PARL cluster."""
    # Cluster topology
    num_actors: int = 4
    num_learners: int = 1
    num_parameter_servers: int = 1
    
    # Communication
    backend: Backend = Backend.RAY
    master_address: str = "localhost"
    master_port: int = 12345
    
    # Resource allocation
    actor_cpus: int = 2
    actor_gpus: float = 0.5
    learner_cpus: int = 4
    learner_gpus: float = 1.0
    
    # Training
    update_interval: int = 10
    sync_interval: int = 100


class PARLCluster:
    """
    PARL-style cluster for distributed RL.
    
    This implements the classic parameter server architecture with
    separate actors and learners, inspired by PARL and IMPALA.
    
    Architecture:
    - Actors: Generate trajectories by interacting with environment
    - Learners: Update policy parameters
    - Parameter Servers: Store and distribute model parameters
    """
    
    def __init__(self, config: Optional[PARLConfig] = None):
        self.config = config or PARLConfig()
        self.logger = get_logger("PARLCluster")
        
        # Initialize communication
        self.communicator = DistributedCommunicator(
            backend=self.config.backend,
        )
        
        # Cluster state
        self.actors: List[Any] = []
        self.learners: List[Any] = []
        self.parameter_servers: List[Any] = []
        
        self._initialized = False
        
    def initialize(self):
        """Initialize the cluster."""
        if self._initialized:
            return
            
        self.logger.info("Initializing PARL cluster")
        self.communicator.initialize()
        
        # Create parameter servers
        self.logger.info(f"Creating {self.config.num_parameter_servers} parameter servers")
        for i in range(self.config.num_parameter_servers):
            ps = self._create_parameter_server(i)
            self.parameter_servers.append(ps)
        
        self._initialized = True
        self.logger.info("PARL cluster initialized")
        
    def _create_parameter_server(self, ps_id: int):
        """Create a parameter server."""
        # Placeholder - actual implementation would create remote actor
        return {"id": ps_id, "type": "parameter_server"}
    
    def create_actor(
        self,
        actor_class: Type[T],
        *args,
        **kwargs
    ) -> T:
        """
        Create an actor instance.
        
        Args:
            actor_class: Class to instantiate as actor
            *args: Positional arguments for actor_class
            **kwargs: Keyword arguments for actor_class
            
        Returns:
            Actor instance (possibly remote)
        """
        if self.config.backend == Backend.RAY:
            try:
                import ray
                
                # Create remote actor
                RemoteActor = ray.remote(
                    num_cpus=self.config.actor_cpus,
                    num_gpus=self.config.actor_gpus,
                )(actor_class)
                
                actor = RemoteActor.remote(*args, **kwargs)
                self.actors.append(actor)
                return actor
                
            except ImportError:
                raise ImportError("Ray is required for Ray backend")
        else:
            # Local actor
            actor = actor_class(*args, **kwargs)
            self.actors.append(actor)
            return actor
    
    def create_learner(
        self,
        learner_class: Type[T],
        *args,
        **kwargs
    ) -> T:
        """
        Create a learner instance.
        
        Args:
            learner_class: Class to instantiate as learner
            *args: Positional arguments for learner_class
            **kwargs: Keyword arguments for learner_class
            
        Returns:
            Learner instance (possibly remote)
        """
        if self.config.backend == Backend.RAY:
            try:
                import ray
                
                # Create remote learner
                RemoteLearner = ray.remote(
                    num_cpus=self.config.learner_cpus,
                    num_gpus=self.config.learner_gpus,
                )(learner_class)
                
                learner = RemoteLearner.remote(*args, **kwargs)
                self.learners.append(learner)
                return learner
                
            except ImportError:
                raise ImportError("Ray is required for Ray backend")
        else:
            # Local learner
            learner = learner_class(*args, **kwargs)
            self.learners.append(learner)
            return learner
    
    def broadcast_parameters(
        self,
        parameters: Dict[str, torch.Tensor],
        exclude: Optional[List[int]] = None,
    ):
        """
        Broadcast parameters to all actors.
        
        Args:
            parameters: Model parameters to broadcast
            exclude: List of actor IDs to exclude
        """
        exclude = exclude or []
        
        for i, actor in enumerate(self.actors):
            if i in exclude:
                continue
                
            if self.config.backend == Backend.RAY:
                # Async parameter update
                actor.update_parameters.remote(parameters)
            else:
                actor.update_parameters(parameters)
    
    def gather_trajectories(
        self,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Gather trajectories from all actors.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of trajectories
        """
        trajectories = []
        
        for actor in self.actors:
            if self.config.backend == Backend.RAY:
                try:
                    import ray
                    traj = ray.get(actor.get_trajectories.remote(), timeout=timeout)
                    trajectories.extend(traj)
                except Exception as e:
                    self.logger.warning(f"Failed to get trajectories: {e}")
            else:
                traj = actor.get_trajectories()
                trajectories.extend(traj)
                
        return trajectories
    
    def shutdown(self):
        """Shutdown the cluster."""
        self.logger.info("Shutting down PARL cluster")
        
        if self.config.backend == Backend.RAY:
            try:
                import ray
                ray.shutdown()
            except:
                pass
                
        self.communicator.finalize()
        self._initialized = False
        
        self.logger.info("PARL cluster shut down")


class PARLRemoteClass:
    """
    Decorator for creating PARL remote classes.
    
    Similar to @ray.remote or @parl.remote in the original PARL framework.
    """
    
    def __init__(
        self,
        num_cpus: int = 1,
        num_gpus: float = 0,
        memory: Optional[int] = None,
    ):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        
    def __call__(self, cls: Type[T]) -> Type[T]:
        """Decorate a class to make it a remote class."""
        # Store configuration on the class
        cls._parl_remote_config = {
            'num_cpus': self.num_cpus,
            'num_gpus': self.num_gpus,
            'memory': self.memory,
        }
        return cls


def parl_remote(
    num_cpus: int = 1,
    num_gpus: float = 0,
    memory: Optional[int] = None,
):
    """
    Decorator to mark a class as a PARL remote class.
    
    Usage:
        @parl_remote(num_cpus=2, num_gpus=1)
        class MyActor:
            def __init__(self):
                pass
    """
    return PARLRemoteClass(num_cpus, num_gpus, memory)


class PARLAgent:
    """
    Base class for PARL agents (actors and learners).
    
    Provides common functionality for distributed training.
    """
    
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.logger = get_logger(f"PARLAgent-{agent_id}")
        self.parameters: Dict[str, torch.Tensor] = {}
        
    def update_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Update local parameters."""
        self.parameters = {
            k: v.clone() for k, v in parameters.items()
        }
        self.logger.debug(f"Agent {self.agent_id}: Parameters updated")
        
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current parameters."""
        return self.parameters


class PARLActor(PARLAgent):
    """
    Base class for PARL actors.
    
    Actors are responsible for generating trajectories.
    """
    
    def __init__(self, actor_id: int):
        super().__init__(actor_id)
        self.trajectory_buffer = []
        
    def generate_trajectories(self, *args, **kwargs) -> List[Any]:
        """Generate trajectories. Override in subclass."""
        raise NotImplementedError
        
    def get_trajectories(self) -> List[Any]:
        """Get and clear trajectory buffer."""
        trajectories = self.trajectory_buffer.copy()
        self.trajectory_buffer.clear()
        return trajectories


class PARLLearner(PARLAgent):
    """
    Base class for PARL learners.
    
    Learners are responsible for updating policy parameters.
    """
    
    def __init__(self, learner_id: int):
        super().__init__(learner_id)
        
    def learn(self, trajectories: List[Any]) -> Dict[str, float]:
        """
        Learn from trajectories. Override in subclass.
        
        Returns:
            Dictionary of training metrics
        """
        raise NotImplementedError
        
    def get_updated_parameters(self) -> Dict[str, torch.Tensor]:
        """Get updated parameters to broadcast."""
        return self.parameters
