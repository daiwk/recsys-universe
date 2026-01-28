"""
Distributed communication utilities for PARL-style training.
Supports both Ray and PyTorch distributed backends.
"""

import torch
import torch.distributed as dist
from typing import Any, Optional, Callable, Dict, List
import pickle
from enum import Enum


class Backend(Enum):
    """Distributed backend types."""
    RAY = "ray"
    TORCH = "torch"
    MPI = "mpi"


class DistributedCommunicator:
    """
    Distributed communicator for PARL-style training.
    
    Supports:
    1. Ray - for actor-learner architecture
    2. PyTorch DDP - for data parallel training
    3. MPI - for high-performance computing environments
    """
    
    def __init__(
        self,
        backend: Backend = Backend.RAY,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        init_method: Optional[str] = None,
    ):
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.init_method = init_method
        
        self._initialized = False
        self._ray = None
        
    def initialize(self):
        """Initialize distributed backend."""
        if self._initialized:
            return
            
        if self.backend == Backend.RAY:
            self._init_ray()
        elif self.backend == Backend.TORCH:
            self._init_torch()
        elif self.backend == Backend.MPI:
            self._init_mpi()
            
        self._initialized = True
        
    def _init_ray(self):
        """Initialize Ray backend."""
        try:
            import ray
            self._ray = ray
            
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
                
            self.world_size = ray.cluster_resources().get("CPU", 1)
            self.rank = 0  # Will be set per-actor
            
        except ImportError:
            raise ImportError("Ray is required for Ray backend. Install with: pip install ray")
    
    def _init_torch(self):
        """Initialize PyTorch distributed backend."""
        if not dist.is_initialized():
            if self.init_method is None:
                self.init_method = "env://"
                
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method=self.init_method,
                world_size=self.world_size,
                rank=self.rank,
            )
            
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
    def _init_mpi(self):
        """Initialize MPI backend."""
        try:
            from mpi4py import MPI
            self._mpi = MPI
            self._comm = MPI.COMM_WORLD
            self.world_size = self._comm.Get_size()
            self.rank = self._comm.Get_rank()
        except ImportError:
            raise ImportError("mpi4py is required for MPI backend")
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source rank to all ranks."""
        if self.backend == Backend.TORCH:
            dist.broadcast(tensor, src=src)
        elif self.backend == Backend.MPI:
            self._comm.Bcast(tensor, root=src)
        return tensor
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum"
    ) -> torch.Tensor:
        """All-reduce operation across all ranks."""
        if self.backend == Backend.TORCH:
            reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
            dist.all_reduce(tensor, op=reduce_op)
        elif self.backend == Backend.MPI:
            mpi_op = self._mpi.SUM if op == "sum" else self._mpi.AVG
            self._comm.Allreduce(self._mpi.IN_PLACE, tensor, op=mpi_op)
        return tensor
    
    def all_gather(
        self,
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Gather tensors from all ranks."""
        if self.backend == Backend.TORCH:
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
            return gathered
        elif self.backend == Backend.MPI:
            return self._comm.allgather(tensor)
        return [tensor]
    
    def send(self, tensor: torch.Tensor, dst: int):
        """Send tensor to destination rank."""
        if self.backend == Backend.TORCH:
            dist.send(tensor, dst=dst)
        elif self.backend == Backend.MPI:
            self._comm.Send(tensor, dest=dst)
    
    def recv(self, tensor: torch.Tensor, src: int):
        """Receive tensor from source rank."""
        if self.backend == Backend.TORCH:
            dist.recv(tensor, src=src)
        elif self.backend == Backend.MPI:
            self._comm.Recv(tensor, source=src)
        return tensor
    
    def barrier(self):
        """Synchronization barrier."""
        if self.backend == Backend.TORCH:
            dist.barrier()
        elif self.backend == Backend.MPI:
            self._comm.Barrier()
    
    def get_rank(self) -> int:
        """Get current process rank."""
        return self.rank
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.world_size
    
    def is_main_process(self) -> bool:
        """Check if current process is the main process."""
        return self.rank == 0
    
    def finalize(self):
        """Finalize distributed backend."""
        if self.backend == Backend.TORCH and dist.is_initialized():
            dist.destroy_process_group()
        elif self.backend == Backend.RAY and self._ray is not None:
            self._ray.shutdown()


class ParameterServer:
    """
    Parameter Server for distributed training.
    
    Implements the PS architecture commonly used in distributed RL.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        communicator: DistributedCommunicator,
    ):
        self.model = model
        self.communicator = communicator
        self.parameters = {name: param.data.clone() for name, param in model.named_parameters()}
        
    def push(self, gradients: Dict[str, torch.Tensor]):
        """Receive gradients from workers and update parameters."""
        with torch.no_grad():
            for name, grad in gradients.items():
                if name in self.parameters:
                    # Simple SGD update (can be extended to other optimizers)
                    self.parameters[name] -= 0.001 * grad
                    
    def pull(self) -> Dict[str, torch.Tensor]:
        """Send current parameters to workers."""
        return {name: param.clone() for name, param in self.parameters.items()}
    
    def sync_model(self):
        """Sync model parameters with the PS."""
        for name, param in self.model.named_parameters():
            if name in self.parameters:
                param.data.copy_(self.parameters[name])


class RingAllReduce:
    """
    Ring All-Reduce implementation for efficient gradient synchronization.
    
    This is used in distributed data parallel training.
    """
    
    def __init__(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        
    def reduce(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform ring all-reduce on a list of tensors.
        
        Args:
            tensors: List of tensors to reduce
            
        Returns:
            List of reduced tensors (same on all ranks)
        """
        rank = self.communicator.get_rank()
        world_size = self.communicator.get_world_size()
        
        # Scatter-reduce phase
        for step in range(world_size - 1):
            send_idx = (rank - step) % world_size
            recv_idx = (rank - step - 1) % world_size
            
            # Send to next rank
            self.communicator.send(tensors[send_idx], (rank + 1) % world_size)
            
            # Receive from previous rank
            recv_tensor = torch.zeros_like(tensors[recv_idx])
            self.communicator.recv(recv_tensor, (rank - 1) % world_size)
            
            # Accumulate
            tensors[recv_idx] += recv_tensor
            
        # All-gather phase
        for step in range(world_size - 1):
            send_idx = (rank - step + 1) % world_size
            recv_idx = (rank - step) % world_size
            
            # Send to next rank
            self.communicator.send(tensors[send_idx], (rank + 1) % world_size)
            
            # Receive from previous rank
            recv_tensor = torch.zeros_like(tensors[recv_idx])
            self.communicator.recv(recv_tensor, (rank - 1) % world_size)
            
            # Copy received tensor
            tensors[recv_idx] = recv_tensor
            
        return tensors
