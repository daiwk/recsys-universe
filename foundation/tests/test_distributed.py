"""
Unit tests for distributed training components.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import queue

from foundation.distributed.communication import (
    DistributedCommunicator, Backend, ParameterServer, RingAllReduce
)
from foundation.distributed.parl_cluster import (
    PARLCluster, PARLConfig, parl_remote, PARLAgent, PARLActor, PARLLearner
)
from foundation.distributed.actor_learner import (
    Actor, Learner, ActorConfig, LearnerConfig, ActorLearnerSystem
)
from foundation.distributed.parameter_server import (
    ParameterServer, ShardedParameterServer, FederatedParameterServer
)


class TestBackend:
    """Tests for Backend enum."""
    
    def test_backend_values(self):
        """Test backend enum values."""
        assert Backend.RAY.value == "ray"
        assert Backend.TORCH.value == "torch"
        assert Backend.MPI.value == "mpi"


class TestDistributedCommunicator:
    """Tests for DistributedCommunicator."""
    
    def test_initialization(self):
        """Test communicator initialization."""
        comm = DistributedCommunicator(backend=Backend.TORCH)
        assert comm.backend == Backend.TORCH
        assert not comm._initialized
        
    def test_get_rank_and_world_size(self):
        """Test getting rank and world size."""
        comm = DistributedCommunicator(
            backend=Backend.TORCH,
            rank=0,
            world_size=4
        )
        assert comm.get_rank() == 0
        assert comm.get_world_size() == 4
        assert comm.is_main_process()
        
    def test_non_main_process(self):
        """Test non-main process detection."""
        comm = DistributedCommunicator(
            backend=Backend.TORCH,
            rank=1,
            world_size=4
        )
        assert not comm.is_main_process()


class TestPARLConfig:
    """Tests for PARLConfig."""
    
    def test_default_config(self):
        """Test default PARL configuration."""
        config = PARLConfig()
        assert config.num_actors == 4
        assert config.num_learners == 1
        assert config.backend == Backend.RAY
        assert config.master_address == "localhost"
        
    def test_custom_config(self):
        """Test custom PARL configuration."""
        config = PARLConfig(
            num_actors=8,
            num_learners=2,
            backend=Backend.TORCH
        )
        assert config.num_actors == 8
        assert config.num_learners == 2
        assert config.backend == Backend.TORCH


class TestPARLCluster:
    """Tests for PARLCluster."""
    
    def test_cluster_initialization(self):
        """Test cluster initialization."""
        config = PARLConfig(backend=Backend.TORCH)
        cluster = PARLCluster(config)
        
        assert cluster.config == config
        assert not cluster._initialized
        assert len(cluster.actors) == 0
        assert len(cluster.learners) == 0


class TestParlRemoteDecorator:
    """Tests for @parl_remote decorator."""
    
    def test_decorator(self):
        """Test the parl_remote decorator."""
        @parl_remote(num_cpus=2, num_gpus=1)
        class TestClass:
            pass
        
        assert hasattr(TestClass, '_parl_remote_config')
        assert TestClass._parl_remote_config['num_cpus'] == 2
        assert TestClass._parl_remote_config['num_gpus'] == 1


class TestPARLAgent:
    """Tests for PARLAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = PARLAgent(agent_id=0)
        assert agent.agent_id == 0
        assert agent.parameters == {}
        
    def test_update_parameters(self):
        """Test parameter update."""
        agent = PARLAgent(agent_id=0)
        params = {
            'weight': torch.randn(10, 10),
            'bias': torch.randn(10)
        }
        
        agent.update_parameters(params)
        
        assert 'weight' in agent.parameters
        assert 'bias' in agent.parameters
        assert torch.equal(agent.parameters['weight'], params['weight'])
        
    def test_get_parameters(self):
        """Test getting parameters."""
        agent = PARLAgent(agent_id=0)
        params = {'weight': torch.randn(5, 5)}
        agent.update_parameters(params)
        
        retrieved = agent.get_parameters()
        assert 'weight' in retrieved


class TestPARLActor:
    """Tests for PARLActor."""
    
    def test_actor_initialization(self):
        """Test actor initialization."""
        actor = PARLActor(actor_id=1)
        assert actor.agent_id == 1
        assert actor.trajectory_buffer == []
        
    def test_get_trajectories(self):
        """Test getting trajectories."""
        actor = PARLActor(actor_id=0)
        actor.trajectory_buffer = ['traj1', 'traj2']
        
        trajectories = actor.get_trajectories()
        assert trajectories == ['traj1', 'traj2']
        assert actor.trajectory_buffer == []


class TestPARLLearner:
    """Tests for PARLLearner."""
    
    def test_learner_initialization(self):
        """Test learner initialization."""
        learner = PARLLearner(learner_id=0)
        assert learner.agent_id == 0
        
    def test_get_updated_parameters(self):
        """Test getting updated parameters."""
        learner = PARLLearner(learner_id=0)
        params = {'weight': torch.randn(5, 5)}
        learner.update_parameters(params)
        
        updated = learner.get_updated_parameters()
        assert 'weight' in updated


class TestActorConfig:
    """Tests for ActorConfig."""
    
    def test_default_config(self):
        """Test default actor configuration."""
        config = ActorConfig()
        assert config.actor_id == 0
        assert config.num_episodes == 1000
        assert config.gamma == 0.99
        assert config.temperature == 0.7


class TestLearnerConfig:
    """Tests for LearnerConfig."""
    
    def test_default_config(self):
        """Test default learner configuration."""
        config = LearnerConfig()
        assert config.learner_id == 0
        assert config.batch_size == 32
        assert config.learning_rate == 1e-6
        assert config.clip_epsilon == 0.2


class TestParameterServer:
    """Tests for ParameterServer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = nn.Linear(10, 5)
        return model
    
    def test_ps_initialization(self, mock_model):
        """Test parameter server initialization."""
        ps = ParameterServer(mock_model, update_rule="sync")
        
        assert ps.update_rule == "sync"
        assert len(ps.parameters) == 2  # weight and bias
        assert 'weight' in ps.parameters
        assert 'bias' in ps.parameters
        
    def test_pull(self, mock_model):
        """Test pulling parameters."""
        ps = ParameterServer(mock_model)
        params = ps.pull()
        
        assert 'weight' in params
        assert 'bias' in params
        assert params['weight'].shape == (5, 10)
        
    def test_push_async(self, mock_model):
        """Test async parameter update."""
        ps = ParameterServer(mock_model, update_rule="async")
        
        initial_weight = ps.parameters['weight'].clone()
        gradients = {'weight': torch.ones_like(initial_weight)}
        
        ps._push_async(gradients)
        
        # Parameters should be updated
        assert not torch.equal(ps.parameters['weight'], initial_weight)
        assert ps.update_count == 1


class TestShardedParameterServer:
    """Tests for ShardedParameterServer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        return model
    
    def test_sharding(self, mock_model):
        """Test parameter sharding."""
        ps = ShardedParameterServer(mock_model, num_shards=2)
        
        assert ps.num_shards == 2
        assert len(ps.shards) == 2
        
    def test_pull_shard(self, mock_model):
        """Test pulling a specific shard."""
        ps = ShardedParameterServer(mock_model, num_shards=2)
        
        shard = ps.pull_shard(0)
        assert isinstance(shard, dict)
        
    def test_pull_all(self, mock_model):
        """Test pulling all parameters."""
        ps = ShardedParameterServer(mock_model, num_shards=2)
        
        all_params = ps.pull_all()
        assert isinstance(all_params, dict)
        assert len(all_params) > 0


class TestFederatedParameterServer:
    """Tests for FederatedParameterServer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = nn.Linear(10, 5)
        return model
    
    def test_fedavg_aggregation(self, mock_model):
        """Test FedAvg aggregation."""
        ps = FederatedParameterServer(mock_model, aggregation_rule="fedavg")
        
        # Create client updates
        client_updates = [
            (1, {'weight': torch.randn(5, 10), 'bias': torch.randn(5)}),
            (1, {'weight': torch.randn(5, 10), 'bias': torch.randn(5)}),
        ]
        
        ps.aggregate(client_updates)
        
        # Global parameters should be updated
        assert 'weight' in ps.global_parameters
        assert 'bias' in ps.global_parameters
        
    def test_pull(self, mock_model):
        """Test pulling global parameters."""
        ps = FederatedParameterServer(mock_model)
        params = ps.pull()
        
        assert 'weight' in params
        assert 'bias' in params


class TestRingAllReduce:
    """Tests for RingAllReduce."""
    
    def test_initialization(self):
        """Test RingAllReduce initialization."""
        mock_comm = Mock()
        ring = RingAllReduce(mock_comm)
        
        assert ring.communicator == mock_comm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
