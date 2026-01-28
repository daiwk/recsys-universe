"""
Unit tests for Agent Swarm (Kimi K2.5 style).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
import time

from foundation.distributed.agent_swarm import (
    AgentSwarm,
    SwarmAgent,
    AgentConfig,
    SwarmTask,
    AgentRole,
    TaskStatus,
)


class TestAgentRole:
    """Tests for AgentRole enum."""
    
    def test_agent_roles(self):
        """Test agent role values."""
        assert AgentRole.ORCHESTRATOR.value == "orchestrator"
        assert AgentRole.WORKER.value == "worker"
        assert AgentRole.CRITIC.value == "critic"
        assert AgentRole.PLANNER.value == "planner"
        assert AgentRole.EXECUTOR.value == "executor"


class TestTaskStatus:
    """Tests for TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test task status values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"


class TestSwarmTask:
    """Tests for SwarmTask dataclass."""
    
    def test_task_creation(self):
        """Test creating a swarm task."""
        task = SwarmTask(
            task_id="task_001",
            task_type="general",
            prompt="Test prompt",
            role=AgentRole.WORKER
        )
        
        assert task.task_id == "task_001"
        assert task.task_type == "general"
        assert task.prompt == "Test prompt"
        assert task.role == AgentRole.WORKER
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.reward == 0.0
        
    def test_task_with_parent(self):
        """Test creating a task with parent."""
        parent_task = SwarmTask(
            task_id="parent_001",
            task_type="general",
            prompt="Parent prompt",
            role=AgentRole.ORCHESTRATOR
        )
        
        child_task = SwarmTask(
            task_id="child_001",
            task_type="subtask",
            prompt="Child prompt",
            role=AgentRole.WORKER,
            parent_task_id=parent_task.task_id
        )
        
        assert child_task.parent_task_id == "parent_001"


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating agent config."""
        mock_model = Mock()
        config = AgentConfig(
            agent_id="agent_001",
            role=AgentRole.WORKER,
            model=mock_model,
            max_concurrent_tasks=5,
            temperature=0.8
        )
        
        assert config.agent_id == "agent_001"
        assert config.role == AgentRole.WORKER
        assert config.model == mock_model
        assert config.max_concurrent_tasks == 5
        assert config.temperature == 0.8


class TestSwarmAgent:
    """Tests for SwarmAgent."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock swarm agent."""
        mock_model = Mock()
        config = AgentConfig(
            agent_id="test_agent",
            role=AgentRole.WORKER,
            model=mock_model
        )
        return SwarmAgent(config)
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.agent_id == "test_agent"
        assert mock_agent.role == AgentRole.WORKER
        assert mock_agent.tasks == {}
        assert mock_agent.tasks_completed == 0
        assert mock_agent.tasks_failed == 0
        
    def test_submit_task(self, mock_agent):
        """Test submitting a task."""
        task = SwarmTask(
            task_id="task_001",
            task_type="test",
            prompt="Test prompt",
            role=AgentRole.WORKER
        )
        
        task_id = mock_agent.submit_task(task)
        
        assert task_id == "task_001"
        assert "task_001" in mock_agent.tasks
        assert mock_agent.tasks["task_001"] == task
        
    def test_plan_task(self, mock_agent):
        """Test task planning."""
        task = SwarmTask(
            task_id="task_001",
            task_type="planning",
            prompt="Plan a complex task",
            role=AgentRole.PLANNER
        )
        
        plan = mock_agent._plan_task(task)
        
        assert "original_task" in plan
        assert "subtasks" in plan
        assert len(plan["subtasks"]) > 0
        
    def test_execute_tool(self, mock_agent):
        """Test tool execution."""
        task = SwarmTask(
            task_id="task_001",
            task_type="execution",
            prompt="Search for something",
            role=AgentRole.EXECUTOR
        )
        
        result = mock_agent._execute_tool(task)
        
        assert "tool" in result
        assert "query" in result
        assert "result" in result
        
    def test_evaluate_task(self, mock_agent):
        """Test task evaluation."""
        task = SwarmTask(
            task_id="task_001",
            task_type="evaluation",
            prompt="Evaluate output",
            role=AgentRole.CRITIC
        )
        
        evaluation = mock_agent._evaluate_task(task)
        
        assert "quality_score" in evaluation
        assert "reward" in evaluation
        assert task.reward == evaluation["reward"]


class TestAgentSwarm:
    """Tests for AgentSwarm."""
    
    @pytest.fixture
    def mock_model_factory(self):
        """Create a mock model factory."""
        return lambda: Mock()
    
    @pytest.fixture
    def swarm(self, mock_model_factory):
        """Create an agent swarm."""
        return AgentSwarm(
            model_factory=mock_model_factory,
            max_agents=10,
            max_concurrent_tasks=100
        )
    
    def test_swarm_initialization(self, swarm):
        """Test swarm initialization."""
        assert swarm.max_agents == 10
        assert swarm.max_concurrent_tasks == 100
        assert swarm.agents == {}
        assert swarm.orchestrator is None
        
    def test_initialize_swarm(self, swarm):
        """Test initializing the swarm."""
        swarm.initialize()
        
        assert "orchestrator" in swarm.agents
        assert swarm.orchestrator is not None
        assert swarm.orchestrator.role == AgentRole.ORCHESTRATOR
        
    def test_create_agent(self, swarm):
        """Test creating an agent."""
        swarm.initialize()
        
        agent_id = swarm.create_agent(AgentRole.WORKER)
        
        assert agent_id in swarm.agents
        assert swarm.agents[agent_id].role == AgentRole.WORKER
        assert swarm.total_agents_created == 1
        
    def test_max_agents_limit(self, swarm):
        """Test maximum agents limit."""
        swarm.initialize()
        
        # Create agents up to limit
        for _ in range(9):  # 9 more + 1 orchestrator = 10
            swarm.create_agent(AgentRole.WORKER)
        
        # Should raise error when exceeding limit
        with pytest.raises(RuntimeError):
            swarm.create_agent(AgentRole.WORKER)
            
    def test_execute_single_task(self, swarm):
        """Test executing a single task."""
        swarm.initialize()
        
        task = swarm.execute_task("Simple task")
        
        assert task.status == TaskStatus.COMPLETED
        assert task.task_id in swarm.completed_tasks
        
    def test_get_statistics(self, swarm):
        """Test getting swarm statistics."""
        swarm.initialize()
        
        # Execute some tasks
        swarm.execute_task("Task 1")
        swarm.execute_task("Task 2")
        
        stats = swarm.get_statistics()
        
        assert "total_agents" in stats
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert stats["total_tasks"] == 2
        
    def test_shutdown(self, swarm):
        """Test shutting down the swarm."""
        swarm.initialize()
        swarm.create_agent(AgentRole.WORKER)
        
        swarm.shutdown()
        
        assert len(swarm.agents) == 0


class TestAgentSwarmParallelExecution:
    """Tests for parallel execution in agent swarm."""
    
    @pytest.fixture
    def mock_model_factory(self):
        """Create a mock model factory."""
        return lambda: Mock()
    
    @pytest.fixture
    def swarm(self, mock_model_factory):
        """Create an agent swarm."""
        return AgentSwarm(
            model_factory=mock_model_factory,
            max_agents=20,
            max_concurrent_tasks=100
        )
    
    def test_parallel_task_execution(self, swarm):
        """Test parallel execution of multiple tasks."""
        swarm.initialize()
        
        # Execute multiple tasks
        start_time = time.time()
        
        task1 = swarm.execute_task("Task 1")
        task2 = swarm.execute_task("Task 2")
        task3 = swarm.execute_task("Task 3")
        
        elapsed = time.time() - start_time
        
        # All tasks should be completed
        assert task1.status == TaskStatus.COMPLETED
        assert task2.status == TaskStatus.COMPLETED
        assert task3.status == TaskStatus.COMPLETED
        
        # Should complete faster than sequential execution
        # (this is a loose check)
        assert elapsed < 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
