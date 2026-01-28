"""
Agent Swarm implementation inspired by Kimi K2.5.

Kimi K2.5 uses a self-directed agent swarm paradigm where:
1. Up to 100 sub-agents can be created automatically
2. Parallel workflows with up to 1,500 tool calls
3. No predefined subagents or workflow - automatically orchestrated
4. Reduces execution time by up to 4.5x compared to single-agent

Reference: https://www.kimi.com/blog/kimi-k2-5.html
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import queue
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from foundation.utils.logging_utils import get_logger


class AgentRole(Enum):
    """Roles for agents in the swarm."""
    ORCHESTRATOR = "orchestrator"  # Main agent that coordinates the swarm
    WORKER = "worker"              # Worker agent that executes tasks
    CRITIC = "critic"              # Critic agent that evaluates outputs
    PLANNER = "planner"            # Planner agent that breaks down tasks
    EXECUTOR = "executor"          # Executor agent that runs tools/calls


class TaskStatus(Enum):
    """Status of a task in the swarm."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SwarmTask:
    """A task in the agent swarm."""
    task_id: str
    task_type: str
    prompt: str
    role: AgentRole
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class AgentConfig:
    """Configuration for an agent in the swarm."""
    agent_id: str
    role: AgentRole
    model: nn.Module
    max_concurrent_tasks: int = 5
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512


class SwarmAgent:
    """
    An agent in the Kimi K2.5 style swarm.
    
    Each agent can:
    1. Execute tasks independently
    2. Spawn sub-agents for complex tasks
    3. Communicate with other agents
    4. Self-direct workflow without predefined patterns
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.role = config.role
        self.model = config.model
        
        self.logger = get_logger(f"SwarmAgent-{self.agent_id}")
        
        # Task management
        self.tasks: Dict[str, SwarmTask] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.running_tasks: Set[str] = set()
        
        # Communication
        self.message_queue: queue.Queue = queue.Queue()
        self.neighbors: List[str] = []  # IDs of neighboring agents
        
        # Statistics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_reward = 0.0
        
        self._running = False
        
    def start(self):
        """Start the agent's main loop."""
        self._running = True
        self.logger.info(f"Agent {self.agent_id} ({self.role.value}) started")
        
        while self._running:
            try:
                # Process incoming messages
                self._process_messages()
                
                # Process tasks
                self._process_tasks()
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Agent {self.agent_id} error: {e}")
                
    def stop(self):
        """Stop the agent."""
        self._running = False
        self.logger.info(f"Agent {self.agent_id} stopped")
        
    def submit_task(self, task: SwarmTask) -> str:
        """
        Submit a task to this agent.
        
        Args:
            task: The task to execute
            
        Returns:
            Task ID
        """
        self.tasks[task.task_id] = task
        self.task_queue.put(task)
        self.logger.debug(f"Task {task.task_id} submitted to agent {self.agent_id}")
        return task.task_id
        
    def _process_messages(self):
        """Process incoming messages from other agents."""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass
            
    def _handle_message(self, message: Dict[str, Any]):
        """Handle a message from another agent."""
        msg_type = message.get('type')
        
        if msg_type == 'task_request':
            # Another agent is requesting help with a task
            task = message.get('task')
            if task and len(self.running_tasks) < self.config.max_concurrent_tasks:
                self.submit_task(task)
                
        elif msg_type == 'task_result':
            # Receive task result from another agent
            task_id = message.get('task_id')
            result = message.get('result')
            if task_id in self.tasks:
                self.tasks[task_id].result = result
                self.tasks[task_id].status = TaskStatus.COMPLETED
                
        elif msg_type == 'coordination':
            # Coordination message from orchestrator
            pass
            
    def _process_tasks(self):
        """Process tasks from the queue."""
        if len(self.running_tasks) >= self.config.max_concurrent_tasks:
            return
            
        try:
            task = self.task_queue.get_nowait()
            self.running_tasks.add(task.task_id)
            task.status = TaskStatus.RUNNING
            
            # Execute task in a separate thread
            thread = threading.Thread(target=self._execute_task, args=(task,))
            thread.start()
            
        except queue.Empty:
            pass
            
    def _execute_task(self, task: SwarmTask):
        """
        Execute a task.
        
        This is where the agent performs its work based on its role.
        """
        try:
            self.logger.info(f"Agent {self.agent_id} executing task {task.task_id}")
            
            if self.role == AgentRole.PLANNER:
                result = self._plan_task(task)
            elif self.role == AgentRole.EXECUTOR:
                result = self._execute_tool(task)
            elif self.role == AgentRole.CRITIC:
                result = self._evaluate_task(task)
            else:
                result = self._generate_response(task)
                
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            
            self.tasks_completed += 1
            self.logger.info(f"Task {task.task_id} completed by agent {self.agent_id}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            self.tasks_failed += 1
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            self.running_tasks.discard(task.task_id)
            
    def _plan_task(self, task: SwarmTask) -> Dict[str, Any]:
        """
        Plan how to execute a complex task by breaking it into subtasks.
        
        This is the key innovation in Kimi K2.5 - automatic task decomposition.
        """
        # Use the model to plan task decomposition
        prompt = f"""Break down the following task into subtasks:
Task: {task.prompt}

Provide a list of subtasks that can be executed in parallel or sequence."""

        # Generate plan (simplified)
        plan = {
            'original_task': task.prompt,
            'subtasks': [
                {'type': 'research', 'description': f'Research: {task.prompt}'},
                {'type': 'analysis', 'description': f'Analyze: {task.prompt}'},
                {'type': 'synthesis', 'description': f'Synthesize: {task.prompt}'},
            ],
            'dependencies': [],
            'estimated_steps': 3
        }
        
        return plan
        
    def _execute_tool(self, task: SwarmTask) -> Any:
        """Execute a tool or external call."""
        # In real implementation, this would call actual tools
        # For now, return a mock result
        return {
            'tool': 'search',
            'query': task.prompt,
            'result': f'Result for: {task.prompt}'
        }
        
    def _evaluate_task(self, task: SwarmTask) -> Dict[str, Any]:
        """Evaluate the quality of a task result."""
        # Critic agent evaluates outputs
        evaluation = {
            'quality_score': 0.85,
            'issues': [],
            'suggestions': ['Improve clarity'],
            'reward': 0.85
        }
        
        task.reward = evaluation['reward']
        self.total_reward += evaluation['reward']
        
        return evaluation
        
    def _generate_response(self, task: SwarmTask) -> str:
        """Generate a response using the model."""
        # In real implementation, use the model to generate
        # For now, return a mock response
        return f"Response to: {task.prompt}"
        
    def send_message(self, target_agent_id: str, message: Dict[str, Any]):
        """Send a message to another agent."""
        # In real implementation, this would use a message bus
        # For now, just log it
        self.logger.debug(f"Message from {self.agent_id} to {target_agent_id}: {message['type']}")


class AgentSwarm:
    """
    Agent Swarm orchestrator inspired by Kimi K2.5.
    
    Key features:
    1. Self-directed: Automatically creates and orchestrates sub-agents
    2. Parallel: Executes up to 100 sub-agents concurrently
    3. Efficient: Reduces execution time by up to 4.5x
    4. Scalable: Supports up to 1,500 tool calls
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        max_agents: int = 100,
        max_concurrent_tasks: int = 1500,
    ):
        """
        Initialize the agent swarm.
        
        Args:
            model_factory: Factory function to create models for agents
            max_agents: Maximum number of agents (Kimi K2.5 uses up to 100)
            max_concurrent_tasks: Maximum concurrent tool calls (Kimi K2.5 uses up to 1,500)
        """
        self.model_factory = model_factory
        self.max_agents = max_agents
        self.max_concurrent_tasks = max_concurrent_tasks
        
        self.logger = get_logger("AgentSwarm")
        
        # Agent management
        self.agents: Dict[str, SwarmAgent] = {}
        self.orchestrator: Optional[SwarmAgent] = None
        
        # Task management
        self.global_task_queue: queue.Queue = queue.Queue()
        self.completed_tasks: Dict[str, SwarmTask] = {}
        
        # Statistics
        self.total_tasks = 0
        self.total_agents_created = 0
        self.start_time: Optional[float] = None
        
    def initialize(self):
        """Initialize the swarm with an orchestrator agent."""
        self.logger.info(f"Initializing Agent Swarm (max_agents={self.max_agents})")
        
        # Create orchestrator
        orchestrator_config = AgentConfig(
            agent_id="orchestrator",
            role=AgentRole.ORCHESTRATOR,
            model=self.model_factory(),
            max_concurrent_tasks=10
        )
        self.orchestrator = SwarmAgent(orchestrator_config)
        self.agents["orchestrator"] = self.orchestrator
        
        self.logger.info("Agent Swarm initialized with orchestrator")
        
    def create_agent(self, role: AgentRole) -> str:
        """
        Create a new agent in the swarm.
        
        Args:
            role: Role of the new agent
            
        Returns:
            Agent ID
        """
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Maximum number of agents ({self.max_agents}) reached")
            
        agent_id = f"agent_{role.value}_{uuid.uuid4().hex[:8]}"
        
        config = AgentConfig(
            agent_id=agent_id,
            role=role,
            model=self.model_factory(),
        )
        
        agent = SwarmAgent(config)
        self.agents[agent_id] = agent
        self.total_agents_created += 1
        
        self.logger.info(f"Created agent {agent_id} with role {role.value}")
        
        return agent_id
        
    def execute_task(self, prompt: str, task_type: str = "general") -> SwarmTask:
        """
        Execute a task using the agent swarm.
        
        This is the main entry point for task execution.
        
        Args:
            prompt: The task prompt
            task_type: Type of task
            
        Returns:
            Completed task with results
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        self.logger.info(f"Executing task: {prompt[:50]}...")
        
        # Create main task
        task = SwarmTask(
            task_id=f"task_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            prompt=prompt,
            role=AgentRole.WORKER
        )
        
        self.total_tasks += 1
        
        # Orchestrator analyzes task and creates sub-agents as needed
        subtasks = self._decompose_task(task)
        
        if len(subtasks) == 1:
            # Simple task - execute directly
            return self._execute_single_task(task)
        else:
            # Complex task - use swarm
            return self._execute_swarm_task(task, subtasks)
            
    def _decompose_task(self, task: SwarmTask) -> List[SwarmTask]:
        """
        Decompose a task into subtasks.
        
        This is where the self-directed nature comes in - the orchestrator
        decides how to break down the task.
        """
        # Use planner agent to decompose task
        planner_id = self.create_agent(AgentRole.PLANNER)
        planner = self.agents[planner_id]
        
        # Submit planning task
        plan_task = SwarmTask(
            task_id=f"plan_{task.task_id}",
            task_type="planning",
            prompt=task.prompt,
            role=AgentRole.PLANNER
        )
        
        planner.submit_task(plan_task)
        
        # Wait for plan (simplified)
        while plan_task.status != TaskStatus.COMPLETED:
            time.sleep(0.1)
            planner._process_tasks()
            
        plan = plan_task.result
        
        # Create subtasks from plan
        subtasks = []
        for i, subtask_info in enumerate(plan.get('subtasks', [])):
            subtask = SwarmTask(
                task_id=f"{task.task_id}_sub_{i}",
                task_type=subtask_info['type'],
                prompt=subtask_info['description'],
                role=AgentRole.WORKER,
                parent_task_id=task.task_id
            )
            subtasks.append(subtask)
            
        # Clean up planner
        del self.agents[planner_id]
        
        return subtasks if subtasks else [task]
        
    def _execute_single_task(self, task: SwarmTask) -> SwarmTask:
        """Execute a single task with one agent."""
        # Create a worker agent
        worker_id = self.create_agent(AgentRole.WORKER)
        worker = self.agents[worker_id]
        
        # Execute task
        worker.submit_task(task)
        
        # Wait for completion (simplified)
        while task.status != TaskStatus.COMPLETED and task.status != TaskStatus.FAILED:
            time.sleep(0.1)
            worker._process_tasks()
            
        self.completed_tasks[task.task_id] = task
        
        # Clean up worker
        del self.agents[worker_id]
        
        return task
        
    def _execute_swarm_task(
        self,
        task: SwarmTask,
        subtasks: List[SwarmTask]
    ) -> SwarmTask:
        """
        Execute a complex task using multiple agents in parallel.
        
        This is the key to Kimi K2.5's 4.5x speedup.
        """
        self.logger.info(f"Executing swarm task with {len(subtasks)} subtasks")
        
        # Create worker agents for each subtask
        with ThreadPoolExecutor(max_workers=min(len(subtasks), 32)) as executor:
            futures = []
            
            for subtask in subtasks:
                worker_id = self.create_agent(AgentRole.WORKER)
                worker = self.agents[worker_id]
                
                # Submit subtask
                future = executor.submit(self._execute_subtask, worker, subtask)
                futures.append((future, subtask))
                
            # Wait for all subtasks to complete
            results = []
            for future, subtask in futures:
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Subtask {subtask.task_id} failed: {e}")
                    
        # Aggregate results
        task.result = {
            'subtask_results': results,
            'num_subtasks': len(subtasks),
            'num_completed': len(results)
        }
        task.status = TaskStatus.COMPLETED
        
        self.completed_tasks[task.task_id] = task
        
        return task
        
    def _execute_subtask(self, worker: SwarmAgent, subtask: SwarmTask) -> Any:
        """Execute a subtask with a worker agent."""
        worker.submit_task(subtask)
        
        # Wait for completion
        while subtask.status != TaskStatus.COMPLETED and subtask.status != TaskStatus.FAILED:
            time.sleep(0.01)
            worker._process_tasks()
            
        return subtask.result
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_agents': len(self.agents),
            'total_agents_created': self.total_agents_created,
            'total_tasks': self.total_tasks,
            'completed_tasks': len(self.completed_tasks),
            'elapsed_time': elapsed,
            'tasks_per_second': self.total_tasks / elapsed if elapsed > 0 else 0,
        }
        
    def shutdown(self):
        """Shutdown all agents in the swarm."""
        self.logger.info("Shutting down Agent Swarm")
        
        for agent in self.agents.values():
            agent.stop()
            
        self.agents.clear()
        self.logger.info("Agent Swarm shut down")
