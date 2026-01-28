"""
Distributed training components with Agent Swarm architecture.

Inspired by Kimi K2.5's self-directed agent swarm paradigm:
- Up to 100 sub-agents can be created automatically
- Parallel workflows with up to 1,500 tool calls
- No predefined subagents or workflow - automatically orchestrated
- Reduces execution time by up to 4.5x compared to single-agent

Reference: https://www.kimi.com/blog/kimi-k2-5.html
"""

# Agent Swarm (Kimi K2.5 style)
from foundation.distributed.agent_swarm import (
    AgentSwarm,
    SwarmAgent,
    AgentConfig,
    SwarmTask,
    AgentRole,
    TaskStatus,
)

# Legacy PARL components (kept for compatibility)
from foundation.distributed.parl_cluster import PARLCluster, PARLConfig
from foundation.distributed.actor_learner import Actor, Learner
from foundation.distributed.parameter_server import ParameterServer
from foundation.distributed.communication import DistributedCommunicator

__all__ = [
    # Agent Swarm (Kimi K2.5 style)
    "AgentSwarm",
    "SwarmAgent",
    "AgentConfig",
    "SwarmTask",
    "AgentRole",
    "TaskStatus",
    # Legacy PARL components
    "PARLCluster",
    "PARLConfig",
    "Actor",
    "Learner",
    "ParameterServer",
    "DistributedCommunicator",
]
