"""
Type definitions for adaptive topology management
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import numpy as np
from uuid import uuid4


class TopologyType(Enum):
    """Different types of topology structures"""
    CENTRALIZED = "centralized"  # Star topology with central coordinator
    DECENTRALIZED = "decentralized"  # Fully connected mesh
    HIERARCHICAL = "hierarchical"  # Tree-like structure
    RING = "ring"  # Ring topology
    HYBRID = "hybrid"  # Mixed topology
    DYNAMIC = "dynamic"  # Dynamically changing topology


class CommunicationMode(Enum):
    """Communication modes for agents"""
    BROADCAST = "broadcast"  # One-to-all communication
    POINT_TO_POINT = "point_to_point"  # Direct agent-to-agent
    HIERARCHICAL_REPORT = "hierarchical_report"  # Bottom-up reporting
    MULTICAST = "multicast"  # One-to-many selective
    GOSSIP = "gossip"  # Peer-to-peer spreading


@dataclass
class AgentCapability:
    """Represents an agent's capabilities and current state"""
    agent_id: str
    name: str
    role: str
    tools: List[str]
    current_load: float = 0.0  # 0.0 to 1.0
    expertise_domains: List[str] = None
    communication_bandwidth: float = 1.0  # Available bandwidth
    processing_power: float = 1.0  # Relative processing capability
    reliability_score: float = 1.0  # Historical reliability
    
    def __post_init__(self):
        if self.expertise_domains is None:
            self.expertise_domains = []


@dataclass
class TaskState:
    """Current state of the task being executed"""
    task_id: str
    phase: str  # e.g., "initialization", "execution", "consolidation"
    complexity: float  # 0.0 to 1.0
    urgency: float  # 0.0 to 1.0
    required_capabilities: List[str]
    current_bottlenecks: List[str] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.current_bottlenecks is None:
            self.current_bottlenecks = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class TopologyMetrics:
    """Metrics for evaluating topology performance"""
    efficiency: float = 0.0  # Task completion efficiency
    communication_cost: float = 0.0  # Total communication overhead
    load_balance: float = 0.0  # How well load is distributed
    fault_tolerance: float = 0.0  # Resilience to agent failures
    adaptability: float = 0.0  # How quickly topology can adapt
    convergence_time: float = 0.0  # Time to reach consensus
    
    def overall_score(self) -> float:
        """Calculate overall topology performance score"""
        weights = {
            'efficiency': 0.3,
            'communication_cost': -0.2,  # Lower is better
            'load_balance': 0.2,
            'fault_tolerance': 0.15,
            'adaptability': 0.1,
            'convergence_time': -0.05  # Lower is better
        }
        
        score = (
            weights['efficiency'] * self.efficiency +
            weights['communication_cost'] * (1.0 - self.communication_cost) +
            weights['load_balance'] * self.load_balance +
            weights['fault_tolerance'] * self.fault_tolerance +
            weights['adaptability'] * self.adaptability +
            weights['convergence_time'] * (1.0 - min(self.convergence_time, 1.0))
        )
        
        return max(0.0, min(1.0, score))


@dataclass
class TopologyGraph:
    """Represents the topology as a graph structure"""
    adjacency_matrix: np.ndarray  # NxN matrix for N agents
    agent_ids: List[str]  # Ordered list of agent IDs
    edge_weights: Optional[np.ndarray] = None  # Communication weights
    topology_type: TopologyType = TopologyType.HYBRID
    
    def __post_init__(self):
        if self.edge_weights is None:
            self.edge_weights = self.adjacency_matrix.copy()
    
    def get_neighbors(self, agent_id: str) -> List[str]:
        """Get direct neighbors of an agent"""
        if agent_id not in self.agent_ids:
            return []
        
        agent_idx = self.agent_ids.index(agent_id)
        neighbors = []
        
        for i, connected in enumerate(self.adjacency_matrix[agent_idx]):
            if connected and i != agent_idx:
                neighbors.append(self.agent_ids[i])
        
        return neighbors
    
    def get_communication_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest communication path between two agents"""
        if source not in self.agent_ids or target not in self.agent_ids:
            return None
        
        # Simple BFS for shortest path
        from collections import deque
        
        source_idx = self.agent_ids.index(source)
        target_idx = self.agent_ids.index(target)
        
        if source_idx == target_idx:
            return [source]
        
        queue = deque([(source_idx, [source_idx])])
        visited = {source_idx}
        
        while queue:
            current_idx, path = queue.popleft()
            
            for next_idx, connected in enumerate(self.adjacency_matrix[current_idx]):
                if connected and next_idx not in visited:
                    new_path = path + [next_idx]
                    
                    if next_idx == target_idx:
                        return [self.agent_ids[i] for i in new_path]
                    
                    queue.append((next_idx, new_path))
                    visited.add(next_idx)
        
        return None  # No path found


@dataclass
class CommunicationMessage:
    """Represents a message in the communication protocol"""
    message_id: str
    sender_id: str
    receiver_ids: List[str]  # Can be multiple for multicast
    content: Any
    message_type: str
    priority: float = 0.5  # 0.0 to 1.0
    timestamp: float = 0.0
    mode: CommunicationMode = CommunicationMode.POINT_TO_POINT
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid4())


@dataclass
class TopologyTransition:
    """Represents a transition from one topology to another"""
    from_topology: TopologyGraph
    to_topology: TopologyGraph
    transition_reason: str
    transition_cost: float
    expected_improvement: float
    timestamp: float
