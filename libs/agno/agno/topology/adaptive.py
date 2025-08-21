"""
Adaptive Topology Management

Main class that integrates GNN topology generation, RL topology search,
and hybrid communication protocols for dynamic multi-agent collaboration.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum

from agno.topology.types import (
    AgentCapability,
    TaskState,
    TopologyGraph,
    TopologyMetrics,
    TopologyTransition,
    TopologyType
)
from agno.topology.gnn_generator import GNNTopologyGenerator, GNNConfig
from agno.topology.rl_search import RLTopologySearch, RLConfig
from agno.topology.communication import HybridCommunicationProtocol
from agno.utils.log import logger


class AdaptationTrigger(Enum):
    """Triggers for topology adaptation"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TASK_PHASE_CHANGE = "task_phase_change"
    AGENT_FAILURE = "agent_failure"
    LOAD_IMBALANCE = "load_imbalance"
    COMMUNICATION_BOTTLENECK = "communication_bottleneck"
    MANUAL = "manual"


@dataclass
class AdaptationConfig:
    """Configuration for adaptive topology management"""
    # Performance thresholds
    min_efficiency_threshold: float = 0.6
    max_communication_cost_threshold: float = 0.8
    min_load_balance_threshold: float = 0.5
    
    # Adaptation parameters
    adaptation_cooldown: float = 30.0  # Seconds between adaptations
    performance_window: int = 10  # Number of measurements to consider
    
    # Method preferences
    prefer_gnn: bool = True  # Prefer GNN over RL for topology generation
    enable_rl_learning: bool = True  # Enable RL learning from adaptations
    
    # Communication settings
    enable_hybrid_communication: bool = True
    communication_bandwidth: float = 10.0


class AdaptiveTopology:
    """
    Adaptive topology management system that combines:
    - GNN-based topology generation
    - RL-based topology search
    - Hybrid communication protocols
    - Dynamic reconfiguration based on performance metrics
    """
    
    def __init__(
        self,
        agents: List[AgentCapability],
        initial_task: TaskState,
        config: Optional[AdaptationConfig] = None,
        gnn_config: Optional[GNNConfig] = None,
        rl_config: Optional[RLConfig] = None
    ):
        self.agents = agents
        self.current_task = initial_task
        self.config = config or AdaptationConfig()
        
        # Initialize topology generators
        self.gnn_generator = GNNTopologyGenerator(gnn_config)
        self.rl_search = RLTopologySearch(rl_config)
        
        # Current topology and communication
        self.current_topology: Optional[TopologyGraph] = None
        self.communication_protocol: Optional[HybridCommunicationProtocol] = None
        
        # Performance tracking
        self.performance_history: List[TopologyMetrics] = []
        self.topology_history: List[TopologyTransition] = []
        self.last_adaptation_time = 0.0
        
        # Callbacks
        self.adaptation_callbacks: List[Callable[[TopologyTransition], None]] = []
        
        # Initialize with default topology
        self._initialize_topology()
        
    def _initialize_topology(self):
        """Initialize with a default topology"""
        # Start with a simple centralized topology
        n_agents = len(self.agents)
        adj_matrix = np.zeros((n_agents, n_agents))
        
        # Connect all agents to the first agent (star topology)
        if n_agents > 1:
            for i in range(1, n_agents):
                adj_matrix[0, i] = 1.0
                adj_matrix[i, 0] = 1.0
        
        self.current_topology = TopologyGraph(
            adjacency_matrix=adj_matrix,
            agent_ids=[agent.agent_id for agent in self.agents],
            topology_type=TopologyType.CENTRALIZED
        )
        
        # Initialize communication protocol
        if self.config.enable_hybrid_communication:
            self.communication_protocol = HybridCommunicationProtocol(
                self.current_topology,
                self.agents,
                self.config.communication_bandwidth
            )
        
        logger.info(f"Initialized adaptive topology with {n_agents} agents")
    
    def update_task_state(self, new_task_state: TaskState):
        """Update current task state and trigger adaptation if needed"""
        old_phase = self.current_task.phase if self.current_task else None
        self.current_task = new_task_state
        
        # Check if task phase changed significantly
        if old_phase and old_phase != new_task_state.phase:
            self._trigger_adaptation(AdaptationTrigger.TASK_PHASE_CHANGE)
    
    def update_agent_capabilities(self, updated_agents: List[AgentCapability]):
        """Update agent capabilities and trigger adaptation if needed"""
        # Check for significant changes in agent capabilities
        significant_change = False
        
        for new_agent in updated_agents:
            old_agent = next(
                (a for a in self.agents if a.agent_id == new_agent.agent_id), 
                None
            )
            
            if old_agent:
                # Check for significant load change
                if abs(new_agent.current_load - old_agent.current_load) > 0.3:
                    significant_change = True
                    break
                
                # Check for capability changes
                if (new_agent.reliability_score < 0.5 and 
                    old_agent.reliability_score >= 0.5):
                    significant_change = True
                    break
        
        self.agents = updated_agents
        
        if significant_change:
            self._trigger_adaptation(AdaptationTrigger.LOAD_IMBALANCE)
    
    def evaluate_current_performance(self) -> TopologyMetrics:
        """Evaluate current topology performance"""
        if not self.current_topology:
            return TopologyMetrics()
        
        # Use RL search for evaluation (it has built-in evaluation)
        metrics = self.rl_search.evaluate_topology(
            self.current_topology,
            self.agents,
            self.current_task
        )
        
        # Add to performance history
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.config.performance_window:
            self.performance_history.pop(0)
        
        return metrics
    
    def should_adapt(self, current_metrics: TopologyMetrics) -> Tuple[bool, AdaptationTrigger]:
        """Determine if topology should be adapted based on current metrics"""
        # Check cooldown period
        if (time.time() - self.last_adaptation_time < 
            self.config.adaptation_cooldown):
            return False, None
        
        # Check performance thresholds
        if current_metrics.efficiency < self.config.min_efficiency_threshold:
            return True, AdaptationTrigger.PERFORMANCE_DEGRADATION
        
        if (current_metrics.communication_cost > 
            self.config.max_communication_cost_threshold):
            return True, AdaptationTrigger.COMMUNICATION_BOTTLENECK
        
        if current_metrics.load_balance < self.config.min_load_balance_threshold:
            return True, AdaptationTrigger.LOAD_IMBALANCE
        
        # Check for performance degradation trend
        if len(self.performance_history) >= 3:
            recent_scores = [m.overall_score() for m in self.performance_history[-3:]]
            if all(recent_scores[i] > recent_scores[i+1] for i in range(len(recent_scores)-1)):
                return True, AdaptationTrigger.PERFORMANCE_DEGRADATION
        
        return False, None
    
    def adapt_topology(
        self, 
        trigger: AdaptationTrigger = AdaptationTrigger.MANUAL
    ) -> Optional[TopologyTransition]:
        """
        Adapt topology based on current conditions
        
        Args:
            trigger: Reason for adaptation
            
        Returns:
            Topology transition if adaptation occurred
        """
        if not self.current_topology:
            return None
        
        logger.info(f"Adapting topology due to: {trigger.value}")
        
        old_topology = self.current_topology
        new_topology = None
        
        # Choose adaptation method
        if self.config.prefer_gnn:
            # Try GNN first
            try:
                new_topology = self.gnn_generator.generate_topology(
                    self.agents,
                    self.current_task,
                    self.current_topology
                )
                logger.info("Generated new topology using GNN")
            except Exception as e:
                logger.warning(f"GNN topology generation failed: {e}")
        
        # Fallback to RL search or if GNN not preferred
        if not new_topology:
            try:
                new_topology, _ = self.rl_search.search_optimal_topology(
                    self.agents,
                    self.current_task,
                    num_episodes=50  # Reduced for real-time adaptation
                )
                logger.info("Generated new topology using RL search")
            except Exception as e:
                logger.error(f"RL topology search failed: {e}")
                return None
        
        if not new_topology:
            logger.error("Failed to generate new topology")
            return None
        
        # Evaluate new topology
        new_metrics = self.rl_search.evaluate_topology(
            new_topology,
            self.agents,
            self.current_task
        )
        
        current_metrics = self.evaluate_current_performance()
        
        # Only adapt if new topology is significantly better
        improvement_threshold = 0.05  # 5% improvement required
        if (new_metrics.overall_score() > 
            current_metrics.overall_score() + improvement_threshold):
            
            # Create transition record
            transition = TopologyTransition(
                from_topology=old_topology,
                to_topology=new_topology,
                transition_reason=trigger.value,
                transition_cost=self._calculate_transition_cost(old_topology, new_topology),
                expected_improvement=new_metrics.overall_score() - current_metrics.overall_score(),
                timestamp=time.time()
            )
            
            # Apply new topology
            self.current_topology = new_topology
            self.last_adaptation_time = time.time()
            
            # Update communication protocol
            if self.communication_protocol:
                self.communication_protocol.update_topology(new_topology)
            
            # Record transition
            self.topology_history.append(transition)
            
            # Notify callbacks
            for callback in self.adaptation_callbacks:
                try:
                    callback(transition)
                except Exception as e:
                    logger.error(f"Adaptation callback failed: {e}")
            
            # Train RL model if enabled
            if self.config.enable_rl_learning:
                try:
                    self.rl_search.train_step()
                except Exception as e:
                    logger.warning(f"RL training step failed: {e}")
            
            logger.info(f"Topology adapted successfully. "
                       f"Expected improvement: {transition.expected_improvement:.3f}")
            
            return transition
        else:
            logger.info("New topology not significantly better, keeping current topology")
            return None
    
    def _trigger_adaptation(self, trigger: AdaptationTrigger):
        """Trigger topology adaptation"""
        current_metrics = self.evaluate_current_performance()
        should_adapt, detected_trigger = self.should_adapt(current_metrics)
        
        if should_adapt or trigger == AdaptationTrigger.MANUAL:
            self.adapt_topology(trigger)
    
    def _calculate_transition_cost(
        self, 
        old_topology: TopologyGraph, 
        new_topology: TopologyGraph
    ) -> float:
        """Calculate cost of transitioning between topologies"""
        # Simple cost based on number of edge changes
        old_adj = old_topology.adjacency_matrix
        new_adj = new_topology.adjacency_matrix
        
        changes = np.sum(np.abs(old_adj - new_adj))
        total_possible_edges = old_adj.shape[0] * (old_adj.shape[0] - 1)
        
        return changes / total_possible_edges if total_possible_edges > 0 else 0.0
    
    def add_adaptation_callback(self, callback: Callable[[TopologyTransition], None]):
        """Add callback to be called when topology adapts"""
        self.adaptation_callbacks.append(callback)
    
    def get_current_topology(self) -> Optional[TopologyGraph]:
        """Get current topology"""
        return self.current_topology
    
    def get_communication_protocol(self) -> Optional[HybridCommunicationProtocol]:
        """Get communication protocol"""
        return self.communication_protocol
    
    def get_performance_history(self) -> List[TopologyMetrics]:
        """Get performance history"""
        return self.performance_history.copy()
    
    def get_topology_history(self) -> List[TopologyTransition]:
        """Get topology transition history"""
        return self.topology_history.copy()
    
    async def run_continuous_adaptation(self, check_interval: float = 10.0):
        """Run continuous topology adaptation in background"""
        logger.info("Starting continuous topology adaptation")
        
        while True:
            try:
                # Evaluate current performance
                current_metrics = self.evaluate_current_performance()
                
                # Check if adaptation is needed
                should_adapt, trigger = self.should_adapt(current_metrics)
                
                if should_adapt:
                    self.adapt_topology(trigger)
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in continuous adaptation: {e}")
                await asyncio.sleep(check_interval)
    
    def save_state(self, path: str):
        """Save adaptive topology state"""
        import pickle
        
        state = {
            'agents': self.agents,
            'current_task': self.current_task,
            'config': self.config,
            'current_topology': self.current_topology,
            'performance_history': self.performance_history,
            'topology_history': self.topology_history,
            'last_adaptation_time': self.last_adaptation_time
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        # Save models
        self.gnn_generator.save_model(f"{path}_gnn.pt")
        self.rl_search.save_model(f"{path}_rl.pt")
    
    def load_state(self, path: str):
        """Load adaptive topology state"""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.agents = state['agents']
        self.current_task = state['current_task']
        self.config = state['config']
        self.current_topology = state['current_topology']
        self.performance_history = state['performance_history']
        self.topology_history = state['topology_history']
        self.last_adaptation_time = state['last_adaptation_time']
        
        # Load models
        try:
            self.gnn_generator.load_model(f"{path}_gnn.pt")
            self.rl_search.load_model(f"{path}_rl.pt")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
        
        # Reinitialize communication protocol
        if self.config.enable_hybrid_communication and self.current_topology:
            self.communication_protocol = HybridCommunicationProtocol(
                self.current_topology,
                self.agents,
                self.config.communication_bandwidth
            )
