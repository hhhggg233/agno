"""
Topology Manager

High-level manager that integrates adaptive topology with agno's existing
Team and Workflow systems, providing seamless topology management.
"""

import asyncio
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
import time

from agno.agent import Agent
from agno.team import Team
from agno.topology.adaptive import AdaptiveTopology, AdaptationConfig
from agno.topology.types import (
    AgentCapability,
    TaskState,
    TopologyGraph,
    TopologyMetrics,
    TopologyTransition,
    CommunicationMessage,
    CommunicationMode
)
from agno.topology.gnn_generator import GNNConfig
from agno.topology.rl_search import RLConfig
from agno.utils.log import logger


@dataclass
class TopologyManagerConfig:
    """Configuration for topology manager"""
    enable_adaptive_topology: bool = True
    auto_start_adaptation: bool = True
    adaptation_check_interval: float = 15.0
    
    # Integration settings
    override_team_communication: bool = True
    log_topology_changes: bool = True
    
    # Performance settings
    performance_monitoring_interval: float = 5.0
    min_agents_for_adaptation: int = 3


class TopologyManager:
    """
    High-level topology manager that integrates with agno's Team system
    to provide adaptive topology management for multi-agent collaboration.
    """
    
    def __init__(
        self,
        config: Optional[TopologyManagerConfig] = None,
        adaptation_config: Optional[AdaptationConfig] = None,
        gnn_config: Optional[GNNConfig] = None,
        rl_config: Optional[RLConfig] = None
    ):
        self.config = config or TopologyManagerConfig()
        self.adaptation_config = adaptation_config
        self.gnn_config = gnn_config
        self.rl_config = rl_config
        
        # Managed topologies
        self.managed_teams: Dict[str, Team] = {}
        self.adaptive_topologies: Dict[str, AdaptiveTopology] = {}
        
        # Performance monitoring
        self.performance_monitors: Dict[str, asyncio.Task] = {}
        self.adaptation_tasks: Dict[str, asyncio.Task] = {}
        
        # Callbacks
        self.topology_change_callbacks: List[Callable[[str, TopologyTransition], None]] = []
        
        # State
        self.is_running = False
        
    def register_team(
        self,
        team: Team,
        initial_task_description: str = "Multi-agent collaboration task",
        task_complexity: float = 0.5,
        task_urgency: float = 0.5
    ) -> bool:
        """
        Register a team for adaptive topology management
        
        Args:
            team: The team to manage
            initial_task_description: Description of the initial task
            task_complexity: Task complexity (0.0 to 1.0)
            task_urgency: Task urgency (0.0 to 1.0)
            
        Returns:
            True if registration successful
        """
        if not team.team_id:
            logger.error("Team must have a team_id to be managed")
            return False
        
        if len(team.members) < self.config.min_agents_for_adaptation:
            logger.warning(f"Team {team.team_id} has too few members for adaptation")
            if not self.config.enable_adaptive_topology:
                return False
        
        # Convert team members to agent capabilities
        agents = self._extract_agent_capabilities(team)
        
        # Create initial task state
        initial_task = TaskState(
            task_id=f"task_{team.team_id}_{int(time.time())}",
            phase="initialization",
            complexity=task_complexity,
            urgency=task_urgency,
            required_capabilities=self._extract_required_capabilities(team)
        )
        
        # Create adaptive topology
        if self.config.enable_adaptive_topology:
            try:
                adaptive_topology = AdaptiveTopology(
                    agents=agents,
                    initial_task=initial_task,
                    config=self.adaptation_config,
                    gnn_config=self.gnn_config,
                    rl_config=self.rl_config
                )
                
                # Add callback for topology changes
                adaptive_topology.add_adaptation_callback(
                    lambda transition: self._on_topology_change(team.team_id, transition)
                )
                
                self.adaptive_topologies[team.team_id] = adaptive_topology
                
                logger.info(f"Registered team {team.team_id} for adaptive topology management")
                
            except Exception as e:
                logger.error(f"Failed to create adaptive topology for team {team.team_id}: {e}")
                return False
        
        # Store team reference
        self.managed_teams[team.team_id] = team
        
        # Start monitoring if auto-start is enabled
        if self.config.auto_start_adaptation and self.is_running:
            self._start_team_monitoring(team.team_id)
        
        return True
    
    def unregister_team(self, team_id: str) -> bool:
        """Unregister a team from topology management"""
        if team_id not in self.managed_teams:
            return False
        
        # Stop monitoring
        self._stop_team_monitoring(team_id)
        
        # Clean up
        self.managed_teams.pop(team_id, None)
        self.adaptive_topologies.pop(team_id, None)
        
        logger.info(f"Unregistered team {team_id} from topology management")
        return True
    
    def update_task_state(
        self,
        team_id: str,
        phase: Optional[str] = None,
        complexity: Optional[float] = None,
        urgency: Optional[float] = None,
        required_capabilities: Optional[List[str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update task state for a managed team"""
        if team_id not in self.adaptive_topologies:
            return False
        
        adaptive_topology = self.adaptive_topologies[team_id]
        current_task = adaptive_topology.current_task
        
        # Create updated task state
        updated_task = TaskState(
            task_id=current_task.task_id,
            phase=phase or current_task.phase,
            complexity=complexity if complexity is not None else current_task.complexity,
            urgency=urgency if urgency is not None else current_task.urgency,
            required_capabilities=required_capabilities or current_task.required_capabilities,
            current_bottlenecks=current_task.current_bottlenecks,
            performance_metrics=performance_metrics or current_task.performance_metrics
        )
        
        # Update task state
        adaptive_topology.update_task_state(updated_task)
        
        logger.debug(f"Updated task state for team {team_id}")
        return True
    
    def update_agent_loads(self, team_id: str, agent_loads: Dict[str, float]) -> bool:
        """Update agent load information"""
        if team_id not in self.adaptive_topologies:
            return False
        
        adaptive_topology = self.adaptive_topologies[team_id]
        
        # Update agent capabilities
        updated_agents = []
        for agent in adaptive_topology.agents:
            new_load = agent_loads.get(agent.agent_id, agent.current_load)
            
            updated_agent = AgentCapability(
                agent_id=agent.agent_id,
                name=agent.name,
                role=agent.role,
                tools=agent.tools,
                current_load=new_load,
                expertise_domains=agent.expertise_domains,
                communication_bandwidth=agent.communication_bandwidth,
                processing_power=agent.processing_power,
                reliability_score=agent.reliability_score
            )
            updated_agents.append(updated_agent)
        
        adaptive_topology.update_agent_capabilities(updated_agents)
        
        logger.debug(f"Updated agent loads for team {team_id}")
        return True
    
    def force_topology_adaptation(self, team_id: str) -> Optional[TopologyTransition]:
        """Force topology adaptation for a team"""
        if team_id not in self.adaptive_topologies:
            return None
        
        adaptive_topology = self.adaptive_topologies[team_id]
        return adaptive_topology.adapt_topology()
    
    def get_current_topology(self, team_id: str) -> Optional[TopologyGraph]:
        """Get current topology for a team"""
        if team_id not in self.adaptive_topologies:
            return None
        
        return self.adaptive_topologies[team_id].get_current_topology()
    
    def get_performance_metrics(self, team_id: str) -> Optional[TopologyMetrics]:
        """Get current performance metrics for a team"""
        if team_id not in self.adaptive_topologies:
            return None
        
        adaptive_topology = self.adaptive_topologies[team_id]
        return adaptive_topology.evaluate_current_performance()
    
    def send_message(
        self,
        team_id: str,
        sender_id: str,
        receiver_ids: List[str],
        content: Any,
        message_type: str = "general",
        priority: float = 0.5
    ) -> bool:
        """Send message through adaptive communication protocol"""
        if team_id not in self.adaptive_topologies:
            return False
        
        adaptive_topology = self.adaptive_topologies[team_id]
        comm_protocol = adaptive_topology.get_communication_protocol()
        
        if not comm_protocol:
            return False
        
        message = CommunicationMessage(
            message_id="",  # Will be auto-generated
            sender_id=sender_id,
            receiver_ids=receiver_ids,
            content=content,
            message_type=message_type,
            priority=priority,
            timestamp=time.time()
        )
        
        return comm_protocol.send_message(message)
    
    def receive_messages(self, team_id: str, agent_id: str) -> List[CommunicationMessage]:
        """Receive messages for an agent"""
        if team_id not in self.adaptive_topologies:
            return []
        
        adaptive_topology = self.adaptive_topologies[team_id]
        comm_protocol = adaptive_topology.get_communication_protocol()
        
        if not comm_protocol:
            return []
        
        return comm_protocol.receive_messages(agent_id)
    
    def start(self):
        """Start topology management"""
        self.is_running = True
        
        # Start monitoring for all registered teams
        for team_id in self.managed_teams.keys():
            self._start_team_monitoring(team_id)
        
        logger.info("Topology manager started")
    
    def stop(self):
        """Stop topology management"""
        self.is_running = False
        
        # Stop all monitoring tasks
        for team_id in list(self.managed_teams.keys()):
            self._stop_team_monitoring(team_id)
        
        logger.info("Topology manager stopped")
    
    def _extract_agent_capabilities(self, team: Team) -> List[AgentCapability]:
        """Extract agent capabilities from team members"""
        capabilities = []
        
        for member in team.members:
            if isinstance(member, Agent):
                # Extract tools
                tools = []
                if hasattr(member, 'tools') and member.tools:
                    tools = [tool.__class__.__name__ for tool in member.tools]
                
                # Extract expertise domains from role/description
                expertise_domains = []
                if member.role:
                    expertise_domains.append(member.role.lower())
                
                capability = AgentCapability(
                    agent_id=member.agent_id or member.name,
                    name=member.name or "Unknown",
                    role=member.role or "General",
                    tools=tools,
                    current_load=0.0,  # Will be updated dynamically
                    expertise_domains=expertise_domains,
                    communication_bandwidth=1.0,  # Default
                    processing_power=1.0,  # Default
                    reliability_score=1.0  # Default
                )
                capabilities.append(capability)
            
            elif isinstance(member, Team):
                # Handle nested teams (flatten for now)
                nested_capabilities = self._extract_agent_capabilities(member)
                capabilities.extend(nested_capabilities)
        
        return capabilities
    
    def _extract_required_capabilities(self, team: Team) -> List[str]:
        """Extract required capabilities from team"""
        capabilities = set()
        
        # Extract from team instructions
        if hasattr(team, 'instructions') and team.instructions:
            if isinstance(team.instructions, list):
                for instruction in team.instructions:
                    if isinstance(instruction, str):
                        # Simple keyword extraction
                        words = instruction.lower().split()
                        capabilities.update(words)
            elif isinstance(team.instructions, str):
                words = team.instructions.lower().split()
                capabilities.update(words)
        
        # Extract from member roles
        for member in team.members:
            if isinstance(member, Agent) and member.role:
                capabilities.add(member.role.lower())
        
        return list(capabilities)[:10]  # Limit to 10 capabilities
    
    def _start_team_monitoring(self, team_id: str):
        """Start monitoring for a team"""
        if team_id in self.performance_monitors:
            return  # Already monitoring
        
        # Start performance monitoring
        self.performance_monitors[team_id] = asyncio.create_task(
            self._monitor_team_performance(team_id)
        )
        
        # Start adaptation task if adaptive topology is enabled
        if team_id in self.adaptive_topologies:
            self.adaptation_tasks[team_id] = asyncio.create_task(
                self.adaptive_topologies[team_id].run_continuous_adaptation(
                    self.config.adaptation_check_interval
                )
            )
        
        logger.debug(f"Started monitoring for team {team_id}")
    
    def _stop_team_monitoring(self, team_id: str):
        """Stop monitoring for a team"""
        # Stop performance monitoring
        if team_id in self.performance_monitors:
            self.performance_monitors[team_id].cancel()
            del self.performance_monitors[team_id]
        
        # Stop adaptation task
        if team_id in self.adaptation_tasks:
            self.adaptation_tasks[team_id].cancel()
            del self.adaptation_tasks[team_id]
        
        logger.debug(f"Stopped monitoring for team {team_id}")
    
    async def _monitor_team_performance(self, team_id: str):
        """Monitor team performance continuously"""
        while self.is_running and team_id in self.managed_teams:
            try:
                # Get current performance metrics
                metrics = self.get_performance_metrics(team_id)
                
                if metrics and self.config.log_topology_changes:
                    logger.debug(f"Team {team_id} performance: "
                               f"efficiency={metrics.efficiency:.3f}, "
                               f"comm_cost={metrics.communication_cost:.3f}, "
                               f"load_balance={metrics.load_balance:.3f}")
                
                await asyncio.sleep(self.config.performance_monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring team {team_id}: {e}")
                await asyncio.sleep(self.config.performance_monitoring_interval)
    
    def _on_topology_change(self, team_id: str, transition: TopologyTransition):
        """Handle topology change event"""
        if self.config.log_topology_changes:
            logger.info(f"Topology changed for team {team_id}: "
                       f"reason={transition.transition_reason}, "
                       f"improvement={transition.expected_improvement:.3f}")
        
        # Notify callbacks
        for callback in self.topology_change_callbacks:
            try:
                callback(team_id, transition)
            except Exception as e:
                logger.error(f"Topology change callback failed: {e}")
    
    def add_topology_change_callback(
        self, 
        callback: Callable[[str, TopologyTransition], None]
    ):
        """Add callback for topology changes"""
        self.topology_change_callbacks.append(callback)
    
    def get_managed_teams(self) -> List[str]:
        """Get list of managed team IDs"""
        return list(self.managed_teams.keys())
    
    def get_topology_history(self, team_id: str) -> List[TopologyTransition]:
        """Get topology history for a team"""
        if team_id not in self.adaptive_topologies:
            return []
        
        return self.adaptive_topologies[team_id].get_topology_history()
