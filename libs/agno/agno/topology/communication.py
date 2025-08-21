"""
Hybrid Communication Protocol

Implements DAMCS-inspired structured communication combined with free communication,
supporting multiple communication modes: broadcast, point-to-point, hierarchical reporting, etc.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any, Union
import time
from enum import Enum
import threading
from queue import Queue, PriorityQueue

from agno.topology.types import (
    CommunicationMessage,
    CommunicationMode,
    TopologyGraph,
    AgentCapability
)
from agno.utils.log import logger


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class CommunicationStats:
    """Statistics for communication performance"""
    total_messages: int = 0
    messages_by_mode: Dict[CommunicationMode, int] = field(default_factory=dict)
    average_latency: float = 0.0
    bandwidth_usage: float = 0.0
    failed_deliveries: int = 0
    congestion_events: int = 0
    
    def update_message_count(self, mode: CommunicationMode):
        """Update message count for a specific mode"""
        self.total_messages += 1
        self.messages_by_mode[mode] = self.messages_by_mode.get(mode, 0) + 1


@dataclass
class CommunicationRule:
    """Rule for determining communication mode based on message characteristics"""
    condition: Callable[[CommunicationMessage], bool]
    preferred_mode: CommunicationMode
    priority_boost: float = 0.0
    description: str = ""


class MessageRouter:
    """Routes messages based on topology and communication rules"""
    
    def __init__(self, topology: TopologyGraph):
        self.topology = topology
        self.routing_table: Dict[str, Dict[str, List[str]]] = {}
        self._build_routing_table()
    
    def _build_routing_table(self):
        """Build routing table for efficient message delivery"""
        for agent_id in self.topology.agent_ids:
            self.routing_table[agent_id] = {}
            
            # Calculate shortest paths to all other agents
            for target_id in self.topology.agent_ids:
                if agent_id != target_id:
                    path = self.topology.get_communication_path(agent_id, target_id)
                    self.routing_table[agent_id][target_id] = path or []
    
    def get_route(self, sender_id: str, receiver_id: str) -> List[str]:
        """Get communication route between two agents"""
        return self.routing_table.get(sender_id, {}).get(receiver_id, [])
    
    def get_broadcast_targets(self, sender_id: str) -> List[str]:
        """Get all agents that can receive broadcast from sender"""
        return [aid for aid in self.topology.agent_ids if aid != sender_id]
    
    def get_multicast_targets(
        self, 
        sender_id: str, 
        target_capabilities: List[str]
    ) -> List[str]:
        """Get agents with specific capabilities for multicast"""
        # This would need access to agent capabilities
        # For now, return all neighbors
        return self.topology.get_neighbors(sender_id)
    
    def update_topology(self, new_topology: TopologyGraph):
        """Update routing table when topology changes"""
        self.topology = new_topology
        self._build_routing_table()


class CommunicationChannel:
    """Represents a communication channel between agents"""
    
    def __init__(
        self, 
        channel_id: str, 
        bandwidth: float = 1.0,
        latency: float = 0.1
    ):
        self.channel_id = channel_id
        self.bandwidth = bandwidth  # Messages per second
        self.latency = latency  # Seconds
        self.message_queue: PriorityQueue = PriorityQueue()
        self.is_active = True
        self.congestion_threshold = 0.8
        
    def send_message(self, message: CommunicationMessage) -> bool:
        """Send message through this channel"""
        if not self.is_active:
            return False
        
        # Check congestion
        current_load = self.message_queue.qsize() / (self.bandwidth * 10)  # Rough estimate
        if current_load > self.congestion_threshold:
            logger.warning(f"Channel {self.channel_id} congested")
            return False
        
        # Add to queue with priority
        priority = MessagePriority.NORMAL.value
        if message.priority > 0.8:
            priority = MessagePriority.CRITICAL.value
        elif message.priority > 0.6:
            priority = MessagePriority.HIGH.value
        elif message.priority < 0.3:
            priority = MessagePriority.LOW.value
        
        self.message_queue.put((priority, time.time(), message))
        return True
    
    def receive_message(self) -> Optional[CommunicationMessage]:
        """Receive message from channel"""
        if self.message_queue.empty():
            return None
        
        try:
            _, timestamp, message = self.message_queue.get_nowait()
            # Simulate latency
            if time.time() - timestamp < self.latency:
                # Put back if not enough time has passed
                self.message_queue.put((MessagePriority.NORMAL.value, timestamp, message))
                return None
            return message
        except:
            return None


class HybridCommunicationProtocol:
    """
    Hybrid communication protocol supporting multiple modes:
    - Broadcast: One-to-all communication
    - Point-to-point: Direct agent-to-agent
    - Hierarchical report: Bottom-up reporting
    - Multicast: One-to-many selective
    - Gossip: Peer-to-peer spreading
    """
    
    def __init__(
        self,
        topology: TopologyGraph,
        agents: List[AgentCapability],
        default_bandwidth: float = 10.0
    ):
        self.topology = topology
        self.agents = {agent.agent_id: agent for agent in agents}
        self.router = MessageRouter(topology)
        
        # Communication channels
        self.channels: Dict[str, CommunicationChannel] = {}
        self._initialize_channels(default_bandwidth)
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Communication rules
        self.communication_rules: List[CommunicationRule] = []
        self._initialize_default_rules()
        
        # Statistics
        self.stats = CommunicationStats()
        
        # Message queues for each agent
        self.agent_inboxes: Dict[str, Queue] = {
            agent_id: Queue() for agent_id in self.agents.keys()
        }
        
        # Active flag
        self.is_active = True
        
    def _initialize_channels(self, default_bandwidth: float):
        """Initialize communication channels based on topology"""
        for i, agent_id in enumerate(self.topology.agent_ids):
            for j, other_id in enumerate(self.topology.agent_ids):
                if i != j and self.topology.adjacency_matrix[i, j] > 0:
                    channel_id = f"{agent_id}->{other_id}"
                    
                    # Bandwidth based on edge weight and agent capabilities
                    bandwidth = default_bandwidth
                    if agent_id in self.agents and other_id in self.agents:
                        sender_bw = self.agents[agent_id].communication_bandwidth
                        receiver_bw = self.agents[other_id].communication_bandwidth
                        bandwidth *= min(sender_bw, receiver_bw)
                    
                    self.channels[channel_id] = CommunicationChannel(
                        channel_id, bandwidth
                    )
    
    def _initialize_default_rules(self):
        """Initialize default communication rules"""
        # Critical messages use point-to-point
        self.communication_rules.append(CommunicationRule(
            condition=lambda msg: msg.priority > 0.8,
            preferred_mode=CommunicationMode.POINT_TO_POINT,
            description="Critical messages use direct communication"
        ))
        
        # Broadcast for general announcements
        self.communication_rules.append(CommunicationRule(
            condition=lambda msg: msg.message_type == "announcement",
            preferred_mode=CommunicationMode.BROADCAST,
            description="Announcements use broadcast"
        ))
        
        # Hierarchical for status reports
        self.communication_rules.append(CommunicationRule(
            condition=lambda msg: msg.message_type == "status_report",
            preferred_mode=CommunicationMode.HIERARCHICAL_REPORT,
            description="Status reports use hierarchical communication"
        ))
    
    def add_communication_rule(self, rule: CommunicationRule):
        """Add a custom communication rule"""
        self.communication_rules.append(rule)
    
    def register_message_handler(
        self, 
        agent_id: str, 
        handler: Callable[[CommunicationMessage], Any]
    ):
        """Register message handler for an agent"""
        self.message_handlers[agent_id] = handler
    
    def send_message(self, message: CommunicationMessage) -> bool:
        """Send message using appropriate communication mode"""
        if not self.is_active:
            return False
        
        # Determine communication mode
        mode = self._determine_communication_mode(message)
        message.mode = mode
        
        # Route message based on mode
        success = False
        if mode == CommunicationMode.BROADCAST:
            success = self._send_broadcast(message)
        elif mode == CommunicationMode.POINT_TO_POINT:
            success = self._send_point_to_point(message)
        elif mode == CommunicationMode.HIERARCHICAL_REPORT:
            success = self._send_hierarchical_report(message)
        elif mode == CommunicationMode.MULTICAST:
            success = self._send_multicast(message)
        elif mode == CommunicationMode.GOSSIP:
            success = self._send_gossip(message)
        
        # Update statistics
        if success:
            self.stats.update_message_count(mode)
        else:
            self.stats.failed_deliveries += 1
        
        return success
    
    def _determine_communication_mode(self, message: CommunicationMessage) -> CommunicationMode:
        """Determine optimal communication mode for message"""
        # Apply communication rules
        for rule in self.communication_rules:
            if rule.condition(message):
                return rule.preferred_mode
        
        # Default mode based on number of receivers
        if len(message.receiver_ids) == 1:
            return CommunicationMode.POINT_TO_POINT
        elif len(message.receiver_ids) > len(self.agents) * 0.7:
            return CommunicationMode.BROADCAST
        else:
            return CommunicationMode.MULTICAST
    
    def _send_broadcast(self, message: CommunicationMessage) -> bool:
        """Send message to all agents"""
        targets = self.router.get_broadcast_targets(message.sender_id)
        success_count = 0
        
        for target_id in targets:
            if self._deliver_message(message, target_id):
                success_count += 1
        
        return success_count > 0
    
    def _send_point_to_point(self, message: CommunicationMessage) -> bool:
        """Send message directly to specific agents"""
        success_count = 0
        
        for receiver_id in message.receiver_ids:
            if self._deliver_message(message, receiver_id):
                success_count += 1
        
        return success_count == len(message.receiver_ids)
    
    def _send_hierarchical_report(self, message: CommunicationMessage) -> bool:
        """Send message up the hierarchy"""
        # Find parent nodes (nodes with higher degree)
        sender_idx = self.topology.agent_ids.index(message.sender_id)
        sender_degree = sum(self.topology.adjacency_matrix[sender_idx])
        
        # Find neighbors with higher degree (potential parents)
        neighbors = self.topology.get_neighbors(message.sender_id)
        parents = []
        
        for neighbor_id in neighbors:
            neighbor_idx = self.topology.agent_ids.index(neighbor_id)
            neighbor_degree = sum(self.topology.adjacency_matrix[neighbor_idx])
            if neighbor_degree > sender_degree:
                parents.append(neighbor_id)
        
        # If no parents found, send to highest degree neighbor
        if not parents and neighbors:
            max_degree = -1
            best_parent = None
            for neighbor_id in neighbors:
                neighbor_idx = self.topology.agent_ids.index(neighbor_id)
                neighbor_degree = sum(self.topology.adjacency_matrix[neighbor_idx])
                if neighbor_degree > max_degree:
                    max_degree = neighbor_degree
                    best_parent = neighbor_id
            if best_parent:
                parents = [best_parent]
        
        # Send to parents
        success_count = 0
        for parent_id in parents:
            if self._deliver_message(message, parent_id):
                success_count += 1
        
        return success_count > 0
    
    def _send_multicast(self, message: CommunicationMessage) -> bool:
        """Send message to selected group of agents"""
        success_count = 0
        
        for receiver_id in message.receiver_ids:
            if self._deliver_message(message, receiver_id):
                success_count += 1
        
        return success_count > 0
    
    def _send_gossip(self, message: CommunicationMessage) -> bool:
        """Send message using gossip protocol"""
        # Send to random subset of neighbors
        neighbors = self.topology.get_neighbors(message.sender_id)
        if not neighbors:
            return False
        
        # Select random subset (typically 2-3 neighbors)
        import random
        gossip_targets = random.sample(neighbors, min(3, len(neighbors)))
        
        success_count = 0
        for target_id in gossip_targets:
            if self._deliver_message(message, target_id):
                success_count += 1
        
        return success_count > 0
    
    def _deliver_message(self, message: CommunicationMessage, receiver_id: str) -> bool:
        """Deliver message to specific agent"""
        if receiver_id not in self.agents:
            return False
        
        # Get communication route
        route = self.router.get_route(message.sender_id, receiver_id)
        if not route:
            return False
        
        # Use direct channel if available
        channel_id = f"{message.sender_id}->{receiver_id}"
        if channel_id in self.channels:
            channel = self.channels[channel_id]
            if channel.send_message(message):
                # Add to receiver's inbox
                self.agent_inboxes[receiver_id].put(message)
                return True
        
        return False
    
    def receive_messages(self, agent_id: str) -> List[CommunicationMessage]:
        """Receive all pending messages for an agent"""
        messages = []
        
        if agent_id in self.agent_inboxes:
            while not self.agent_inboxes[agent_id].empty():
                try:
                    message = self.agent_inboxes[agent_id].get_nowait()
                    messages.append(message)
                except:
                    break
        
        return messages
    
    def update_topology(self, new_topology: TopologyGraph):
        """Update communication protocol when topology changes"""
        self.topology = new_topology
        self.router.update_topology(new_topology)
        
        # Reinitialize channels
        self.channels.clear()
        self._initialize_channels(10.0)  # Default bandwidth
        
        logger.info(f"Communication protocol updated for new topology")
    
    def get_statistics(self) -> CommunicationStats:
        """Get communication statistics"""
        return self.stats
    
    def shutdown(self):
        """Shutdown communication protocol"""
        self.is_active = False
        logger.info("Communication protocol shutdown")
