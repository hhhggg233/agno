"""
Reinforcement Learning-based Topology Search

Implements RL approach where topology structure is part of the action space,
allowing agents to learn optimal communication structures through trial and error.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass

from agno.topology.types import (
    AgentCapability,
    TaskState,
    TopologyGraph,
    TopologyMetrics,
    TopologyType
)
from agno.utils.log import logger


# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])


@dataclass
class RLConfig:
    """Configuration for RL topology search"""
    state_dim: int = 128
    action_dim: int = 64
    hidden_dim: int = 256
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100
    max_agents: int = 20


class TopologyEnvironment:
    """Environment for topology optimization using RL"""
    
    def __init__(self, max_agents: int = 20):
        self.max_agents = max_agents
        self.current_agents: List[AgentCapability] = []
        self.current_task: Optional[TaskState] = None
        self.current_topology: Optional[TopologyGraph] = None
        self.step_count = 0
        self.max_steps = 100
        
    def reset(
        self, 
        agents: List[AgentCapability], 
        task_state: TaskState
    ) -> np.ndarray:
        """Reset environment with new agents and task"""
        self.current_agents = agents[:self.max_agents]  # Limit agents
        self.current_task = task_state
        self.step_count = 0
        
        # Initialize with random topology
        n_agents = len(self.current_agents)
        adj_matrix = np.random.rand(n_agents, n_agents)
        adj_matrix = (adj_matrix > 0.5).astype(float)
        np.fill_diagonal(adj_matrix, 0)  # No self-connections
        
        self.current_topology = TopologyGraph(
            adjacency_matrix=adj_matrix,
            agent_ids=[agent.agent_id for agent in self.current_agents]
        )
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return next state, reward, done, info"""
        self.step_count += 1
        
        # Decode action to topology changes
        new_topology = self._decode_action(action)
        
        # Calculate reward based on topology performance
        reward = self._calculate_reward(new_topology)
        
        # Update current topology
        self.current_topology = new_topology
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'topology_metrics': self._evaluate_topology(new_topology),
            'step_count': self.step_count
        }
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if not self.current_agents or not self.current_task or not self.current_topology:
            return np.zeros(128)  # Default state size
        
        # Agent features
        agent_features = []
        for agent in self.current_agents:
            features = [
                agent.current_load,
                agent.communication_bandwidth,
                agent.processing_power,
                agent.reliability_score,
                len(agent.tools) / 10.0,  # Normalize
                len(agent.expertise_domains) / 5.0  # Normalize
            ]
            agent_features.extend(features)
        
        # Pad or truncate to fixed size
        while len(agent_features) < 60:  # 10 agents * 6 features
            agent_features.append(0.0)
        agent_features = agent_features[:60]
        
        # Task features
        task_features = [
            self.current_task.complexity,
            self.current_task.urgency,
            len(self.current_task.required_capabilities) / 10.0,
            len(self.current_task.current_bottlenecks) / 5.0,
        ]
        
        # Topology features
        adj_matrix = self.current_topology.adjacency_matrix
        topo_features = [
            np.sum(adj_matrix) / (adj_matrix.shape[0] ** 2),  # Density
            np.std(np.sum(adj_matrix, axis=1)),  # Degree variance
            self._calculate_clustering_coefficient(adj_matrix),
            self._calculate_path_length(adj_matrix)
        ]
        
        # Combine all features
        state = np.array(agent_features + task_features + topo_features, dtype=np.float32)
        
        # Pad to fixed state dimension
        if len(state) < 128:
            state = np.pad(state, (0, 128 - len(state)), 'constant')
        else:
            state = state[:128]
        
        return state
    
    def _decode_action(self, action: np.ndarray) -> TopologyGraph:
        """Decode RL action to topology modification"""
        n_agents = len(self.current_agents)
        
        # Action represents edge modification probabilities
        # Reshape action to adjacency matrix format
        action_matrix = action[:n_agents * n_agents].reshape(n_agents, n_agents)
        
        # Current adjacency matrix
        current_adj = self.current_topology.adjacency_matrix.copy()
        
        # Apply modifications based on action
        for i in range(n_agents):
            for j in range(i + 1, n_agents):  # Upper triangular
                # Action value determines whether to add/remove edge
                if action_matrix[i, j] > 0.5:
                    current_adj[i, j] = 1.0
                    current_adj[j, i] = 1.0
                else:
                    current_adj[i, j] = 0.0
                    current_adj[j, i] = 0.0
        
        # Ensure no self-connections
        np.fill_diagonal(current_adj, 0)
        
        return TopologyGraph(
            adjacency_matrix=current_adj,
            agent_ids=self.current_topology.agent_ids,
            topology_type=self._classify_topology(current_adj)
        )
    
    def _calculate_reward(self, topology: TopologyGraph) -> float:
        """Calculate reward for the given topology"""
        metrics = self._evaluate_topology(topology)
        
        # Base reward from topology performance
        base_reward = metrics.overall_score()
        
        # Penalty for excessive communication overhead
        density = np.sum(topology.adjacency_matrix) / (topology.adjacency_matrix.shape[0] ** 2)
        communication_penalty = -0.1 * density if density > 0.7 else 0.0
        
        # Bonus for balanced load distribution
        degrees = np.sum(topology.adjacency_matrix, axis=1)
        degree_variance = np.var(degrees)
        balance_bonus = 0.1 / (1.0 + degree_variance)
        
        total_reward = base_reward + communication_penalty + balance_bonus
        
        return np.clip(total_reward, -1.0, 1.0)
    
    def _evaluate_topology(self, topology: TopologyGraph) -> TopologyMetrics:
        """Evaluate topology performance (simplified)"""
        adj_matrix = topology.adjacency_matrix
        n_agents = adj_matrix.shape[0]
        
        # Calculate basic metrics
        density = np.sum(adj_matrix) / (n_agents * (n_agents - 1))
        degrees = np.sum(adj_matrix, axis=1)
        
        # Efficiency: based on connectivity and balance
        efficiency = density * (1.0 - np.var(degrees) / np.mean(degrees + 1e-6))
        
        # Communication cost: proportional to number of edges
        communication_cost = density
        
        # Load balance: inverse of degree variance
        load_balance = 1.0 / (1.0 + np.var(degrees))
        
        # Fault tolerance: based on minimum cut (simplified)
        fault_tolerance = min(1.0, np.min(degrees + 1e-6) / n_agents)
        
        # Adaptability: based on clustering coefficient
        adaptability = self._calculate_clustering_coefficient(adj_matrix)
        
        # Convergence time: based on path length
        convergence_time = self._calculate_path_length(adj_matrix) / n_agents
        
        return TopologyMetrics(
            efficiency=efficiency,
            communication_cost=communication_cost,
            load_balance=load_balance,
            fault_tolerance=fault_tolerance,
            adaptability=adaptability,
            convergence_time=convergence_time
        )
    
    def _calculate_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """Calculate average clustering coefficient"""
        n = adj_matrix.shape[0]
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adj_matrix[i] == 1)[0]
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[k]] == 1:
                        triangles += 1
            
            # Clustering coefficient for node i
            possible_triangles = len(neighbors) * (len(neighbors) - 1) / 2
            clustering_coeffs.append(triangles / possible_triangles if possible_triangles > 0 else 0.0)
        
        return np.mean(clustering_coeffs)
    
    def _calculate_path_length(self, adj_matrix: np.ndarray) -> float:
        """Calculate average shortest path length"""
        n = adj_matrix.shape[0]
        
        # Floyd-Warshall algorithm for all-pairs shortest paths
        dist = adj_matrix.copy()
        dist[dist == 0] = np.inf
        np.fill_diagonal(dist, 0)
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
        
        # Calculate average path length (excluding infinite distances)
        finite_distances = dist[dist != np.inf]
        return np.mean(finite_distances) if len(finite_distances) > 0 else float('inf')
    
    def _classify_topology(self, adj_matrix: np.ndarray) -> TopologyType:
        """Classify topology type"""
        n = adj_matrix.shape[0]
        total_edges = np.sum(adj_matrix) / 2
        
        if total_edges == n - 1:
            degrees = np.sum(adj_matrix, axis=1)
            if np.max(degrees) == n - 1:
                return TopologyType.CENTRALIZED
            else:
                return TopologyType.HIERARCHICAL
        elif total_edges == n * (n - 1) / 2:
            return TopologyType.DECENTRALIZED
        elif total_edges == n:
            return TopologyType.RING
        else:
            return TopologyType.HYBRID


class DQNNetwork(nn.Module):
    """Deep Q-Network for topology optimization"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class RLTopologySearch:
    """Reinforcement Learning-based topology search agent"""

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.env = TopologyEnvironment(self.config.max_agents)

        # Neural networks
        self.q_network = DQNNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        )
        self.target_network = DQNNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        )

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and replay buffer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # Training state
        self.epsilon = self.config.epsilon
        self.step_count = 0
        self.episode_count = 0
        self.training_history = []

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.rand(self.config.action_dim)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)

                # Convert Q-values to action probabilities
                action_probs = torch.sigmoid(q_values).squeeze(0).numpy()
                return action_probs

    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Convert to tensors
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.FloatTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])

        # Current Q-values
        current_q_values = self.q_network(states)

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            target_q_values = rewards + (self.config.gamma * next_q_values.max(1)[0] * ~dones)

        # Compute loss (simplified - in practice, you'd need more sophisticated loss)
        loss = F.mse_loss(current_q_values.mean(1), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.step_count % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        self.step_count += 1
        loss_value = loss.item()
        self.training_history.append(loss_value)

        return loss_value

    def search_optimal_topology(
        self,
        agents: List[AgentCapability],
        task_state: TaskState,
        num_episodes: int = 100
    ) -> Tuple[TopologyGraph, List[float]]:
        """
        Search for optimal topology using RL

        Args:
            agents: List of agent capabilities
            task_state: Current task state
            num_episodes: Number of training episodes

        Returns:
            Best topology found and episode rewards
        """
        episode_rewards = []
        best_topology = None
        best_reward = float('-inf')

        for episode in range(num_episodes):
            state = self.env.reset(agents, task_state)
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action = self.select_action(state, training=True)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)

                # Store experience
                experience = Experience(state, action, reward, next_state, done)
                self.replay_buffer.push(experience)

                # Train
                loss = self.train_step()

                # Update state and reward
                state = next_state
                episode_reward += reward

                # Track best topology
                if reward > best_reward:
                    best_reward = reward
                    best_topology = self.env.current_topology

            episode_rewards.append(episode_reward)
            self.episode_count += 1

            if episode % 10 == 0:
                logger.info(f"RL Episode {episode}, Reward: {episode_reward:.3f}, "
                          f"Epsilon: {self.epsilon:.3f}")

        return best_topology or self.env.current_topology, episode_rewards

    def evaluate_topology(
        self,
        topology: TopologyGraph,
        agents: List[AgentCapability],
        task_state: TaskState
    ) -> TopologyMetrics:
        """Evaluate a topology without training"""
        # Set up environment
        self.env.reset(agents, task_state)
        self.env.current_topology = topology

        # Get metrics
        return self.env._evaluate_topology(topology)

    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'training_history': self.training_history
        }, path)

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.training_history = checkpoint.get('training_history', [])
