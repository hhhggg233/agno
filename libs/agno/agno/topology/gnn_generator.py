"""
Graph Neural Network-based Topology Generator

Implements the G-Designer approach for generating optimal communication topologies
based on current task state and agent capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from agno.topology.types import (
    AgentCapability, 
    TaskState, 
    TopologyGraph, 
    TopologyType,
    TopologyMetrics
)
from agno.utils.log import logger


@dataclass
class GNNConfig:
    """Configuration for GNN topology generator"""
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    max_agents: int = 50
    feature_dim: int = 64


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for processing agent relationships"""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GAT layer

        Args:
            h: Node features [N, in_features]
            adj: Adjacency matrix [N, N]

        Returns:
            Updated node features [N, out_features]
        """
        Wh = self.W(h)  # [N, out_features]
        N = Wh.size(0)

        # Compute attention coefficients
        Wh1 = Wh.repeat(1, N).view(N * N, -1)  # [N*N, out_features]
        Wh2 = Wh.repeat(N, 1)  # [N*N, out_features]

        e = self.a(torch.cat([Wh1, Wh2], dim=1)).squeeze(1)  # [N*N]
        e = e.view(N, N)  # [N, N]

        # Apply adjacency mask
        e = e.masked_fill(adj == 0, -1e9)

        # Compute attention weights
        attention = F.softmax(e, dim=1)  # [N, N]
        attention = self.dropout_layer(attention)

        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]

        return self.leakyrelu(h_prime)


class TopologyGNN(nn.Module):
    """Graph Neural Network for topology generation"""

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # Agent feature encoder
        self.agent_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Task state encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(10, config.hidden_dim),  # Task features
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(config.hidden_dim, config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Topology decoder
        self.topology_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def encode_agent_features(self, agents: List[AgentCapability]) -> torch.Tensor:
        """Encode agent capabilities into feature vectors"""
        features = []

        for agent in agents:
            # Create feature vector from agent capabilities
            feature_vec = [
                agent.current_load,
                agent.communication_bandwidth,
                agent.processing_power,
                agent.reliability_score,
                len(agent.tools),
                len(agent.expertise_domains),
            ]

            # Pad to feature_dim
            while len(feature_vec) < self.config.feature_dim:
                feature_vec.append(0.0)

            features.append(feature_vec[:self.config.feature_dim])

        return torch.tensor(features, dtype=torch.float32)

    def encode_task_state(self, task: TaskState) -> torch.Tensor:
        """Encode task state into feature vector"""
        task_features = [
            task.complexity,
            task.urgency,
            len(task.required_capabilities),
            len(task.current_bottlenecks),
            task.performance_metrics.get('efficiency', 0.0),
            task.performance_metrics.get('communication_cost', 0.0),
            task.performance_metrics.get('load_balance', 0.0),
            task.performance_metrics.get('fault_tolerance', 0.0),
            task.performance_metrics.get('adaptability', 0.0),
            task.performance_metrics.get('convergence_time', 0.0),
        ]

        return torch.tensor(task_features, dtype=torch.float32).unsqueeze(0)
    
    def forward(
        self,
        agent_features: torch.Tensor,
        task_features: torch.Tensor,
        current_adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate topology adjacency matrix

        Args:
            agent_features: Agent capability features [N, feature_dim]
            task_features: Task state features [1, task_feature_dim]
            current_adj: Current adjacency matrix [N, N]

        Returns:
            New adjacency matrix [N, N]
        """
        N = agent_features.size(0)

        # Encode features
        agent_h = self.agent_encoder(agent_features)  # [N, hidden_dim]
        task_h = self.task_encoder(task_features)  # [1, hidden_dim]

        # Broadcast task features to all agents
        task_h_broadcast = task_h.repeat(N, 1)  # [N, hidden_dim]

        # Combine agent and task features
        combined_h = agent_h + task_h_broadcast  # [N, hidden_dim]

        # Apply graph attention layers
        h = combined_h
        for gat_layer in self.gat_layers:
            h = gat_layer(h, current_adj)

        # Generate pairwise connection probabilities
        adj_probs = torch.zeros(N, N)

        for i in range(N):
            for j in range(i + 1, N):  # Upper triangular
                # Concatenate node features
                pair_features = torch.cat([h[i], h[j]], dim=0)

                # Predict connection probability
                prob = self.topology_decoder(pair_features)
                adj_probs[i, j] = prob
                adj_probs[j, i] = prob  # Symmetric

        return adj_probs


class GNNTopologyGenerator:
    """GNN-based topology generator implementing G-Designer approach"""
    
    def __init__(self, config: Optional[GNNConfig] = None):
        self.config = config or GNNConfig()
        self.model = TopologyGNN(self.config)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.training_history = []
        
    def generate_topology(
        self,
        agents: List[AgentCapability],
        task_state: TaskState,
        current_topology: Optional[TopologyGraph] = None
    ) -> TopologyGraph:
        """
        Generate optimal topology for given agents and task state
        
        Args:
            agents: List of agent capabilities
            task_state: Current task state
            current_topology: Current topology (if any)
            
        Returns:
            Generated topology graph
        """
        self.model.eval()

        with torch.no_grad():
            # Encode inputs
            agent_features = self.model.encode_agent_features(agents)
            task_features = self.model.encode_task_state(task_state)

            # Current adjacency matrix
            if current_topology:
                current_adj = torch.tensor(
                    current_topology.adjacency_matrix,
                    dtype=torch.float32
                )
            else:
                # Start with fully connected
                N = len(agents)
                current_adj = torch.ones(N, N) - torch.eye(N)

            # Generate new topology
            adj_probs = self.model(agent_features, task_features, current_adj)

            # Convert probabilities to binary adjacency matrix
            threshold = 0.5  # Could be adaptive
            new_adj = (adj_probs > threshold).float().numpy()

            # Ensure no self-connections
            np.fill_diagonal(new_adj, 0)

            # Create topology graph
            agent_ids = [agent.agent_id for agent in agents]

            return TopologyGraph(
                adjacency_matrix=new_adj,
                agent_ids=agent_ids,
                edge_weights=adj_probs.numpy(),
                topology_type=self._classify_topology(new_adj)
            )
    
    def _classify_topology(self, adj_matrix: np.ndarray) -> TopologyType:
        """Classify the topology type based on adjacency matrix"""
        N = adj_matrix.shape[0]
        total_edges = np.sum(adj_matrix) / 2  # Undirected graph
        
        # Check for specific patterns
        if total_edges == N - 1:
            # Could be star or tree
            degrees = np.sum(adj_matrix, axis=1)
            if np.max(degrees) == N - 1:
                return TopologyType.CENTRALIZED  # Star topology
            else:
                return TopologyType.HIERARCHICAL  # Tree topology
        elif total_edges == N * (N - 1) / 2:
            return TopologyType.DECENTRALIZED  # Fully connected
        elif total_edges == N:
            return TopologyType.RING  # Ring topology
        else:
            return TopologyType.HYBRID
    
    def train_step(
        self,
        agents: List[AgentCapability],
        task_state: TaskState,
        current_topology: TopologyGraph,
        target_metrics: TopologyMetrics
    ) -> float:
        """
        Single training step for the GNN model

        Args:
            agents: Agent capabilities
            task_state: Task state
            current_topology: Current topology
            target_metrics: Target performance metrics

        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Encode inputs
        agent_features = self.model.encode_agent_features(agents)
        task_features = self.model.encode_task_state(task_state)
        current_adj = torch.tensor(
            current_topology.adjacency_matrix,
            dtype=torch.float32
        )

        # Forward pass
        pred_adj_probs = self.model(agent_features, task_features, current_adj)

        # Create target adjacency matrix based on metrics
        # This is a simplified approach - in practice, you'd need more sophisticated
        # reward shaping based on actual topology performance
        target_score = target_metrics.overall_score()

        # Loss function: encourage topologies that lead to better performance
        # This is a placeholder - real implementation would need more sophisticated loss
        loss = F.mse_loss(
            pred_adj_probs.mean(),
            torch.tensor(target_score, dtype=torch.float32)
        )

        # Backward pass
        loss.backward()
        self.optimizer.step()

        loss_value = loss.item()
        self.training_history.append(loss_value)

        logger.debug(f"GNN training step loss: {loss_value:.4f}")

        return loss_value

    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)

    def load_model(self, path: str):
        """Load a trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.training_history = checkpoint.get('training_history', [])
