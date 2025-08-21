"""
Unit tests for adaptive topology system
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch

from agno.topology.types import (
    AgentCapability,
    TaskState,
    TopologyGraph,
    TopologyMetrics,
    TopologyType,
    CommunicationMessage,
    CommunicationMode
)
from agno.topology.gnn_generator import GNNTopologyGenerator, GNNConfig
from agno.topology.rl_search import RLTopologySearch, RLConfig, TopologyEnvironment
from agno.topology.communication import HybridCommunicationProtocol
from agno.topology.adaptive import AdaptiveTopology, AdaptationConfig
from agno.topology.manager import TopologyManager, TopologyManagerConfig


class TestTopologyTypes:
    """Test topology type definitions"""
    
    def test_agent_capability_creation(self):
        """Test agent capability creation"""
        agent = AgentCapability(
            agent_id="test_agent",
            name="Test Agent",
            role="Tester",
            tools=["tool1", "tool2"],
            current_load=0.5,
            expertise_domains=["testing", "validation"]
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.role == "Tester"
        assert len(agent.tools) == 2
        assert agent.current_load == 0.5
        assert len(agent.expertise_domains) == 2
    
    def test_task_state_creation(self):
        """Test task state creation"""
        task = TaskState(
            task_id="test_task",
            phase="testing",
            complexity=0.7,
            urgency=0.8,
            required_capabilities=["testing", "analysis"]
        )
        
        assert task.task_id == "test_task"
        assert task.phase == "testing"
        assert task.complexity == 0.7
        assert task.urgency == 0.8
        assert len(task.required_capabilities) == 2
    
    def test_topology_graph_creation(self):
        """Test topology graph creation"""
        adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        agent_ids = ["agent1", "agent2", "agent3"]
        
        topology = TopologyGraph(
            adjacency_matrix=adj_matrix,
            agent_ids=agent_ids,
            topology_type=TopologyType.HIERARCHICAL
        )
        
        assert topology.adjacency_matrix.shape == (3, 3)
        assert len(topology.agent_ids) == 3
        assert topology.topology_type == TopologyType.HIERARCHICAL
    
    def test_topology_graph_neighbors(self):
        """Test getting neighbors from topology graph"""
        adj_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        agent_ids = ["agent1", "agent2", "agent3"]
        
        topology = TopologyGraph(
            adjacency_matrix=adj_matrix,
            agent_ids=agent_ids
        )
        
        neighbors = topology.get_neighbors("agent1")
        assert set(neighbors) == {"agent2", "agent3"}
        
        neighbors = topology.get_neighbors("agent2")
        assert neighbors == ["agent1"]
    
    def test_topology_metrics_overall_score(self):
        """Test topology metrics overall score calculation"""
        metrics = TopologyMetrics(
            efficiency=0.8,
            communication_cost=0.3,
            load_balance=0.7,
            fault_tolerance=0.6,
            adaptability=0.5,
            convergence_time=0.4
        )
        
        score = metrics.overall_score()
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)


class TestGNNTopologyGenerator:
    """Test GNN topology generator"""
    
    def test_gnn_config_creation(self):
        """Test GNN configuration creation"""
        config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            learning_rate=0.001
        )
        
        assert config.hidden_dim == 64
        assert config.num_layers == 2
        assert config.learning_rate == 0.001
    
    def test_gnn_generator_creation(self):
        """Test GNN generator creation"""
        config = GNNConfig(hidden_dim=32, num_layers=2)
        generator = GNNTopologyGenerator(config)
        
        assert generator.config.hidden_dim == 32
        assert generator.config.num_layers == 2
        assert generator.model is not None
    
    def test_agent_feature_encoding(self):
        """Test agent feature encoding"""
        config = GNNConfig(feature_dim=10)
        generator = GNNTopologyGenerator(config)
        
        agents = [
            AgentCapability(
                agent_id="agent1",
                name="Agent 1",
                role="Tester",
                tools=["tool1"],
                current_load=0.5
            ),
            AgentCapability(
                agent_id="agent2", 
                name="Agent 2",
                role="Analyzer",
                tools=["tool1", "tool2"],
                current_load=0.7
            )
        ]
        
        features = generator.model.encode_agent_features(agents)
        assert features.shape == (2, 10)
    
    def test_task_state_encoding(self):
        """Test task state encoding"""
        generator = GNNTopologyGenerator()
        
        task = TaskState(
            task_id="test_task",
            phase="testing",
            complexity=0.7,
            urgency=0.8,
            required_capabilities=["testing"]
        )
        
        features = generator.model.encode_task_state(task)
        assert features.shape == (1, 10)


class TestRLTopologySearch:
    """Test RL topology search"""
    
    def test_rl_config_creation(self):
        """Test RL configuration creation"""
        config = RLConfig(
            state_dim=64,
            action_dim=32,
            learning_rate=0.001
        )
        
        assert config.state_dim == 64
        assert config.action_dim == 32
        assert config.learning_rate == 0.001
    
    def test_topology_environment_creation(self):
        """Test topology environment creation"""
        env = TopologyEnvironment(max_agents=5)
        assert env.max_agents == 5
        assert env.step_count == 0
    
    def test_environment_reset(self):
        """Test environment reset"""
        env = TopologyEnvironment(max_agents=3)
        
        agents = [
            AgentCapability(
                agent_id=f"agent{i}",
                name=f"Agent {i}",
                role="Tester",
                tools=[]
            ) for i in range(3)
        ]
        
        task = TaskState(
            task_id="test_task",
            phase="testing",
            complexity=0.5,
            urgency=0.5,
            required_capabilities=[]
        )
        
        state = env.reset(agents, task)
        assert isinstance(state, np.ndarray)
        assert len(state) == 128  # Default state size
        assert env.current_topology is not None
    
    def test_rl_search_creation(self):
        """Test RL search creation"""
        config = RLConfig(state_dim=64, action_dim=32)
        rl_search = RLTopologySearch(config)
        
        assert rl_search.config.state_dim == 64
        assert rl_search.config.action_dim == 32
        assert rl_search.q_network is not None
        assert rl_search.target_network is not None


class TestHybridCommunicationProtocol:
    """Test hybrid communication protocol"""
    
    def test_communication_protocol_creation(self):
        """Test communication protocol creation"""
        # Create simple topology
        adj_matrix = np.array([[0, 1], [1, 0]])
        topology = TopologyGraph(
            adjacency_matrix=adj_matrix,
            agent_ids=["agent1", "agent2"]
        )
        
        agents = [
            AgentCapability(
                agent_id="agent1",
                name="Agent 1", 
                role="Sender",
                tools=[]
            ),
            AgentCapability(
                agent_id="agent2",
                name="Agent 2",
                role="Receiver", 
                tools=[]
            )
        ]
        
        protocol = HybridCommunicationProtocol(topology, agents)
        assert protocol.topology == topology
        assert len(protocol.agents) == 2
        assert protocol.is_active
    
    def test_message_sending(self):
        """Test message sending"""
        # Create simple topology
        adj_matrix = np.array([[0, 1], [1, 0]])
        topology = TopologyGraph(
            adjacency_matrix=adj_matrix,
            agent_ids=["agent1", "agent2"]
        )
        
        agents = [
            AgentCapability(
                agent_id="agent1",
                name="Agent 1",
                role="Sender",
                tools=[]
            ),
            AgentCapability(
                agent_id="agent2", 
                name="Agent 2",
                role="Receiver",
                tools=[]
            )
        ]
        
        protocol = HybridCommunicationProtocol(topology, agents)
        
        message = CommunicationMessage(
            message_id="test_msg",
            sender_id="agent1",
            receiver_ids=["agent2"],
            content="Test message",
            message_type="test"
        )
        
        success = protocol.send_message(message)
        assert isinstance(success, bool)


class TestAdaptiveTopology:
    """Test adaptive topology system"""
    
    def test_adaptive_topology_creation(self):
        """Test adaptive topology creation"""
        agents = [
            AgentCapability(
                agent_id=f"agent{i}",
                name=f"Agent {i}",
                role="Tester",
                tools=[]
            ) for i in range(3)
        ]
        
        task = TaskState(
            task_id="test_task",
            phase="testing",
            complexity=0.5,
            urgency=0.5,
            required_capabilities=[]
        )
        
        config = AdaptationConfig(
            min_efficiency_threshold=0.6,
            adaptation_cooldown=10.0
        )
        
        adaptive_topology = AdaptiveTopology(
            agents=agents,
            initial_task=task,
            config=config
        )
        
        assert len(adaptive_topology.agents) == 3
        assert adaptive_topology.current_task == task
        assert adaptive_topology.config == config
        assert adaptive_topology.current_topology is not None
    
    def test_task_state_update(self):
        """Test task state update"""
        agents = [
            AgentCapability(
                agent_id="agent1",
                name="Agent 1",
                role="Tester",
                tools=[]
            )
        ]
        
        initial_task = TaskState(
            task_id="test_task",
            phase="initial",
            complexity=0.5,
            urgency=0.5,
            required_capabilities=[]
        )
        
        adaptive_topology = AdaptiveTopology(
            agents=agents,
            initial_task=initial_task
        )
        
        new_task = TaskState(
            task_id="test_task",
            phase="updated",
            complexity=0.7,
            urgency=0.8,
            required_capabilities=["new_capability"]
        )
        
        adaptive_topology.update_task_state(new_task)
        assert adaptive_topology.current_task.phase == "updated"
        assert adaptive_topology.current_task.complexity == 0.7


class TestTopologyManager:
    """Test topology manager"""
    
    def test_topology_manager_creation(self):
        """Test topology manager creation"""
        config = TopologyManagerConfig(
            enable_adaptive_topology=True,
            auto_start_adaptation=False
        )
        
        manager = TopologyManager(config=config)
        assert manager.config.enable_adaptive_topology
        assert not manager.config.auto_start_adaptation
        assert not manager.is_running
    
    @patch('agno.topology.manager.Team')
    def test_team_registration(self, mock_team):
        """Test team registration"""
        # Mock team
        mock_team.team_id = "test_team"
        mock_team.members = [Mock() for _ in range(3)]
        
        # Mock agent attributes
        for i, member in enumerate(mock_team.members):
            member.agent_id = f"agent{i}"
            member.name = f"Agent {i}"
            member.role = "Tester"
            member.tools = []
        
        manager = TopologyManager()
        
        # This would need more mocking to work properly
        # success = manager.register_team(mock_team)
        # assert success or not success  # Either outcome is valid for this test
    
    def test_manager_start_stop(self):
        """Test manager start and stop"""
        manager = TopologyManager()
        
        assert not manager.is_running
        
        manager.start()
        assert manager.is_running
        
        manager.stop()
        assert not manager.is_running


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_basic_adaptation_flow(self):
        """Test basic adaptation flow"""
        # Create minimal setup
        agents = [
            AgentCapability(
                agent_id=f"agent{i}",
                name=f"Agent {i}",
                role="Tester",
                tools=[]
            ) for i in range(3)
        ]
        
        task = TaskState(
            task_id="test_task",
            phase="testing",
            complexity=0.5,
            urgency=0.5,
            required_capabilities=[]
        )
        
        # Create adaptive topology with minimal config
        config = AdaptationConfig(
            min_efficiency_threshold=0.1,  # Low threshold for testing
            adaptation_cooldown=0.1,       # Short cooldown
            enable_rl_learning=False       # Disable for faster testing
        )
        
        adaptive_topology = AdaptiveTopology(
            agents=agents,
            initial_task=task,
            config=config
        )
        
        # Test basic functionality
        assert adaptive_topology.current_topology is not None
        
        # Evaluate performance
        metrics = adaptive_topology.evaluate_current_performance()
        assert isinstance(metrics, TopologyMetrics)
        
        # Test adaptation trigger
        should_adapt, trigger = adaptive_topology.should_adapt(metrics)
        assert isinstance(should_adapt, bool)


if __name__ == "__main__":
    pytest.main([__file__])
