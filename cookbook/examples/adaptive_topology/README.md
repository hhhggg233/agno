# Adaptive and Hybrid Topology for Multi-Agent Systems

This module implements adaptive topology structures for multi-agent collaboration in the agno framework, based on the research direction of **Adaptive and Hybrid Topologies**.

## Overview

The adaptive topology system automatically optimizes communication structures between agents based on:
- Current task characteristics and phase
- Agent capabilities and load distribution
- Performance metrics and bottlenecks
- Historical adaptation patterns

## Key Features

### 🧠 Graph Neural Network (GNN) Topology Generation
- Implements G-Designer approach for meta-controller design
- Generates optimal communication graphs based on task state and agent capabilities
- Uses Graph Attention Networks (GAT) for processing agent relationships
- Supports real-time topology optimization

### 🎯 Reinforcement Learning Topology Search
- Treats topology structure as part of the action space
- Learns optimal communication structures through trial and error
- Uses Deep Q-Networks (DQN) for topology optimization
- Maximizes task completion rewards and efficiency

### 📡 Hybrid Communication Protocol
- Combines DAMCS structured communication with free communication
- Supports multiple communication modes:
  - **Broadcast**: One-to-all communication
  - **Point-to-point**: Direct agent-to-agent
  - **Hierarchical reporting**: Bottom-up information flow
  - **Multicast**: One-to-many selective
  - **Gossip**: Peer-to-peer information spreading

### 🔄 Dynamic Topology Reconfiguration
- Automatically adapts topology during task execution
- Responds to efficiency degradation, load imbalances, and bottlenecks
- Supports transitions between different topology types:
  - Centralized (star topology)
  - Decentralized (fully connected)
  - Hierarchical (tree structure)
  - Ring topology
  - Hybrid structures

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TopologyManager                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Team Registry  │  │ Performance     │  │ Adaptation  │ │
│  │                 │  │ Monitoring      │  │ Control     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  AdaptiveTopology                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ GNN Topology    │  │ RL Topology     │  │ Hybrid      │ │
│  │ Generator       │  │ Search          │  │ Communication│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Agent Teams & Workflows                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Research    │  │ Analysis    │  │ Synthesis           │ │
│  │ Team        │  │ Team        │  │ Team                │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
import asyncio
from agno.agent import Agent
from agno.team import Team
from agno.topology.manager import TopologyManager
from agno.models.openai import OpenAIChat

# Create agents
agents = [
    Agent(name="Researcher", agent_id="researcher_001", 
          model=OpenAIChat(id="gpt-4o-mini")),
    Agent(name="Analyst", agent_id="analyst_001", 
          model=OpenAIChat(id="gpt-4o-mini")),
    Agent(name="Writer", agent_id="writer_001", 
          model=OpenAIChat(id="gpt-4o-mini"))
]

# Create team
team = Team(
    name="Research Team",
    team_id="research_team_001",
    members=agents,
    mode="coordinate"
)

# Create topology manager
topology_manager = TopologyManager()

# Register team for adaptive topology management
topology_manager.register_team(
    team=team,
    initial_task_description="Research and analysis task",
    task_complexity=0.7,
    task_urgency=0.6
)

# Start adaptive management
topology_manager.start()

# The topology will now automatically adapt based on:
# - Task phase changes
# - Agent load variations
# - Performance metrics
# - Communication patterns

# Update task state to trigger adaptation
topology_manager.update_task_state(
    team_id=team.team_id,
    phase="analysis_phase",
    complexity=0.9,
    urgency=0.8
)

# Stop when done
topology_manager.stop()
```

### Advanced Configuration

```python
from agno.topology.manager import TopologyManagerConfig
from agno.topology.adaptive import AdaptationConfig
from agno.topology.gnn_generator import GNNConfig
from agno.topology.rl_search import RLConfig

# Configure topology manager
manager_config = TopologyManagerConfig(
    enable_adaptive_topology=True,
    auto_start_adaptation=True,
    adaptation_check_interval=15.0,  # Check every 15 seconds
    performance_monitoring_interval=5.0,  # Monitor every 5 seconds
    log_topology_changes=True
)

# Configure adaptation behavior
adaptation_config = AdaptationConfig(
    min_efficiency_threshold=0.65,
    max_communication_cost_threshold=0.8,
    adaptation_cooldown=30.0,  # Wait 30s between adaptations
    prefer_gnn=True,  # Prefer GNN over RL
    enable_rl_learning=True
)

# Configure GNN topology generator
gnn_config = GNNConfig(
    hidden_dim=128,
    num_layers=3,
    learning_rate=0.001,
    dropout=0.1
)

# Configure RL topology search
rl_config = RLConfig(
    state_dim=128,
    action_dim=64,
    learning_rate=0.001,
    epsilon=0.1,  # Exploration rate
    batch_size=32
)

# Create manager with custom configuration
topology_manager = TopologyManager(
    config=manager_config,
    adaptation_config=adaptation_config,
    gnn_config=gnn_config,
    rl_config=rl_config
)
```

## Examples

### 1. Basic Adaptive Team
See `basic_adaptive_team.py` for a complete example showing:
- Team creation with diverse agent capabilities
- Automatic topology adaptation during different task phases
- Performance monitoring and metrics
- Communication protocol demonstration

### 2. Workflow Integration
See `workflow_integration.py` for advanced usage with:
- Multi-phase workflow execution
- Specialized teams for different phases
- Dynamic team reconfiguration
- Parallel processing with adaptive topologies

## Performance Metrics

The system tracks several key metrics:

- **Efficiency**: Task completion effectiveness
- **Communication Cost**: Overhead from agent communication
- **Load Balance**: Distribution of work across agents
- **Fault Tolerance**: Resilience to agent failures
- **Adaptability**: Speed of topology reconfiguration
- **Convergence Time**: Time to reach consensus

## Topology Types

The system can automatically transition between:

1. **Centralized (Star)**: One central coordinator, efficient for simple tasks
2. **Decentralized (Mesh)**: Fully connected, robust but high communication cost
3. **Hierarchical (Tree)**: Multi-level structure, good for complex tasks
4. **Ring**: Sequential processing, good for pipeline tasks
5. **Hybrid**: Mixed structures optimized for specific requirements

## Communication Modes

Agents can communicate using different modes based on message characteristics:

- **Broadcast**: General announcements and status updates
- **Point-to-point**: Critical messages and direct coordination
- **Hierarchical**: Status reports and escalations
- **Multicast**: Targeted group communications
- **Gossip**: Information spreading and consensus building

## Integration with Existing agno Features

The adaptive topology system seamlessly integrates with:
- **Teams**: Automatic topology management for existing teams
- **Workflows**: Dynamic reconfiguration during workflow execution
- **Agents**: Capability-based topology optimization
- **Models**: Support for different LLM providers
- **Tools**: Tool-aware topology generation

## Research Background

This implementation is based on research in:
- **DyLAN**: Dynamic LLM-powered agent networks
- **Flow**: Modularized agentic workflow automation
- **G-Designer**: Graph neural network topology generation
- **DAMCS**: Structured multi-agent communication systems

## Future Enhancements

Planned improvements include:
- Integration with more sophisticated GNN architectures
- Advanced RL algorithms (PPO, A3C)
- Real-time performance prediction
- Multi-objective topology optimization
- Integration with external monitoring systems

## Contributing

To contribute to the adaptive topology system:
1. Add new topology generation algorithms
2. Implement additional communication protocols
3. Create performance optimization strategies
4. Add integration with other agno components
5. Improve testing and documentation

## Testing

Run the test suite:
```bash
python -m pytest libs/agno/tests/unit/topology/
```

For integration testing:
```bash
python cookbook/examples/adaptive_topology/basic_adaptive_team.py
python cookbook/examples/adaptive_topology/workflow_integration.py
```
