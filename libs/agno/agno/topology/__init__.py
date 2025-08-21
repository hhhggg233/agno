"""
Adaptive and Hybrid Topology Management for Multi-Agent Systems

This module implements adaptive topology structures for multi-agent collaboration,
including:
- Graph Neural Network-based topology generation
- Reinforcement Learning topology search
- Hybrid communication protocols
- Dynamic topology reconfiguration

Usage:
    from agno.topology import TopologyManager, ProductionTopologyManager
    from agno.topology.config import ProductionTopologyConfig
"""

# Import core types (no heavy dependencies)
from agno.topology.types import (
    TopologyType,
    CommunicationMode,
    TopologyMetrics,
    AgentCapability,
    TaskState,
    TopologyGraph,
    CommunicationMessage,
    TopologyTransition,
)

# Import configuration
from agno.topology.config import ProductionTopologyConfig, ConfigManager

# Import main managers
from agno.topology.manager import TopologyManager, TopologyManagerConfig
from agno.topology.production_manager import ProductionTopologyManager

# Import adaptive components (these require PyTorch)
try:
    from agno.topology.adaptive import AdaptiveTopology, AdaptationConfig
    from agno.topology.gnn_generator import GNNTopologyGenerator, GNNConfig
    from agno.topology.rl_search import RLTopologySearch, RLConfig
    from agno.topology.communication import HybridCommunicationProtocol
    from agno.topology.evaluator import TopologyEvaluator

    _PYTORCH_AVAILABLE = True
except ImportError as e:
    _PYTORCH_AVAILABLE = False
    _PYTORCH_ERROR = str(e)

    # Provide stub classes for graceful degradation
    class _PyTorchRequired:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PyTorch is required for this functionality. "
                f"Install with: pip install torch\n"
                f"Original error: {_PYTORCH_ERROR}"
            )

    AdaptiveTopology = _PyTorchRequired
    AdaptationConfig = _PyTorchRequired
    GNNTopologyGenerator = _PyTorchRequired
    GNNConfig = _PyTorchRequired
    RLTopologySearch = _PyTorchRequired
    RLConfig = _PyTorchRequired
    HybridCommunicationProtocol = _PyTorchRequired
    TopologyEvaluator = _PyTorchRequired


def check_pytorch_availability() -> bool:
    """Check if PyTorch is available for advanced features"""
    return _PYTORCH_AVAILABLE


def get_pytorch_error() -> str:
    """Get PyTorch import error message"""
    return _PYTORCH_ERROR if not _PYTORCH_AVAILABLE else ""


__all__ = [
    # Core types
    "TopologyType",
    "CommunicationMode",
    "TopologyMetrics",
    "AgentCapability",
    "TaskState",
    "TopologyGraph",
    "CommunicationMessage",
    "TopologyTransition",

    # Configuration
    "ProductionTopologyConfig",
    "ConfigManager",

    # Managers
    "TopologyManager",
    "TopologyManagerConfig",
    "ProductionTopologyManager",

    # Advanced components (require PyTorch)
    "AdaptiveTopology",
    "AdaptationConfig",
    "GNNTopologyGenerator",
    "GNNConfig",
    "RLTopologySearch",
    "RLConfig",
    "HybridCommunicationProtocol",
    "TopologyEvaluator",

    # Utilities
    "check_pytorch_availability",
    "get_pytorch_error",
]
