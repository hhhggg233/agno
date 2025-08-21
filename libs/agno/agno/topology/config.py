"""
Adaptive Topology Configuration Management

Provides configuration classes and utilities for managing adaptive topology
settings in production environments.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from agno.topology.adaptive import AdaptationConfig
from agno.topology.gnn_generator import GNNConfig
from agno.topology.rl_search import RLConfig
from agno.topology.manager import TopologyManagerConfig


@dataclass
class ProductionTopologyConfig:
    """Production configuration for adaptive topology system"""
    
    # Environment settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Model persistence
    model_save_dir: str = "./models/topology"
    checkpoint_interval: int = 100  # Save every N adaptations
    
    # Performance settings
    max_agents_per_team: int = 20
    max_concurrent_teams: int = 10
    adaptation_timeout: float = 300.0  # 5 minutes
    
    # Monitoring
    enable_metrics_collection: bool = True
    metrics_export_interval: float = 60.0  # 1 minute
    performance_history_size: int = 1000
    
    # Safety settings
    max_adaptations_per_hour: int = 10
    emergency_fallback_topology: str = "centralized"
    
    # Resource limits
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: float = 80.0
    
    # Manager configuration
    manager_config: TopologyManagerConfig = None
    adaptation_config: AdaptationConfig = None
    gnn_config: GNNConfig = None
    rl_config: RLConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations if not provided"""
        if self.manager_config is None:
            self.manager_config = TopologyManagerConfig(
                enable_adaptive_topology=True,
                auto_start_adaptation=True,
                adaptation_check_interval=30.0,
                performance_monitoring_interval=15.0,
                log_topology_changes=True,
                min_agents_for_adaptation=3
            )
        
        if self.adaptation_config is None:
            self.adaptation_config = AdaptationConfig(
                min_efficiency_threshold=0.65,
                max_communication_cost_threshold=0.75,
                min_load_balance_threshold=0.4,
                adaptation_cooldown=60.0,
                performance_window=10,
                prefer_gnn=True,
                enable_rl_learning=True,
                enable_hybrid_communication=True,
                communication_bandwidth=10.0
            )
        
        if self.gnn_config is None:
            self.gnn_config = GNNConfig(
                feature_dim=10,
                hidden_dim=128,
                num_layers=3,
                learning_rate=0.001,
                dropout=0.1,
                batch_size=32,
                max_epochs=100
            )
        
        if self.rl_config is None:
            self.rl_config = RLConfig(
                state_dim=128,
                action_dim=64,
                hidden_dim=256,
                learning_rate=0.001,
                epsilon=0.1,
                epsilon_min=0.01,
                epsilon_decay=0.995,
                gamma=0.95,
                batch_size=32,
                buffer_size=10000,
                target_update_freq=100,
                max_agents=self.max_agents_per_team
            )


class ConfigManager:
    """Configuration manager for adaptive topology system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[ProductionTopologyConfig] = None
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        # Try environment variable first
        if "AGNO_TOPOLOGY_CONFIG" in os.environ:
            return os.environ["AGNO_TOPOLOGY_CONFIG"]
        
        # Try current directory
        current_dir = Path.cwd()
        config_file = current_dir / "topology_config.json"
        if config_file.exists():
            return str(config_file)
        
        # Try user home directory
        home_dir = Path.home()
        config_file = home_dir / ".agno" / "topology_config.json"
        if config_file.exists():
            return str(config_file)
        
        # Default path
        return str(current_dir / "topology_config.json")
    
    def load_config(self) -> ProductionTopologyConfig:
        """Load configuration from file"""
        if self._config is not None:
            return self._config
        
        config_path = Path(self.config_path)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Convert nested dictionaries to dataclass instances
                config_data = self._convert_config_data(config_data)
                self._config = ProductionTopologyConfig(**config_data)
                
                print(f"✅ Loaded configuration from {config_path}")
                return self._config
                
            except Exception as e:
                print(f"⚠️ Failed to load config from {config_path}: {e}")
                print("Using default configuration")
        else:
            print(f"📝 Config file not found at {config_path}")
            print("Using default configuration")
        
        # Use default configuration
        self._config = ProductionTopologyConfig()
        return self._config
    
    def save_config(self, config: Optional[ProductionTopologyConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self._config
        
        if config is None:
            raise ValueError("No configuration to save")
        
        config_path = Path(self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = self._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Saved configuration to {config_path}")
    
    def _convert_config_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert configuration data from JSON to appropriate types"""
        converted = data.copy()
        
        # Convert sub-configurations
        if 'manager_config' in converted and converted['manager_config']:
            converted['manager_config'] = TopologyManagerConfig(**converted['manager_config'])
        
        if 'adaptation_config' in converted and converted['adaptation_config']:
            converted['adaptation_config'] = AdaptationConfig(**converted['adaptation_config'])
        
        if 'gnn_config' in converted and converted['gnn_config']:
            converted['gnn_config'] = GNNConfig(**converted['gnn_config'])
        
        if 'rl_config' in converted and converted['rl_config']:
            converted['rl_config'] = RLConfig(**converted['rl_config'])
        
        return converted
    
    def _config_to_dict(self, config: ProductionTopologyConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization"""
        config_dict = asdict(config)
        
        # Convert dataclass instances to dictionaries
        if config_dict['manager_config']:
            config_dict['manager_config'] = asdict(config_dict['manager_config'])
        
        if config_dict['adaptation_config']:
            config_dict['adaptation_config'] = asdict(config_dict['adaptation_config'])
        
        if config_dict['gnn_config']:
            config_dict['gnn_config'] = asdict(config_dict['gnn_config'])
        
        if config_dict['rl_config']:
            config_dict['rl_config'] = asdict(config_dict['rl_config'])
        
        return config_dict
    
    def create_sample_config(self, output_path: Optional[str] = None):
        """Create a sample configuration file"""
        if output_path is None:
            output_path = "topology_config_sample.json"
        
        sample_config = ProductionTopologyConfig(
            environment="development",
            debug_mode=True,
            log_level="DEBUG"
        )
        
        config_dict = self._config_to_dict(sample_config)
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"📄 Created sample configuration at {output_path}")
        print("Edit this file and rename to 'topology_config.json' to use")


def get_production_config() -> ProductionTopologyConfig:
    """Get production configuration (singleton pattern)"""
    if not hasattr(get_production_config, '_config_manager'):
        get_production_config._config_manager = ConfigManager()
    
    return get_production_config._config_manager.load_config()


def validate_config(config: ProductionTopologyConfig) -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Validate basic settings
    if config.max_agents_per_team < 2:
        errors.append("max_agents_per_team must be at least 2")
    
    if config.max_concurrent_teams < 1:
        errors.append("max_concurrent_teams must be at least 1")
    
    if config.adaptation_timeout <= 0:
        errors.append("adaptation_timeout must be positive")
    
    # Validate thresholds
    if not (0.0 <= config.adaptation_config.min_efficiency_threshold <= 1.0):
        errors.append("min_efficiency_threshold must be between 0 and 1")
    
    if not (0.0 <= config.adaptation_config.max_communication_cost_threshold <= 1.0):
        errors.append("max_communication_cost_threshold must be between 0 and 1")
    
    # Validate model parameters
    if config.gnn_config.hidden_dim < 1:
        errors.append("GNN hidden_dim must be positive")
    
    if config.rl_config.batch_size < 1:
        errors.append("RL batch_size must be positive")
    
    if errors:
        print("❌ Configuration validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True


if __name__ == "__main__":
    # Create sample configuration
    manager = ConfigManager()
    manager.create_sample_config()
    
    # Load and validate default configuration
    config = get_production_config()
    validate_config(config)
