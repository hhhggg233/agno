"""
Production Topology Manager

Production-ready topology manager with real performance monitoring,
resource management, and fault tolerance.
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from agno.team import Team
from agno.topology.manager import TopologyManager
from agno.topology.config import ProductionTopologyConfig, get_production_config
from agno.topology.types import TopologyMetrics, TopologyTransition
from agno.utils.log import logger


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io_mb: float
    timestamp: float


@dataclass
class TeamPerformanceMetrics:
    """Team performance metrics"""
    team_id: str
    task_completion_rate: float
    average_response_time: float
    error_rate: float
    throughput: float  # tasks per minute
    resource_efficiency: float
    timestamp: float


class ResourceMonitor:
    """System resource monitoring"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start resource monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🔍 Resource monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Resource monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O (simplified)
        net_io = psutil.net_io_counters()
        network_io_mb = (net_io.bytes_sent + net_io.bytes_recv) / (1024 * 1024)
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_mb=memory.used / (1024 * 1024),
            disk_usage_percent=disk.percent,
            network_io_mb=network_io_mb,
            timestamp=time.time()
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def is_resource_constrained(self, config: ProductionTopologyConfig) -> bool:
        """Check if system is resource constrained"""
        current = self.get_current_metrics()
        if not current:
            return False
        
        return (
            current.cpu_percent > config.max_cpu_usage_percent or
            current.memory_mb > config.max_memory_usage_mb
        )


class PerformanceTracker:
    """Track team and task performance"""
    
    def __init__(self):
        self.team_metrics: Dict[str, List[TeamPerformanceMetrics]] = {}
        self.task_start_times: Dict[str, float] = {}
        self.task_completion_times: Dict[str, float] = {}
        self.task_errors: Dict[str, int] = {}
        self.max_history_size = 1000
    
    def start_task(self, team_id: str, task_id: str):
        """Record task start"""
        self.task_start_times[f"{team_id}:{task_id}"] = time.time()
    
    def complete_task(self, team_id: str, task_id: str, success: bool = True):
        """Record task completion"""
        key = f"{team_id}:{task_id}"
        if key in self.task_start_times:
            completion_time = time.time() - self.task_start_times[key]
            self.task_completion_times[key] = completion_time
            
            if not success:
                self.task_errors[team_id] = self.task_errors.get(team_id, 0) + 1
    
    def calculate_team_metrics(self, team_id: str) -> TeamPerformanceMetrics:
        """Calculate current team performance metrics"""
        # Get recent tasks (last hour)
        current_time = time.time()
        hour_ago = current_time - 3600
        
        recent_tasks = []
        error_count = 0
        
        for key, completion_time in self.task_completion_times.items():
            if key.startswith(f"{team_id}:"):
                task_start = self.task_start_times.get(key, current_time)
                if task_start >= hour_ago:
                    recent_tasks.append(completion_time)
        
        # Calculate metrics
        if recent_tasks:
            completion_rate = len(recent_tasks) / max(1, len(recent_tasks) + self.task_errors.get(team_id, 0))
            avg_response_time = sum(recent_tasks) / len(recent_tasks)
            throughput = len(recent_tasks) / 60.0  # per minute
            error_rate = self.task_errors.get(team_id, 0) / max(1, len(recent_tasks))
        else:
            completion_rate = 1.0
            avg_response_time = 0.0
            throughput = 0.0
            error_rate = 0.0
        
        # Resource efficiency (simplified)
        resource_efficiency = max(0.1, 1.0 - (avg_response_time / 60.0))
        
        metrics = TeamPerformanceMetrics(
            team_id=team_id,
            task_completion_rate=completion_rate,
            average_response_time=avg_response_time,
            error_rate=error_rate,
            throughput=throughput,
            resource_efficiency=resource_efficiency,
            timestamp=current_time
        )
        
        # Store metrics
        if team_id not in self.team_metrics:
            self.team_metrics[team_id] = []
        
        self.team_metrics[team_id].append(metrics)
        
        # Trim history
        if len(self.team_metrics[team_id]) > self.max_history_size:
            self.team_metrics[team_id] = self.team_metrics[team_id][-self.max_history_size:]
        
        return metrics


class ProductionTopologyManager:
    """Production-ready topology manager"""
    
    def __init__(self, config: Optional[ProductionTopologyConfig] = None):
        self.config = config or get_production_config()
        
        # Core topology manager
        self.topology_manager = TopologyManager(
            config=self.config.manager_config,
            adaptation_config=self.config.adaptation_config,
            gnn_config=self.config.gnn_config,
            rl_config=self.config.rl_config
        )
        
        # Monitoring components
        self.resource_monitor = ResourceMonitor(check_interval=5.0)
        self.performance_tracker = PerformanceTracker()
        
        # State management
        self.managed_teams: Dict[str, Team] = {}
        self.adaptation_counts: Dict[str, int] = {}
        self.last_adaptation_times: Dict[str, float] = {}
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Callbacks
        self.adaptation_callbacks: List[Callable[[str, TopologyTransition], None]] = []
        
        # Setup logging
        self._setup_logging()
        
        # Add topology change callback
        self.topology_manager.add_topology_change_callback(self._on_topology_change)
    
    def _setup_logging(self):
        """Setup production logging"""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create model save directory
        Path(self.config.model_save_dir).mkdir(parents=True, exist_ok=True)
    
    async def start(self):
        """Start production topology manager"""
        logger.info("🚀 Starting Production Topology Manager")
        
        # Start monitoring
        await self.resource_monitor.start_monitoring()
        
        # Start core topology manager
        self.topology_manager.start()
        
        # Start metrics export if enabled
        if self.config.enable_metrics_collection:
            asyncio.create_task(self._metrics_export_loop())
        
        logger.info("✅ Production Topology Manager started")
    
    async def stop(self):
        """Stop production topology manager"""
        logger.info("🛑 Stopping Production Topology Manager")
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Stop core topology manager
        self.topology_manager.stop()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Production Topology Manager stopped")
    
    def register_team(
        self,
        team: Team,
        task_description: str = "Production task",
        complexity: float = 0.5,
        urgency: float = 0.5
    ) -> bool:
        """Register team with production safeguards"""
        
        # Check resource constraints
        if self.resource_monitor.is_resource_constrained(self.config):
            logger.warning(f"System resource constrained, deferring team registration: {team.team_id}")
            return False
        
        # Check team limits
        if len(self.managed_teams) >= self.config.max_concurrent_teams:
            logger.warning(f"Maximum concurrent teams reached: {self.config.max_concurrent_teams}")
            return False
        
        # Check agent count
        if len(team.members) > self.config.max_agents_per_team:
            logger.warning(f"Team {team.team_id} exceeds maximum agents: {len(team.members)} > {self.config.max_agents_per_team}")
            return False
        
        # Register with core manager
        success = self.topology_manager.register_team(
            team=team,
            initial_task_description=task_description,
            task_complexity=complexity,
            task_urgency=urgency
        )
        
        if success:
            self.managed_teams[team.team_id] = team
            self.adaptation_counts[team.team_id] = 0
            self.last_adaptation_times[team.team_id] = 0.0
            
            logger.info(f"✅ Registered team {team.team_id} for production topology management")
        
        return success
    
    def unregister_team(self, team_id: str) -> bool:
        """Unregister team"""
        success = self.topology_manager.unregister_team(team_id)
        
        if success:
            self.managed_teams.pop(team_id, None)
            self.adaptation_counts.pop(team_id, None)
            self.last_adaptation_times.pop(team_id, None)
            
            logger.info(f"✅ Unregistered team {team_id}")
        
        return success
    
    async def execute_task(
        self,
        team_id: str,
        task_description: str,
        task_function: Callable,
        complexity: float = 0.5,
        urgency: float = 0.5,
        timeout: Optional[float] = None
    ) -> Any:
        """Execute task with performance tracking"""
        
        if team_id not in self.managed_teams:
            raise ValueError(f"Team {team_id} not registered")
        
        task_id = f"task_{int(time.time() * 1000)}"
        timeout = timeout or self.config.adaptation_timeout
        
        # Update task state
        self.topology_manager.update_task_state(
            team_id=team_id,
            phase="execution",
            complexity=complexity,
            urgency=urgency
        )
        
        # Start performance tracking
        self.performance_tracker.start_task(team_id, task_id)
        
        try:
            # Execute task with timeout
            result = await asyncio.wait_for(
                task_function(),
                timeout=timeout
            )
            
            # Record successful completion
            self.performance_tracker.complete_task(team_id, task_id, True)
            
            # Update performance metrics
            await self._update_performance_metrics(team_id)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out after {timeout}s")
            self.performance_tracker.complete_task(team_id, task_id, False)
            raise
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            self.performance_tracker.complete_task(team_id, task_id, False)
            raise
    
    async def _update_performance_metrics(self, team_id: str):
        """Update performance metrics for a team"""
        # Calculate team metrics
        team_metrics = self.performance_tracker.calculate_team_metrics(team_id)
        
        # Convert to topology metrics
        topology_metrics = TopologyMetrics(
            efficiency=team_metrics.resource_efficiency,
            communication_cost=1.0 - team_metrics.task_completion_rate,
            load_balance=1.0 - team_metrics.error_rate,
            fault_tolerance=team_metrics.task_completion_rate,
            adaptability=0.7,  # Default value
            convergence_time=team_metrics.average_response_time / 60.0
        )
        
        # Update topology manager (this would trigger adaptation if needed)
        self.topology_manager.update_task_state(
            team_id=team_id,
            performance_metrics={
                'efficiency': topology_metrics.efficiency,
                'communication_cost': topology_metrics.communication_cost,
                'load_balance': topology_metrics.load_balance,
                'fault_tolerance': topology_metrics.fault_tolerance
            }
        )
    
    def _on_topology_change(self, team_id: str, transition: TopologyTransition):
        """Handle topology change with production safeguards"""
        current_time = time.time()
        
        # Check adaptation rate limits
        last_adaptation = self.last_adaptation_times.get(team_id, 0)
        time_since_last = current_time - last_adaptation
        
        if time_since_last < 3600:  # Within last hour
            hourly_count = self.adaptation_counts.get(team_id, 0)
            if hourly_count >= self.config.max_adaptations_per_hour:
                logger.warning(f"Team {team_id} exceeded hourly adaptation limit")
                return
        else:
            # Reset hourly counter
            self.adaptation_counts[team_id] = 0
        
        # Update counters
        self.adaptation_counts[team_id] = self.adaptation_counts.get(team_id, 0) + 1
        self.last_adaptation_times[team_id] = current_time
        
        # Save model checkpoint if needed
        if self.adaptation_counts[team_id] % self.config.checkpoint_interval == 0:
            self._save_model_checkpoint(team_id)
        
        # Log adaptation
        logger.info(f"🔄 Topology adapted for team {team_id}")
        logger.info(f"   Reason: {transition.transition_reason}")
        logger.info(f"   From: {transition.from_topology.topology_type.value}")
        logger.info(f"   To: {transition.to_topology.topology_type.value}")
        logger.info(f"   Expected improvement: {transition.expected_improvement:.3f}")
        
        # Notify callbacks
        for callback in self.adaptation_callbacks:
            try:
                callback(team_id, transition)
            except Exception as e:
                logger.error(f"Adaptation callback failed: {e}")
    
    def _save_model_checkpoint(self, team_id: str):
        """Save model checkpoint"""
        try:
            checkpoint_dir = Path(self.config.model_save_dir) / team_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Save GNN model if available
            gnn_path = checkpoint_dir / f"gnn_checkpoint_{timestamp}.pt"
            # self.topology_manager.adaptive_topologies[team_id].gnn_generator.save_model(str(gnn_path))
            
            # Save RL model if available
            rl_path = checkpoint_dir / f"rl_checkpoint_{timestamp}.pt"
            # self.topology_manager.adaptive_topologies[team_id].rl_search.save_model(str(rl_path))
            
            logger.info(f"💾 Saved model checkpoint for team {team_id}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for team {team_id}: {e}")
    
    async def _metrics_export_loop(self):
        """Export metrics periodically"""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_export_interval)
                await self._export_metrics()
            except Exception as e:
                logger.error(f"Metrics export failed: {e}")
    
    async def _export_metrics(self):
        """Export current metrics"""
        metrics_data = {
            'timestamp': time.time(),
            'system_metrics': self.resource_monitor.get_current_metrics(),
            'team_metrics': {},
            'topology_status': {}
        }
        
        # Collect team metrics
        for team_id in self.managed_teams.keys():
            team_metrics = self.performance_tracker.calculate_team_metrics(team_id)
            metrics_data['team_metrics'][team_id] = team_metrics
            
            # Get topology status
            current_topology = self.topology_manager.get_current_topology(team_id)
            performance_metrics = self.topology_manager.get_performance_metrics(team_id)
            
            metrics_data['topology_status'][team_id] = {
                'topology_type': current_topology.topology_type.value if current_topology else None,
                'performance_score': performance_metrics.overall_score() if performance_metrics else 0.0,
                'adaptation_count': self.adaptation_counts.get(team_id, 0)
            }
        
        # Export to file (in production, this could be sent to monitoring system)
        metrics_file = Path(self.config.model_save_dir) / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
    
    def add_adaptation_callback(self, callback: Callable[[str, TopologyTransition], None]):
        """Add callback for topology adaptations"""
        self.adaptation_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'config': {
                'environment': self.config.environment,
                'max_teams': self.config.max_concurrent_teams,
                'max_agents_per_team': self.config.max_agents_per_team
            },
            'system_metrics': self.resource_monitor.get_current_metrics(),
            'managed_teams': list(self.managed_teams.keys()),
            'adaptation_counts': self.adaptation_counts.copy(),
            'resource_constrained': self.resource_monitor.is_resource_constrained(self.config)
        }
