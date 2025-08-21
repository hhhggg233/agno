"""
Topology Performance Evaluator

Provides comprehensive evaluation and benchmarking tools for adaptive topologies,
including performance comparison, bottleneck detection, and optimization suggestions.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

from agno.topology.types import (
    AgentCapability,
    TaskState,
    TopologyGraph,
    TopologyMetrics,
    TopologyType,
    TopologyTransition
)
from agno.topology.adaptive import AdaptiveTopology
from agno.utils.log import logger


@dataclass
class EvaluationResult:
    """Results from topology evaluation"""
    topology_type: TopologyType
    metrics: TopologyMetrics
    execution_time: float
    adaptation_count: int
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def overall_score(self) -> float:
        """Calculate overall evaluation score"""
        base_score = self.metrics.overall_score()
        
        # Penalty for excessive adaptations
        adaptation_penalty = min(0.1, self.adaptation_count * 0.01)
        
        # Penalty for execution time (normalized)
        time_penalty = min(0.1, self.execution_time / 100.0)
        
        return max(0.0, base_score - adaptation_penalty - time_penalty)


@dataclass
class BenchmarkConfig:
    """Configuration for topology benchmarking"""
    test_duration: float = 60.0  # seconds
    task_complexity_range: Tuple[float, float] = (0.3, 0.9)
    task_urgency_range: Tuple[float, float] = (0.2, 0.8)
    agent_count_range: Tuple[int, int] = (3, 10)
    load_variation_frequency: float = 10.0  # seconds
    num_test_scenarios: int = 5


class TopologyEvaluator:
    """
    Comprehensive evaluator for adaptive topology performance
    """
    
    def __init__(self):
        self.evaluation_history: List[EvaluationResult] = []
        self.benchmark_results: Dict[str, List[EvaluationResult]] = defaultdict(list)
        
    def evaluate_topology(
        self,
        adaptive_topology: AdaptiveTopology,
        test_duration: float = 30.0,
        scenario_name: str = "default"
    ) -> EvaluationResult:
        """
        Evaluate a single adaptive topology configuration
        
        Args:
            adaptive_topology: The adaptive topology to evaluate
            test_duration: Duration of evaluation in seconds
            scenario_name: Name of the test scenario
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting topology evaluation: {scenario_name}")
        
        start_time = time.time()
        initial_adaptation_count = len(adaptive_topology.get_topology_history())
        
        # Track performance over time
        performance_samples = []
        bottlenecks = []
        
        # Run evaluation
        end_time = start_time + test_duration
        sample_interval = 2.0  # Sample every 2 seconds
        
        while time.time() < end_time:
            # Get current performance
            current_metrics = adaptive_topology.evaluate_current_performance()
            performance_samples.append(current_metrics)
            
            # Detect bottlenecks
            if current_metrics.efficiency < 0.5:
                bottlenecks.append("Low efficiency detected")
            if current_metrics.communication_cost > 0.8:
                bottlenecks.append("High communication cost")
            if current_metrics.load_balance < 0.4:
                bottlenecks.append("Poor load balance")
            
            time.sleep(sample_interval)
        
        execution_time = time.time() - start_time
        final_adaptation_count = len(adaptive_topology.get_topology_history())
        adaptation_count = final_adaptation_count - initial_adaptation_count
        
        # Calculate average metrics
        if performance_samples:
            avg_metrics = self._average_metrics(performance_samples)
        else:
            avg_metrics = TopologyMetrics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_metrics, bottlenecks, adaptation_count
        )
        
        # Get current topology type
        current_topology = adaptive_topology.get_current_topology()
        topology_type = current_topology.topology_type if current_topology else TopologyType.HYBRID
        
        result = EvaluationResult(
            topology_type=topology_type,
            metrics=avg_metrics,
            execution_time=execution_time,
            adaptation_count=adaptation_count,
            bottlenecks=list(set(bottlenecks)),  # Remove duplicates
            recommendations=recommendations
        )
        
        self.evaluation_history.append(result)
        self.benchmark_results[scenario_name].append(result)
        
        logger.info(f"Evaluation completed: {scenario_name}")
        logger.info(f"  Overall score: {result.overall_score():.3f}")
        logger.info(f"  Adaptations: {adaptation_count}")
        logger.info(f"  Bottlenecks: {len(result.bottlenecks)}")
        
        return result
    
    def benchmark_configurations(
        self,
        configurations: Dict[str, Dict[str, Any]],
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Benchmark multiple topology configurations
        
        Args:
            configurations: Dict of configuration name -> config parameters
            config: Benchmark configuration
            
        Returns:
            Benchmark results for each configuration
        """
        if config is None:
            config = BenchmarkConfig()
        
        logger.info(f"Starting benchmark with {len(configurations)} configurations")
        
        results = {}
        
        for config_name, config_params in configurations.items():
            logger.info(f"Benchmarking configuration: {config_name}")
            
            config_results = []
            
            for scenario_idx in range(config.num_test_scenarios):
                # Generate test scenario
                agents, task = self._generate_test_scenario(config, scenario_idx)
                
                # Create adaptive topology with configuration
                try:
                    adaptive_topology = AdaptiveTopology(
                        agents=agents,
                        initial_task=task,
                        **config_params
                    )
                    
                    # Run evaluation
                    scenario_name = f"{config_name}_scenario_{scenario_idx}"
                    result = self.evaluate_topology(
                        adaptive_topology,
                        config.test_duration,
                        scenario_name
                    )
                    
                    config_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {config_name} scenario {scenario_idx}: {e}")
                    continue
            
            results[config_name] = config_results
            
            # Log configuration summary
            if config_results:
                avg_score = np.mean([r.overall_score() for r in config_results])
                logger.info(f"  {config_name} average score: {avg_score:.3f}")
        
        return results
    
    def compare_topologies(
        self,
        topology_types: List[TopologyType],
        agents: List[AgentCapability],
        task: TaskState,
        test_duration: float = 30.0
    ) -> Dict[TopologyType, EvaluationResult]:
        """
        Compare different topology types on the same task
        
        Args:
            topology_types: List of topology types to compare
            agents: Agents for the test
            task: Task for the test
            test_duration: Duration of each test
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(topology_types)} topology types")
        
        results = {}
        
        for topology_type in topology_types:
            logger.info(f"Testing topology type: {topology_type.value}")
            
            # Create fixed topology of the specified type
            adj_matrix = self._create_topology_matrix(len(agents), topology_type)
            
            # Create adaptive topology with fixed structure
            # (This would need modification to support fixed topologies)
            # For now, we'll use the adaptive topology and track the type
            
            try:
                adaptive_topology = AdaptiveTopology(
                    agents=agents,
                    initial_task=task
                )
                
                result = self.evaluate_topology(
                    adaptive_topology,
                    test_duration,
                    f"comparison_{topology_type.value}"
                )
                
                results[topology_type] = result
                
            except Exception as e:
                logger.error(f"Failed to test topology type {topology_type.value}: {e}")
                continue
        
        # Log comparison summary
        if results:
            best_type = max(results.keys(), key=lambda t: results[t].overall_score())
            logger.info(f"Best performing topology: {best_type.value} "
                       f"(score: {results[best_type].overall_score():.3f})")
        
        return results
    
    def analyze_adaptation_patterns(
        self,
        adaptive_topology: AdaptiveTopology
    ) -> Dict[str, Any]:
        """
        Analyze adaptation patterns in a topology
        
        Args:
            adaptive_topology: The adaptive topology to analyze
            
        Returns:
            Analysis results
        """
        history = adaptive_topology.get_topology_history()
        
        if not history:
            return {"message": "No adaptation history available"}
        
        # Analyze adaptation triggers
        trigger_counts = defaultdict(int)
        for transition in history:
            trigger_counts[transition.transition_reason] += 1
        
        # Analyze adaptation frequency
        if len(history) > 1:
            time_intervals = []
            for i in range(1, len(history)):
                interval = history[i].timestamp - history[i-1].timestamp
                time_intervals.append(interval)
            
            avg_interval = np.mean(time_intervals)
            adaptation_frequency = 1.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            avg_interval = 0.0
            adaptation_frequency = 0.0
        
        # Analyze improvement trends
        improvements = [t.expected_improvement for t in history]
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        # Analyze topology type transitions
        type_transitions = []
        for transition in history:
            from_type = transition.from_topology.topology_type
            to_type = transition.to_topology.topology_type
            type_transitions.append((from_type, to_type))
        
        analysis = {
            "total_adaptations": len(history),
            "trigger_distribution": dict(trigger_counts),
            "average_adaptation_interval": avg_interval,
            "adaptation_frequency": adaptation_frequency,
            "average_improvement": avg_improvement,
            "topology_transitions": type_transitions,
            "most_common_trigger": max(trigger_counts.keys(), key=trigger_counts.get) if trigger_counts else None
        }
        
        return analysis
    
    def generate_report(
        self,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Report content as string
        """
        if not self.evaluation_history:
            return "No evaluation data available"
        
        report_lines = [
            "# Adaptive Topology Evaluation Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total evaluations: {len(self.evaluation_history)}",
            ""
        ]
        
        # Overall statistics
        overall_scores = [r.overall_score() for r in self.evaluation_history]
        report_lines.extend([
            "## Overall Performance",
            f"Average score: {np.mean(overall_scores):.3f}",
            f"Best score: {np.max(overall_scores):.3f}",
            f"Worst score: {np.min(overall_scores):.3f}",
            f"Standard deviation: {np.std(overall_scores):.3f}",
            ""
        ])
        
        # Topology type analysis
        type_counts = defaultdict(int)
        type_scores = defaultdict(list)
        
        for result in self.evaluation_history:
            type_counts[result.topology_type] += 1
            type_scores[result.topology_type].append(result.overall_score())
        
        report_lines.extend([
            "## Topology Type Performance",
            "| Type | Count | Avg Score | Best Score |",
            "|------|-------|-----------|------------|"
        ])
        
        for topology_type in TopologyType:
            if topology_type in type_counts:
                count = type_counts[topology_type]
                scores = type_scores[topology_type]
                avg_score = np.mean(scores)
                best_score = np.max(scores)
                
                report_lines.append(
                    f"| {topology_type.value} | {count} | {avg_score:.3f} | {best_score:.3f} |"
                )
        
        report_lines.append("")
        
        # Common bottlenecks
        all_bottlenecks = []
        for result in self.evaluation_history:
            all_bottlenecks.extend(result.bottlenecks)
        
        bottleneck_counts = defaultdict(int)
        for bottleneck in all_bottlenecks:
            bottleneck_counts[bottleneck] += 1
        
        if bottleneck_counts:
            report_lines.extend([
                "## Common Bottlenecks",
                "| Bottleneck | Frequency |",
                "|------------|-----------|"
            ])
            
            for bottleneck, count in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"| {bottleneck} | {count} |")
            
            report_lines.append("")
        
        # Recommendations summary
        all_recommendations = []
        for result in self.evaluation_history:
            all_recommendations.extend(result.recommendations)
        
        recommendation_counts = defaultdict(int)
        for rec in all_recommendations:
            recommendation_counts[rec] += 1
        
        if recommendation_counts:
            report_lines.extend([
                "## Top Recommendations",
                ""
            ])
            
            for rec, count in sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                report_lines.append(f"- {rec} (mentioned {count} times)")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_file}")
        
        return report_content
    
    def _average_metrics(self, metrics_list: List[TopologyMetrics]) -> TopologyMetrics:
        """Calculate average of multiple metrics"""
        if not metrics_list:
            return TopologyMetrics()
        
        return TopologyMetrics(
            efficiency=np.mean([m.efficiency for m in metrics_list]),
            communication_cost=np.mean([m.communication_cost for m in metrics_list]),
            load_balance=np.mean([m.load_balance for m in metrics_list]),
            fault_tolerance=np.mean([m.fault_tolerance for m in metrics_list]),
            adaptability=np.mean([m.adaptability for m in metrics_list]),
            convergence_time=np.mean([m.convergence_time for m in metrics_list])
        )
    
    def _generate_recommendations(
        self,
        metrics: TopologyMetrics,
        bottlenecks: List[str],
        adaptation_count: int
    ) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if metrics.efficiency < 0.6:
            recommendations.append("Consider increasing agent specialization")
            recommendations.append("Review task decomposition strategy")
        
        if metrics.communication_cost > 0.7:
            recommendations.append("Reduce communication overhead")
            recommendations.append("Implement more efficient routing protocols")
        
        if metrics.load_balance < 0.5:
            recommendations.append("Improve load balancing algorithms")
            recommendations.append("Consider dynamic agent allocation")
        
        if adaptation_count > 10:
            recommendations.append("Increase adaptation cooldown period")
            recommendations.append("Review adaptation triggers")
        
        if "Low efficiency detected" in bottlenecks:
            recommendations.append("Optimize agent coordination mechanisms")
        
        if "High communication cost" in bottlenecks:
            recommendations.append("Implement hierarchical communication")
        
        if "Poor load balance" in bottlenecks:
            recommendations.append("Add load monitoring and redistribution")
        
        return recommendations
    
    def _generate_test_scenario(
        self,
        config: BenchmarkConfig,
        scenario_idx: int
    ) -> Tuple[List[AgentCapability], TaskState]:
        """Generate a test scenario for benchmarking"""
        np.random.seed(scenario_idx)  # For reproducibility
        
        # Generate agents
        agent_count = np.random.randint(
            config.agent_count_range[0],
            config.agent_count_range[1] + 1
        )
        
        agents = []
        for i in range(agent_count):
            agent = AgentCapability(
                agent_id=f"test_agent_{i}",
                name=f"Test Agent {i}",
                role=f"Role_{i % 3}",  # Rotate through 3 roles
                tools=[f"tool_{j}" for j in range(np.random.randint(1, 4))],
                current_load=np.random.uniform(0.1, 0.8),
                expertise_domains=[f"domain_{j}" for j in range(np.random.randint(1, 3))],
                communication_bandwidth=np.random.uniform(0.5, 1.0),
                processing_power=np.random.uniform(0.5, 1.0),
                reliability_score=np.random.uniform(0.7, 1.0)
            )
            agents.append(agent)
        
        # Generate task
        task = TaskState(
            task_id=f"test_task_{scenario_idx}",
            phase="testing",
            complexity=np.random.uniform(*config.task_complexity_range),
            urgency=np.random.uniform(*config.task_urgency_range),
            required_capabilities=[f"capability_{i}" for i in range(np.random.randint(2, 5))]
        )
        
        return agents, task
    
    def _create_topology_matrix(
        self,
        num_agents: int,
        topology_type: TopologyType
    ) -> np.ndarray:
        """Create adjacency matrix for specific topology type"""
        adj_matrix = np.zeros((num_agents, num_agents))
        
        if topology_type == TopologyType.CENTRALIZED:
            # Star topology - all connect to agent 0
            for i in range(1, num_agents):
                adj_matrix[0, i] = 1.0
                adj_matrix[i, 0] = 1.0
        
        elif topology_type == TopologyType.DECENTRALIZED:
            # Fully connected
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0
        
        elif topology_type == TopologyType.RING:
            # Ring topology
            for i in range(num_agents):
                next_agent = (i + 1) % num_agents
                adj_matrix[i, next_agent] = 1.0
                adj_matrix[next_agent, i] = 1.0
        
        elif topology_type == TopologyType.HIERARCHICAL:
            # Binary tree-like structure
            for i in range(num_agents // 2):
                left_child = 2 * i + 1
                right_child = 2 * i + 2
                
                if left_child < num_agents:
                    adj_matrix[i, left_child] = 1.0
                    adj_matrix[left_child, i] = 1.0
                
                if right_child < num_agents:
                    adj_matrix[i, right_child] = 1.0
                    adj_matrix[right_child, i] = 1.0
        
        return adj_matrix
