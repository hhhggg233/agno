"""
Performance Evaluation Example

Comprehensive evaluation and benchmarking of adaptive topology configurations
to demonstrate the effectiveness of different approaches and identify optimal settings.
"""

import asyncio
import time
from typing import Dict, Any

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.topology.manager import TopologyManager, TopologyManagerConfig
from agno.topology.adaptive import AdaptiveTopology, AdaptationConfig
from agno.topology.gnn_generator import GNNConfig
from agno.topology.rl_search import RLConfig
from agno.topology.evaluator import TopologyEvaluator, BenchmarkConfig
from agno.topology.types import AgentCapability, TaskState, TopologyType
from agno.utils.log import logger


def create_test_agents(count: int = 5) -> list[AgentCapability]:
    """Create test agents with varying capabilities"""
    agents = []
    
    roles = ["Researcher", "Analyst", "Writer", "Reviewer", "Coordinator"]
    tools_sets = [
        ["search", "web_scraping"],
        ["data_analysis", "statistics"],
        ["text_generation", "editing"],
        ["quality_check", "validation"],
        ["coordination", "planning"]
    ]
    
    for i in range(count):
        role = roles[i % len(roles)]
        tools = tools_sets[i % len(tools_sets)]
        
        agent = AgentCapability(
            agent_id=f"agent_{i:03d}",
            name=f"{role} {i+1}",
            role=role,
            tools=tools,
            current_load=0.3 + (i * 0.1) % 0.5,  # Varying loads
            expertise_domains=[role.lower(), "collaboration"],
            communication_bandwidth=0.8 + (i * 0.05) % 0.2,
            processing_power=0.7 + (i * 0.1) % 0.3,
            reliability_score=0.85 + (i * 0.03) % 0.15
        )
        agents.append(agent)
    
    return agents


def create_test_task(complexity: float = 0.6, urgency: float = 0.5) -> TaskState:
    """Create a test task with specified characteristics"""
    return TaskState(
        task_id=f"eval_task_{int(time.time())}",
        phase="evaluation",
        complexity=complexity,
        urgency=urgency,
        required_capabilities=["research", "analysis", "writing", "review"],
        performance_metrics={
            "efficiency": 0.5,
            "communication_cost": 0.4,
            "load_balance": 0.6
        }
    )


async def evaluate_single_configuration():
    """Evaluate a single adaptive topology configuration"""
    logger.info("=== Single Configuration Evaluation ===")
    
    # Create test setup
    agents = create_test_agents(6)
    task = create_test_task(complexity=0.7, urgency=0.6)
    
    # Create adaptive topology with specific configuration
    config = AdaptationConfig(
        min_efficiency_threshold=0.6,
        max_communication_cost_threshold=0.8,
        adaptation_cooldown=15.0,
        prefer_gnn=True,
        enable_rl_learning=True
    )
    
    gnn_config = GNNConfig(
        hidden_dim=64,
        num_layers=2,
        learning_rate=0.001
    )
    
    rl_config = RLConfig(
        state_dim=128,
        action_dim=64,
        learning_rate=0.001,
        epsilon=0.2
    )
    
    adaptive_topology = AdaptiveTopology(
        agents=agents,
        initial_task=task,
        config=config,
        gnn_config=gnn_config,
        rl_config=rl_config
    )
    
    # Create evaluator
    evaluator = TopologyEvaluator()
    
    # Run evaluation
    result = evaluator.evaluate_topology(
        adaptive_topology=adaptive_topology,
        test_duration=30.0,
        scenario_name="single_config_test"
    )
    
    # Display results
    logger.info("📊 Evaluation Results:")
    logger.info(f"  Overall Score: {result.overall_score():.3f}")
    logger.info(f"  Topology Type: {result.topology_type.value}")
    logger.info(f"  Efficiency: {result.metrics.efficiency:.3f}")
    logger.info(f"  Communication Cost: {result.metrics.communication_cost:.3f}")
    logger.info(f"  Load Balance: {result.metrics.load_balance:.3f}")
    logger.info(f"  Adaptations: {result.adaptation_count}")
    
    if result.bottlenecks:
        logger.info(f"  Bottlenecks: {', '.join(result.bottlenecks)}")
    
    if result.recommendations:
        logger.info("  Recommendations:")
        for rec in result.recommendations[:3]:  # Show top 3
            logger.info(f"    - {rec}")
    
    # Analyze adaptation patterns
    analysis = evaluator.analyze_adaptation_patterns(adaptive_topology)
    logger.info(f"  Adaptation Analysis:")
    logger.info(f"    Total adaptations: {analysis['total_adaptations']}")
    logger.info(f"    Average improvement: {analysis['average_improvement']:.3f}")
    if analysis['most_common_trigger']:
        logger.info(f"    Most common trigger: {analysis['most_common_trigger']}")
    
    return result


async def benchmark_multiple_configurations():
    """Benchmark multiple topology configurations"""
    logger.info("=== Multi-Configuration Benchmark ===")
    
    # Define different configurations to test
    configurations = {
        "gnn_preferred": {
            "config": AdaptationConfig(
                min_efficiency_threshold=0.6,
                adaptation_cooldown=10.0,
                prefer_gnn=True,
                enable_rl_learning=False
            ),
            "gnn_config": GNNConfig(hidden_dim=128, num_layers=3),
            "rl_config": None
        },
        
        "rl_preferred": {
            "config": AdaptationConfig(
                min_efficiency_threshold=0.6,
                adaptation_cooldown=10.0,
                prefer_gnn=False,
                enable_rl_learning=True
            ),
            "gnn_config": None,
            "rl_config": RLConfig(state_dim=128, action_dim=64, epsilon=0.3)
        },
        
        "hybrid_approach": {
            "config": AdaptationConfig(
                min_efficiency_threshold=0.65,
                adaptation_cooldown=15.0,
                prefer_gnn=True,
                enable_rl_learning=True
            ),
            "gnn_config": GNNConfig(hidden_dim=64, num_layers=2),
            "rl_config": RLConfig(state_dim=64, action_dim=32, epsilon=0.1)
        },
        
        "conservative": {
            "config": AdaptationConfig(
                min_efficiency_threshold=0.5,
                adaptation_cooldown=30.0,
                prefer_gnn=True,
                enable_rl_learning=False
            ),
            "gnn_config": GNNConfig(hidden_dim=32, num_layers=1),
            "rl_config": None
        },
        
        "aggressive": {
            "config": AdaptationConfig(
                min_efficiency_threshold=0.7,
                adaptation_cooldown=5.0,
                prefer_gnn=False,
                enable_rl_learning=True
            ),
            "gnn_config": None,
            "rl_config": RLConfig(state_dim=128, action_dim=64, epsilon=0.5)
        }
    }
    
    # Create evaluator
    evaluator = TopologyEvaluator()
    
    # Configure benchmark
    benchmark_config = BenchmarkConfig(
        test_duration=20.0,  # Shorter for demo
        task_complexity_range=(0.4, 0.8),
        task_urgency_range=(0.3, 0.7),
        agent_count_range=(4, 7),
        num_test_scenarios=3  # Fewer scenarios for demo
    )
    
    # Run benchmark
    results = evaluator.benchmark_configurations(configurations, benchmark_config)
    
    # Analyze results
    logger.info("📊 Benchmark Results Summary:")
    
    config_scores = {}
    for config_name, config_results in results.items():
        if config_results:
            scores = [r.overall_score() for r in config_results]
            avg_score = sum(scores) / len(scores)
            config_scores[config_name] = avg_score
            
            logger.info(f"  {config_name}:")
            logger.info(f"    Average Score: {avg_score:.3f}")
            logger.info(f"    Best Score: {max(scores):.3f}")
            logger.info(f"    Scenarios: {len(config_results)}")
            
            # Average adaptations
            avg_adaptations = sum(r.adaptation_count for r in config_results) / len(config_results)
            logger.info(f"    Avg Adaptations: {avg_adaptations:.1f}")
    
    # Find best configuration
    if config_scores:
        best_config = max(config_scores.keys(), key=lambda k: config_scores[k])
        logger.info(f"🏆 Best Configuration: {best_config} (score: {config_scores[best_config]:.3f})")
    
    # Generate report
    report = evaluator.generate_report("topology_benchmark_report.md")
    logger.info("📄 Detailed report saved to: topology_benchmark_report.md")
    
    return results


async def compare_topology_types():
    """Compare different fixed topology types"""
    logger.info("=== Topology Type Comparison ===")
    
    # Create test setup
    agents = create_test_agents(5)
    task = create_test_task(complexity=0.6, urgency=0.5)
    
    # Create evaluator
    evaluator = TopologyEvaluator()
    
    # Compare different topology types
    topology_types = [
        TopologyType.CENTRALIZED,
        TopologyType.DECENTRALIZED,
        TopologyType.HIERARCHICAL,
        TopologyType.RING
    ]
    
    comparison_results = evaluator.compare_topologies(
        topology_types=topology_types,
        agents=agents,
        task=task,
        test_duration=15.0
    )
    
    # Display comparison results
    logger.info("📊 Topology Type Comparison:")
    
    for topology_type, result in comparison_results.items():
        logger.info(f"  {topology_type.value}:")
        logger.info(f"    Overall Score: {result.overall_score():.3f}")
        logger.info(f"    Efficiency: {result.metrics.efficiency:.3f}")
        logger.info(f"    Comm Cost: {result.metrics.communication_cost:.3f}")
        logger.info(f"    Load Balance: {result.metrics.load_balance:.3f}")
    
    # Find best topology type
    if comparison_results:
        best_type = max(comparison_results.keys(), 
                       key=lambda t: comparison_results[t].overall_score())
        best_score = comparison_results[best_type].overall_score()
        logger.info(f"🏆 Best Topology Type: {best_type.value} (score: {best_score:.3f})")
    
    return comparison_results


async def stress_test_adaptation():
    """Stress test the adaptation mechanism with rapid changes"""
    logger.info("=== Adaptation Stress Test ===")
    
    # Create test setup
    agents = create_test_agents(8)
    initial_task = create_test_task(complexity=0.5, urgency=0.5)
    
    # Create adaptive topology with aggressive settings
    config = AdaptationConfig(
        min_efficiency_threshold=0.7,  # High threshold
        adaptation_cooldown=2.0,       # Very short cooldown
        prefer_gnn=True,
        enable_rl_learning=True
    )
    
    adaptive_topology = AdaptiveTopology(
        agents=agents,
        initial_task=initial_task,
        config=config
    )
    
    # Create evaluator
    evaluator = TopologyEvaluator()
    
    logger.info("🔥 Starting stress test with rapid task changes...")
    
    # Simulate rapid task changes
    start_time = time.time()
    test_duration = 30.0
    change_interval = 3.0  # Change task every 3 seconds
    
    last_change_time = start_time
    phase_counter = 0
    
    while (time.time() - start_time) < test_duration:
        current_time = time.time()
        
        # Change task characteristics every few seconds
        if (current_time - last_change_time) >= change_interval:
            phase_counter += 1
            
            # Create new task with different characteristics
            new_complexity = 0.3 + (phase_counter * 0.15) % 0.6
            new_urgency = 0.2 + (phase_counter * 0.2) % 0.7
            
            new_task = TaskState(
                task_id=f"stress_task_{phase_counter}",
                phase=f"stress_phase_{phase_counter}",
                complexity=new_complexity,
                urgency=new_urgency,
                required_capabilities=[f"capability_{i}" for i in range(phase_counter % 3 + 2)]
            )
            
            adaptive_topology.update_task_state(new_task)
            
            # Also update agent loads randomly
            import random
            updated_agents = []
            for agent in adaptive_topology.agents:
                new_load = random.uniform(0.1, 0.9)
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
            
            logger.info(f"  Phase {phase_counter}: complexity={new_complexity:.2f}, "
                       f"urgency={new_urgency:.2f}")
            
            last_change_time = current_time
        
        await asyncio.sleep(0.5)  # Check frequently
    
    # Evaluate final state
    result = evaluator.evaluate_topology(
        adaptive_topology=adaptive_topology,
        test_duration=5.0,  # Short final evaluation
        scenario_name="stress_test"
    )
    
    # Analyze stress test results
    logger.info("🔥 Stress Test Results:")
    logger.info(f"  Total phases: {phase_counter}")
    logger.info(f"  Final score: {result.overall_score():.3f}")
    logger.info(f"  Total adaptations: {result.adaptation_count}")
    logger.info(f"  Adaptations per phase: {result.adaptation_count / max(1, phase_counter):.1f}")
    
    # Check if system remained stable
    if result.overall_score() > 0.5:
        logger.info("✅ System remained stable under stress")
    else:
        logger.info("⚠️ System showed signs of instability")
    
    return result


async def main():
    """Run all evaluation examples"""
    logger.info("🚀 Starting Adaptive Topology Performance Evaluation")
    
    try:
        # 1. Single configuration evaluation
        await evaluate_single_configuration()
        await asyncio.sleep(2)
        
        # 2. Multi-configuration benchmark
        await benchmark_multiple_configurations()
        await asyncio.sleep(2)
        
        # 3. Topology type comparison
        await compare_topology_types()
        await asyncio.sleep(2)
        
        # 4. Stress test
        await stress_test_adaptation()
        
        logger.info("✅ All evaluations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
