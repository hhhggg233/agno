#!/usr/bin/env python3
"""
Production Adaptive Topology Example

This example demonstrates how to use the adaptive topology system in a
production environment with real performance monitoring, resource management,
and fault tolerance.

Requirements:
- PyTorch installed (pip install torch)
- agno framework properly configured
- OpenAI API key (for real agent execution)

Usage:
    python examples/production_topology_example.py
"""

import asyncio
import logging
import os
import sys
from typing import List, Dict, Any
import time

# Add agno to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs', 'agno'))

# Check dependencies
try:
    import torch
    print("✅ PyTorch available")
except ImportError:
    print("❌ PyTorch not available. Install with: pip install torch")
    sys.exit(1)

try:
    import psutil
    print("✅ psutil available")
except ImportError:
    print("❌ psutil not available. Install with: pip install psutil")
    sys.exit(1)

# agno imports
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat

# Topology imports
from agno.topology.production_manager import ProductionTopologyManager
from agno.topology.config import ProductionTopologyConfig, ConfigManager
from agno.topology.types import TopologyTransition

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTaskExecutor:
    """Executes real tasks using agno agents"""
    
    def __init__(self, team: Team):
        self.team = team
        self.task_counter = 0
    
    async def research_task(self, topic: str) -> str:
        """Execute a research task"""
        self.task_counter += 1
        task_id = f"research_{self.task_counter}"
        
        logger.info(f"🔍 Starting research task: {topic}")
        
        # In a real implementation, this would use team.run() or similar
        # For now, we simulate the task execution
        start_time = time.time()
        
        # Simulate research work
        await asyncio.sleep(2 + len(topic) * 0.1)  # Variable time based on topic complexity
        
        execution_time = time.time() - start_time
        result = f"Research completed on '{topic}' in {execution_time:.2f}s"
        
        logger.info(f"✅ {result}")
        return result
    
    async def analysis_task(self, data: str) -> str:
        """Execute an analysis task"""
        self.task_counter += 1
        task_id = f"analysis_{self.task_counter}"
        
        logger.info(f"📊 Starting analysis task")
        
        start_time = time.time()
        
        # Simulate analysis work
        await asyncio.sleep(3 + len(data) * 0.05)
        
        execution_time = time.time() - start_time
        result = f"Analysis completed in {execution_time:.2f}s - Found 3 key insights"
        
        logger.info(f"✅ {result}")
        return result
    
    async def writing_task(self, content: str) -> str:
        """Execute a writing task"""
        self.task_counter += 1
        task_id = f"writing_{self.task_counter}"
        
        logger.info(f"✍️ Starting writing task")
        
        start_time = time.time()
        
        # Simulate writing work
        await asyncio.sleep(4 + len(content) * 0.02)
        
        execution_time = time.time() - start_time
        result = f"Document written in {execution_time:.2f}s - 1500 words"
        
        logger.info(f"✅ {result}")
        return result


def create_production_team() -> Team:
    """Create a production-ready team"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY not set, using mock model")
        # In production, you'd want to fail here or use alternative models
    
    # Create specialized agents
    researcher = Agent(
        name="Senior Researcher",
        agent_id="prod_researcher_001",
        role="Research and data collection specialist",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Conduct thorough research on assigned topics",
            "Gather data from multiple reliable sources",
            "Provide comprehensive analysis and insights",
            "Collaborate effectively with team members"
        ]
    )
    
    analyst = Agent(
        name="Data Analyst",
        agent_id="prod_analyst_001", 
        role="Data analysis and pattern recognition",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Analyze complex datasets and identify patterns",
            "Create visualizations and statistical summaries",
            "Provide actionable insights and recommendations",
            "Validate findings with rigorous methods"
        ]
    )
    
    writer = Agent(
        name="Technical Writer",
        agent_id="prod_writer_001",
        role="Technical documentation and communication",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Create clear, comprehensive technical documentation",
            "Synthesize complex information into readable formats",
            "Ensure accuracy and consistency in all outputs",
            "Adapt writing style to target audience"
        ]
    )
    
    reviewer = Agent(
        name="Quality Reviewer",
        agent_id="prod_reviewer_001",
        role="Quality assurance and validation",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Review all team outputs for quality and accuracy",
            "Provide constructive feedback and suggestions",
            "Ensure deliverables meet quality standards",
            "Validate technical accuracy and completeness"
        ]
    )
    
    # Create production team
    team = Team(
        name="Production Research Team",
        team_id="prod_team_001",
        mode="coordinate",
        members=[researcher, analyst, writer, reviewer],
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "Deliver high-quality research and analysis outputs",
            "Maintain efficient communication and coordination",
            "Adapt to changing requirements and priorities",
            "Ensure all deliverables meet production standards"
        ]
    )
    
    return team


def setup_production_config() -> ProductionTopologyConfig:
    """Setup production configuration"""
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Try to load existing config, create sample if not found
    try:
        config = config_manager.load_config()
    except:
        logger.info("Creating sample configuration...")
        config_manager.create_sample_config("topology_config.json")
        config = config_manager.load_config()
    
    # Customize for this example
    config.environment = "production_demo"
    config.debug_mode = True
    config.log_level = "INFO"
    config.max_concurrent_teams = 5
    config.max_agents_per_team = 10
    config.enable_metrics_collection = True
    
    # Adjust adaptation settings for demo
    config.adaptation_config.adaptation_cooldown = 30.0  # 30 seconds for demo
    config.adaptation_config.min_efficiency_threshold = 0.6
    config.manager_config.adaptation_check_interval = 15.0  # Check every 15 seconds
    
    return config


async def run_production_demo():
    """Run the production topology demo"""
    
    logger.info("🚀 Starting Production Adaptive Topology Demo")
    logger.info("=" * 70)
    
    # Setup configuration
    config = setup_production_config()
    logger.info(f"📋 Configuration: {config.environment} mode")
    
    # Create production topology manager
    prod_manager = ProductionTopologyManager(config)
    
    # Add adaptation callback
    def on_adaptation(team_id: str, transition: TopologyTransition):
        logger.info(f"🔄 ADAPTATION DETECTED for {team_id}")
        logger.info(f"   Reason: {transition.transition_reason}")
        logger.info(f"   Performance improvement: {transition.expected_improvement:.3f}")
    
    prod_manager.add_adaptation_callback(on_adaptation)
    
    # Start production manager
    await prod_manager.start()
    
    try:
        # Create and register production team
        team = create_production_team()
        
        success = prod_manager.register_team(
            team=team,
            task_description="Production research and analysis workflow",
            complexity=0.7,
            urgency=0.6
        )
        
        if not success:
            logger.error("❌ Failed to register team")
            return
        
        logger.info(f"✅ Registered team: {team.name}")
        
        # Create task executor
        executor = ProductionTaskExecutor(team)
        
        # Execute production workflow
        workflow_tasks = [
            ("AI Ethics and Safety Research", 0.8, executor.research_task),
            ("Market Analysis of AI Tools", 0.6, executor.research_task),
            ("Technical Performance Analysis", 0.9, executor.analysis_task),
            ("Competitive Landscape Report", 0.7, executor.writing_task),
            ("Executive Summary Creation", 0.5, executor.writing_task)
        ]
        
        results = []
        
        for i, (task_name, complexity, task_func) in enumerate(workflow_tasks, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"📋 Production Task {i}: {task_name}")
            logger.info(f"   Complexity: {complexity}")
            logger.info(f"{'='*70}")
            
            try:
                # Execute task with production manager
                result = await prod_manager.execute_task(
                    team_id=team.team_id,
                    task_description=task_name,
                    task_function=lambda: task_func(task_name),
                    complexity=complexity,
                    urgency=0.5 + (i * 0.1),  # Increasing urgency
                    timeout=60.0  # 1 minute timeout
                )
                
                results.append({
                    'task': task_name,
                    'result': result,
                    'success': True
                })
                
                logger.info(f"✅ Task completed: {result}")
                
            except Exception as e:
                logger.error(f"❌ Task failed: {e}")
                results.append({
                    'task': task_name,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
            
            # Show system status
            status = prod_manager.get_system_status()
            logger.info(f"\n📊 System Status:")
            logger.info(f"   Managed Teams: {len(status['managed_teams'])}")
            logger.info(f"   Resource Constrained: {status['resource_constrained']}")
            
            if status['system_metrics']:
                metrics = status['system_metrics']
                logger.info(f"   CPU: {metrics.cpu_percent:.1f}%")
                logger.info(f"   Memory: {metrics.memory_percent:.1f}%")
            
            # Show adaptation counts
            for team_id, count in status['adaptation_counts'].items():
                logger.info(f"   Adaptations for {team_id}: {count}")
            
            # Wait between tasks
            await asyncio.sleep(10)
        
        # Final summary
        logger.info(f"\n{'='*70}")
        logger.info("🎉 Production Demo Completed")
        logger.info(f"{'='*70}")
        
        successful_tasks = [r for r in results if r['success']]
        failed_tasks = [r for r in results if not r['success']]
        
        logger.info(f"✅ Successful tasks: {len(successful_tasks)}/{len(results)}")
        logger.info(f"❌ Failed tasks: {len(failed_tasks)}")
        
        if failed_tasks:
            logger.info("Failed tasks:")
            for task in failed_tasks:
                logger.info(f"   - {task['task']}: {task.get('error', 'Unknown error')}")
        
        # Show final system status
        final_status = prod_manager.get_system_status()
        total_adaptations = sum(final_status['adaptation_counts'].values())
        logger.info(f"🔄 Total topology adaptations: {total_adaptations}")
        
        # Unregister team
        prod_manager.unregister_team(team.team_id)
        
    finally:
        # Stop production manager
        await prod_manager.stop()
    
    logger.info("✅ Production demo completed successfully!")


if __name__ == "__main__":
    # Check environment
    logger.info("🔧 Production Adaptive Topology System")
    logger.info("Checking environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        sys.exit(1)
    
    logger.info(f"✅ Python {sys.version}")
    
    # Check available memory
    memory = psutil.virtual_memory()
    if memory.available < 1024 * 1024 * 1024:  # 1GB
        logger.warning("⚠️ Low available memory, performance may be affected")
    
    logger.info(f"💾 Available memory: {memory.available / (1024**3):.1f} GB")
    
    # Run the demo
    try:
        asyncio.run(run_production_demo())
    except KeyboardInterrupt:
        logger.info("🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
