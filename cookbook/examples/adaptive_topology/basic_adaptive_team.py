"""
Basic Adaptive Topology Example

Demonstrates how to use the adaptive topology system with agno teams
to automatically optimize communication structures based on task requirements.
"""

import asyncio
import time
from typing import List

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.topology.manager import TopologyManager, TopologyManagerConfig
from agno.topology.adaptive import AdaptationConfig
from agno.topology.gnn_generator import GNNConfig
from agno.topology.rl_search import RLConfig
from agno.utils.log import logger


def create_research_team() -> Team:
    """Create a research team with diverse capabilities"""
    
    # Web researcher
    web_researcher = Agent(
        name="Web Researcher",
        agent_id="web_researcher_001",
        role="Information gathering from web sources",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Search for the latest information on given topics",
            "Provide comprehensive summaries with sources",
            "Focus on recent developments and trends"
        ]
    )
    
    # Data analyst
    data_analyst = Agent(
        name="Data Analyst",
        agent_id="data_analyst_001", 
        role="Data analysis and pattern recognition",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Analyze data patterns and trends",
            "Create visualizations and summaries",
            "Identify key insights and correlations"
        ]
    )
    
    # Technical writer
    tech_writer = Agent(
        name="Technical Writer",
        agent_id="tech_writer_001",
        role="Technical documentation and reporting",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Create clear, comprehensive technical reports",
            "Synthesize information from multiple sources",
            "Ensure accuracy and readability"
        ]
    )
    
    # Domain expert
    domain_expert = Agent(
        name="Domain Expert",
        agent_id="domain_expert_001",
        role="Subject matter expertise and validation",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Provide expert knowledge and context",
            "Validate findings and recommendations",
            "Offer strategic insights"
        ]
    )
    
    # Quality assurance
    qa_agent = Agent(
        name="QA Agent",
        agent_id="qa_agent_001",
        role="Quality assurance and review",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Review all outputs for quality and accuracy",
            "Ensure consistency across team deliverables",
            "Provide feedback and improvement suggestions"
        ]
    )
    
    # Create team
    research_team = Team(
        name="Adaptive Research Team",
        team_id="adaptive_research_team_001",
        mode="coordinate",
        members=[web_researcher, data_analyst, tech_writer, domain_expert, qa_agent],
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "Collaborate effectively to produce high-quality research outputs",
            "Adapt communication patterns based on task requirements",
            "Optimize for both speed and quality"
        ],
        success_criteria="Comprehensive, accurate, and well-structured research deliverable"
    )
    
    return research_team


async def demonstrate_adaptive_topology():
    """Demonstrate adaptive topology management"""
    
    logger.info("=== Adaptive Topology Demonstration ===")
    
    # Create configuration
    manager_config = TopologyManagerConfig(
        enable_adaptive_topology=True,
        auto_start_adaptation=True,
        adaptation_check_interval=10.0,  # Check every 10 seconds
        performance_monitoring_interval=5.0,  # Monitor every 5 seconds
        log_topology_changes=True
    )
    
    adaptation_config = AdaptationConfig(
        min_efficiency_threshold=0.6,
        max_communication_cost_threshold=0.8,
        adaptation_cooldown=15.0,  # 15 seconds between adaptations
        prefer_gnn=True,
        enable_rl_learning=True
    )
    
    gnn_config = GNNConfig(
        hidden_dim=64,  # Smaller for demo
        num_layers=2,
        learning_rate=0.001
    )
    
    rl_config = RLConfig(
        state_dim=128,
        action_dim=64,
        learning_rate=0.001,
        epsilon=0.3,  # More exploration for demo
        batch_size=16
    )
    
    # Create topology manager
    topology_manager = TopologyManager(
        config=manager_config,
        adaptation_config=adaptation_config,
        gnn_config=gnn_config,
        rl_config=rl_config
    )
    
    # Add callback to log topology changes
    def on_topology_change(team_id: str, transition):
        logger.info(f"🔄 Topology adapted for {team_id}:")
        logger.info(f"   Reason: {transition.transition_reason}")
        logger.info(f"   Expected improvement: {transition.expected_improvement:.3f}")
        logger.info(f"   Transition cost: {transition.transition_cost:.3f}")
    
    topology_manager.add_topology_change_callback(on_topology_change)
    
    # Create and register team
    research_team = create_research_team()
    
    success = topology_manager.register_team(
        team=research_team,
        initial_task_description="AI research and analysis task",
        task_complexity=0.7,  # Moderately complex
        task_urgency=0.6       # Moderate urgency
    )
    
    if not success:
        logger.error("Failed to register team")
        return
    
    # Start topology management
    topology_manager.start()
    
    logger.info("🚀 Topology management started")
    
    # Simulate different task phases
    phases = [
        ("initialization", 0.5, 0.4, "Setting up research parameters"),
        ("data_collection", 0.8, 0.7, "Intensive data gathering phase"),
        ("analysis", 0.9, 0.5, "Deep analysis and pattern recognition"),
        ("synthesis", 0.6, 0.8, "Urgent synthesis and report generation"),
        ("review", 0.4, 0.3, "Final review and quality assurance")
    ]
    
    for i, (phase, complexity, urgency, description) in enumerate(phases):
        logger.info(f"\n📋 Phase {i+1}: {phase.upper()}")
        logger.info(f"   Description: {description}")
        logger.info(f"   Complexity: {complexity}, Urgency: {urgency}")
        
        # Update task state
        topology_manager.update_task_state(
            team_id=research_team.team_id,
            phase=phase,
            complexity=complexity,
            urgency=urgency,
            required_capabilities=[phase, "analysis", "communication"]
        )
        
        # Simulate varying agent loads
        if phase == "data_collection":
            # High load on web researcher
            topology_manager.update_agent_loads(
                team_id=research_team.team_id,
                agent_loads={
                    "web_researcher_001": 0.9,
                    "data_analyst_001": 0.7,
                    "tech_writer_001": 0.3,
                    "domain_expert_001": 0.4,
                    "qa_agent_001": 0.2
                }
            )
        elif phase == "analysis":
            # High load on data analyst and domain expert
            topology_manager.update_agent_loads(
                team_id=research_team.team_id,
                agent_loads={
                    "web_researcher_001": 0.3,
                    "data_analyst_001": 0.9,
                    "tech_writer_001": 0.4,
                    "domain_expert_001": 0.8,
                    "qa_agent_001": 0.3
                }
            )
        elif phase == "synthesis":
            # High load on technical writer
            topology_manager.update_agent_loads(
                team_id=research_team.team_id,
                agent_loads={
                    "web_researcher_001": 0.2,
                    "data_analyst_001": 0.5,
                    "tech_writer_001": 0.9,
                    "domain_expert_001": 0.6,
                    "qa_agent_001": 0.4
                }
            )
        elif phase == "review":
            # High load on QA agent
            topology_manager.update_agent_loads(
                team_id=research_team.team_id,
                agent_loads={
                    "web_researcher_001": 0.1,
                    "data_analyst_001": 0.3,
                    "tech_writer_001": 0.4,
                    "domain_expert_001": 0.5,
                    "qa_agent_001": 0.9
                }
            )
        
        # Get current topology and performance
        current_topology = topology_manager.get_current_topology(research_team.team_id)
        performance = topology_manager.get_performance_metrics(research_team.team_id)
        
        if current_topology and performance:
            logger.info(f"   Current topology type: {current_topology.topology_type.value}")
            logger.info(f"   Performance score: {performance.overall_score():.3f}")
            logger.info(f"   Efficiency: {performance.efficiency:.3f}")
            logger.info(f"   Communication cost: {performance.communication_cost:.3f}")
            logger.info(f"   Load balance: {performance.load_balance:.3f}")
        
        # Wait for adaptation to occur
        await asyncio.sleep(20)  # 20 seconds per phase
        
        # Optionally force adaptation to demonstrate
        if i == 2:  # Force adaptation during analysis phase
            logger.info("🔧 Forcing topology adaptation...")
            transition = topology_manager.force_topology_adaptation(research_team.team_id)
            if transition:
                logger.info(f"   Forced adaptation successful!")
    
    # Final summary
    logger.info("\n📊 FINAL SUMMARY")
    
    # Get topology history
    history = topology_manager.get_topology_history(research_team.team_id)
    logger.info(f"Total topology adaptations: {len(history)}")
    
    for i, transition in enumerate(history):
        logger.info(f"  Adaptation {i+1}: {transition.transition_reason} "
                   f"(improvement: {transition.expected_improvement:.3f})")
    
    # Get final performance
    final_performance = topology_manager.get_performance_metrics(research_team.team_id)
    if final_performance:
        logger.info(f"Final performance score: {final_performance.overall_score():.3f}")
    
    # Stop topology management
    topology_manager.stop()
    logger.info("🛑 Topology management stopped")


async def demonstrate_communication_protocol():
    """Demonstrate the hybrid communication protocol"""
    
    logger.info("\n=== Communication Protocol Demonstration ===")
    
    # Create a simple team for communication demo
    research_team = create_research_team()
    
    # Create topology manager
    topology_manager = TopologyManager()
    topology_manager.register_team(research_team)
    topology_manager.start()
    
    team_id = research_team.team_id
    
    # Demonstrate different communication modes
    
    # 1. Broadcast message
    logger.info("📢 Sending broadcast message...")
    topology_manager.send_message(
        team_id=team_id,
        sender_id="web_researcher_001",
        receiver_ids=["data_analyst_001", "tech_writer_001", "domain_expert_001", "qa_agent_001"],
        content="Starting new research project on AI trends",
        message_type="announcement",
        priority=0.7
    )
    
    # 2. Point-to-point critical message
    logger.info("🎯 Sending critical point-to-point message...")
    topology_manager.send_message(
        team_id=team_id,
        sender_id="domain_expert_001",
        receiver_ids=["data_analyst_001"],
        content="Urgent: Found critical data inconsistency",
        message_type="alert",
        priority=0.9
    )
    
    # 3. Status report (hierarchical)
    logger.info("📋 Sending status report...")
    topology_manager.send_message(
        team_id=team_id,
        sender_id="qa_agent_001",
        receiver_ids=["domain_expert_001"],
        content="Quality review completed - 3 issues found",
        message_type="status_report",
        priority=0.6
    )
    
    # Simulate message processing
    await asyncio.sleep(2)
    
    # Check received messages
    for agent_id in ["data_analyst_001", "tech_writer_001", "domain_expert_001", "qa_agent_001"]:
        messages = topology_manager.receive_messages(team_id, agent_id)
        if messages:
            logger.info(f"📨 {agent_id} received {len(messages)} messages:")
            for msg in messages:
                logger.info(f"   - {msg.message_type}: {msg.content[:50]}...")
    
    topology_manager.stop()


if __name__ == "__main__":
    # Run demonstrations
    asyncio.run(demonstrate_adaptive_topology())
    asyncio.run(demonstrate_communication_protocol())
