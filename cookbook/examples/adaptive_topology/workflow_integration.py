"""
Workflow Integration Example

Demonstrates how to integrate adaptive topology with agno workflows
for complex multi-stage tasks with dynamic team reconfiguration.
"""

import asyncio
from typing import Dict, Any

from agno.agent import Agent
from agno.team import Team
from agno.workflow.v2 import Workflow, Step, Parallel, Router
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.topology.manager import TopologyManager, TopologyManagerConfig
from agno.topology.adaptive import AdaptationConfig
from agno.topology.types import TopologyTransition
from agno.utils.log import logger


class AdaptiveWorkflowManager:
    """Manager that coordinates adaptive topology with workflow execution"""
    
    def __init__(self):
        self.topology_manager = TopologyManager(
            config=TopologyManagerConfig(
                enable_adaptive_topology=True,
                auto_start_adaptation=True,
                adaptation_check_interval=15.0,
                log_topology_changes=True
            ),
            adaptation_config=AdaptationConfig(
                min_efficiency_threshold=0.65,
                adaptation_cooldown=20.0,
                prefer_gnn=True
            )
        )
        
        self.workflow_state: Dict[str, Any] = {}
        self.active_teams: Dict[str, Team] = {}
        
        # Add topology change callback
        self.topology_manager.add_topology_change_callback(
            self._on_topology_change
        )
    
    def _on_topology_change(self, team_id: str, transition: TopologyTransition):
        """Handle topology changes during workflow execution"""
        logger.info(f"🔄 Workflow topology adapted for {team_id}")
        logger.info(f"   Reason: {transition.transition_reason}")
        logger.info(f"   Performance improvement: {transition.expected_improvement:.3f}")
        
        # Update workflow state based on topology change
        if team_id in self.workflow_state:
            self.workflow_state[team_id]['last_adaptation'] = transition.timestamp
            self.workflow_state[team_id]['adaptation_count'] = (
                self.workflow_state[team_id].get('adaptation_count', 0) + 1
            )
    
    def register_workflow_team(
        self, 
        team: Team, 
        workflow_phase: str,
        complexity: float = 0.5,
        urgency: float = 0.5
    ):
        """Register a team for a specific workflow phase"""
        success = self.topology_manager.register_team(
            team=team,
            initial_task_description=f"Workflow phase: {workflow_phase}",
            task_complexity=complexity,
            task_urgency=urgency
        )
        
        if success:
            self.active_teams[team.team_id] = team
            self.workflow_state[team.team_id] = {
                'phase': workflow_phase,
                'complexity': complexity,
                'urgency': urgency,
                'adaptation_count': 0
            }
            logger.info(f"✅ Registered team {team.team_id} for phase: {workflow_phase}")
        
        return success
    
    def update_workflow_phase(
        self, 
        team_id: str, 
        new_phase: str,
        complexity: float = None,
        urgency: float = None
    ):
        """Update workflow phase and trigger topology adaptation"""
        if team_id not in self.workflow_state:
            return False
        
        old_state = self.workflow_state[team_id]
        
        # Update state
        self.workflow_state[team_id]['phase'] = new_phase
        if complexity is not None:
            self.workflow_state[team_id]['complexity'] = complexity
        if urgency is not None:
            self.workflow_state[team_id]['urgency'] = urgency
        
        # Update topology manager
        self.topology_manager.update_task_state(
            team_id=team_id,
            phase=new_phase,
            complexity=complexity or old_state['complexity'],
            urgency=urgency or old_state['urgency'],
            required_capabilities=[new_phase, "collaboration", "analysis"]
        )
        
        logger.info(f"📋 Updated workflow phase for {team_id}: {new_phase}")
        return True
    
    def start(self):
        """Start adaptive workflow management"""
        self.topology_manager.start()
        logger.info("🚀 Adaptive workflow manager started")
    
    def stop(self):
        """Stop adaptive workflow management"""
        self.topology_manager.stop()
        logger.info("🛑 Adaptive workflow manager stopped")


def create_specialized_teams():
    """Create specialized teams for different workflow phases"""
    
    # Research Team
    research_team = Team(
        name="Research Team",
        team_id="research_team_001",
        mode="coordinate",
        members=[
            Agent(
                name="Senior Researcher",
                agent_id="senior_researcher",
                role="Lead research and strategy",
                model=OpenAIChat(id="gpt-4o"),
                tools=[DuckDuckGoTools()],
                instructions=["Lead research initiatives", "Provide strategic direction"]
            ),
            Agent(
                name="Data Collector",
                agent_id="data_collector",
                role="Gather and organize data",
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[DuckDuckGoTools()],
                instructions=["Collect comprehensive data", "Organize findings systematically"]
            ),
            Agent(
                name="Research Analyst",
                agent_id="research_analyst",
                role="Analyze research findings",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=["Analyze data patterns", "Identify key insights"]
            )
        ],
        model=OpenAIChat(id="gpt-4o"),
        instructions=["Conduct thorough research", "Collaborate effectively"]
    )
    
    # Analysis Team
    analysis_team = Team(
        name="Analysis Team",
        team_id="analysis_team_001",
        mode="collaborate",
        members=[
            Agent(
                name="Data Scientist",
                agent_id="data_scientist",
                role="Advanced data analysis",
                model=OpenAIChat(id="gpt-4o"),
                instructions=["Perform statistical analysis", "Create data models"]
            ),
            Agent(
                name="Pattern Analyst",
                agent_id="pattern_analyst",
                role="Pattern recognition and trends",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=["Identify patterns and trends", "Predict future developments"]
            ),
            Agent(
                name="Validation Expert",
                agent_id="validation_expert",
                role="Validate analysis results",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=["Validate findings", "Ensure accuracy and reliability"]
            )
        ],
        model=OpenAIChat(id="gpt-4o"),
        instructions=["Perform deep analysis", "Ensure high accuracy"]
    )
    
    # Synthesis Team
    synthesis_team = Team(
        name="Synthesis Team",
        team_id="synthesis_team_001",
        mode="coordinate",
        members=[
            Agent(
                name="Content Synthesizer",
                agent_id="content_synthesizer",
                role="Synthesize information from multiple sources",
                model=OpenAIChat(id="gpt-4o"),
                instructions=["Combine insights from all sources", "Create coherent narratives"]
            ),
            Agent(
                name="Technical Writer",
                agent_id="technical_writer",
                role="Create technical documentation",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=["Write clear technical content", "Ensure readability"]
            ),
            Agent(
                name="Quality Reviewer",
                agent_id="quality_reviewer",
                role="Review and improve content quality",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=["Review all content", "Suggest improvements"]
            )
        ],
        model=OpenAIChat(id="gpt-4o"),
        instructions=["Create comprehensive synthesis", "Maintain high quality standards"]
    )
    
    return research_team, analysis_team, synthesis_team


async def adaptive_research_workflow():
    """Execute a research workflow with adaptive topology management"""
    
    logger.info("=== Adaptive Research Workflow ===")
    
    # Create workflow manager
    workflow_manager = AdaptiveWorkflowManager()
    
    # Create specialized teams
    research_team, analysis_team, synthesis_team = create_specialized_teams()
    
    # Register teams with different characteristics
    workflow_manager.register_workflow_team(
        research_team, 
        "research_phase",
        complexity=0.6,  # Moderate complexity
        urgency=0.7      # High urgency for research
    )
    
    workflow_manager.register_workflow_team(
        analysis_team,
        "analysis_phase", 
        complexity=0.9,  # High complexity
        urgency=0.5      # Moderate urgency
    )
    
    workflow_manager.register_workflow_team(
        synthesis_team,
        "synthesis_phase",
        complexity=0.7,  # Moderate-high complexity
        urgency=0.8      # High urgency for delivery
    )
    
    # Start workflow management
    workflow_manager.start()
    
    # Define workflow phases with different requirements
    workflow_phases = [
        {
            "name": "initial_research",
            "teams": [research_team.team_id],
            "duration": 25,
            "complexity": 0.6,
            "urgency": 0.7,
            "description": "Initial research and data gathering"
        },
        {
            "name": "deep_analysis", 
            "teams": [analysis_team.team_id],
            "duration": 30,
            "complexity": 0.9,
            "urgency": 0.5,
            "description": "Deep analysis of collected data"
        },
        {
            "name": "parallel_processing",
            "teams": [research_team.team_id, analysis_team.team_id],
            "duration": 20,
            "complexity": 0.8,
            "urgency": 0.6,
            "description": "Parallel research and analysis"
        },
        {
            "name": "synthesis_preparation",
            "teams": [synthesis_team.team_id],
            "duration": 15,
            "complexity": 0.7,
            "urgency": 0.8,
            "description": "Prepare for final synthesis"
        },
        {
            "name": "final_synthesis",
            "teams": [synthesis_team.team_id],
            "duration": 20,
            "complexity": 0.8,
            "urgency": 0.9,
            "description": "Final synthesis and report generation"
        }
    ]
    
    # Execute workflow phases
    for i, phase in enumerate(workflow_phases):
        logger.info(f"\n🔄 Phase {i+1}: {phase['name'].upper()}")
        logger.info(f"   Description: {phase['description']}")
        logger.info(f"   Teams: {', '.join(phase['teams'])}")
        logger.info(f"   Duration: {phase['duration']}s")
        
        # Update all involved teams for this phase
        for team_id in phase['teams']:
            workflow_manager.update_workflow_phase(
                team_id=team_id,
                new_phase=phase['name'],
                complexity=phase['complexity'],
                urgency=phase['urgency']
            )
            
            # Simulate varying loads during different phases
            if phase['name'] == 'initial_research':
                # High load on research team
                workflow_manager.topology_manager.update_agent_loads(
                    team_id, {
                        "senior_researcher": 0.9,
                        "data_collector": 0.8,
                        "research_analyst": 0.6
                    }
                )
            elif phase['name'] == 'deep_analysis':
                # High load on analysis team
                workflow_manager.topology_manager.update_agent_loads(
                    team_id, {
                        "data_scientist": 0.9,
                        "pattern_analyst": 0.8,
                        "validation_expert": 0.7
                    }
                )
            elif phase['name'] == 'final_synthesis':
                # High load on synthesis team
                workflow_manager.topology_manager.update_agent_loads(
                    team_id, {
                        "content_synthesizer": 0.9,
                        "technical_writer": 0.8,
                        "quality_reviewer": 0.7
                    }
                )
        
        # Monitor topology during phase execution
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < phase['duration']:
            # Log current topology status
            for team_id in phase['teams']:
                topology = workflow_manager.topology_manager.get_current_topology(team_id)
                performance = workflow_manager.topology_manager.get_performance_metrics(team_id)
                
                if topology and performance:
                    logger.debug(f"   {team_id}: {topology.topology_type.value} "
                               f"(score: {performance.overall_score():.3f})")
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        logger.info(f"✅ Phase {phase['name']} completed")
    
    # Final workflow summary
    logger.info("\n📊 WORKFLOW SUMMARY")
    
    total_adaptations = 0
    for team_id, state in workflow_manager.workflow_state.items():
        adaptations = state.get('adaptation_count', 0)
        total_adaptations += adaptations
        logger.info(f"  {team_id}: {adaptations} adaptations")
        
        # Get final topology
        final_topology = workflow_manager.topology_manager.get_current_topology(team_id)
        final_performance = workflow_manager.topology_manager.get_performance_metrics(team_id)
        
        if final_topology and final_performance:
            logger.info(f"    Final topology: {final_topology.topology_type.value}")
            logger.info(f"    Final performance: {final_performance.overall_score():.3f}")
    
    logger.info(f"Total topology adaptations across workflow: {total_adaptations}")
    
    # Stop workflow management
    workflow_manager.stop()


if __name__ == "__main__":
    # Run the adaptive workflow demonstration
    asyncio.run(adaptive_research_workflow())
