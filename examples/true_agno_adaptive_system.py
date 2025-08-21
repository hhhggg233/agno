#!/usr/bin/env python3
"""
真正使用agno执行逻辑的自适应拓扑系统

这个版本真正调用agno的team.run()和agent.run()方法，
不再使用模拟执行，而是使用agno框架的实际执行能力。
"""

import asyncio
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional

# Add agno to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs', 'agno'))

# agno imports
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

# Topology imports
from agno.topology import TopologyManager, TopologyType
from agno.topology.types import AgentCapability, TaskState

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrueAgnoAdaptiveSystem:
    """真正使用agno执行逻辑的自适应拓扑系统"""
    
    def __init__(self):
        self.topology_manager = None
    
    def analyze_requirement(self, requirement: str) -> Dict[str, Any]:
        """分析用户需求"""
        req_lower = requirement.lower()
        
        # 分析复杂度
        complexity = 0.5
        if any(word in req_lower for word in ['简单', 'simple', 'basic', '快速']):
            complexity = 0.3
        elif any(word in req_lower for word in ['复杂', 'complex', '详细', 'detailed', '深入']):
            complexity = 0.8
        elif any(word in req_lower for word in ['全面', 'comprehensive', '系统', 'systematic']):
            complexity = 0.9
        
        # 分析任务类型
        task_types = []
        if any(word in req_lower for word in ['研究', 'research', '调研', '分析', 'analyze']):
            task_types.append('research')
        if any(word in req_lower for word in ['写', 'write', '撰写', '文档', 'document']):
            task_types.append('writing')
        if any(word in req_lower for word in ['创作', 'create', '设计', 'design']):
            task_types.append('creative')
        if any(word in req_lower for word in ['技术', 'technical', '开发', 'develop']):
            task_types.append('technical')
        
        if not task_types:
            task_types = ['research']
        
        # 确定智能体数量和拓扑
        agent_count = min(2 + len(task_types), 4)
        
        if complexity < 0.4:
            topology = TopologyType.CENTRALIZED
        elif complexity < 0.7:
            topology = TopologyType.HIERARCHICAL
        else:
            topology = TopologyType.DECENTRALIZED
        
        return {
            'complexity': complexity,
            'task_types': task_types,
            'agent_count': agent_count,
            'initial_topology': topology
        }
    
    def create_specialized_agents(self, analysis: Dict[str, Any]) -> List[Agent]:
        """创建专业化的智能体"""
        agents = []
        
        # 研究员 - 总是包含
        researcher = Agent(
            name="Senior Researcher",
            agent_id="researcher_001",
            role="Conduct comprehensive research and gather information",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[DuckDuckGoTools()],
            instructions=[
                "You are a senior researcher with expertise in information gathering and analysis.",
                "Conduct thorough research using available tools and provide comprehensive insights.",
                "Always cite sources and verify information accuracy.",
                "Focus on finding the most current and relevant information."
            ]
        )
        agents.append(researcher)
        
        # 根据任务类型添加专门智能体
        if 'writing' in analysis['task_types']:
            writer = Agent(
                name="Technical Writer",
                agent_id="writer_001",
                role="Create clear, well-structured content and documentation",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "You are a professional technical writer specializing in clear communication.",
                    "Create well-structured, readable content that serves the intended audience.",
                    "Ensure accuracy, clarity, and logical flow in all written materials.",
                    "Adapt writing style based on the target audience and purpose."
                ]
            )
            agents.append(writer)
        
        if 'creative' in analysis['task_types']:
            creative = Agent(
                name="Creative Strategist",
                agent_id="creative_001",
                role="Generate innovative ideas and creative solutions",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "You are a creative strategist with expertise in innovation and design thinking.",
                    "Generate original, practical, and innovative solutions to problems.",
                    "Think outside the box while maintaining feasibility and user focus.",
                    "Provide multiple creative alternatives and explain their benefits."
                ]
            )
            agents.append(creative)
        
        if 'technical' in analysis['task_types']:
            technical = Agent(
                name="Technical Specialist",
                agent_id="technical_001",
                role="Provide technical expertise and implementation guidance",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "You are a technical specialist with deep expertise in technology implementation.",
                    "Analyze technical requirements and provide feasible solutions.",
                    "Consider scalability, security, and best practices in all recommendations.",
                    "Explain technical concepts clearly for both technical and non-technical audiences."
                ]
            )
            agents.append(technical)
        
        # 如果复杂度高，添加分析师
        if analysis['complexity'] > 0.6:
            analyst = Agent(
                name="Strategic Analyst",
                agent_id="analyst_001",
                role="Analyze data, identify patterns, and provide strategic insights",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "You are a strategic analyst with expertise in data analysis and pattern recognition.",
                    "Analyze complex information to identify key insights and trends.",
                    "Provide actionable recommendations based on thorough analysis.",
                    "Present findings in a clear, structured manner with supporting evidence."
                ]
            )
            agents.append(analyst)
        
        return agents
    
    async def execute_requirement(self, requirement: str) -> Dict[str, Any]:
        """执行用户需求 - 使用真正的agno执行逻辑"""
        
        print(f"\n🎯 Processing: {requirement}")
        print("=" * 60)
        
        # 1. 分析需求
        print("📋 Step 1: Analyzing requirement...")
        analysis = self.analyze_requirement(requirement)
        
        print(f"   Complexity: {analysis['complexity']:.2f}")
        print(f"   Task types: {', '.join(analysis['task_types'])}")
        print(f"   Initial topology: {analysis['initial_topology'].value}")
        
        # 2. 创建智能体和团队
        print("\n🤖 Step 2: Creating specialized agents...")
        agents = self.create_specialized_agents(analysis)
        
        print(f"   Created {len(agents)} agents:")
        for agent in agents:
            print(f"     - {agent.name}: {agent.role}")
        
        # 3. 创建团队
        print("\n🏗️ Step 3: Setting up team...")
        
        team = Team(
            name="Adaptive Research Team",
            team_id=f"adaptive_team_{int(time.time())}",
            mode="coordinate",  # 协调模式
            members=agents,
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "You are working as a coordinated team to address user requirements.",
                "Each team member should contribute their expertise to deliver comprehensive results.",
                "Collaborate effectively and build upon each other's contributions.",
                "Ensure the final output is complete, accurate, and well-structured."
            ]
        )
        
        # 4. 设置拓扑管理
        print("\n🔗 Step 4: Setting up adaptive topology...")
        
        self.topology_manager = TopologyManager()
        self.topology_manager.start()
        
        success = self.topology_manager.register_team(
            team=team,
            initial_task_description=requirement,
            task_complexity=analysis['complexity'],
            task_urgency=0.5
        )
        
        if not success:
            raise RuntimeError("Failed to register team with topology manager")
        
        print(f"   Team registered with {analysis['initial_topology'].value} topology")
        
        # 5. 执行任务 - 使用真正的agno执行
        print("\n🚀 Step 5: Executing with agno framework...")
        
        start_time = time.time()
        
        try:
            # 更新任务状态
            self.topology_manager.update_task_state(
                team_id=team.team_id,
                phase="execution",
                complexity=analysis['complexity'],
                urgency=0.5
            )
            
            # 构建详细的任务提示
            task_prompt = f"""
用户需求: {requirement}

任务分析:
- 复杂度: {analysis['complexity']:.2f}
- 任务类型: {', '.join(analysis['task_types'])}
- 团队规模: {len(agents)} 个专业智能体

团队成员及职责:
"""
            for agent in agents:
                task_prompt += f"- {agent.name}: {agent.role}\n"
            
            task_prompt += f"""
请作为一个协调的团队来完成这个需求。每个成员应该:
1. 发挥自己的专业优势
2. 与其他成员协作
3. 确保输出的完整性和质量

最终请提供一个综合的、高质量的回答来满足用户需求。
"""
            
            print("     🤖 Team executing with agno framework...")
            
            # 真正使用agno的team.run()方法
            team_response = team.run(
                message=task_prompt,
                stream=False  # 不使用流式输出以便获取完整结果
            )
            
            # 提取结果
            if hasattr(team_response, 'content'):
                result = team_response.content
            elif hasattr(team_response, 'message'):
                result = team_response.message
            else:
                result = str(team_response)
            
            execution_time = time.time() - start_time
            
            # 获取拓扑统计
            current_topology = self.topology_manager.get_current_topology(team.team_id)
            performance_metrics = self.topology_manager.get_performance_metrics(team.team_id)
            
            print(f"\n✅ Task completed in {execution_time:.2f} seconds")
            print(f"   Final topology: {current_topology.topology_type.value if current_topology else 'Unknown'}")
            
            if performance_metrics:
                print(f"   Performance score: {performance_metrics.overall_score():.3f}")
            
            return {
                'requirement': requirement,
                'analysis': analysis,
                'agents': [{'name': a.name, 'role': a.role} for a in agents],
                'result': result,
                'execution_time': execution_time,
                'final_topology': current_topology.topology_type.value if current_topology else None,
                'performance_score': performance_metrics.overall_score() if performance_metrics else 0.0,
                'team_response': team_response,  # 保留原始响应
                'success': True
            }
            
        except Exception as e:
            print(f"\n❌ Task execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'requirement': requirement,
                'analysis': analysis,
                'error': str(e),
                'success': False
            }
        
        finally:
            # 清理
            if self.topology_manager:
                self.topology_manager.stop()
    
    async def execute_with_individual_agents(self, requirement: str, agents: List[Agent]) -> Dict[str, str]:
        """使用单个智能体执行（备选方案）"""
        results = {}
        
        for agent in agents:
            try:
                print(f"     🤖 {agent.name} working...")
                
                agent_prompt = f"""
作为{agent.role}，请处理以下用户需求：

{requirement}

请根据你的专业领域提供高质量的回答。
"""
                
                # 使用agent.run()方法
                agent_response = agent.run(
                    message=agent_prompt,
                    stream=False
                )
                
                if hasattr(agent_response, 'content'):
                    agent_result = agent_response.content
                elif hasattr(agent_response, 'message'):
                    agent_result = agent_response.message
                else:
                    agent_result = str(agent_response)
                
                results[agent.name] = agent_result
                print(f"     ✅ {agent.name} completed")
                
            except Exception as e:
                print(f"     ❌ {agent.name} failed: {e}")
                results[agent.name] = f"Error: {str(e)}"
        
        return results


async def main():
    """主函数"""
    
    print("🚀 True Agno Adaptive Topology System")
    print("=" * 50)
    print("This system uses REAL agno team.run() and agent.run() methods!")
    print("Enter your requirement and watch the system:")
    print("1. Analyze complexity and select specialized agents")
    print("2. Set up adaptive topology")
    print("3. Execute using agno framework")
    print("4. Provide real results from agno execution")
    print("=" * 50)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    system = TrueAgnoAdaptiveSystem()
    
    # 示例需求
    examples = [
        "研究人工智能在教育领域的最新应用和发展趋势",
        "设计一个帮助小企业管理客户关系的创新解决方案",
        "分析当前远程工作模式对企业生产力的影响",
        "撰写一份关于可持续能源技术的综合报告"
    ]
    
    while True:
        print("\n" + "=" * 50)
        print("💬 Enter your requirement:")
        print("   (or 'example' for demo, 'quit' to exit)")
        
        user_input = input(">>> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'example':
            import random
            user_input = random.choice(examples)
            print(f"Using example: {user_input}")
        
        if not user_input:
            print("❌ Please enter a valid requirement.")
            continue
        
        try:
            result = await system.execute_requirement(user_input)
            
            print("\n" + "=" * 60)
            print("🎉 AGNO EXECUTION COMPLETED")
            print("=" * 60)
            
            if result['success']:
                print(f"✅ Success: Real agno execution completed")
                print(f"📊 Performance: {result.get('performance_score', 0):.3f}")
                print(f"⏱️ Time: {result.get('execution_time', 0):.2f}s")
                print(f"🔗 Final topology: {result.get('final_topology', 'Unknown')}")
                print(f"🤖 Agents used: {len(result.get('agents', []))}")
                
                print(f"\n📋 AGNO TEAM RESULT:")
                print("-" * 40)
                print(result['result'])
                
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"\n❌ System error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
