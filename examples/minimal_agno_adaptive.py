#!/usr/bin/env python3
"""
最小化的agno自适应拓扑系统

专注于展示真正的agno执行逻辑，去除复杂的分析和管理，
直接使用team.run()和agent.run()方法。
"""

import asyncio
import os
import sys
import time

# Add agno to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'libs', 'agno'))

# agno imports
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools


def create_research_team() -> Team:
    """创建一个简单的研究团队"""
    
    # 研究员
    researcher = Agent(
        name="Researcher",
        agent_id="researcher_001",
        role="Research and gather information",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        instructions=[
            "You are a professional researcher.",
            "Conduct thorough research and provide accurate information.",
            "Use web search tools when needed to find current information."
        ]
    )
    
    # 分析师
    analyst = Agent(
        name="Analyst",
        agent_id="analyst_001",
        role="Analyze information and provide insights",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "You are a data analyst and strategic thinker.",
            "Analyze information to identify patterns and insights.",
            "Provide clear, actionable recommendations."
        ]
    )
    
    # 撰写员
    writer = Agent(
        name="Writer",
        agent_id="writer_001",
        role="Create clear, well-structured content",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "You are a professional writer and communicator.",
            "Create clear, engaging, and well-structured content.",
            "Ensure information is accessible to the target audience."
        ]
    )
    
    # 创建团队
    team = Team(
        name="Research Team",
        team_id="research_team_001",
        mode="coordinate",
        members=[researcher, analyst, writer],
        model=OpenAIChat(id="gpt-4o"),
        instructions=[
            "Work together as a coordinated team.",
            "Each member should contribute their expertise.",
            "Deliver comprehensive, high-quality results.",
            "Build upon each other's work to create the best possible output."
        ]
    )
    
    return team


def execute_with_team(team: Team, requirement: str) -> str:
    """使用团队执行任务 - 真正的agno执行"""
    
    print(f"🤖 Team executing: {requirement}")
    print("   Using agno team.run() method...")
    
    # 构建任务提示
    task_prompt = f"""
用户需求: {requirement}

请作为一个协调的团队来完成这个需求。

团队成员:
- Researcher: 负责信息研究和收集
- Analyst: 负责数据分析和洞察
- Writer: 负责内容撰写和整理

请协作提供一个完整、专业的回答。
"""
    
    try:
        # 真正使用agno的team.run()方法
        start_time = time.time()
        
        response = team.run(
            message=task_prompt,
            stream=False
        )
        
        execution_time = time.time() - start_time
        
        # 提取结果
        if hasattr(response, 'content'):
            result = response.content
        elif hasattr(response, 'message'):
            result = response.message
        else:
            result = str(response)
        
        print(f"✅ Team execution completed in {execution_time:.2f}s")
        return result
        
    except Exception as e:
        print(f"❌ Team execution failed: {e}")
        return f"Team execution failed: {str(e)}"


def execute_with_individual_agents(team: Team, requirement: str) -> str:
    """使用单个智能体执行任务"""
    
    print(f"🤖 Individual agents executing: {requirement}")
    
    results = []
    
    for agent in team.members:
        try:
            print(f"   {agent.name} working...")
            
            agent_prompt = f"""
作为{agent.role}，请处理以下需求：

{requirement}

请根据你的专业领域提供高质量的回答。
"""
            
            # 使用agent.run()方法
            start_time = time.time()
            
            response = agent.run(
                message=agent_prompt,
                stream=False
            )
            
            execution_time = time.time() - start_time
            
            # 提取结果
            if hasattr(response, 'content'):
                agent_result = response.content
            elif hasattr(response, 'message'):
                agent_result = response.message
            else:
                agent_result = str(response)
            
            results.append(f"**{agent.name}的贡献:**\n{agent_result}")
            print(f"   ✅ {agent.name} completed in {execution_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ {agent.name} failed: {e}")
            results.append(f"**{agent.name}的贡献:**\n执行失败: {str(e)}")
    
    # 整合结果
    final_result = f"# 团队协作结果\n\n" + "\n\n".join(results)
    return final_result


async def main():
    """主函数"""
    
    print("🚀 Minimal Agno Adaptive System")
    print("=" * 50)
    print("This system demonstrates REAL agno execution:")
    print("1. Uses actual team.run() method")
    print("2. Uses actual agent.run() methods")
    print("3. No simulation - real agno framework execution")
    print("=" * 50)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # 创建团队
    print("\n🏗️ Creating research team...")
    team = create_research_team()
    
    print(f"✅ Team created with {len(team.members)} members:")
    for member in team.members:
        print(f"   - {member.name}: {member.role}")
    
    # 示例需求
    examples = [
        "研究人工智能在医疗诊断中的应用",
        "分析电动汽车市场的发展趋势",
        "撰写一份关于远程工作的综合报告",
        "设计一个环保的城市交通解决方案"
    ]
    
    while True:
        print("\n" + "=" * 50)
        print("💬 Choose execution mode:")
        print("1. Team execution (team.run())")
        print("2. Individual agents (agent.run())")
        print("3. Enter custom requirement")
        print("4. Use example requirement")
        print("5. Quit")
        
        choice = input(">>> ").strip()
        
        if choice == '5' or choice.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        # 获取需求
        if choice == '4':
            import random
            requirement = random.choice(examples)
            print(f"Using example: {requirement}")
        elif choice == '3':
            requirement = input("Enter your requirement: ").strip()
            if not requirement:
                print("❌ Please enter a valid requirement.")
                continue
        else:
            # 使用默认需求
            requirement = "研究人工智能在教育领域的应用和发展前景"
            print(f"Using default: {requirement}")
        
        print(f"\n🎯 Processing: {requirement}")
        print("-" * 50)
        
        try:
            if choice == '1':
                # 团队执行
                print("📋 Mode: Team Execution (team.run())")
                result = execute_with_team(team, requirement)
                
            elif choice == '2':
                # 单个智能体执行
                print("📋 Mode: Individual Agents (agent.run())")
                result = execute_with_individual_agents(team, requirement)
                
            else:
                # 默认使用团队执行
                print("📋 Mode: Team Execution (team.run())")
                result = execute_with_team(team, requirement)
            
            print("\n" + "=" * 60)
            print("🎉 AGNO EXECUTION COMPLETED")
            print("=" * 60)
            print("📋 RESULT:")
            print("-" * 30)
            print(result)
            
        except Exception as e:
            print(f"\n❌ Execution failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
