# 自适应拓扑系统安装和使用指南

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.8+
- **操作系统**: Linux, macOS, Windows
- **内存**: 至少 2GB 可用内存
- **GPU**: 可选，用于加速深度学习训练

### 2. 安装依赖

#### 方法一：使用远程服务器（推荐）

如果您的本地环境没有PyTorch，可以使用VSCode的Remote SSH插件连接到有PyTorch的服务器：

```bash
# 在远程服务器上
cd /path/to/agno
pip install -r libs/agno/requirements-topology.txt
```

#### 方法二：本地安装

```bash
# 安装PyTorch (根据您的系统选择)
# CPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU版本 (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r libs/agno/requirements-topology.txt

# 安装agno (开发模式)
cd libs/agno
pip install -e .
```

### 3. 验证安装

```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "from agno.topology.types import TopologyType; print('Topology types loaded successfully')"
```

## 📋 配置设置

### 1. 创建配置文件

```bash
cd /path/to/agno
python -c "from agno.topology.config import ConfigManager; ConfigManager().create_sample_config()"
```

这将创建 `topology_config_sample.json` 文件。

### 2. 自定义配置

将 `topology_config_sample.json` 重命名为 `topology_config.json` 并根据需要修改：

```json
{
  "environment": "production",
  "debug_mode": false,
  "log_level": "INFO",
  "max_agents_per_team": 10,
  "max_concurrent_teams": 5,
  "adaptation_config": {
    "min_efficiency_threshold": 0.65,
    "adaptation_cooldown": 60.0,
    "prefer_gnn": true,
    "enable_rl_learning": true
  }
}
```

### 3. 环境变量

设置必要的环境变量：

```bash
# OpenAI API密钥（如果使用OpenAI模型）
export OPENAI_API_KEY="your-api-key-here"

# 配置文件路径（可选）
export AGNO_TOPOLOGY_CONFIG="/path/to/topology_config.json"
```

## 🏃‍♂️ 运行示例

### 1. 生产环境示例

```bash
cd /path/to/agno
python examples/production_topology_example.py
```

这个示例展示了：
- ✅ 真实的性能监控
- ✅ 资源管理和限制
- ✅ 拓扑自动适应
- ✅ 错误处理和恢复
- ✅ 生产级日志记录

### 2. 基础集成示例

```python
import asyncio
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.topology.production_manager import ProductionTopologyManager

async def basic_example():
    # 创建智能体
    agent1 = Agent(
        name="Researcher",
        agent_id="researcher_001",
        model=OpenAIChat(id="gpt-4o-mini")
    )
    
    agent2 = Agent(
        name="Analyst", 
        agent_id="analyst_001",
        model=OpenAIChat(id="gpt-4o-mini")
    )
    
    # 创建团队
    team = Team(
        name="Research Team",
        team_id="team_001",
        members=[agent1, agent2],
        mode="coordinate"
    )
    
    # 创建拓扑管理器
    manager = ProductionTopologyManager()
    await manager.start()
    
    # 注册团队
    success = manager.register_team(
        team=team,
        task_description="Research task",
        complexity=0.6
    )
    
    if success:
        print("✅ Team registered successfully")
        
        # 执行任务
        async def research_task():
            # 这里放置您的实际任务逻辑
            await asyncio.sleep(2)
            return "Research completed"
        
        result = await manager.execute_task(
            team_id=team.team_id,
            task_description="AI research",
            task_function=research_task,
            complexity=0.7
        )
        
        print(f"Task result: {result}")
    
    await manager.stop()

# 运行示例
asyncio.run(basic_example())
```

## 🔧 高级配置

### 1. 自定义拓扑适应策略

```python
from agno.topology.adaptive import AdaptationConfig

# 创建自定义适应配置
adaptation_config = AdaptationConfig(
    min_efficiency_threshold=0.7,        # 更高的效率要求
    adaptation_cooldown=30.0,            # 更频繁的适应
    prefer_gnn=True,                     # 优先使用GNN
    enable_rl_learning=True,             # 启用强化学习
    max_communication_cost_threshold=0.6  # 更严格的通信成本控制
)
```

### 2. 自定义GNN配置

```python
from agno.topology.gnn_generator import GNNConfig

gnn_config = GNNConfig(
    hidden_dim=256,      # 更大的隐藏层
    num_layers=4,        # 更深的网络
    learning_rate=0.0005, # 更小的学习率
    dropout=0.2          # 更高的dropout
)
```

### 3. 自定义RL配置

```python
from agno.topology.rl_search import RLConfig

rl_config = RLConfig(
    state_dim=256,       # 更大的状态空间
    action_dim=128,      # 更大的动作空间
    epsilon=0.05,        # 更少的探索
    batch_size=64,       # 更大的批次
    buffer_size=50000    # 更大的经验回放缓冲区
)
```

## 📊 监控和调试

### 1. 启用详细日志

```python
import logging

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 或在配置文件中设置
{
  "log_level": "DEBUG",
  "debug_mode": true
}
```

### 2. 性能监控

系统会自动收集以下指标：
- CPU和内存使用率
- 任务完成时间
- 拓扑适应频率
- 通信成本
- 负载平衡度

监控数据保存在 `./models/topology/metrics.json`

### 3. 模型检查点

模型会自动保存检查点：
- GNN模型: `./models/topology/{team_id}/gnn_checkpoint_{timestamp}.pt`
- RL模型: `./models/topology/{team_id}/rl_checkpoint_{timestamp}.pt`

## 🚨 故障排除

### 1. 常见问题

**问题**: `ImportError: No module named 'torch'`
**解决**: 安装PyTorch
```bash
pip install torch torchvision torchaudio
```

**问题**: `CUDA out of memory`
**解决**: 减少批次大小或使用CPU
```python
# 在配置中设置
{
  "gnn_config": {
    "batch_size": 16  # 减少批次大小
  }
}
```

**问题**: 拓扑适应过于频繁
**解决**: 增加冷却时间
```python
{
  "adaptation_config": {
    "adaptation_cooldown": 120.0  # 2分钟冷却
  }
}
```

### 2. 性能优化

**内存优化**:
```python
{
  "max_memory_usage_mb": 1024,  # 限制内存使用
  "performance_history_size": 500  # 减少历史记录
}
```

**CPU优化**:
```python
{
  "max_cpu_usage_percent": 70.0,  # 限制CPU使用
  "adaptation_check_interval": 60.0  # 减少检查频率
}
```

## 🔗 集成到现有项目

### 1. 最小集成

```python
from agno.topology.manager import TopologyManager

# 在现有团队中启用自适应拓扑
topology_manager = TopologyManager()
topology_manager.register_team(your_existing_team)
topology_manager.start()

# 在任务执行前更新状态
topology_manager.update_task_state(
    team_id=your_team.team_id,
    complexity=task_complexity,
    urgency=task_urgency
)
```

### 2. 完整集成

参考 `examples/production_topology_example.py` 中的完整示例。

## 📚 API参考

详细的API文档请参考：
- [TopologyManager API](libs/agno/agno/topology/manager.py)
- [ProductionTopologyManager API](libs/agno/agno/topology/production_manager.py)
- [配置选项](libs/agno/agno/topology/config.py)

## 🤝 贡献

如果您想为自适应拓扑系统做出贡献：

1. Fork 项目
2. 创建功能分支
3. 添加测试
4. 提交 Pull Request

## 📄 许可证

本项目遵循 agno 框架的许可证条款。
