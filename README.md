# recsys-universe

基于 Claude-style Skills 架构的多智能体电影推荐系统 Demo。

## 功能特点

- **Claude-style Skills 架构**: 使用 Skills 替代传统 LangGraph 编排
- **多智能体协作**: Planner、Profile、Content、Collab、Merge、Final 六个专业技能
- **本地 LLM 支持**: 兼容 OpenAI 接口，支持本地 VLLM 部署（Qwen3-1.7B）
- **中文 Prompt**: 全流程中文交互，丰富的调试日志

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export RECSYS_DEBUG="false"  # 可选，设为 "true" 开启调试模式
```

### 3. 启动 VLLM 服务

```bash
./run_server.sh
```

### 4. 运行示例

```python
from multiagents_movielens import run_recommendation

# 基本调用
recommendations = run_recommendation(
    user_id=123,
    query="我想看一点黑暗风格的科幻片，最好有一点赛博朋克的味道"
)

# 遍历结果
for movie in recommendations:
    print(f"{movie['title']} - {movie['reason']}")
```

## 项目结构

```
recsys-universe/
├── config.py                    # 统一配置管理
├── multiagents_movielens.py     # 主入口
├── skills_coordinator.py        # Skills 协调器
├── skills/
│   ├── base_skill.py            # BaseSkill 基类
│   ├── skill_registry.py        # Skill 注册表
│   ├── planner_skill.py         # 规划调度技能
│   ├── profile_skill.py         # 用户画像技能
│   ├── content_skill.py         # 内容检索技能
│   ├── collab_skill.py          # 协同过滤技能
│   ├── merge_skill.py           # 候选合并技能
│   ├── final_skill.py           # 最终推荐技能
│   └── data_utils.py            # 数据工具函数
├── tests/                       # 单元测试
│   ├── test_config.py
│   ├── test_data_utils.py
│   └── test_skills_coordinator.py
└── run_server.sh                # VLLM 启动脚本
```

## 配置说明

所有配置通过 `config.py` 管理，支持环境变量覆盖：

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `debug` | `RECSYS_DEBUG` | `false` | 调试模式 |
| `max_steps` | - | `10` | 最大规划步数 |
| `llm.api_key` | `OPENAI_API_KEY` | - | API Key |
| `llm.base_url` | `OPENAI_BASE_URL` | `http://localhost:8000/v1` | API 地址 |
| `llm.model` | - | `Qwen/Qwen3-1.7B` | 模型名称 |
| `llm.temperature` | - | `0.3` | LLM 温度 |

## 测试

```bash
pytest tests/ -v
```

## 改进日志

### v2.0.0 (本次更新)

- 新增统一配置管理 (`config.py`)
- 重构 LLM 客户端，支持连接缓存和复用
- 优化 Skills 协调器，使用动态调度替代 if-elif 链
- 改进数据工具层，添加缓存管理和 lru_cache
- 添加输入验证和类型检查
- 完善单元测试覆盖
- 移除硬编码的安全隐患
- 统一使用 logging 模块替代 print
