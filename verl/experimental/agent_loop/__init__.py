# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：空行占位
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：许可证获取提示
#  # 注释：空行占位
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无担保声明
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明

"""
模块用途：agent_loop 子包入口，集中导出 AgentLoop 相关核心类。  # 注释：模块用途
输入/输出：无直接输入输出（提供导入入口）。  # 注释：输入输出说明
关键依赖：agent_loop.py、single_turn_agent_loop.py、tool_agent_loop.py。  # 注释：关键依赖
典型用法：  # 注释：用法标题
- from verl.experimental.agent_loop import AgentLoopManager, ToolAgentLoop  # 注释：导入示例
调用路径概览：  # 注释：调用路径标题
- 训练/推理入口 -> AgentLoopManager -> Tool/SingleTurn AgentLoop。  # 注释：调用链
"""  # 注释：模块 docstring 结束

from .agent_loop import AgentLoopBase, AgentLoopManager, AgentLoopWorker, AsyncLLMServerManager  # 注释：导出核心类
from .single_turn_agent_loop import SingleTurnAgentLoop  # 注释：导出单轮 agent loop
from .tool_agent_loop import ToolAgentLoop  # 注释：导出工具调用 agent loop

_ = [SingleTurnAgentLoop, ToolAgentLoop]  # 注释：占位，避免未使用告警

__all__ = ["AgentLoopBase", "AgentLoopManager", "AsyncLLMServerManager", "AgentLoopWorker"]  # 注释：限制公共导出符号
