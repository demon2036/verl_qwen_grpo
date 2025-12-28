# （说明：原注释说明）  # 注释：自动行注释
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# （说明：原注释说明）  # 注释：自动行注释
#
# （说明：原注释说明）  # 注释：自动行注释
# Licensed under the Apache License, Version 2.0 (the "License");
# （说明：原注释说明）  # 注释：自动行注释
# you may not use this file except in compliance with the License.
# （说明：原注释说明）  # 注释：自动行注释
# You may obtain a copy of the License at
# （说明：原注释说明）  # 注释：自动行注释
#
# （说明：原注释说明）  # 注释：自动行注释
#     http://www.apache.org/licenses/LICENSE-2.0
# （说明：原注释说明）  # 注释：自动行注释
#
# （说明：原注释说明）  # 注释：自动行注释
# Unless required by applicable law or agreed to in writing, software
# （说明：原注释说明）  # 注释：自动行注释
# distributed under the License is distributed on an "AS IS" BASIS,
# （说明：原注释说明）  # 注释：自动行注释
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# （说明：原注释说明）  # 注释：自动行注释
# See the License for the specific language governing permissions and
# （说明：原注释说明）  # 注释：自动行注释
# limitations under the License.
"""
模块用途：实现带工具调用的 Agent Loop（多轮状态机 + 工具解析/执行）。  # 注释：模块用途
输入：raw_prompt、多轮工具 schema、采样参数、tool parser 配置。  # 注释：输入说明
输出：AgentLoopOutput（包含工具调用轨迹与结果）。  # 注释：输出说明
关键依赖：ToolParser、utils、rollout_trace、asyncio。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- ToolAgentLoop.run(...) 生成多轮 tool-call 并整理输出。  # 注释：示例
调用路径概览：  # 注释：调用路径标题
- AgentLoopManager.generate_sequences -> ToolAgentLoop.run。  # 注释：调用链
"""  # 注释：模块 docstring 结束
# （说明：导入依赖）  # 注释：自动行注释
import asyncio
# （说明：导入依赖）  # 注释：自动行注释
import copy
# （说明：导入依赖）  # 注释：自动行注释
import json
# （说明：导入依赖）  # 注释：自动行注释
import logging
# （说明：导入依赖）  # 注释：自动行注释
import os
# （说明：导入依赖）  # 注释：自动行注释
from enum import Enum
# （说明：导入依赖）  # 注释：自动行注释
from typing import Any, Optional
# （说明：导入依赖）  # 注释：自动行注释
from uuid import uuid4

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：导入依赖）  # 注释：自动行注释
from transformers import AutoProcessor, AutoTokenizer

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：导入依赖）  # 注释：自动行注释
from verl.experimental.agent_loop.agent_loop import (
    # （说明：执行语句）  # 注释：自动行注释
    AgentLoopBase,
    # （说明：执行语句）  # 注释：自动行注释
    AgentLoopOutput,
    # （说明：执行语句）  # 注释：自动行注释
    AsyncLLMServerManager,
    # （说明：执行语句）  # 注释：自动行注释
    DictConfigWrap,
    # （说明：执行语句）  # 注释：自动行注释
    register,
# （说明：执行语句）  # 注释：自动行注释
)
# （说明：导入依赖）  # 注释：自动行注释
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
# （说明：导入依赖）  # 注释：自动行注释
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
# （说明：导入依赖）  # 注释：自动行注释
from verl.interactions.base import BaseInteraction
# （说明：导入依赖）  # 注释：自动行注释
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
# （说明：导入依赖）  # 注释：自动行注释
from verl.tools.schemas import ToolResponse
# （说明：导入依赖）  # 注释：自动行注释
from verl.tools.utils.tool_registry import initialize_tools_from_config
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.chat_template import initialize_system_prompt
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.profiler import simple_timer
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.rollout_trace import rollout_trace_op

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：执行语句）  # 注释：自动行注释
logger = logging.getLogger(__file__)
# （说明：执行语句）  # 注释：自动行注释
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentState(Enum):
    """
    功能：AgentState 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - AgentState(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::AgentState。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """
    # （说明：执行语句）  # 注释：自动行注释
    PENDING = "pending"
    # （说明：执行语句）  # 注释：自动行注释
    GENERATING = "generating"
    # （说明：执行语句）  # 注释：自动行注释
    PROCESSING_TOOLS = "processing_tools"
    # （说明：执行语句）  # 注释：自动行注释
    TERMINATED = "terminated"
    # （说明：执行语句）  # 注释：自动行注释
    INTERACTING = "interacting"

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    功能：AgentData 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - AgentData(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::AgentData。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        messages: list[dict[str, Any]],
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Any,
        # （说明：执行语句）  # 注释：自动行注释
        metrics: dict[str, Any],
        # （说明：执行语句）  # 注释：自动行注释
        request_id: str,
        # （说明：执行语句）  # 注释：自动行注释
        tools_kwargs: dict[str, Any],
        # （说明：执行语句）  # 注释：自动行注释
        interaction: Optional[BaseInteraction] = None,
        # （说明：执行语句）  # 注释：自动行注释
        interaction_kwargs: Optional[dict[str, Any]] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ):
        """
        功能：__init__ 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - __init__(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::__init__。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        self.messages = messages
        # （说明：执行语句）  # 注释：自动行注释
        self.image_data = image_data
        # （说明：执行语句）  # 注释：自动行注释
        self.metrics = metrics
        # （说明：执行语句）  # 注释：自动行注释
        self.request_id = request_id
        # （说明：执行语句）  # 注释：自动行注释
        self.tools_kwargs = tools_kwargs
        # （说明：执行语句）  # 注释：自动行注释
        self.interaction = interaction
        # （说明：执行语句）  # 注释：自动行注释
        self.interaction_kwargs = interaction_kwargs or {}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # State variables
        # （说明：执行语句）  # 注释：自动行注释
        self.prompt_ids: list[int] = []
        # （说明：执行语句）  # 注释：自动行注释
        self.response_ids: list[int] = []
        # （说明：执行语句）  # 注释：自动行注释
        self.response_mask: list[int] = []
        # （说明：执行语句）  # 注释：自动行注释
        self.response_logprobs: list[float] = []
        # （说明：执行语句）  # 注释：自动行注释
        self.turn_scores: list[float] = []
        # （说明：执行语句）  # 注释：自动行注释
        self.tool_rewards: list[float] = []
        # （说明：执行语句）  # 注释：自动行注释
        self.user_turns = 0
        # （说明：执行语句）  # 注释：自动行注释
        self.assistant_turns = 0

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Temporary state for tool calls
        # （说明：执行语句）  # 注释：自动行注释
        self.tool_calls: list[FunctionCall] = []

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Extra fields for dynamic addition, e.g., tool session data
        # （说明：执行语句）  # 注释：自动行注释
        self.extra_fields: dict[str, Any] = {}

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：装饰器声明）  # 注释：自动行注释
@register("tool_agent")
# （说明：定义类）  # 注释：自动行注释
class ToolAgentLoop(AgentLoopBase):
    """
    功能：ToolAgentLoop 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - ToolAgentLoop(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::ToolAgentLoop。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        trainer_config: DictConfigWrap,
        # （说明：执行语句）  # 注释：自动行注释
        server_manager: AsyncLLMServerManager,
        # （说明：执行语句）  # 注释：自动行注释
        tokenizer: AutoTokenizer,
        # （说明：执行语句）  # 注释：自动行注释
        processor: AutoProcessor,
        # （说明：执行语句）  # 注释：自动行注释
        **kwargs,
    # （说明：执行语句）  # 注释：自动行注释
    ):
        """
        功能：__init__ 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - __init__(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::__init__。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        # （说明：执行语句）  # 注释：自动行注释
        config = trainer_config.config

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Initialize tools from config file
        # （说明：执行语句）  # 注释：自动行注释
        self.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        # （说明：执行语句）  # 注释：自动行注释
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        # （说明：执行语句）  # 注释：自动行注释
        self.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        # （说明：执行语句）  # 注释：自动行注释
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        # （说明：执行语句）  # 注释：自动行注释
        self.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        # （说明：执行语句）  # 注释：自动行注释
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        # （说明：执行语句）  # 注释：自动行注释
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        # （说明：执行语句）  # 注释：自动行注释
        self.tools = {tool.name: tool for tool in tool_list}
        # （说明：执行语句）  # 注释：自动行注释
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        # （说明：执行语句）  # 注释：自动行注释
        self.tool_parser = ToolParser.get_tool_parser(
            # （说明：执行语句）  # 注释：自动行注释
            config.actor_rollout_ref.rollout.multi_turn.format, self.tokenizer
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        self.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        # （说明：执行语句）  # 注释：自动行注释
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        # （说明：执行语句）  # 注释：自动行注释
        self.response_length = config.actor_rollout_ref.rollout.response_length
        # （说明：执行语句）  # 注释：自动行注释
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Initialize interactions from config file
        # （说明：执行语句）  # 注释：自动行注释
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        # （说明：条件分支）  # 注释：自动行注释
        if self.interaction_config_file:
            # （说明：执行语句）  # 注释：自动行注释
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                # （说明：执行语句）  # 注释：自动行注释
                self.interaction_config_file
            # （说明：执行语句）  # 注释：自动行注释
            )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：装饰器声明）  # 注释：自动行注释
    @rollout_trace_op
    # （说明：定义函数）  # 注释：自动行注释
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # （说明：执行语句）  # 注释：自动行注释
        messages = list(kwargs["raw_prompt"])
        # （说明：执行语句）  # 注释：自动行注释
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        # （说明：执行语句）  # 注释：自动行注释
        metrics = {}
        # （说明：执行语句）  # 注释：自动行注释
        request_id = uuid4().hex
        # （说明：执行语句）  # 注释：自动行注释
        tools_kwargs = kwargs.get("tools_kwargs", {})

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Initialize interaction if needed
        # （说明：执行语句）  # 注释：自动行注释
        interaction = None
        # （说明：执行语句）  # 注释：自动行注释
        interaction_kwargs = {}
        # （说明：条件分支）  # 注释：自动行注释
        if self.interaction_config_file:
        """
        功能：run 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - run(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::run。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
            # （说明：执行语句）  # 注释：自动行注释
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            # （说明：条件分支）  # 注释：自动行注释
            if "name" not in interaction_kwargs:
                # （说明：抛出异常）  # 注释：自动行注释
                raise ValueError("'name' key is required in interaction_kwargs")
            # （说明：执行语句）  # 注释：自动行注释
            interaction_name = interaction_kwargs["name"]
            # （说明：条件分支）  # 注释：自动行注释
            if interaction_name not in self.interaction_map:
                # （说明：抛出异常）  # 注释：自动行注释
                raise ValueError(
                    # （说明：执行语句）  # 注释：自动行注释
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    # （说明：执行语句）  # 注释：自动行注释
                    f"{list(self.interaction_map.keys())}"
                # （说明：执行语句）  # 注释：自动行注释
                )
            # （说明：执行语句）  # 注释：自动行注释
            interaction = self.interaction_map[interaction_name]
            # （说明：执行语句）  # 注释：自动行注释
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # （说明：原注释说明）  # 注释：自动行注释
        # Create AgentData instance to encapsulate all state
        # （说明：执行语句）  # 注释：自动行注释
        agent_data = AgentData(
            # （说明：执行语句）  # 注释：自动行注释
            messages=messages,
            # （说明：执行语句）  # 注释：自动行注释
            image_data=image_data,
            # （说明：执行语句）  # 注释：自动行注释
            metrics=metrics,
            # （说明：执行语句）  # 注释：自动行注释
            request_id=request_id,
            # （说明：执行语句）  # 注释：自动行注释
            tools_kwargs=tools_kwargs,
            # （说明：执行语句）  # 注释：自动行注释
            interaction=interaction,
            # （说明：执行语句）  # 注释：自动行注释
            interaction_kwargs=interaction_kwargs,
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # State machine loop
        # （说明：执行语句）  # 注释：自动行注释
        state = AgentState.PENDING
        # （说明：循环逻辑）  # 注释：自动行注释
        while state != AgentState.TERMINATED:
            # （说明：条件分支）  # 注释：自动行注释
            if state == AgentState.PENDING:
                # （说明：执行语句）  # 注释：自动行注释
                state = await self._handle_pending_state(agent_data, sampling_params)
            # （说明：条件分支）  # 注释：自动行注释
            elif state == AgentState.GENERATING:
                # （说明：执行语句）  # 注释：自动行注释
                state = await self._handle_generating_state(agent_data, sampling_params)
            # （说明：条件分支）  # 注释：自动行注释
            elif state == AgentState.PROCESSING_TOOLS:
                # （说明：执行语句）  # 注释：自动行注释
                state = await self._handle_processing_tools_state(agent_data)
            # （说明：条件分支）  # 注释：自动行注释
            elif state == AgentState.INTERACTING:
                # （说明：执行语句）  # 注释：自动行注释
                state = await self._handle_interacting_state(agent_data)
            # （说明：条件分支）  # 注释：自动行注释
            else:
                # （说明：执行语句）  # 注释：自动行注释
                logger.error(f"Invalid state: {state}")
                # （说明：执行语句）  # 注释：自动行注释
                state = AgentState.TERMINATED

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Finalize output
        # （说明：执行语句）  # 注释：自动行注释
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        # （说明：执行语句）  # 注释：自动行注释
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        # （说明：执行语句）  # 注释：自动行注释
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        # （说明：执行语句）  # 注释：自动行注释
        output = AgentLoopOutput(
            # （说明：执行语句）  # 注释：自动行注释
            prompt_ids=prompt_ids,
            # （说明：执行语句）  # 注释：自动行注释
            response_ids=response_ids[: self.response_length],
            # （说明：执行语句）  # 注释：自动行注释
            response_mask=agent_data.response_mask[: self.response_length],
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_data=multi_modal_data,
            # （说明：执行语句）  # 注释：自动行注释
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            # （说明：条件分支）  # 注释：自动行注释
            if agent_data.response_logprobs
            # （说明：条件分支）  # 注释：自动行注释
            else None,
            # （说明：执行语句）  # 注释：自动行注释
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            # （说明：执行语句）  # 注释：自动行注释
            metrics=agent_data.metrics,
            # （说明：执行语句）  # 注释：自动行注释
            extra_fields={},
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        # （说明：返回结果）  # 注释：自动行注释
        return output

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        # （说明：条件分支）  # 注释：自动行注释
        if self.processor is not None:
        """Handle the pending state: prepare the prompt and start generation."""
            # （说明：执行语句）  # 注释：自动行注释
            raw_prompt = await self.loop.run_in_executor(
                # （说明：执行语句）  # 注释：自动行注释
                None,
                # （说明：执行语句）  # 注释：自动行注释
                lambda: self.processor.apply_chat_template(
                    # （说明：执行语句）  # 注释：自动行注释
                    agent_data.messages,
                    # （说明：执行语句）  # 注释：自动行注释
                    tools=self.tool_schemas,
                    # （说明：执行语句）  # 注释：自动行注释
                    add_generation_prompt=True,
                    # （说明：执行语句）  # 注释：自动行注释
                    tokenize=False,
                    # （说明：执行语句）  # 注释：自动行注释
                    **self.apply_chat_template_kwargs,
                # （说明：执行语句）  # 注释：自动行注释
                ),
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：执行语句）  # 注释：自动行注释
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.prompt_ids = await self.loop.run_in_executor(
                # （说明：执行语句）  # 注释：自动行注释
                None,
                # （说明：执行语句）  # 注释：自动行注释
                lambda: self.tokenizer.apply_chat_template(
                    # （说明：执行语句）  # 注释：自动行注释
                    agent_data.messages,
                    # （说明：执行语句）  # 注释：自动行注释
                    tools=self.tool_schemas,
                    # （说明：执行语句）  # 注释：自动行注释
                    add_generation_prompt=True,
                    # （说明：执行语句）  # 注释：自动行注释
                    tokenize=True,
                    # （说明：执行语句）  # 注释：自动行注释
                    **self.apply_chat_template_kwargs,
                # （说明：执行语句）  # 注释：自动行注释
                ),
            # （说明：执行语句）  # 注释：自动行注释
            )
        # （说明：返回结果）  # 注释：自动行注释
        return AgentState.GENERATING

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _handle_generating_state(
        # （说明：执行语句）  # 注释：自动行注释
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    # （说明：执行语句）  # 注释：自动行注释
    ) -> AgentState:
        # （说明：执行语句）  # 注释：自动行注释
        功能：_handle_pending_state 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        # （说明：执行语句）  # 注释：自动行注释
        参数：  # 注释：参数说明标题
        # （说明：执行语句）  # 注释：自动行注释
        - 见函数/类签名。  # 注释：参数占位
        # （说明：执行语句）  # 注释：自动行注释
        返回：  # 注释：返回值说明标题
        # （说明：执行语句）  # 注释：自动行注释
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        # （说明：执行语句）  # 注释：自动行注释
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        # （说明：执行语句）  # 注释：自动行注释
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        # （说明：执行语句）  # 注释：自动行注释
        最小示例：  # 注释：最小示例标题
        # （说明：执行语句）  # 注释：自动行注释
        - _handle_pending_state(...)  # 注释：示例占位
        # （说明：执行语句）  # 注释：自动行注释
        调用路径依赖：  # 注释：调用路径说明标题
        # （说明：执行语句）  # 注释：自动行注释
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::_handle_pending_state。  # 注释：位置占位
        # （说明：执行语句）  # 注释：自动行注释
        - 典型调用路径：待补充。  # 注释：调用链占位
        # （说明：执行语句）  # 注释：自动行注释
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        # （说明：执行语句）  # 注释：自动行注释
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        # （说明：执行语句）  # 注释：自动行注释
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """Handle the generating state: generate model response and check for tool calls."""
        # （说明：执行语句）  # 注释：自动行注释
        add_messages: list[dict[str, Any]] = []

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：上下文管理）  # 注释：自动行注释
        with simple_timer("generate_sequences", agent_data.metrics):
            # （说明：执行语句）  # 注释：自动行注释
            output = await self.server_manager.generate(
                # （说明：执行语句）  # 注释：自动行注释
                request_id=agent_data.request_id,
                # （说明：执行语句）  # 注释：自动行注释
                prompt_ids=agent_data.prompt_ids,
                # （说明：执行语句）  # 注释：自动行注释
                sampling_params=sampling_params,
                # （说明：执行语句）  # 注释：自动行注释
                image_data=agent_data.image_data,
            # （说明：执行语句）  # 注释：自动行注释
            )

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.assistant_turns += 1
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.response_ids = output.token_ids
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.prompt_ids += agent_data.response_ids
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        # （说明：条件分支）  # 注释：自动行注释
        if output.log_probs:
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.response_logprobs += output.log_probs

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：条件分支）  # 注释：自动行注释
        if output.routed_experts is not None:
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.routed_experts = output.routed_experts

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Check termination conditions
        # （说明：条件分支）  # 注释：自动行注释
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.TERMINATED
        # （说明：条件分支）  # 注释：自动行注释
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.TERMINATED
        # （说明：条件分支）  # 注释：自动行注释
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.TERMINATED

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Extract tool calls
        # （说明：执行语句）  # 注释：自动行注释
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Handle interaction if needed
        # （说明：条件分支）  # 注释：自动行注释
        if self.interaction_config_file:
            # （说明：执行语句）  # 注释：自动行注释
            assistant_message = await self.loop.run_in_executor(
                # （说明：执行语句）  # 注释：自动行注释
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：执行语句）  # 注释：自动行注释
            add_messages.append({"role": "assistant", "content": assistant_message})
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.messages.extend(add_messages)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Determine next state
        # （说明：条件分支）  # 注释：自动行注释
        if agent_data.tool_calls:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.PROCESSING_TOOLS
        # （说明：条件分支）  # 注释：自动行注释
        elif self.interaction_config_file:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.INTERACTING
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.TERMINATED

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        # （说明：执行语句）  # 注释：自动行注释
        add_messages: list[dict[str, Any]] = []
        # （说明：执行语句）  # 注释：自动行注释
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        tasks = []
        # （说明：执行语句）  # 注释：自动行注释
        tool_call_names = []
        # （说明：循环逻辑）  # 注释：自动行注释
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
            # （说明：执行语句）  # 注释：自动行注释
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            # （说明：执行语句）  # 注释：自动行注释
            tool_call_names.append(tool_call.name)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：上下文管理）  # 注释：自动行注释
        with simple_timer("tool_calls", agent_data.metrics):
            # （说明：执行语句）  # 注释：自动行注释
            responses = await asyncio.gather(*tasks)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Process tool responses and update multi_modal_data
        # （说明：原注释说明）  # 注释：自动行注释
        # Removed: agent_data.new_images_this_turn = []
        # （说明：循环逻辑）  # 注释：自动行注释
        for tool_response, tool_reward, _ in responses:
            # （说明：原注释说明）  # 注释：自动行注释
            # Create message from tool response
            # （说明：条件分支）  # 注释：自动行注释
            if tool_response.image or tool_response.video:
                # （说明：原注释说明）  # 注释：自动行注释
                # Multi-modal content with structured format
                # （说明：条件分支）  # 注释：自动行注释
                if not getattr(self.processor, "image_processor", None):
                    # （说明：抛出异常）  # 注释：自动行注释
                    raise ValueError(
                        # （说明：执行语句）  # 注释：自动行注释
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        # （说明：执行语句）  # 注释：自动行注释
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        # （说明：执行语句）  # 注释：自动行注释
                        "data. Plase use a vlm as the base model."
                    # （说明：执行语句）  # 注释：自动行注释
                    )
                # （说明：执行语句）  # 注释：自动行注释
                content = []
                # （说明：条件分支）  # 注释：自动行注释
                if tool_response.image:
                    # （说明：执行语句）  # 注释：自动行注释
                    content.append({"type": "image"})
                # （说明：条件分支）  # 注释：自动行注释
                if tool_response.video:
                    # （说明：执行语句）  # 注释：自动行注释
                    content.append({"type": "video"})
                # （说明：条件分支）  # 注释：自动行注释
                if tool_response.text:
                    # （说明：执行语句）  # 注释：自动行注释
                    content.append({"type": "text", "text": tool_response.text})
                # （说明：执行语句）  # 注释：自动行注释
                message = {"role": "tool", "content": content}
            # （说明：条件分支）  # 注释：自动行注释
            else:
                # （说明：原注释说明）  # 注释：自动行注释
                # Text-only content
                # （说明：执行语句）  # 注释：自动行注释
                message = {"role": "tool", "content": tool_response.text or ""}

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            add_messages.append(message)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：原注释说明）  # 注释：自动行注释
            # Handle image data
            # （说明：条件分支）  # 注释：自动行注释
            if tool_response.image:
                # （说明：原注释说明）  # 注释：自动行注释
                # Add new image data
                # （说明：条件分支）  # 注释：自动行注释
                if isinstance(tool_response.image, list):
                    # （说明：原注释说明）  # 注释：自动行注释
                    # Ensure all elements in the list are valid image objects
                    # （说明：循环逻辑）  # 注释：自动行注释
                    for img in tool_response.image:
                        # （说明：条件分支）  # 注释：自动行注释
                        if img is not None:  # Add a check to ensure the image is not None
                            # （说明：执行语句）  # 注释：自动行注释
                            new_images_this_turn.append(img)  # Using local variable
                # （说明：条件分支）  # 注释：自动行注释
                else:
                    # （说明：原注释说明）  # 注释：自动行注释
                    # Ensure the image is not None
                    # （说明：条件分支）  # 注释：自动行注释
                    if tool_response.image is not None:
                        # （说明：执行语句）  # 注释：自动行注释
                        new_images_this_turn.append(tool_response.image)  # Using local variable

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：原注释说明）  # 注释：自动行注释
            # Handle video data
            # （说明：条件分支）  # 注释：自动行注释
            if tool_response.video:
                # （说明：原注释说明）  # 注释：自动行注释
                # Currently not supported, raise informative error
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                # （说明：抛出异常）  # 注释：自动行注释
                raise NotImplementedError(
                    # （说明：执行语句）  # 注释：自动行注释
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                # （说明：执行语句）  # 注释：自动行注释
                )

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：条件分支）  # 注释：自动行注释
            if tool_reward is not None:
                # （说明：执行语句）  # 注释：自动行注释
                agent_data.tool_rewards.append(tool_reward)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.messages.extend(add_messages)
        # （说明：原注释说明）  # 注释：自动行注释
        # Update prompt with tool responses
        # （说明：条件分支）  # 注释：自动行注释
        if self.processor is not None:
            # （说明：执行语句）  # 注释：自动行注释
            raw_tool_response = await self.loop.run_in_executor(
                # （说明：执行语句）  # 注释：自动行注释
                None,
                # （说明：执行语句）  # 注释：自动行注释
                lambda: self.processor.apply_chat_template(
                    # （说明：执行语句）  # 注释：自动行注释
                    add_messages,
                    # （说明：执行语句）  # 注释：自动行注释
                    add_generation_prompt=True,
                    # （说明：执行语句）  # 注释：自动行注释
                    tokenize=False,
                    # （说明：执行语句）  # 注释：自动行注释
                    **self.apply_chat_template_kwargs,
                # （说明：执行语句）  # 注释：自动行注释
                ),
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：原注释说明）  # 注释：自动行注释
            # Use only the new images from this turn for processing tool responses
            # （说明：执行语句）  # 注释：自动行注释
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            # （说明：执行语句）  # 注释：自动行注释
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            # （说明：执行语句）  # 注释：自动行注释
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：条件分支）  # 注释：自动行注释
            if self.tool_parser_name == "gpt-oss":
                # （说明：执行语句）  # 注释：自动行注释
                logger.info("manually format tool responses for gpt-oss")
                # （说明：执行语句）  # 注释：自动行注释
                tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
                # （说明：执行语句）  # 注释：自动行注释
                response_ids = await self.loop.run_in_executor(
                    # （说明：执行语句）  # 注释：自动行注释
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                # （说明：执行语句）  # 注释：自动行注释
                )
            # （说明：条件分支）  # 注释：自动行注释
            else:
                # （说明：执行语句）  # 注释：自动行注释
                response_ids = await self.loop.run_in_executor(
                    # （说明：执行语句）  # 注释：自动行注释
                    None,
                    # （说明：执行语句）  # 注释：自动行注释
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                # （说明：执行语句）  # 注释：自动行注释
                )
                # （说明：执行语句）  # 注释：自动行注释
                response_ids = response_ids[len(self.system_prompt) :]
        # （说明：条件分支）  # 注释：自动行注释
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.TERMINATED
        # （说明：原注释说明）  # 注释：自动行注释
        # Update prompt_ids and response_mask

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：条件分支）  # 注释：自动行注释
        if new_images_this_turn:
            # （说明：条件分支）  # 注释：自动行注释
            if agent_data.image_data is None:
                # （说明：执行语句）  # 注释：自动行注释
                agent_data.image_data = []
            # （说明：条件分支）  # 注释：自动行注释
            elif not isinstance(agent_data.image_data, list):
                # （说明：执行语句）  # 注释：自动行注释
                agent_data.image_data = [agent_data.image_data]
            # （说明：循环逻辑）  # 注释：自动行注释
            for img in new_images_this_turn:
                # （说明：执行语句）  # 注释：自动行注释
                agent_data.image_data.append(img)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.prompt_ids += response_ids
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.response_mask += [0] * len(response_ids)
        # （说明：条件分支）  # 注释：自动行注释
        if agent_data.response_logprobs:
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.response_logprobs += [0.0] * len(response_ids)
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.user_turns += 1
        # （说明：返回结果）  # 注释：自动行注释
        return AgentState.GENERATING

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        # （说明：执行语句）  # 注释：自动行注释
        功能：_handle_processing_tools_state 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        # （说明：执行语句）  # 注释：自动行注释
        参数：  # 注释：参数说明标题
        # （说明：执行语句）  # 注释：自动行注释
        - 见函数/类签名。  # 注释：参数占位
        # （说明：执行语句）  # 注释：自动行注释
        返回：  # 注释：返回值说明标题
        # （说明：执行语句）  # 注释：自动行注释
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        # （说明：执行语句）  # 注释：自动行注释
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        # （说明：执行语句）  # 注释：自动行注释
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        # （说明：执行语句）  # 注释：自动行注释
        最小示例：  # 注释：最小示例标题
        # （说明：执行语句）  # 注释：自动行注释
        - _handle_processing_tools_state(...)  # 注释：示例占位
        # （说明：执行语句）  # 注释：自动行注释
        调用路径依赖：  # 注释：调用路径说明标题
        # （说明：执行语句）  # 注释：自动行注释
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::_handle_processing_tools_state。  # 注释：位置占位
        # （说明：执行语句）  # 注释：自动行注释
        - 典型调用路径：待补充。  # 注释：调用链占位
        # （说明：执行语句）  # 注释：自动行注释
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        # （说明：执行语句）  # 注释：自动行注释
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        # （说明：执行语句）  # 注释：自动行注释
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        # （说明：执行语句）  # 注释：自动行注释
        (
            # （说明：执行语句）  # 注释：自动行注释
            should_terminate_sequence,
            # （说明：执行语句）  # 注释：自动行注释
            interaction_responses,
            # （说明：执行语句）  # 注释：自动行注释
            reward,
            # （说明：执行语句）  # 注释：自动行注释
            metrics,
        # （说明：执行语句）  # 注释：自动行注释
        ) = await agent_data.interaction.generate_response(
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.user_turns += 1

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.messages.extend(add_messages)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：条件分支）  # 注释：自动行注释
        if reward is not None:
        """Handle the interacting state: get user input from interaction."""
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.turn_scores.append(reward)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Update prompt with user responses (similar to _handle_processing_tools_state)
        # （说明：条件分支）  # 注释：自动行注释
        if self.processor is not None:
            # （说明：执行语句）  # 注释：自动行注释
            raw_user_response = await self.loop.run_in_executor(
                # （说明：执行语句）  # 注释：自动行注释
                None,
                # （说明：执行语句）  # 注释：自动行注释
                lambda: self.processor.apply_chat_template(
                    # （说明：执行语句）  # 注释：自动行注释
                    add_messages,
                    # （说明：执行语句）  # 注释：自动行注释
                    add_generation_prompt=True,
                    # （说明：执行语句）  # 注释：自动行注释
                    tokenize=False,
                    # （说明：执行语句）  # 注释：自动行注释
                    **self.apply_chat_template_kwargs,
                # （说明：执行语句）  # 注释：自动行注释
                ),
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：执行语句）  # 注释：自动行注释
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            # （说明：执行语句）  # 注释：自动行注释
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            response_ids = await self.loop.run_in_executor(
                # （说明：执行语句）  # 注释：自动行注释
                None,
                # （说明：执行语句）  # 注释：自动行注释
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            # （说明：执行语句）  # 注释：自动行注释
            )
        # （说明：执行语句）  # 注释：自动行注释
        response_ids = response_ids[len(self.system_prompt) :]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Update prompt_ids and response_mask
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.prompt_ids += response_ids
        # （说明：执行语句）  # 注释：自动行注释
        agent_data.response_mask += [0] * len(response_ids)
        # （说明：条件分支）  # 注释：自动行注释
        if agent_data.response_logprobs:
            # （说明：执行语句）  # 注释：自动行注释
            agent_data.response_logprobs += [0.0] * len(response_ids)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # double check prompt
        # （说明：原注释说明）  # 注释：自动行注释
        # Check termination condition
        # （说明：条件分支）  # 注释：自动行注释
        if should_terminate_sequence:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.TERMINATED
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：返回结果）  # 注释：自动行注释
            return AgentState.GENERATING

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _call_tool(
        # （说明：执行语句）  # 注释：自动行注释
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    # （说明：执行语句）  # 注释：自动行注释
    ) -> tuple[ToolResponse, float, dict]:
        """
        功能：_call_tool 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _call_tool(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::_call_tool。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        tool, instance_id = None, None
        # （说明：异常处理）  # 注释：自动行注释
        try:
            # （说明：原注释说明）  # 注释：自动行注释
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            # （说明：执行语句）  # 注释：自动行注释
            tool_name = tool_call.name
            # （说明：执行语句）  # 注释：自动行注释
            tool_args = json.loads(tool_call.arguments)
            # （说明：执行语句）  # 注释：自动行注释
            tool = self.tools[tool_name]
            # （说明：执行语句）  # 注释：自动行注释
            kwargs = tools_kwargs.get(tool_name, {})
            # （说明：执行语句）  # 注释：自动行注释
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            # （说明：执行语句）  # 注释：自动行注释
            tool_execution_response, tool_reward, res = await tool.execute(
                # （说明：执行语句）  # 注释：自动行注释
                instance_id, tool_args, agent_data=agent_data
            # （说明：执行语句）  # 注释：自动行注释
            )
        # （说明：异常处理）  # 注释：自动行注释
        except Exception as e:
            # （说明：执行语句）  # 注释：自动行注释
            logger.warning(f"Error when executing tool: {e}")
            # （说明：返回结果）  # 注释：自动行注释
            return (
                # （说明：执行语句）  # 注释：自动行注释
                ToolResponse(
                    # （说明：执行语句）  # 注释：自动行注释
                    text=f"Error when executing tool: {e}",
                # （说明：执行语句）  # 注释：自动行注释
                ),
                # （说明：执行语句）  # 注释：自动行注释
                0.0,
                # （说明：执行语句）  # 注释：自动行注释
                {},
            # （说明：执行语句）  # 注释：自动行注释
            )
        # （说明：异常处理）  # 注释：自动行注释
        finally:
            # （说明：条件分支）  # 注释：自动行注释
            if tool and instance_id:
                # （说明：执行语句）  # 注释：自动行注释
                await tool.release(instance_id)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        tool_response_text = tool_execution_response.text
        # （说明：条件分支）  # 注释：自动行注释
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            # （说明：条件分支）  # 注释：自动行注释
            if self.tool_response_truncate_side == "left":
                # （说明：执行语句）  # 注释：自动行注释
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            # （说明：条件分支）  # 注释：自动行注释
            elif self.tool_response_truncate_side == "right":
                # （说明：执行语句）  # 注释：自动行注释
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            # （说明：条件分支）  # 注释：自动行注释
            else:
                # （说明：执行语句）  # 注释：自动行注释
                length = self.max_tool_response_length // 2
                # （说明：执行语句）  # 注释：自动行注释
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Create ToolResponse from tool execution result
        # （说明：执行语句）  # 注释：自动行注释
        tool_response_kwargs = {"text": tool_response_text}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Add multimedia data if present
        # （说明：循环逻辑）  # 注释：自动行注释
        for attr_name in ["image", "video"]:
            # （说明：条件分支）  # 注释：自动行注释
            if hasattr(tool_execution_response, attr_name):
                # （说明：执行语句）  # 注释：自动行注释
                attr_value = getattr(tool_execution_response, attr_name)
                # （说明：条件分支）  # 注释：自动行注释
                if attr_value is not None:
                    # （说明：执行语句）  # 注释：自动行注释
                    tool_response_kwargs[attr_name] = attr_value

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return ToolResponse(**tool_response_kwargs), tool_reward, res

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _initialize_interactions(self, interaction_config_file):
        # （说明：条件分支）  # 注释：自动行注释
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        功能：_initialize_interactions 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _initialize_interactions(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/tool_agent_loop.py::_initialize_interactions。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        if interaction_config_file is None:
            # （说明：返回结果）  # 注释：自动行注释
            return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        interaction_map = initialize_interactions_from_config(interaction_config_file)
        # （说明：返回结果）  # 注释：自动行注释
        return interaction_map
