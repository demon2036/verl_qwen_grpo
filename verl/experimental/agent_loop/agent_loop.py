# （说明：原注释说明）  # 注释：自动行注释
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
模块用途：实现 Agent Loop 的核心逻辑与管理器（异步 server 管理、流程调度）。  # 注释：模块用途
输入：Hydra 配置、prompt 数据、采样参数、server actor 句柄。  # 注释：输入说明
输出：AgentLoopOutput、DataProto、指标统计。  # 注释：输出说明
关键依赖：ray、hydra、torch、numpy、pydantic。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- AgentLoopManager.generate_sequences(prompts) -> AgentLoopBase.run。  # 注释：示例
调用路径概览：  # 注释：调用路径标题
- Ray worker -> AgentLoopWorker.generate_sequences -> AgentLoopBase.run。  # 注释：调用链
"""  # 注释：模块 docstring 结束
# （说明：导入依赖）  # 注释：自动行注释
import asyncio
# （说明：导入依赖）  # 注释：自动行注释
import heapq
# （说明：导入依赖）  # 注释：自动行注释
import logging
# （说明：导入依赖）  # 注释：自动行注释
import os
# （说明：导入依赖）  # 注释：自动行注释
import random
# （说明：导入依赖）  # 注释：自动行注释
from abc import ABC, abstractmethod
# （说明：导入依赖）  # 注释：自动行注释
from typing import Any, Optional
# （说明：导入依赖）  # 注释：自动行注释
from uuid import uuid4

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：导入依赖）  # 注释：自动行注释
import hydra
# （说明：导入依赖）  # 注释：自动行注释
import numpy as np
# （说明：导入依赖）  # 注释：自动行注释
import ray
# （说明：导入依赖）  # 注释：自动行注释
import torch
# （说明：导入依赖）  # 注释：自动行注释
from cachetools import LRUCache
# （说明：导入依赖）  # 注释：自动行注释
from omegaconf import DictConfig, OmegaConf
# （说明：导入依赖）  # 注释：自动行注释
from pydantic import BaseModel, ConfigDict
# （说明：导入依赖）  # 注释：自动行注释
from tensordict import TensorDict
# （说明：导入依赖）  # 注释：自动行注释
from transformers import AutoProcessor, AutoTokenizer

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：导入依赖）  # 注释：自动行注释
from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
# （说明：导入依赖）  # 注释：自动行注释
from verl.experimental.agent_loop.utils import resolve_config_path
# （说明：导入依赖）  # 注释：自动行注释
from verl.experimental.reward_loop import RewardLoopWorker
# （说明：导入依赖）  # 注释：自动行注释
from verl.protocol import DataProto
# （说明：导入依赖）  # 注释：自动行注释
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils import hf_processor, hf_tokenizer
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.fs import copy_to_local
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.model import compute_position_id_with_mask
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.ray_utils import get_event_loop
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.rollout_trace import (
    # （说明：执行语句）  # 注释：自动行注释
    RolloutTraceConfig,
    # （说明：执行语句）  # 注释：自动行注释
    rollout_trace_attr,
    # （说明：执行语句）  # 注释：自动行注释
    rollout_trace_op,
# （说明：执行语句）  # 注释：自动行注释
)
# （说明：导入依赖）  # 注释：自动行注释
from verl.utils.transferqueue_utils import tqbridge
# （说明：导入依赖）  # 注释：自动行注释
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：执行语句）  # 注释：自动行注释
logger = logging.getLogger(__file__)
# （说明：执行语句）  # 注释：自动行注释
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    功能：AsyncLLMServerManager 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - AsyncLLMServerManager(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/agent_loop.py::AsyncLLMServerManager。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        # （说明：执行语句）  # 注释：自动行注释
        self.config = config
        # （说明：执行语句）  # 注释：自动行注释
        self.server_handles = server_handles
        # （说明：执行语句）  # 注释：自动行注释
        random.shuffle(self.server_handles)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Least requests load balancing
        # （说明：执行语句）  # 注释：自动行注释
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(self.server_handles)]
        # （说明：执行语句）  # 注释：自动行注释
        heapq.heapify(self.weighted_serveres)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # LRU cache to map request_id to server
        # （说明：执行语句）  # 注释：自动行注释
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
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
        - 所在位置：verl/experimental/agent_loop/agent_loop.py::__init__。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：原注释说明）  # 注释：自动行注释
        # TODO: implement server pressure awareness load balancing
        """
        功能：_choose_server 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _choose_server(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/agent_loop.py::_choose_server。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：条件分支）  # 注释：自动行注释
        if request_id in self.request_id_to_server:
            # （说明：返回结果）  # 注释：自动行注释
            return self.request_id_to_server[request_id]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        _, _, server = self.weighted_serveres[0]
        # （说明：执行语句）  # 注释：自动行注释
        self.weighted_serveres[0][0] += 1
        # （说明：执行语句）  # 注释：自动行注释
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        # （说明：执行语句）  # 注释：自动行注释
        self.request_id_to_server[request_id] = server
        # （说明：返回结果）  # 注释：自动行注释
        return server

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：装饰器声明）  # 注释：自动行注释
    @rollout_trace_op
    # （说明：定义函数）  # 注释：自动行注释
    async def generate(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        request_id,
        # （说明：执行语句）  # 注释：自动行注释
        *,
        # （说明：执行语句）  # 注释：自动行注释
        prompt_ids: list[int],
        # （说明：执行语句）  # 注释：自动行注释
        sampling_params: dict[str, Any],
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[list[Any]] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> TokenOutput:
        """
        功能：generate 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - generate(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/agent_loop.py::generate。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        server = self._choose_server(request_id)
        # （说明：执行语句）  # 注释：自动行注释
        output = await server.generate.remote(
            # （说明：执行语句）  # 注释：自动行注释
            request_id=uuid4().hex,  # use new request_id for each turn
            # （说明：执行语句）  # 注释：自动行注释
            prompt_ids=prompt_ids,
            # （说明：执行语句）  # 注释：自动行注释
            sampling_params=sampling_params,
            # （说明：执行语句）  # 注释：自动行注释
            image_data=image_data,
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：返回结果）  # 注释：自动行注释
        return output

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：执行语句）  # 注释：自动行注释
    generate_sequences: float = 0.0
    # （说明：执行语句）  # 注释：自动行注释
    tool_calls: float = 0.0

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentLoopOutput(BaseModel):
    # （说明：执行语句）  # 注释：自动行注释
    功能：AgentLoopMetrics 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
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
    - AgentLoopMetrics(...)  # 注释：示例占位
    # （说明：执行语句）  # 注释：自动行注释
    调用路径依赖：  # 注释：调用路径说明标题
    # （说明：执行语句）  # 注释：自动行注释
    - 所在位置：verl/experimental/agent_loop/agent_loop.py::AgentLoopMetrics。  # 注释：位置占位
    # （说明：执行语句）  # 注释：自动行注释
    - 典型调用路径：待补充。  # 注释：调用链占位
    # （说明：执行语句）  # 注释：自动行注释
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    # （说明：执行语句）  # 注释：自动行注释
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    # （说明：执行语句）  # 注释：自动行注释
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """Agent loop output."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：执行语句）  # 注释：自动行注释
    prompt_ids: list[int]
    """Prompt token ids."""
    # （说明：执行语句）  # 注释：自动行注释
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    # （说明：执行语句）  # 注释：自动行注释
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    # （说明：执行语句）  # 注释：自动行注释
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    # （说明：执行语句）  # 注释：自动行注释
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    # （说明：执行语句）  # 注释：自动行注释
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    # （说明：执行语句）  # 注释：自动行注释
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    # （说明：执行语句）  # 注释：自动行注释
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    # （说明：执行语句）  # 注释：自动行注释
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    # （说明：执行语句）  # 注释：自动行注释
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：执行语句）  # 注释：自动行注释
    model_config = ConfigDict(arbitrary_types_allowed=True)

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：执行语句）  # 注释：自动行注释
    prompt_ids: torch.Tensor
    # （说明：执行语句）  # 注释：自动行注释
    功能：_InternalAgentLoopOutput 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
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
    - _InternalAgentLoopOutput(...)  # 注释：示例占位
    # （说明：执行语句）  # 注释：自动行注释
    调用路径依赖：  # 注释：调用路径说明标题
    # （说明：执行语句）  # 注释：自动行注释
    - 所在位置：verl/experimental/agent_loop/agent_loop.py::_InternalAgentLoopOutput。  # 注释：位置占位
    # （说明：执行语句）  # 注释：自动行注释
    - 典型调用路径：待补充。  # 注释：调用链占位
    # （说明：执行语句）  # 注释：自动行注释
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    # （说明：执行语句）  # 注释：自动行注释
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    # （说明：执行语句）  # 注释：自动行注释
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """Padded prompt token ids."""
    # （说明：执行语句）  # 注释：自动行注释
    response_ids: torch.Tensor
    """Padded response token ids."""
    # （说明：执行语句）  # 注释：自动行注释
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    # （说明：执行语句）  # 注释：自动行注释
    position_ids: torch.Tensor
    """Padded position ids."""
    # （说明：执行语句）  # 注释：自动行注释
    response_mask: torch.Tensor
    """Padded response mask."""
    # （说明：执行语句）  # 注释：自动行注释
    attention_mask: torch.Tensor
    """Padded attention mask."""
    # （说明：执行语句）  # 注释：自动行注释
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    # （说明：执行语句）  # 注释：自动行注释
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    # （说明：执行语句）  # 注释：自动行注释
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    # （说明：执行语句）  # 注释：自动行注释
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(self, config: DictConfig):
        # （说明：执行语句）  # 注释：自动行注释
        self.config = config

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentLoopBase(ABC):
    # （说明：执行语句）  # 注释：自动行注释
    功能：DictConfigWrap 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
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
    - DictConfigWrap(...)  # 注释：示例占位
    # （说明：执行语句）  # 注释：自动行注释
    调用路径依赖：  # 注释：调用路径说明标题
    # （说明：执行语句）  # 注释：自动行注释
    - 所在位置：verl/experimental/agent_loop/agent_loop.py::DictConfigWrap。  # 注释：位置占位
    # （说明：执行语句）  # 注释：自动行注释
    - 典型调用路径：待补充。  # 注释：调用链占位
    # （说明：执行语句）  # 注释：自动行注释
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    # （说明：执行语句）  # 注释：自动行注释
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    # （说明：执行语句）  # 注释：自动行注释
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

# （空行说明：保持段落分隔）  # 注释：空行占位
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
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
        """
        # （说明：执行语句）  # 注释：自动行注释
        self.config = trainer_config.config
        # （说明：执行语句）  # 注释：自动行注释
        self.server_manager = server_manager
        # （说明：执行语句）  # 注释：自动行注释
        self.tokenizer = tokenizer
        # （说明：执行语句）  # 注释：自动行注释
        self.processor = processor
        # （说明：执行语句）  # 注释：自动行注释
        self.loop = get_event_loop()

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：装饰器声明）  # 注释：自动行注释
    @abstractmethod
    # （说明：定义函数）  # 注释：自动行注释
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # （说明：抛出异常）  # 注释：自动行注释
        raise NotImplementedError

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：执行语句）  # 注释：自动行注释
_agent_loop_registry: dict[str, dict] = {}

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义函数）  # 注释：自动行注释
def register(agent_name: str):
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
    """Register agent loop class."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        # （说明：执行语句）  # 注释：自动行注释
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        # （说明：执行语句）  # 注释：自动行注释
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        # （说明：返回结果）  # 注释：自动行注释
        return subclass

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：返回结果）  # 注释：自动行注释
    return decorator

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentLoopWorkerBase:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        config: DictConfig,
        # （说明：执行语句）  # 注释：自动行注释
        server_handles: list[ray.actor.ActorHandle],
        # （说明：执行语句）  # 注释：自动行注释
        reward_router_address: str = None,
    # （说明：执行语句）  # 注释：自动行注释
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        # （说明：执行语句）  # 注释：自动行注释
        self.config = config

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # for recipe to change
        # （说明：条件分支）  # 注释：自动行注释
        if not hasattr(self, "server_manager"):
            # （说明：执行语句）  # 注释：自动行注释
            self.server_manager = AsyncLLMServerManager(config, server_handles)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        self.reward_router_address = reward_router_address

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        model_path = config.actor_rollout_ref.model.path
        # （说明：执行语句）  # 注释：自动行注释
        self.model_name = "/".join(model_path.split("/")[-2:])
        # （说明：执行语句）  # 注释：自动行注释
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        # （说明：执行语句）  # 注释：自动行注释
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        # （说明：执行语句）  # 注释：自动行注释
        self.processor = hf_processor(local_path, trust_remote_code=True)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        # （说明：条件分支）  # 注释：自动行注释
        if agent_loop_config_path:
            # （说明：执行语句）  # 注释：自动行注释
            resolved_path = resolve_config_path(agent_loop_config_path)
            # （说明：执行语句）  # 注释：自动行注释
            agent_loop_configs = OmegaConf.load(resolved_path)
            # （说明：循环逻辑）  # 注释：自动行注释
            for agent_loop_config in agent_loop_configs:
                # （说明：执行语句）  # 注释：自动行注释
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        # （说明：条件分支）  # 注释：自动行注释
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            # （说明：条件分支）  # 注释：自动行注释
            if self.processor is not None:
                # （说明：执行语句）  # 注释：自动行注释
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            # （说明：执行语句）  # 注释：自动行注释
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        use_reward_loop = True if self.config.reward_model.use_reward_loop else None
        # （说明：执行语句）  # 注释：自动行注释
        self.use_reward_loop = use_reward_loop
        # （说明：条件分支）  # 注释：自动行注释
        if use_reward_loop and not hasattr(self, "reward_loop_worker"):
            # （说明：执行语句）  # 注释：自动行注释
            self.reward_loop_worker = RewardLoopWorker.options(
                # （说明：执行语句）  # 注释：自动行注释
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    # （说明：执行语句）  # 注释：自动行注释
                    node_id=ray.get_runtime_context().get_node_id(),
                    # （说明：执行语句）  # 注释：自动行注释
                    soft=False,
                # （说明：执行语句）  # 注释：自动行注释
                ),
            # （说明：执行语句）  # 注释：自动行注释
            ).remote(self.config, self.reward_router_address)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        # （说明：执行语句）  # 注释：自动行注释
        RolloutTraceConfig.init(
            # （说明：执行语句）  # 注释：自动行注释
            self.config.trainer.project_name,
            # （说明：执行语句）  # 注释：自动行注释
            self.config.trainer.experiment_name,
            # （说明：执行语句）  # 注释：自动行注释
            trace_config.get("backend"),
            # （说明：执行语句）  # 注释：自动行注释
            trace_config.get("token2text", False),
            # （说明：执行语句）  # 注释：自动行注释
            trace_config.get("max_samples_per_step_per_worker", None),
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：装饰器声明）  # 注释：自动行注释
    @tqbridge()
    # （说明：定义函数）  # 注释：自动行注释
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        # （说明：执行语句）  # 注释：自动行注释
        config = self.config.actor_rollout_ref.rollout
        # （说明：执行语句）  # 注释：自动行注释
        sampling_params = dict(
            # （说明：执行语句）  # 注释：自动行注释
            temperature=config.temperature,
            # （说明：执行语句）  # 注释：自动行注释
            top_p=config.top_p,
            # （说明：执行语句）  # 注释：自动行注释
            repetition_penalty=1.0,
            # （说明：执行语句）  # 注释：自动行注释
            logprobs=config.calculate_log_probs,
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # override sampling params for validation
        # （说明：条件分支）  # 注释：自动行注释
        if batch.meta_info.get("validate", False):
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
            # （说明：执行语句）  # 注释：自动行注释
            sampling_params["top_p"] = config.val_kwargs.top_p
            # （说明：执行语句）  # 注释：自动行注释
            sampling_params["temperature"] = config.val_kwargs.temperature

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # by default, we assume it's a single turn agent
        # （说明：条件分支）  # 注释：自动行注释
        if "agent_name" not in batch.non_tensor_batch:
            # （说明：执行语句）  # 注释：自动行注释
            default_agent_loop = config.agent.default_agent_loop
            # （说明：执行语句）  # 注释：自动行注释
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：条件分支）  # 注释：自动行注释
        if "index" in batch.non_tensor_batch:
            # （说明：执行语句）  # 注释：自动行注释
            index = batch.non_tensor_batch["index"]
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            index = np.arange(len(batch))

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # For n rollouts per sample, we trace all n rollouts for selected samples
        # （说明：原注释说明）  # 注释：自动行注释
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        # （说明：条件分支）  # 注释：自动行注释
        if max_samples_per_worker is not None:
            # （说明：执行语句）  # 注释：自动行注释
            unique_sample_indices = np.unique(index)
            # （说明：条件分支）  # 注释：自动行注释
            if max_samples_per_worker < len(unique_sample_indices):
                # （说明：执行语句）  # 注释：自动行注释
                selected_samples = set(
                    # （说明：执行语句）  # 注释：自动行注释
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                # （说明：执行语句）  # 注释：自动行注释
                )
                # （说明：执行语句）  # 注释：自动行注释
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            # （说明：条件分支）  # 注释：自动行注释
            else:
                # （说明：执行语句）  # 注释：自动行注释
                traced_indices = set(range(len(batch)))
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            traced_indices = set(range(len(batch)))

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        trajectory_info = await get_trajectory_info(
            # （说明：执行语句）  # 注释：自动行注释
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        tasks = []
        # （说明：循环逻辑）  # 注释：自动行注释
        for i in range(len(batch)):
            # （说明：执行语句）  # 注释：自动行注释
            trace_this_sample = i in traced_indices
            # （说明：执行语句）  # 注释：自动行注释
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # （说明：执行语句）  # 注释：自动行注释
            tasks.append(
                # （说明：执行语句）  # 注释：自动行注释
                asyncio.create_task(
                    # （说明：执行语句）  # 注释：自动行注释
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                # （说明：执行语句）  # 注释：自动行注释
                )
            # （说明：执行语句）  # 注释：自动行注释
            )
        # （说明：执行语句）  # 注释：自动行注释
        outputs = await asyncio.gather(*tasks)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        output = self._postprocess(outputs)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return output

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _run_agent_loop(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        sampling_params: dict[str, Any],
        # （说明：执行语句）  # 注释：自动行注释
        trajectory: dict[str, Any],
        # （说明：执行语句）  # 注释：自动行注释
        *,
        # （说明：执行语句）  # 注释：自动行注释
        agent_name: str,
        # （说明：执行语句）  # 注释：自动行注释
        trace: bool = True,
        # （说明：执行语句）  # 注释：自动行注释
        **kwargs,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> _InternalAgentLoopOutput:
        # （说明：上下文管理）  # 注释：自动行注释
        with rollout_trace_attr(
            # （说明：执行语句）  # 注释：自动行注释
            step=trajectory["step"],
            # （说明：执行语句）  # 注释：自动行注释
            sample_index=trajectory["sample_index"],
            # （说明：执行语句）  # 注释：自动行注释
            rollout_n=trajectory["rollout_n"],
            # （说明：执行语句）  # 注释：自动行注释
            validate=trajectory["validate"],
            # （说明：执行语句）  # 注释：自动行注释
            name="agent_loop",
            # （说明：执行语句）  # 注释：自动行注释
            trace=trace,
        # （说明：执行语句）  # 注释：自动行注释
        ):
            # （说明：断言检查）  # 注释：自动行注释
            assert agent_name in _agent_loop_registry, (
                # （说明：执行语句）  # 注释：自动行注释
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            # （说明：执行语句）  # 注释：自动行注释
            )

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            agent_loop_config = _agent_loop_registry[agent_name]
            # （说明：执行语句）  # 注释：自动行注释
            agent_loop = hydra.utils.instantiate(
                # （说明：执行语句）  # 注释：自动行注释
                config=agent_loop_config,
                # （说明：执行语句）  # 注释：自动行注释
                trainer_config=DictConfigWrap(config=self.config),
                # （说明：执行语句）  # 注释：自动行注释
                server_manager=self.server_manager,
                # （说明：执行语句）  # 注释：自动行注释
                tokenizer=self.tokenizer,
                # （说明：执行语句）  # 注释：自动行注释
                processor=self.processor,
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：执行语句）  # 注释：自动行注释
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            # （说明：返回结果）  # 注释：自动行注释
            return await self._agent_loop_postprocess(output, **kwargs)

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        # （说明：执行语句）  # 注释：自动行注释
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Some AgentLoop may have already computed the reward score, e.g SWE-agent.

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # NOTE: consistent with the legacy batch version of generate_sequences that existed in the
        # （说明：原注释说明）  # 注释：自动行注释
        # deprecated vLLM SPMD rollout implementation.
        # （说明：原注释说明）  # 注释：自动行注释
        # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
        # （说明：原注释说明）  # 注释：自动行注释
        # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
        # （说明：原注释说明）  # 注释：自动行注释
        # input_ids: concatenation of prompt + response
        # （说明：原注释说明）  # 注释：自动行注释
        # Mask:
        """Perform post-processing operations on the output of each individual agent loop."""
        # （说明：原注释说明）  # 注释：自动行注释
        # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
        # （说明：原注释说明）  # 注释：自动行注释
        # - prompt_attention_mask: 0s for padding, 1s for tokens
        # （说明：原注释说明）  # 注释：自动行注释
        #   e.g., [0,0,0,0,1,1,1,1]
        # （说明：原注释说明）  # 注释：自动行注释
        # - response_attention_mask: 0s for padding, 1s for tokens
        # （说明：原注释说明）  # 注释：自动行注释
        #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
        # （说明：原注释说明）  # 注释：自动行注释
        # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
        # （说明：原注释说明）  # 注释：自动行注释
        #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
        # （说明：原注释说明）  # 注释：自动行注释
        # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
        # （说明：原注释说明）  # 注释：自动行注释
        #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
        # （说明：原注释说明）  # 注释：自动行注释
        # - position_ids: sequential positions for tokens, starting at 0
        # （说明：原注释说明）  # 注释：自动行注释
        #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        self.tokenizer.padding_side = "left"
        # （说明：执行语句）  # 注释：自动行注释
        prompt_output = self.tokenizer.pad(
            # （说明：执行语句）  # 注释：自动行注释
            {"input_ids": output.prompt_ids},
            # （说明：执行语句）  # 注释：自动行注释
            padding="max_length",
            # （说明：执行语句）  # 注释：自动行注释
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            # （说明：执行语句）  # 注释：自动行注释
            return_tensors="pt",
            # （说明：执行语句）  # 注释：自动行注释
            return_attention_mask=True,
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：条件分支）  # 注释：自动行注释
        if prompt_output["input_ids"].dim() == 1:
            # （说明：执行语句）  # 注释：自动行注释
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            # （说明：执行语句）  # 注释：自动行注释
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        self.tokenizer.padding_side = "right"
        # （说明：执行语句）  # 注释：自动行注释
        response_output = self.tokenizer.pad(
            # （说明：执行语句）  # 注释：自动行注释
            {"input_ids": output.response_ids},
            # （说明：执行语句）  # 注释：自动行注释
            padding="max_length",
            # （说明：执行语句）  # 注释：自动行注释
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            # （说明：执行语句）  # 注释：自动行注释
            return_tensors="pt",
            # （说明：执行语句）  # 注释：自动行注释
            return_attention_mask=True,
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：条件分支）  # 注释：自动行注释
        if response_output["input_ids"].dim() == 1:
            # （说明：执行语句）  # 注释：自动行注释
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            # （说明：执行语句）  # 注释：自动行注释
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        response_mask_output = self.tokenizer.pad(
            # （说明：执行语句）  # 注释：自动行注释
            {"input_ids": output.response_mask},
            # （说明：执行语句）  # 注释：自动行注释
            padding="max_length",
            # （说明：执行语句）  # 注释：自动行注释
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            # （说明：执行语句）  # 注释：自动行注释
            return_tensors="pt",
            # （说明：执行语句）  # 注释：自动行注释
            return_attention_mask=False,
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：条件分支）  # 注释：自动行注释
        if response_mask_output["input_ids"].dim() == 1:
            # （说明：执行语句）  # 注释：自动行注释
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        response_logprobs = None
        # （说明：条件分支）  # 注释：自动行注释
        if output.response_logprobs is not None:
            # （说明：执行语句）  # 注释：自动行注释
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            # （说明：执行语句）  # 注释：自动行注释
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        # （说明：执行语句）  # 注释：自动行注释
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        # （说明：执行语句）  # 注释：自动行注释
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        routed_experts = None
        # （说明：条件分支）  # 注释：自动行注释
        if output.routed_experts is not None:
            # （说明：执行语句）  # 注释：自动行注释
            total_length = input_ids.shape[1]
            # （说明：执行语句）  # 注释：自动行注释
            length, layer_num, topk_num = output.routed_experts.shape
            # （说明：执行语句）  # 注释：自动行注释
            experts_tensor = torch.from_numpy(output.routed_experts)
            # （说明：执行语句）  # 注释：自动行注释
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：原注释说明）  # 注释：自动行注释
            # Calculate start position: left padding means original prompt starts at the end
            # （说明：执行语句）  # 注释：自动行注释
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            # （说明：执行语句）  # 注释：自动行注释
            end_pos = min(start_pos + length, total_length)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：原注释说明）  # 注释：自动行注释
            # Add boundary checks for robustness
            # （说明：条件分支）  # 注释：自动行注释
            if start_pos < 0 or end_pos > total_length:
                # （说明：抛出异常）  # 注释：自动行注释
                raise ValueError(
                    # （说明：执行语句）  # 注释：自动行注释
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                # （说明：执行语句）  # 注释：自动行注释
                )

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Handle multi-modal inputs and position_ids calculation
        # （说明：原注释说明）  # 注释：自动行注释
        # Only support Qwen2VLImageProcessor for multi-modal processing currently
        # （说明：原注释说明）  # 注释：自动行注释
        # TODO: support other multi-modal inputs
        # （说明：执行语句）  # 注释：自动行注释
        multi_modal_inputs = None
        # （说明：条件分支）  # 注释：自动行注释
        if self.processor is not None:
            # （说明：执行语句）  # 注释：自动行注释
            images = getattr(output, "multi_modal_data", {}).get("image", None)
            # （说明：执行语句）  # 注释：自动行注释
            current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_inputs.pop("input_ids", None)
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_inputs.pop("attention_mask", None)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：原注释说明）  # 注释：自动行注释
            # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
            # （说明：原注释说明）  # 注释：自动行注释
            # because np.array() only keeps the keys for BatchFeature.
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        # （说明：条件分支）  # 注释：自动行注释
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # （说明：导入依赖）  # 注释：自动行注释
            from verl.models.transformers.qwen2_vl import get_rope_index

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            image_grid_thw = multi_modal_inputs.get("image_grid_thw")
            # （说明：执行语句）  # 注释：自动行注释
            video_grid_thw = multi_modal_inputs.get("video_grid_thw")
            # （说明：执行语句）  # 注释：自动行注释
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            vision_position_ids = get_rope_index(
                # （说明：执行语句）  # 注释：自动行注释
                self.processor,
                # （说明：执行语句）  # 注释：自动行注释
                input_ids=input_ids.squeeze(0),
                # （说明：执行语句）  # 注释：自动行注释
                image_grid_thw=image_grid_thw,
                # （说明：执行语句）  # 注释：自动行注释
                video_grid_thw=video_grid_thw,
                # （说明：执行语句）  # 注释：自动行注释
                second_per_grid_ts=second_per_grid_ts,
                # （说明：执行语句）  # 注释：自动行注释
                attention_mask=attention_mask.squeeze(0),
            # （说明：执行语句）  # 注释：自动行注释
            ).unsqueeze(0)  # (1, 3, seq_len)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            valid_mask = attention_mask[0].bool()
            # （说明：执行语句）  # 注释：自动行注释
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            # （说明：执行语句）  # 注释：自动行注释
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            # （说明：执行语句）  # 注释：自动行注释
            text_position_ids = text_position_ids.unsqueeze(0)
            # （说明：执行语句）  # 注释：自动行注释
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)
        # （说明：执行语句）  # 注释：自动行注释
        enable_async_reward = (
            # （说明：执行语句）  # 注释：自动行注释
            self.reward_router_address is not None and self.config.reward_model.enable_resource_pool
        # （说明：执行语句）  # 注释：自动行注释
        ) or not self.config.reward_model.enable
        # （说明：条件分支）  # 注释：自动行注释
        if output.reward_score is None and enable_async_reward and self.use_reward_loop:
            # （说明：执行语句）  # 注释：自动行注释
            batch = TensorDict(
                # （说明：执行语句）  # 注释：自动行注释
                {
                    # （说明：执行语句）  # 注释：自动行注释
                    "prompts": prompt_output["input_ids"],  # [1, prompt_length]
                    # （说明：执行语句）  # 注释：自动行注释
                    "responses": response_output["input_ids"],  # [1, response_length]
                    # （说明：执行语句）  # 注释：自动行注释
                    "attention_mask": attention_mask,  # [1, prompt_length + response_length]
                    # （说明：执行语句）  # 注释：自动行注释
                    "input_ids": input_ids,  # [1, prompt_length + response_length]
                    # （说明：执行语句）  # 注释：自动行注释
                    "position_ids": position_ids,
                # （说明：执行语句）  # 注释：自动行注释
                },
                # （说明：执行语句）  # 注释：自动行注释
                batch_size=1,
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：执行语句）  # 注释：自动行注释
            non_tensor_batch = {
                # （说明：执行语句）  # 注释：自动行注释
                **{k: np.array([v]) for k, v in kwargs.items()},
                # （说明：执行语句）  # 注释：自动行注释
                "__num_turns__": np.array([output.num_turns]),
                # （说明：执行语句）  # 注释：自动行注释
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            # （说明：执行语句）  # 注释：自动行注释
            }

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            data = DataProto(
                # （说明：执行语句）  # 注释：自动行注释
                batch=batch,
                # （说明：执行语句）  # 注释：自动行注释
                non_tensor_batch=non_tensor_batch,
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：执行语句）  # 注释：自动行注释
            result = await self.reward_loop_worker.compute_score.remote(data)
            # （说明：执行语句）  # 注释：自动行注释
            output.reward_score = result["reward_score"]
            # （说明：执行语句）  # 注释：自动行注释
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return _InternalAgentLoopOutput(
            # （说明：执行语句）  # 注释：自动行注释
            prompt_ids=prompt_output["input_ids"],
            # （说明：执行语句）  # 注释：自动行注释
            response_ids=response_output["input_ids"],
            # （说明：执行语句）  # 注释：自动行注释
            input_ids=input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            position_ids=position_ids,
            # （说明：执行语句）  # 注释：自动行注释
            response_mask=response_mask,
            # （说明：执行语句）  # 注释：自动行注释
            attention_mask=attention_mask,
            # （说明：执行语句）  # 注释：自动行注释
            response_logprobs=response_logprobs,
            # （说明：执行语句）  # 注释：自动行注释
            routed_experts=routed_experts,
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_inputs=multi_modal_inputs,
            # （说明：执行语句）  # 注释：自动行注释
            multi_modal_data=output.multi_modal_data,
            # （说明：执行语句）  # 注释：自动行注释
            reward_score=output.reward_score,
            # （说明：执行语句）  # 注释：自动行注释
            num_turns=output.num_turns,
            # （说明：执行语句）  # 注释：自动行注释
            metrics=output.metrics,
            # （说明：执行语句）  # 注释：自动行注释
            extra_fields=output.extra_fields,
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        # （说明：原注释说明）  # 注释：自动行注释
        # Convert lists back to tensors and stack them to create a batch.
        # （说明：执行语句）  # 注释：自动行注释
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        # （说明：执行语句）  # 注释：自动行注释
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        # （说明：执行语句）  # 注释：自动行注释
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        # （说明：执行语句）  # 注释：自动行注释
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        # （说明：执行语句）  # 注释：自动行注释
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        # （说明：执行语句）  # 注释：自动行注释
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        # （说明：执行语句）  # 注释：自动行注释
        optional_outputs = {}
        # （说明：条件分支）  # 注释：自动行注释
        if inputs[0].response_logprobs is not None:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
            # （说明：执行语句）  # 注释：自动行注释
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        # （说明：条件分支）  # 注释：自动行注释
        if inputs[0].routed_experts is not None:
            # （说明：执行语句）  # 注释：自动行注释
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        batch = TensorDict(
            # （说明：执行语句）  # 注释：自动行注释
            {
                # （说明：执行语句）  # 注释：自动行注释
                "prompts": prompt_ids,  # [bsz, prompt_length]
                # （说明：执行语句）  # 注释：自动行注释
                "responses": response_ids,  # [bsz, response_length]
                # （说明：执行语句）  # 注释：自动行注释
                "response_mask": response_mask,  # [bsz, response_length]
                # （说明：执行语句）  # 注释：自动行注释
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                # （说明：执行语句）  # 注释：自动行注释
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                # （说明：原注释说明）  # 注释：自动行注释
                # position_ids: [bsz, 3, prompt_length + response_length] or [bsz, prompt_length + response_length]
                # （说明：执行语句）  # 注释：自动行注释
                "position_ids": position_ids,
                # （说明：执行语句）  # 注释：自动行注释
                **optional_outputs,
            # （说明：执行语句）  # 注释：自动行注释
            },
            # （说明：执行语句）  # 注释：自动行注释
            batch_size=len(inputs),
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        scores = [input.reward_score for input in inputs]
        # （说明：条件分支）  # 注释：自动行注释
        if all(score is not None for score in scores):
            # （说明：执行语句）  # 注释：自动行注释
            prompt_length = prompt_ids.size(1)
            # （说明：执行语句）  # 注释：自动行注释
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            # （说明：执行语句）  # 注释：自动行注释
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            # （说明：执行语句）  # 注释：自动行注释
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            # （说明：执行语句）  # 注释：自动行注释
            batch["rm_scores"] = rm_scores

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        non_tensor_batch = {
            # （说明：执行语句）  # 注释：自动行注释
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        # （说明：执行语句）  # 注释：自动行注释
        }

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # add reward_extra_info to non_tensor_batch
        # （说明：执行语句）  # 注释：自动行注释
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        # （说明：执行语句）  # 注释：自动行注释
        reward_extra_keys = list(reward_extra_infos[0].keys())
        # （说明：循环逻辑）  # 注释：自动行注释
        for key in reward_extra_keys:
            # （说明：执行语句）  # 注释：自动行注释
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Add multi_modal_inputs to non_tensor_batch if any samples have them
        # （说明：执行语句）  # 注释：自动行注释
        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        # （说明：条件分支）  # 注释：自动行注释
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            # （说明：执行语句）  # 注释：自动行注释
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        metrics = [input.metrics.model_dump() for input in inputs]
        # （说明：原注释说明）  # 注释：自动行注释
        # Collect extra fields from all inputs and convert them to np.ndarray
        # （说明：执行语句）  # 注释：自动行注释
        extra_fields = {}
        # （说明：执行语句）  # 注释：自动行注释
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        # （说明：循环逻辑）  # 注释：自动行注释
        for key in all_keys:
            # （说明：执行语句）  # 注释：自动行注释
            temp_arr = np.empty(len(inputs), dtype=object)
            # （说明：执行语句）  # 注释：自动行注释
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            # （说明：执行语句）  # 注释：自动行注释
            extra_fields[key] = temp_arr

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        non_tensor_batch.update(extra_fields)
        # （说明：返回结果）  # 注释：自动行注释
        return DataProto(
            # （说明：执行语句）  # 注释：自动行注释
            batch=batch,
            # （说明：执行语句）  # 注释：自动行注释
            non_tensor_batch=non_tensor_batch,
            # （说明：执行语句）  # 注释：自动行注释
            meta_info={"metrics": metrics, "reward_extra_keys": reward_extra_keys},
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def create_transferqueue_client(
        # （说明：执行语句）  # 注释：自动行注释
        self,
    # （说明：执行语句）  # 注释：自动行注释
    ):
        """Create a client for data system (TransferQueue)."""
        # （说明：导入依赖）  # 注释：自动行注释
        from verl.single_controller.ray.base import get_random_string
        # （说明：导入依赖）  # 注释：自动行注释
        from verl.utils.transferqueue_utils import create_transferqueue_client

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        client_name = get_random_string(length=6)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        self.tq_client = create_transferqueue_client(
            # （说明：执行语句）  # 注释：自动行注释
            client_id=f"AgentLoopWorker_{client_name}",
            # （说明：执行语句）  # 注释：自动行注释
            config=self.config.transfer_queue,
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：装饰器声明）  # 注释：自动行注释
@ray.remote
# （说明：定义类）  # 注释：自动行注释
class AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None
    # （说明：执行语句）  # 注释：自动行注释
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_router_address (str): reward router address.
        """
        # （说明：执行语句）  # 注释：自动行注释
        super().__init__(config, server_handles, reward_router_address)

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义函数）  # 注释：自动行注释
async def get_trajectory_info(step, index, validate):
    # （说明：执行语句）  # 注释：自动行注释
    trajectory_info = []
    # （说明：执行语句）  # 注释：自动行注释
    rollout_n = 0
    # （说明：循环逻辑）  # 注释：自动行注释
    for i in range(len(index)):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
        # （说明：条件分支）  # 注释：自动行注释
        if i > 0 and index[i - 1] == index[i]:
            # （说明：执行语句）  # 注释：自动行注释
            rollout_n += 1
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            rollout_n = 0
        # （说明：执行语句）  # 注释：自动行注释
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    # （说明：返回结果）  # 注释：自动行注释
    return trajectory_info

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_resource_pool: RayResourcePool = None
    # （说明：执行语句）  # 注释：自动行注释
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        # （说明：执行语句）  # 注释：自动行注释
        self.config = config
        # （说明：执行语句）  # 注释：自动行注释
        self.worker_group = worker_group
        # （说明：执行语句）  # 注释：自动行注释
        self.reward_model_manager = None
        # （说明：执行语句）  # 注释：自动行注释
        self.reward_router_address = None
        # （说明：条件分支）  # 注释：自动行注释
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            # （说明：导入依赖）  # 注释：自动行注释
            from verl.experimental.reward_loop import RewardModelManager

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            # （说明：执行语句）  # 注释：自动行注释
            self.reward_router_address = self.reward_model_manager.get_router_address()

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # for recipe to change
        # （说明：条件分支）  # 注释：自动行注释
        if not hasattr(self, "rollout_replica_class"):
            # （说明：执行语句）  # 注释：自动行注释
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        # （说明：条件分支）  # 注释：自动行注释
        if not hasattr(self, "agent_loop_workers_class"):
            # （说明：执行语句）  # 注释：自动行注释
            self.agent_loop_workers_class = AgentLoopWorker

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        self._initialize_llm_servers()
        # （说明：执行语句）  # 注释：自动行注释
        self._init_agent_loop_workers()

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Initially we're in sleep mode.
        # （说明：条件分支）  # 注释：自动行注释
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            # （说明：执行语句）  # 注释：自动行注释
            self.sleep()

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _initialize_llm_servers(self):
        # （说明：执行语句）  # 注释：自动行注释
        rollout_world_size = (
            # （说明：执行语句）  # 注释：自动行注释
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            # （说明：执行语句）  # 注释：自动行注释
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            # （说明：执行语句）  # 注释：自动行注释
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        world_size = (
            # （说明：执行语句）  # 注释：自动行注释
            self.worker_group.world_size
            # （说明：条件分支）  # 注释：自动行注释
            if self.worker_group
            # （说明：条件分支）  # 注释：自动行注释
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        num_replicas = world_size // rollout_world_size

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        rollout_config = self.config.actor_rollout_ref.rollout
        # （说明：执行语句）  # 注释：自动行注释
        model_config = self.config.actor_rollout_ref.model
        # （说明：执行语句）  # 注释：自动行注释
        self.rollout_replicas = [
            # （说明：执行语句）  # 注释：自动行注释
            self.rollout_replica_class(
                # （说明：执行语句）  # 注释：自动行注释
                replica_rank=replica_rank,
                # （说明：执行语句）  # 注释：自动行注释
                config=rollout_config,
                # （说明：执行语句）  # 注释：自动行注释
                model_config=model_config,
                # （说明：执行语句）  # 注释：自动行注释
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            # （说明：执行语句）  # 注释：自动行注释
            )
            # （说明：循环逻辑）  # 注释：自动行注释
            for replica_rank in range(num_replicas)
        # （说明：执行语句）  # 注释：自动行注释
        ]
        # （说明：条件分支）  # 注释：自动行注释
        if self.worker_group:
            # （说明：执行语句）  # 注释：自动行注释
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        # （说明：执行语句）  # 注释：自动行注释
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        # （说明：执行语句）  # 注释：自动行注释
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        print(f"AgentLoopManager: {self.server_addresses}")

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Update Prometheus configuration with server addresses
        # （说明：条件分支）  # 注释：自动行注释
        if rollout_config.prometheus.enable:
            # （说明：条件分支）  # 注释：自动行注释
            if rollout_config.disable_log_stats:
                # （说明：抛出异常）  # 注释：自动行注释
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            # （说明：执行语句）  # 注释：自动行注释
            update_prometheus_config(rollout_config.prometheus, self.server_addresses)

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _init_agent_loop_workers(self):
        # （说明：执行语句）  # 注释：自动行注释
        self.agent_loop_workers = []
        # （说明：执行语句）  # 注释：自动行注释
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        # （说明：循环逻辑）  # 注释：自动行注释
        for i in range(num_workers):
            # （说明：原注释说明）  # 注释：自动行注释
            # Round-robin scheduling over the all nodes
            # （说明：执行语句）  # 注释：自动行注释
            node_id = node_ids[i % len(node_ids)]
            # （说明：执行语句）  # 注释：自动行注释
            self.agent_loop_workers.append(
                # （说明：执行语句）  # 注释：自动行注释
                self.agent_loop_workers_class.options(
                    # （说明：执行语句）  # 注释：自动行注释
                    name=f"agent_loop_worker_{i}",
                    # （说明：执行语句）  # 注释：自动行注释
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        # （说明：执行语句）  # 注释：自动行注释
                        node_id=node_id, soft=True
                    # （说明：执行语句）  # 注释：自动行注释
                    ),
                # （说明：执行语句）  # 注释：自动行注释
                ).remote(self.config, self.server_handles, self.reward_router_address)
            # （说明：执行语句）  # 注释：自动行注释
            )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def generate_sequences(self, prompts: DataProto) -> DataProto:

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Fix for Issue #4147: Always call wake_up() to ensure weight sync
        # （说明：原注释说明）  # 注释：自动行注释
        # The wake_up()/sleep() methods internally check free_cache_engine
        # （说明：执行语句）  # 注释：自动行注释
        self.wake_up()
        # （说明：条件分支）  # 注释：自动行注释
        if self.reward_model_manager:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
            # （说明：执行语句）  # 注释：自动行注释
            self.reward_model_manager.wake_up()

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        # （说明：执行语句）  # 注释：自动行注释
        outputs = ray.get(
            # （说明：执行语句）  # 注释：自动行注释
            [
                # （说明：执行语句）  # 注释：自动行注释
                worker.generate_sequences.remote(chunk)
                # （说明：循环逻辑）  # 注释：自动行注释
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            # （说明：执行语句）  # 注释：自动行注释
            ]
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        output = DataProto.concat(outputs)
        # （说明：原注释说明）  # 注释：自动行注释
        # Fix for Issue #4147: Always call sleep() to ensure proper cleanup
        # （说明：执行语句）  # 注释：自动行注释
        self.sleep()
        # （说明：条件分支）  # 注释：自动行注释
        if self.reward_model_manager:
            # （说明：执行语句）  # 注释：自动行注释
            self.reward_model_manager.sleep()

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # calculate performance metrics
        # （说明：执行语句）  # 注释：自动行注释
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        # （说明：执行语句）  # 注释：自动行注释
        timing = self._performance_metrics(metrics, output)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        # （说明：返回结果）  # 注释：自动行注释
        return output

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        # （说明：执行语句）  # 注释：自动行注释
        timing = {}
        # （说明：执行语句）  # 注释：自动行注释
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        # （说明：执行语句）  # 注释：自动行注释
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # batch sequence generation is bounded by the slowest sample
        # （说明：执行语句）  # 注释：自动行注释
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        # （说明：执行语句）  # 注释：自动行注释
        attention_mask = output.batch["attention_mask"][slowest]
        # （说明：执行语句）  # 注释：自动行注释
        prompt_length = output.batch["prompts"].shape[1]
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        # （说明：执行语句）  # 注释：自动行注释
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return timing

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def wake_up(self):
        """Wake up all rollout replica instances."""
        # （说明：执行语句）  # 注释：自动行注释
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def sleep(self):
        # （说明：执行语句）  # 注释：自动行注释
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def clear_kv_cache(self):
        """Sleep all rollout replica instances."""
        """Clear all rollout kv cache, but don`t sleep."""
        # （说明：执行语句）  # 注释：自动行注释
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _run_all(self, tasks: list[asyncio.Task]):
        # （说明：定义函数）  # 注释：自动行注释
        async def run_all():
            # （说明：执行语句）  # 注释：自动行注释
            await asyncio.gather(*tasks)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        asyncio.run(run_all())
