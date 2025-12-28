# （说明：原注释说明）  # 注释：自动行注释
# Copyright 2025 z.ai
# （说明：原注释说明）  # 注释：自动行注释
# Copyright 2023-2024 SGLang Team
# （说明：原注释说明）  # 注释：自动行注释
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# （说明：原注释说明）  # 注释：自动行注释
#
# （说明：原注释说明）  # 注释：自动行注释
# This file is adapted from multiple sources:
# （说明：原注释说明）  # 注释：自动行注释
# 1. THUDM/slime project
# （说明：原注释说明）  # 注释：自动行注释
#    Original source: https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/http_server_engine.py
# （说明：原注释说明）  # 注释：自动行注释
#    Copyright 2025 z.ai
# （说明：原注释说明）  # 注释：自动行注释
#    Licensed under the Apache License, Version 2.0
# （说明：原注释说明）  # 注释：自动行注释
# 2. SGLang project
# （说明：原注释说明）  # 注释：自动行注释
#    Original source: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server_engine.py
# （说明：原注释说明）  # 注释：自动行注释
#    Copyright 2023-2024 SGLang Team
# （说明：原注释说明）  # 注释：自动行注释
#    Licensed under the Apache License, Version 2.0
# （说明：原注释说明）  # 注释：自动行注释
#
# （说明：原注释说明）  # 注释：自动行注释
# Modifications made by z.ai and ModelBest Inc. include but are not limited to:
# （说明：原注释说明）  # 注释：自动行注释
# - Enhanced error handling and retry logic
# （说明：原注释说明）  # 注释：自动行注释
# - Added async support with connection pooling
# （说明：原注释说明）  # 注释：自动行注释
# - Extended functionality for distributed weight updates
# （说明：原注释说明）  # 注释：自动行注释
# - Improved logging and monitoring capabilities
# （说明：原注释说明）  # 注释：自动行注释
# - Additional configuration options and optimizations

# （空行说明：保持段落分隔）  # 注释：空行占位
"""
模块用途：提供 SGLang HTTP server 的同步/异步适配器与请求封装。  # 注释：模块用途
输入：server 地址、权重/生成请求、HTTP 参数等。  # 注释：输入说明
输出：请求响应（JSON）与生成结果。  # 注释：输出说明
关键依赖：requests/aiohttp、sglang.srt、torch。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- ServerAdapter -> AsyncHttpServerAdapter.generate/reward_score。  # 注释：示例
调用路径概览：  # 注释：调用路径标题
- sglang_rollout.ServerAdapter -> HttpServerAdapter/AsyncHttpServerAdapter。  # 注释：调用链

HTTP Server Engine Adapter for SGLang.

This module provides HTTP-based adapters for SGLang engines, allowing communication
with SGLang servers through HTTP requests instead of direct engine calls.

Classes:
    HttpServerAdapter: Synchronous HTTP adapter for SGLang engines
    AsyncHttpServerAdapter: Asynchronous HTTP adapter for SGLang engines

Functions:
    launch_server_process: Launch and initialize an SGLang HTTP server process
"""

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：导入依赖）  # 注释：自动行注释
import asyncio
# （说明：导入依赖）  # 注释：自动行注释
import logging
# （说明：导入依赖）  # 注释：自动行注释
import multiprocessing
# （说明：导入依赖）  # 注释：自动行注释
import os
# （说明：导入依赖）  # 注释：自动行注释
import time
# （说明：导入依赖）  # 注释：自动行注释
from contextlib import asynccontextmanager
# （说明：导入依赖）  # 注释：自动行注释
from typing import Any, Callable, Optional

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：导入依赖）  # 注释：自动行注释
import aiohttp
# （说明：导入依赖）  # 注释：自动行注释
import requests
# （说明：导入依赖）  # 注释：自动行注释
from sglang.srt.entrypoints.EngineBase import EngineBase
# （说明：导入依赖）  # 注释：自动行注释
from sglang.srt.entrypoints.http_server import launch_server
# （说明：导入依赖）  # 注释：自动行注释
from sglang.srt.managers.io_struct import (
    # （说明：执行语句）  # 注释：自动行注释
    UpdateWeightsFromTensorReqInput,
# （说明：执行语句）  # 注释：自动行注释
)
# （说明：导入依赖）  # 注释：自动行注释
from sglang.srt.server_args import ServerArgs
# （说明：导入依赖）  # 注释：自动行注释
from sglang.srt.utils import kill_process_tree

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：原注释说明）  # 注释：自动行注释
# Configure logger
# （说明：执行语句）  # 注释：自动行注释
logger = logging.getLogger(__name__)
# （说明：执行语句）  # 注释：自动行注释
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：原注释说明）  # 注释：自动行注释
# Default configuration constants
# （说明：执行语句）  # 注释：自动行注释
DEFAULT_TIMEOUT = 60.0
# （说明：执行语句）  # 注释：自动行注释
DEFAULT_MAX_ATTEMPTS = 3
# （说明：执行语句）  # 注释：自动行注释
DEFAULT_RETRY_DELAY = 2.0
# （说明：执行语句）  # 注释：自动行注释
DEFAULT_MAX_CONNECTIONS = 2000
# （说明：执行语句）  # 注释：自动行注释
DEFAULT_MAX_WAIT_TIME = 300.0

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义函数）  # 注释：自动行注释
def _read_response(response: requests.Response):
    # （说明：条件分支）  # 注释：自动行注释
    """
    功能：_read_response 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - _read_response(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::_read_response。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """
    if response.status_code == 204 or not response.content:
        # （说明：返回结果）  # 注释：自动行注释
        return {}
    # （说明：异常处理）  # 注释：自动行注释
    try:
        # （说明：返回结果）  # 注释：自动行注释
        return response.json()
    # （说明：异常处理）  # 注释：自动行注释
    except ValueError:
        # （说明：返回结果）  # 注释：自动行注释
        return {
            # （说明：执行语句）  # 注释：自动行注释
            "content_type": response.headers.get("Content-Type", ""),
            # （说明：执行语句）  # 注释：自动行注释
            "text": response.text,
        # （说明：执行语句）  # 注释：自动行注释
        }

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义函数）  # 注释：自动行注释
async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    # （说明：条件分支）  # 注释：自动行注释
    """
    功能：_read_async_response 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - _read_async_response(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::_read_async_response。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """
    if resp.status == 204 or (resp.content_length == 0):
        # （说明：返回结果）  # 注释：自动行注释
        return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：异常处理）  # 注释：自动行注释
    try:
        # （说明：返回结果）  # 注释：自动行注释
        return await resp.json(content_type=None)
    # （说明：异常处理）  # 注释：自动行注释
    except Exception:
        # （说明：异常处理）  # 注释：自动行注释
        try:
            # （说明：执行语句）  # 注释：自动行注释
            text = await resp.text()
        # （说明：异常处理）  # 注释：自动行注释
        except Exception:
            # （说明：返回结果）  # 注释：自动行注释
            return {}
        # （说明：返回结果）  # 注释：自动行注释
        return {
            # （说明：执行语句）  # 注释：自动行注释
            "content_type": (resp.headers.get("Content-Type") or ""),
            # （说明：执行语句）  # 注释：自动行注释
            "text": text,
        # （说明：执行语句）  # 注释：自动行注释
        }

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义函数）  # 注释：自动行注释
def launch_server_process(
    # （说明：执行语句）  # 注释：自动行注释
    server_args: ServerArgs,
    # （说明：执行语句）  # 注释：自动行注释
    timeout: float = DEFAULT_TIMEOUT,
    # （说明：执行语句）  # 注释：自动行注释
    max_wait_time=DEFAULT_MAX_WAIT_TIME,
    # （说明：执行语句）  # 注释：自动行注释
    first_rank_in_node=False,
# （说明：执行语句）  # 注释：自动行注释
) -> multiprocessing.Process:
    """
    功能：launch_server_process 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - launch_server_process(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::launch_server_process。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """
    # （说明：执行语句）  # 注释：自动行注释
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    # （说明：条件分支）  # 注释：自动行注释
    if server_args.node_rank != 0 or not first_rank_in_node:
        # （说明：执行语句）  # 注释：自动行注释
        logger.info(f"Server process started with PID {p.pid} for node rank {server_args.node_rank}", flush=True)
        # （说明：返回结果）  # 注释：自动行注释
        return p

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：执行语句）  # 注释：自动行注释
    p.start()

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：执行语句）  # 注释：自动行注释
    base_url = server_args.url()
    # （说明：执行语句）  # 注释：自动行注释
    headers = {
        # （说明：执行语句）  # 注释：自动行注释
        "Content-Type": "application/json; charset=utf-8",
        # （说明：执行语句）  # 注释：自动行注释
        "Authorization": f"Bearer {server_args.api_key}",
    # （说明：执行语句）  # 注释：自动行注释
    }

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：原注释说明）  # 注释：自动行注释
    # Health check with overall timeout
    # （说明：执行语句）  # 注释：自动行注释
    start_time = time.time()

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：上下文管理）  # 注释：自动行注释
    with requests.Session() as session:
        # （说明：循环逻辑）  # 注释：自动行注释
        while time.time() - start_time < max_wait_time:
            # （说明：条件分支）  # 注释：自动行注释
            if not p.is_alive():
                # （说明：抛出异常）  # 注释：自动行注释
                raise RuntimeError("Server process terminated unexpectedly during startup")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：条件分支）  # 注释：自动行注释
                if server_args.is_embedding:
                    # （说明：执行语句）  # 注释：自动行注释
                    response = session.get(f"{base_url}/health", headers=headers, timeout=timeout)
                # （说明：条件分支）  # 注释：自动行注释
                else:
                    # （说明：执行语句）  # 注释：自动行注释
                    response = session.get(f"{base_url}/health_generate", headers=headers, timeout=timeout)
                # （说明：条件分支）  # 注释：自动行注释
                if response.status_code == 200:
                    # （说明：执行语句）  # 注释：自动行注释
                    break
            # （说明：异常处理）  # 注释：自动行注释
            except requests.RequestException as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.debug(f"Health check failed: {e}")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            time.sleep(2)
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            p.terminate()
            # （说明：执行语句）  # 注释：自动行注释
            logger.error(f"Server in {base_url} failed to become healthy within timeout period")
            # （说明：抛出异常）  # 注释：自动行注释
            raise TimeoutError("Server failed to become healthy within timeout period")

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Ensure cache is ready
        # （说明：循环逻辑）  # 注释：自动行注释
        while time.time() - start_time < max_wait_time:
            # （说明：条件分支）  # 注释：自动行注释
            if not p.is_alive():
                # （说明：抛出异常）  # 注释：自动行注释
                raise RuntimeError("Server process terminated unexpectedly during cache flush")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：执行语句）  # 注释：自动行注释
                response = session.get(f"{base_url}/flush_cache", headers=headers, timeout=timeout)
                # （说明：条件分支）  # 注释：自动行注释
                if response.status_code == 200:
                    # （说明：执行语句）  # 注释：自动行注释
                    break
            # （说明：异常处理）  # 注释：自动行注释
            except requests.RequestException as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.debug(f"Cache flush check failed: {e}")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            time.sleep(2)
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            p.terminate()
            # （说明：抛出异常）  # 注释：自动行注释
            raise TimeoutError("Server cache flush failed within timeout period")

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：返回结果）  # 注释：自动行注释
    return p

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class HttpServerAdapter(EngineBase):
    """HTTP-based adapter for SGLang engines.

    This adapter allows interaction with SGLang engines through HTTP requests
    instead of direct engine calls. It launches an HTTP server process and
    provides methods to communicate with it via REST API calls.

    You can use this class to launch a server from a HttpServerAdapter instance.
    We recommend using this class only when you need to use http server.
    Otherwise, you can use Engine directly.

    Attributes:
        router_ip (Optional[str]): IP address of the router for worker registration
        router_port (Optional[int]): Port of the router for worker registration
        server_args (ServerArgs): Server configuration arguments
        node_rank (int): Rank of this node in distributed setup
        process (multiprocessing.Process): The launched server process
        timeout (float): HTTP request timeout in seconds
        max_attempts (int): Maximum number of attempts for requests
        retry_delay (float): Base delay between retries in seconds
    功能：HttpServerAdapter 的自动中文说明（需按实际逻辑细化）。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - 见函数/类签名。  # 注释：参数占位
    返回：  # 注释：返回值说明标题
    - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
    副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
    异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
    最小示例：  # 注释：最小示例标题
    - HttpServerAdapter(...)  # 注释：示例占位
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::HttpServerAdapter。  # 注释：位置占位
    - 典型调用路径：待补充。  # 注释：调用链占位
    - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
    - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
    - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
    """

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        router_ip: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        router_port: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        timeout: float = DEFAULT_TIMEOUT,
        # （说明：执行语句）  # 注释：自动行注释
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        # （说明：执行语句）  # 注释：自动行注释
        retry_delay: float = DEFAULT_RETRY_DELAY,
        # （说明：执行语句）  # 注释：自动行注释
        first_rank_in_node: bool = False,
        # （说明：执行语句）  # 注释：自动行注释
        max_start_wait_time: float = DEFAULT_MAX_WAIT_TIME,
        # （说明：执行语句）  # 注释：自动行注释
        launch_server: bool = True,
        # （说明：执行语句）  # 注释：自动行注释
        **kwargs: Any,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> None:
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
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::__init__。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        self.router_ip: Optional[str] = router_ip
        # （说明：执行语句）  # 注释：自动行注释
        self.router_port: Optional[int] = router_port
        # （说明：执行语句）  # 注释：自动行注释
        self.timeout: float = timeout
        # （说明：执行语句）  # 注释：自动行注释
        self.max_attempts: int = max_attempts
        # （说明：执行语句）  # 注释：自动行注释
        self.retry_delay: float = retry_delay
        # （说明：执行语句）  # 注释：自动行注释
        self.server_args: ServerArgs = ServerArgs(**kwargs)
        # （说明：执行语句）  # 注释：自动行注释
        self.node_rank: int = self.server_args.node_rank
        # （说明：执行语句）  # 注释：自动行注释
        self.max_start_wait_time: float = max_start_wait_time

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        logger.info(
            # （说明：执行语句）  # 注释：自动行注释
            f"Launch HttpServerAdapter at: {self.server_args.host}:{self.server_args.port} with {first_rank_in_node}"
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：条件分支）  # 注释：自动行注释
        if launch_server:
            # （说明：执行语句）  # 注释：自动行注释
            self.process: multiprocessing.Process = launch_server_process(
                # （说明：执行语句）  # 注释：自动行注释
                self.server_args, self.timeout, self.max_start_wait_time, first_rank_in_node
            # （说明：执行语句）  # 注释：自动行注释
            )

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：条件分支）  # 注释：自动行注释
        if self.node_rank == 0 and self.router_ip and self.router_port:
            # （说明：执行语句）  # 注释：自动行注释
            self._register_with_router()

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _register_with_router(self) -> None:
        # （说明：异常处理）  # 注释：自动行注释
        """Register worker with router with error handling.

        This method attempts to register the current worker with a router service.
        If registration fails, it logs an error but does not raise an exception,
        allowing the server to continue operating without router integration.

        Raises:
            Does not raise exceptions - all errors are logged and handled gracefully.
        功能：_register_with_router 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _register_with_router(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::_register_with_router。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        try:
            # （说明：执行语句）  # 注释：自动行注释
            url = f"http://{self.router_ip}:{self.router_port}/add_worker"
            # （说明：执行语句）  # 注释：自动行注释
            params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
            # （说明：执行语句）  # 注释：自动行注释
            response = requests.post(url, params=params, timeout=self.timeout)
            # （说明：执行语句）  # 注释：自动行注释
            response.raise_for_status()
            # （说明：执行语句）  # 注释：自动行注释
            logger.info("Successfully registered with router")
        # （说明：异常处理）  # 注释：自动行注释
        except Exception as e:
            # （说明：执行语句）  # 注释：自动行注释
            logger.error(f"Failed to register with router: {e}")
            # （说明：原注释说明）  # 注释：自动行注释
            # Don't raise here - server can still work without router

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def _make_request(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        endpoint: str,
        # （说明：执行语句）  # 注释：自动行注释
        payload: Optional[dict[str, Any]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        method: str = "POST",
        # （说明：执行语句）  # 注释：自动行注释
        timeout: float = DEFAULT_TIMEOUT,
        # （说明：执行语句）  # 注释：自动行注释
        only_master: bool = True,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：_make_request 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _make_request(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::_make_request。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：条件分支）  # 注释：自动行注释
        if only_master and self.node_rank != 0:
            # （说明：返回结果）  # 注释：自动行注释
            return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：循环逻辑）  # 注释：自动行注释
        for attempt in range(self.max_attempts):
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：条件分支）  # 注释：自动行注释
                if method.upper() == "GET":
                    # （说明：执行语句）  # 注释：自动行注释
                    response = requests.get(url, timeout=self.timeout)
                # （说明：条件分支）  # 注释：自动行注释
                else:
                    # （说明：执行语句）  # 注释：自动行注释
                    response = requests.post(url, json=payload or {}, timeout=self.timeout)

# （空行说明：保持段落分隔）  # 注释：空行占位
                # （说明：执行语句）  # 注释：自动行注释
                response.raise_for_status()
                # （说明：返回结果）  # 注释：自动行注释
                return _read_response(response)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：异常处理）  # 注释：自动行注释
            except requests.exceptions.Timeout:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Request to {endpoint} timed out (attempt {attempt + 1})")
            # （说明：异常处理）  # 注释：自动行注释
            except requests.exceptions.ConnectionError:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            # （说明：异常处理）  # 注释：自动行注释
            except requests.exceptions.HTTPError as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.error(f"HTTP error for {endpoint}: {e}")
                # （说明：执行语句）  # 注释：自动行注释
                raise
            # （说明：异常处理）  # 注释：自动行注释
            except Exception as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.error(f"Unexpected error for {endpoint}: {e}")
                # （说明：条件分支）  # 注释：自动行注释
                if attempt == self.max_attempts - 1:
                    # （说明：执行语句）  # 注释：自动行注释
                    raise

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：条件分支）  # 注释：自动行注释
            if attempt < self.max_attempts - 1:
                # （说明：执行语句）  # 注释：自动行注释
                time.sleep(self.retry_delay * (2**attempt))

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：抛出异常）  # 注释：自动行注释
        raise RuntimeError(f"Failed to complete request to {endpoint} after {self.max_attempts} attempts")

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def update_weights_from_tensor(self, req: UpdateWeightsFromTensorReqInput) -> dict[str, Any]:
        # （说明：导入依赖）  # 注释：自动行注释
        import base64

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        named_tensors = req.serialized_named_tensors
        # （说明：执行语句）  # 注释：自动行注释
        load_format = req.load_format
        # （说明：执行语句）  # 注释：自动行注释
        flush_cache = req.flush_cache

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：条件分支）  # 注释：自动行注释
        if named_tensors:
        """Update model weights from tensor data.

        The HTTP server will only post meta data, and the real weights will be
        copied directly from GPUs.

        Args:
            serialized_named_tensors (List[str]): List of serialized tensor data
            load_format (Optional[str], optional): Format specification for loading weights.
                Defaults to None.
            flush_cache (bool, optional): Whether to flush cache after updating weights.
                Defaults to False.

        Returns:
            Dict[str, Any]: Server response containing update status

        Note:
            The model should be on GPUs rather than CPU for this functionality to work properly.
            If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        功能：update_weights_from_tensor 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - update_weights_from_tensor(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::update_weights_from_tensor。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
            # （说明：执行语句）  # 注释：自动行注释
            serialized_named_tensors = [
                # （说明：执行语句）  # 注释：自动行注释
                base64.b64encode(named_tensor).decode("utf-8") for named_tensor in named_tensors
            # （说明：执行语句）  # 注释：自动行注释
            ]
        # （说明：条件分支）  # 注释：自动行注释
        else:
            # （说明：执行语句）  # 注释：自动行注释
            serialized_named_tensors = []

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return self._make_request(
            # （说明：执行语句）  # 注释：自动行注释
            "update_weights_from_tensor",
            # （说明：执行语句）  # 注释：自动行注释
            {
                # （说明：执行语句）  # 注释：自动行注释
                "serialized_named_tensors": serialized_named_tensors,
                # （说明：执行语句）  # 注释：自动行注释
                "load_format": load_format,
                # （说明：执行语句）  # 注释：自动行注释
                "flush_cache": flush_cache,
            # （说明：执行语句）  # 注释：自动行注释
            },
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def shutdown(self) -> None:
        # （说明：原注释说明）  # 注释：自动行注释
        # Unregister from router
        # （说明：条件分支）  # 注释：自动行注释
        """Shutdown the HTTP server and clean up resources.

        This method performs the following cleanup operations:
        1. Unregisters the worker from the router (if configured)
        2. Terminates the server process tree

        All operations are performed with error handling to ensure graceful shutdown
        even if individual steps fail.

        Note:
            This method should be called when the adapter is no longer needed
            to ensure proper cleanup of resources and processes.
        功能：shutdown 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - shutdown(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::shutdown。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        if self.router_ip and self.router_port:
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：执行语句）  # 注释：自动行注释
                url = f"http://{self.router_ip}:{self.router_port}/remove_worker"
                # （说明：执行语句）  # 注释：自动行注释
                params = {"url": f"http://{self.server_args.host}:{self.server_args.port}"}
                # （说明：执行语句）  # 注释：自动行注释
                requests.post(url, params=params, timeout=5.0)  # Short timeout for shutdown
                # （说明：执行语句）  # 注释：自动行注释
                logger.info("Successfully unregistered from router")
            # （说明：异常处理）  # 注释：自动行注释
            except Exception as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Failed to unregister from router: {e}")

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Kill server process
        # （说明：条件分支）  # 注释：自动行注释
        if hasattr(self, "process") and self.process is not None:
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：执行语句）  # 注释：自动行注释
                kill_process_tree(self.process.pid)
                # （说明：执行语句）  # 注释：自动行注释
                logger.info("Server process terminated")
            # （说明：异常处理）  # 注释：自动行注释
            except Exception as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.error(f"Failed to terminate server process: {e}")

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def generate(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        prompt: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        sampling_params: Optional[dict[str, Any]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        input_ids: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[Any] = None,
        # （说明：执行语句）  # 注释：自动行注释
        return_logprob: bool = False,
        # （说明：执行语句）  # 注释：自动行注释
        logprob_start_len: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        top_logprobs_num: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        token_ids_logprob: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        lora_path: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        custom_logit_processor: Optional[Callable] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
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
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::generate。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        payload = {
            # （说明：执行语句）  # 注释：自动行注释
            "text": prompt,
            # （说明：执行语句）  # 注释：自动行注释
            "sampling_params": sampling_params,
            # （说明：执行语句）  # 注释：自动行注释
            "input_ids": input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            "image_data": image_data,
            # （说明：执行语句）  # 注释：自动行注释
            "return_logprob": return_logprob,
            # （说明：执行语句）  # 注释：自动行注释
            "logprob_start_len": logprob_start_len,
            # （说明：执行语句）  # 注释：自动行注释
            "top_logprobs_num": top_logprobs_num,
            # （说明：执行语句）  # 注释：自动行注释
            "token_ids_logprob": token_ids_logprob,
            # （说明：执行语句）  # 注释：自动行注释
            "lora_path": lora_path,
            # （说明：执行语句）  # 注释：自动行注释
            "custom_logit_processor": custom_logit_processor,
        # （说明：执行语句）  # 注释：自动行注释
        }
        # （说明：原注释说明）  # 注释：自动行注释
        # Filter out None values
        # （说明：执行语句）  # 注释：自动行注释
        payload = {k: v for k, v in payload.items() if v is not None}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return self._make_request("generate", payload, only_master=False)

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def reward_score(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        prompt: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        input_ids: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[Any] = None,
        # （说明：执行语句）  # 注释：自动行注释
        lora_path: Optional[str] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：reward_score 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - reward_score(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::reward_score。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：断言检查）  # 注释：自动行注释
        assert self.server_args.is_embedding, "Score is only supported for embedding models"
        # （说明：执行语句）  # 注释：自动行注释
        payload = {
            # （说明：执行语句）  # 注释：自动行注释
            "text": prompt,
            # （说明：执行语句）  # 注释：自动行注释
            "input_ids": input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            "image_data": image_data,
            # （说明：执行语句）  # 注释：自动行注释
            "lora_path": lora_path,
        # （说明：执行语句）  # 注释：自动行注释
        }
        # （说明：原注释说明）  # 注释：自动行注释
        # Filter out None values
        # （说明：执行语句）  # 注释：自动行注释
        payload = {k: v for k, v in payload.items() if v is not None}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return self._make_request("classify", payload, only_master=False)

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def flush_cache(self) -> dict[str, Any]:
        # （说明：条件分支）  # 注释：自动行注释
        """Flush the cache of the server.

        This method repeatedly attempts to flush the server cache until successful.
        The flush operation will not return status 200 when there are pending requests.

        Returns:
            Dict[str, Any]: Server response indicating cache flush status.
                For non-master nodes, returns empty dict.

        Note:
            Uses retry logic with limited attempts (max_attempts * 2) to avoid infinite loops.
            Each retry includes a delay to allow pending requests to complete.
        功能：flush_cache 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - flush_cache(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::flush_cache。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        if self.node_rank != 0:
            # （说明：返回结果）  # 注释：自动行注释
            return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Use retry logic with limited attempts to avoid infinite loops
        # （说明：循环逻辑）  # 注释：自动行注释
        for attempt in range(self.max_attempts * 2):  # Allow more retries for cache flush
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：执行语句）  # 注释：自动行注释
                response = requests.get(
                    # （说明：执行语句）  # 注释：自动行注释
                    f"http://{self.server_args.host}:{self.server_args.port}/flush_cache", timeout=self.timeout
                # （说明：执行语句）  # 注释：自动行注释
                )
                # （说明：条件分支）  # 注释：自动行注释
                if response.status_code == 200:
                    # （说明：返回结果）  # 注释：自动行注释
                    return _read_response(response)
            # （说明：异常处理）  # 注释：自动行注释
            except Exception as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            time.sleep(self.retry_delay)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        logger.error("Failed to flush cache after maximum attempts")
        # （说明：返回结果）  # 注释：自动行注释
        return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        # （说明：返回结果）  # 注释：自动行注释
        return self._make_request("release_memory_occupation", {"tags": tags})

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        """Release GPU memory occupation temporarily.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        功能：release_memory_occupation 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - release_memory_occupation(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::release_memory_occupation。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：返回结果）  # 注释：自动行注释
        return self._make_request("resume_memory_occupation", {"tags": tags})

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def abort_request(self, rid: str = "", abort_all: bool = False) -> dict[str, Any]:
        # （说明：返回结果）  # 注释：自动行注释
        return self._make_request("abort_request", {"rid": rid, "abort_all": abort_all})

# （空行说明：保持段落分隔）  # 注释：空行占位

# （空行说明：保持段落分隔）  # 注释：空行占位
# （说明：定义类）  # 注释：自动行注释
class AsyncHttpServerAdapter(HttpServerAdapter):
        """Abort a request.

        Args:
            rid (str): The ID of the request to abort
            abort_all (bool, optional): Whether to abort all requests. Defaults to False.

        Returns:
            Dict[str, Any]: Server response indicating abort status
        功能：abort_request 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - abort_request(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::abort_request。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    def __init__(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        router_ip: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        router_port: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        timeout: float = DEFAULT_TIMEOUT,
        # （说明：执行语句）  # 注释：自动行注释
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        # （说明：执行语句）  # 注释：自动行注释
        retry_delay: float = DEFAULT_RETRY_DELAY,
        # （说明：执行语句）  # 注释：自动行注释
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        # （说明：执行语句）  # 注释：自动行注释
        first_rank_in_node: bool = False,
        # （说明：执行语句）  # 注释：自动行注释
        launch_server: bool = True,
        # （说明：执行语句）  # 注释：自动行注释
        **kwargs: Any,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> None:
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
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::__init__。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        super().__init__(
            # （说明：执行语句）  # 注释：自动行注释
            router_ip,
            # （说明：执行语句）  # 注释：自动行注释
            router_port,
            # （说明：执行语句）  # 注释：自动行注释
            timeout,
            # （说明：执行语句）  # 注释：自动行注释
            max_attempts,
            # （说明：执行语句）  # 注释：自动行注释
            retry_delay,
            # （说明：执行语句）  # 注释：自动行注释
            first_rank_in_node,
            # （说明：执行语句）  # 注释：自动行注释
            launch_server=launch_server,
            # （说明：执行语句）  # 注释：自动行注释
            **kwargs,
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        self.max_connections: int = max_connections

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：装饰器声明）  # 注释：自动行注释
    @asynccontextmanager
    # （说明：定义函数）  # 注释：自动行注释
    async def _get_session(self) -> aiohttp.ClientSession:
        # （说明：原注释说明）  # 注释：自动行注释
        # Create a new session for each request to avoid resource competition
        # （说明：执行语句）  # 注释：自动行注释
        connector = aiohttp.TCPConnector(
            # （说明：执行语句）  # 注释：自动行注释
            limit=self.max_connections,
            # （说明：执行语句）  # 注释：自动行注释
            limit_per_host=self.max_connections // 4,
            # （说明：执行语句）  # 注释：自动行注释
            ttl_dns_cache=300,
            # （说明：执行语句）  # 注释：自动行注释
            use_dns_cache=True,
        # （说明：执行语句）  # 注释：自动行注释
        )
        # （说明：执行语句）  # 注释：自动行注释
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        # （说明：执行语句）  # 注释：自动行注释
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：异常处理）  # 注释：自动行注释
        try:
        """Context manager for safe session access with proper connection pooling.

        Yields:
            aiohttp.ClientSession: Session instance for making HTTP requests

        Note:
            This method creates a new session for each request to avoid resource competition
            while still maintaining proper connection pooling through the shared connector.
        功能：_get_session 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _get_session(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::_get_session。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
            # （说明：执行语句）  # 注释：自动行注释
            yield session
        # （说明：异常处理）  # 注释：自动行注释
        finally:
            # （说明：原注释说明）  # 注释：自动行注释
            # Always close the session to free up resources
            # （说明：条件分支）  # 注释：自动行注释
            if not session.closed:
                # （说明：执行语句）  # 注释：自动行注释
                await session.close()

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def _make_async_request(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        endpoint: str,
        # （说明：执行语句）  # 注释：自动行注释
        payload: Optional[dict[str, Any]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        method: str = "POST",
        # （说明：执行语句）  # 注释：自动行注释
        timeout: float = DEFAULT_TIMEOUT,
        # （说明：执行语句）  # 注释：自动行注释
        only_master: bool = True,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：_make_async_request 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - _make_async_request(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::_make_async_request。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：条件分支）  # 注释：自动行注释
        if only_master and self.node_rank != 0:
            # （说明：返回结果）  # 注释：自动行注释
            return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：循环逻辑）  # 注释：自动行注释
        for attempt in range(self.max_attempts):
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：执行语句）  # 注释：自动行注释
                async with self._get_session() as session:
                    # （说明：条件分支）  # 注释：自动行注释
                    if method.upper() == "GET":
                        # （说明：执行语句）  # 注释：自动行注释
                        async with session.get(url, timeout=timeout) as response:
                            # （说明：执行语句）  # 注释：自动行注释
                            response.raise_for_status()
                            # （说明：返回结果）  # 注释：自动行注释
                            return await _read_async_response(response)
                    # （说明：条件分支）  # 注释：自动行注释
                    else:
                        # （说明：执行语句）  # 注释：自动行注释
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            # （说明：执行语句）  # 注释：自动行注释
                            response.raise_for_status()
                            # （说明：返回结果）  # 注释：自动行注释
                            return await _read_async_response(response)

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：异常处理）  # 注释：自动行注释
            except asyncio.TimeoutError:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            # （说明：异常处理）  # 注释：自动行注释
            except aiohttp.ClientConnectorError:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            # （说明：异常处理）  # 注释：自动行注释
            except aiohttp.ClientResponseError as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.error(f"HTTP error for {endpoint}: {e}")
                # （说明：执行语句）  # 注释：自动行注释
                raise
            # （说明：异常处理）  # 注释：自动行注释
            except Exception as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.error(f"Unexpected error for {endpoint}: {e}")
                # （说明：条件分支）  # 注释：自动行注释
                if attempt == self.max_attempts - 1:
                    # （说明：执行语句）  # 注释：自动行注释
                    raise

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：条件分支）  # 注释：自动行注释
            if attempt < self.max_attempts - 1:
                # （说明：执行语句）  # 注释：自动行注释
                await asyncio.sleep(self.retry_delay * (2**attempt))

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：抛出异常）  # 注释：自动行注释
        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts")

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def release_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        # （说明：返回结果）  # 注释：自动行注释
        return await self._make_async_request("release_memory_occupation", {"tags": tags})

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def resume_memory_occupation(self, tags: Optional[list[str]] = None) -> dict[str, Any]:
        """Release GPU memory occupation temporarily (async version).

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        功能：release_memory_occupation 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - release_memory_occupation(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::release_memory_occupation。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：返回结果）  # 注释：自动行注释
        return await self._make_async_request("resume_memory_occupation", {"tags": tags})

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def update_weights_from_tensor(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        req: UpdateWeightsFromTensorReqInput,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：update_weights_from_tensor 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - update_weights_from_tensor(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::update_weights_from_tensor。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：导入依赖）  # 注释：自动行注释
        import base64

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        named_tensors = req.serialized_named_tensors
        # （说明：执行语句）  # 注释：自动行注释
        load_format = req.load_format
        # （说明：执行语句）  # 注释：自动行注释
        flush_cache = req.flush_cache

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        serialized_named_tensors = [base64.b64encode(named_tensor).decode("utf-8") for named_tensor in named_tensors]
        # （说明：返回结果）  # 注释：自动行注释
        return await self._make_async_request(
            # （说明：执行语句）  # 注释：自动行注释
            "update_weights_from_tensor",
            # （说明：执行语句）  # 注释：自动行注释
            {
                # （说明：执行语句）  # 注释：自动行注释
                "serialized_named_tensors": serialized_named_tensors,
                # （说明：执行语句）  # 注释：自动行注释
                "load_format": load_format,
                # （说明：执行语句）  # 注释：自动行注释
                "flush_cache": flush_cache,
            # （说明：执行语句）  # 注释：自动行注释
            },
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def flush_cache(self) -> dict[str, Any]:
        # （说明：条件分支）  # 注释：自动行注释
        """Flush the cache of the server asynchronously.

        Similar to the sync version, this method retries until the cache
        is successfully flushed. It uses async sleep between retries.

        Returns:
            Dict[str, Any]: Server response indicating cache flush status.
                For non-master nodes, returns empty dict.

        Note:
            Uses retry logic with limited attempts (max_attempts * 4) to avoid infinite loops.
            Each retry includes an async delay to allow pending requests to complete.
        功能：flush_cache 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - flush_cache(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::flush_cache。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        if self.node_rank != 0:
            # （说明：返回结果）  # 注释：自动行注释
            return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Use retry logic with limited attempts to avoid infinite loops
        # （说明：循环逻辑）  # 注释：自动行注释
        for attempt in range(self.max_attempts * 4):  # Allow more retries for cache flush
            # （说明：异常处理）  # 注释：自动行注释
            try:
                # （说明：执行语句）  # 注释：自动行注释
                async with self._get_session() as session:
                    # （说明：执行语句）  # 注释：自动行注释
                    url = f"http://{self.server_args.host}:{self.server_args.port}/flush_cache"
                    # （说明：执行语句）  # 注释：自动行注释
                    async with session.get(url) as response:
                        # （说明：条件分支）  # 注释：自动行注释
                        if response.status == 200:
                            # （说明：返回结果）  # 注释：自动行注释
                            return await _read_async_response(response)
            # （说明：异常处理）  # 注释：自动行注释
            except Exception as e:
                # （说明：执行语句）  # 注释：自动行注释
                logger.warning(f"Error flushing cache (attempt {attempt + 1}): {e}")

# （空行说明：保持段落分隔）  # 注释：空行占位
            # （说明：执行语句）  # 注释：自动行注释
            await asyncio.sleep(self.retry_delay)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        logger.error("Failed to flush cache after maximum attempts")
        # （说明：返回结果）  # 注释：自动行注释
        return {}

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def generate(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        prompt: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        sampling_params: Optional[dict[str, Any]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        input_ids: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[Any] = None,
        # （说明：执行语句）  # 注释：自动行注释
        return_logprob: bool = False,
        # （说明：执行语句）  # 注释：自动行注释
        logprob_start_len: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        top_logprobs_num: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        token_ids_logprob: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        lora_path: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        custom_logit_processor: Optional[Callable] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
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
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::generate。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        logger.info("generate() started")

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：执行语句）  # 注释：自动行注释
        payload = {
            # （说明：执行语句）  # 注释：自动行注释
            "text": prompt,
            # （说明：执行语句）  # 注释：自动行注释
            "sampling_params": sampling_params,
            # （说明：执行语句）  # 注释：自动行注释
            "input_ids": input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            "image_data": image_data,
            # （说明：执行语句）  # 注释：自动行注释
            "return_logprob": return_logprob,
            # （说明：执行语句）  # 注释：自动行注释
            "logprob_start_len": logprob_start_len,
            # （说明：执行语句）  # 注释：自动行注释
            "top_logprobs_num": top_logprobs_num,
            # （说明：执行语句）  # 注释：自动行注释
            "token_ids_logprob": token_ids_logprob,
            # （说明：执行语句）  # 注释：自动行注释
            "lora_path": lora_path,
            # （说明：执行语句）  # 注释：自动行注释
            "custom_logit_processor": custom_logit_processor,
        # （说明：执行语句）  # 注释：自动行注释
        }

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Filter out None values
        # （说明：执行语句）  # 注释：自动行注释
        payload = {k: v for k, v in payload.items() if v is not None}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Send request
        # （说明：执行语句）  # 注释：自动行注释
        response = await self._make_async_request("generate", payload, timeout=self.timeout, only_master=False)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return response

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def async_generate(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        prompt: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        sampling_params: Optional[dict[str, Any]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        input_ids: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[Any] = None,
        # （说明：执行语句）  # 注释：自动行注释
        return_logprob: bool = False,
        # （说明：执行语句）  # 注释：自动行注释
        logprob_start_len: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        top_logprobs_num: Optional[int] = None,
        # （说明：执行语句）  # 注释：自动行注释
        token_ids_logprob: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        lora_path: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        custom_logit_processor: Optional[Callable] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：async_generate 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - async_generate(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::async_generate。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：返回结果）  # 注释：自动行注释
        return await self.generate(
            # （说明：执行语句）  # 注释：自动行注释
            prompt=prompt,
            # （说明：执行语句）  # 注释：自动行注释
            sampling_params=sampling_params,
            # （说明：执行语句）  # 注释：自动行注释
            input_ids=input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            image_data=image_data,
            # （说明：执行语句）  # 注释：自动行注释
            return_logprob=return_logprob,
            # （说明：执行语句）  # 注释：自动行注释
            logprob_start_len=logprob_start_len,
            # （说明：执行语句）  # 注释：自动行注释
            top_logprobs_num=top_logprobs_num,
            # （说明：执行语句）  # 注释：自动行注释
            token_ids_logprob=token_ids_logprob,
            # （说明：执行语句）  # 注释：自动行注释
            lora_path=lora_path,
            # （说明：执行语句）  # 注释：自动行注释
            custom_logit_processor=custom_logit_processor,
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def reward_score(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        prompt: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        input_ids: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[Any] = None,
        # （说明：执行语句）  # 注释：自动行注释
        lora_path: Optional[str] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：reward_score 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - reward_score(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::reward_score。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：执行语句）  # 注释：自动行注释
        logger.info("reward_score() started")
        # （说明：执行语句）  # 注释：自动行注释
        payload = {
            # （说明：执行语句）  # 注释：自动行注释
            "text": prompt,
            # （说明：执行语句）  # 注释：自动行注释
            "input_ids": input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            "image_data": image_data,
            # （说明：执行语句）  # 注释：自动行注释
            "lora_path": lora_path,
        # （说明：执行语句）  # 注释：自动行注释
        }
        # （说明：原注释说明）  # 注释：自动行注释
        # Filter out None values
        # （说明：执行语句）  # 注释：自动行注释
        payload = {k: v for k, v in payload.items() if v is not None}

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：原注释说明）  # 注释：自动行注释
        # Send request
        # （说明：执行语句）  # 注释：自动行注释
        response = await self._make_async_request("classify", payload, timeout=self.timeout, only_master=False)

# （空行说明：保持段落分隔）  # 注释：空行占位
        # （说明：返回结果）  # 注释：自动行注释
        return response

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def async_reward_score(
        # （说明：执行语句）  # 注释：自动行注释
        self,
        # （说明：执行语句）  # 注释：自动行注释
        prompt: Optional[str] = None,
        # （说明：执行语句）  # 注释：自动行注释
        input_ids: Optional[list[int]] = None,
        # （说明：执行语句）  # 注释：自动行注释
        image_data: Optional[Any] = None,
        # （说明：执行语句）  # 注释：自动行注释
        lora_path: Optional[str] = None,
    # （说明：执行语句）  # 注释：自动行注释
    ) -> dict[str, Any]:
        """
        功能：async_reward_score 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - async_reward_score(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::async_reward_score。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：返回结果）  # 注释：自动行注释
        return await self.reward_score(
            # （说明：执行语句）  # 注释：自动行注释
            prompt=prompt,
            # （说明：执行语句）  # 注释：自动行注释
            input_ids=input_ids,
            # （说明：执行语句）  # 注释：自动行注释
            image_data=image_data,
            # （说明：执行语句）  # 注释：自动行注释
            lora_path=lora_path,
        # （说明：执行语句）  # 注释：自动行注释
        )

# （空行说明：保持段落分隔）  # 注释：空行占位
    # （说明：定义函数）  # 注释：自动行注释
    async def abort_request(self, rid: str = "", abort_all: bool = False) -> dict[str, Any]:
        """Abort a request asynchronously.

        Args:
            rid (str): The ID of the request to abort
            abort_all (bool, optional): Whether to abort all requests. Defaults to False.

        Returns:
            Dict[str, Any]: Server response indicating abort status
        功能：abort_request 的自动中文说明（需按实际逻辑细化）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - 见函数/类签名。  # 注释：参数占位
        返回：  # 注释：返回值说明标题
        - 详见实现（可能为 None 或结构体）。  # 注释：返回值占位
        副作用：可能执行 I/O/远程调用。  # 注释：副作用占位
        异常/边界条件：参数不合法可能抛异常。  # 注释：异常占位
        最小示例：  # 注释：最小示例标题
        - abort_request(...)  # 注释：示例占位
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/sglang_rollout/http_server_engine.py::abort_request。  # 注释：位置占位
        - 典型调用路径：待补充。  # 注释：调用链占位
        - 被谁调用：本文件内或上层组件。  # 注释：调用方占位
        - 调用了谁（项目内）：详见函数体。  # 注释：依赖占位
        - 调用了谁（关键外部依赖）：详见函数体。  # 注释：外部依赖占位
        """
        # （说明：返回结果）  # 注释：自动行注释
        return await self._make_async_request("abort_request", {"rid": rid, "abort_all": abort_all})
