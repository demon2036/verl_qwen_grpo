# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
模块用途：提供简单的异步负载均衡路由（FastAPI + aiohttp）。（注释：模块功能概述）
输入：worker_urls、HTTP 请求。（注释：输入形态说明）
输出：转发后的响应 JSON。（注释：输出形态说明）
关键依赖：fastapi、aiohttp、uvicorn、ray。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - addr, proc = launch_router_process(worker_urls=[...])
调用路径概览：（注释：全局调用关系）
  - reward_model.py::_initialize_router -> launch_router_process
"""

import asyncio  # 注释：异步调度与等待
import logging  # 注释：日志记录
import multiprocessing  # 注释：启动独立进程
import os  # 注释：环境变量
import time  # 注释：时间等待
from typing import Any  # 注释：类型提示

import aiohttp  # 注释：异步 HTTP 客户端
import ray  # 注释：获取本机 IP
import uvicorn  # 注释：运行 FastAPI 服务
from fastapi import FastAPI, Request  # 注释：Web 框架
from fastapi.responses import JSONResponse  # 注释：JSON 响应

from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address  # 注释：端口与 IPv6 工具

logger = logging.getLogger(__name__)  # 注释：模块级日志器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：日志级别


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:  # 注释：读取响应体
    """
    功能：安全解析 aiohttp 响应为 JSON 或文本结构。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - resp (aiohttp.ClientResponse): HTTP 响应对象。（注释：响应对象）
    返回：（注释：返回值说明）
      - dict[str, Any]：解析后的内容（可能为空字典）。（注释：返回结构）
    副作用：（注释：副作用说明）
      - 读取响应体。（注释：I/O 副作用）
    异常/边界条件：（注释：异常与边界）
      - JSON 解析失败时回退到文本。（注释：回退逻辑）
    最小示例：（注释：最小可理解示例）
      - 输入：resp.status=200 且 JSON 响应
      - 输出：解析后的 dict
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::_read_async_response`
      - 典型调用路径：`NaiveRouter._make_async_request` -> `_read_async_response`
      - 被谁调用：`NaiveRouter._make_async_request`
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：`aiohttp.ClientResponse.json`/`text`
    """
    if resp.status == 204 or (resp.content_length == 0):  # 注释：无内容响应直接返回空字典
        return {}  # 注释：空响应

    try:  # 注释：优先解析 JSON
        return await resp.json(content_type=None)  # 注释：忽略 content_type 限制
    except Exception:  # 注释：JSON 解析失败
        try:  # 注释：回退读取文本
            text = await resp.text()  # 注释：读取文本
        except Exception:  # 注释：文本读取失败
            return {}  # 注释：返回空字典
        return {  # 注释：返回包含文本与内容类型
            "content_type": (resp.headers.get("Content-Type") or ""),
            "text": text,
        }


def launch_router_process(  # 注释：启动 router 子进程
    worker_urls: list[str],
):
    """
    功能：启动独立进程运行 NaiveRouter，并返回地址与进程句柄。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - worker_urls (list[str]): 下游 worker URL 列表。（注释：worker 地址）
    返回：（注释：返回值说明）
      - (router_address, router_process)：路由地址与进程对象。（注释：返回结构）
    副作用：（注释：副作用说明）
      - 启动新进程并占用端口。（注释：进程/端口副作用）
    异常/边界条件：（注释：异常与边界）
      - 进程启动失败会触发断言。（注释：启动检查）
    最小示例：（注释：最小可理解示例）
      - 输入：launch_router_process(["http://127.0.0.1:8000"])
      - 输出：("127.0.0.1:9000", Process(...))
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::launch_router_process`
      - 典型调用路径：`RewardModelManager._initialize_router` -> `launch_router_process`
      - 被谁调用：`RewardModelManager`
      - 调用了谁（项目内）：`run_router`
      - 调用了谁（外部依赖）：`multiprocessing.Process`
    """
    router_ip = ray.util.get_node_ip_address().strip("[]")  # 注释：获取本机 IP
    router_port, _ = get_free_port(router_ip)  # 注释：分配空闲端口
    router_address = (  # 注释：构造地址字符串（兼容 IPv6）
        f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
    )

    router_process = multiprocessing.Process(  # 注释：创建进程运行 router
        target=run_router,
        args=(
            router_ip,
            router_port,
            worker_urls,
        ),
    )
    router_process.daemon = True  # 注释：随主进程退出
    router_process.start()  # 注释：启动进程
    time.sleep(3)  # 注释：等待进程就绪
    assert router_process.is_alive()  # 注释：确认进程存活

    logger.info(f"Router is running on {router_address}")  # 注释：记录 router 地址
    return router_address, router_process  # 注释：返回地址与进程


def run_router(router_ip: str, router_port: int, worker_urls: list[str]):  # 注释：运行 router 服务
    """
    功能：在当前进程中启动 NaiveRouter 的 FastAPI 服务。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - router_ip (str): 监听 IP。（注释：监听地址）
      - router_port (int): 监听端口。（注释：端口）
      - worker_urls (list[str]): 下游 worker URL。（注释：worker 列表）
    返回：（注释：返回值说明）
      - None。（注释：uvicorn 运行阻塞）
    副作用：（注释：副作用说明）
      - 启动 HTTP 服务并占用端口。（注释：网络副作用）
    异常/边界条件：（注释：异常与边界）
      - 端口占用会导致启动失败。（注释：端口冲突）
    最小示例：（注释：最小可理解示例）
      - 输入：run_router("127.0.0.1", 8000, ["http://..."])
      - 输出：启动 uvicorn 服务
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::run_router`
      - 典型调用路径：`launch_router_process` -> `run_router`
      - 被谁调用：`launch_router_process`
      - 调用了谁（项目内）：`NaiveRouter`
      - 调用了谁（外部依赖）：`uvicorn.run`
    """
    router = NaiveRouter(worker_urls=worker_urls, verbose=False)  # 注释：创建 router 实例
    uvicorn.run(router.app, host=router_ip, port=router_port, log_level="warning")  # 注释：启动服务


class NaiveRouter:  # 注释：简单异步负载均衡路由器
    """
    功能：将请求转发到多个 worker，并进行简单的负载均衡。（注释：类职责）
    输入：worker_urls、路由配置参数。（注释：输入形态说明）
    输出：转发后的响应。（注释：输出形态说明）
    关键依赖：FastAPI、aiohttp。（注释：关键依赖）
    典型用法：（注释：最小使用示例）
      - router = NaiveRouter(worker_urls=[...])
      - uvicorn.run(router.app, ...)
    调用路径概览：（注释：全局调用关系）
      - run_router -> NaiveRouter -> FastAPI app
    """

    def __init__(  # 注释：初始化路由器
        self,
        worker_urls: list[str],
        max_connections: int = 1024,
        timeout: int = 60,
        max_attempts: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = False,
    ) -> None:
        """
        功能：初始化 FastAPI 应用与连接池配置。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - worker_urls (list[str]): 下游 worker URL。（注释：worker 列表）
          - max_connections (int): 连接池上限。（注释：连接上限）
          - timeout (int): 请求超时秒数。（注释：超时配置）
          - max_attempts (int): 最大重试次数。（注释：重试上限）
          - retry_delay (float): 重试基础等待秒数。（注释：退避基数）
          - verbose (bool): 是否输出详细日志。（注释：日志开关）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 创建 FastAPI 应用并注册事件。（注释：应用副作用）
        异常/边界条件：（注释：异常与边界）
          - worker_urls 为空时请求会返回 503。（注释：空列表）
        最小示例：（注释：最小可理解示例）
          - 输入：NaiveRouter(["http://127.0.0.1:8000"])
          - 输出：router.app 可被 uvicorn 运行
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::NaiveRouter.__init__`
          - 典型调用路径：`run_router` -> `NaiveRouter`
          - 被谁调用：`run_router`
          - 调用了谁（项目内）：`_on_startup`、`_on_shutdown`、`_make_async_request`
          - 调用了谁（外部依赖）：`FastAPI`
        """
        self.verbose = verbose  # 注释：保存日志开关
        self.app = FastAPI()  # 注释：创建 FastAPI 应用
        self.worker_urls = worker_urls  # 注释：保存 worker 列表
        self.request_counts = {url: 0 for url in worker_urls}  # 注释：请求计数（负载均衡）

        self.max_connections = max_connections  # 注释：连接池上限
        self.timeout = timeout  # 注释：请求超时
        self.max_attempts = max_attempts  # 注释：最大重试次数
        self.retry_delay = retry_delay  # 注释：重试延迟

        self.app = FastAPI()  # 注释：重新创建 FastAPI 应用（保持与原逻辑一致）

        # Register startup / shutdown hooks  # 注释：注册启动/关闭事件
        self.app.on_event("startup")(self._on_startup)  # 注释：注册启动回调
        self.app.on_event("shutdown")(self._on_shutdown)  # 注释：注册关闭回调

        # Catch-all proxy route  # 注释：注册转发路由
        self.app.api_route("/{endpoint:path}", methods=["GET", "POST"])(self._make_async_request)  # 注释：所有路径代理

        # Placeholder for aiohttp client  # 注释：aiohttp 客户端占位
        self.client = None  # 注释：将在 startup 初始化

    async def _on_startup(self):  # 注释：应用启动回调
        """
        功能：初始化 aiohttp 客户端与连接池。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 创建 aiohttp.ClientSession。（注释：网络资源）
        异常/边界条件：（注释：异常与边界）
          - 连接池创建失败会抛异常。（注释：初始化失败）
        最小示例：（注释：最小可理解示例）
          - 输入：FastAPI startup 触发
          - 输出：self.client 可用
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::NaiveRouter._on_startup`
          - 典型调用路径：FastAPI startup -> `_on_startup`
          - 被谁调用：FastAPI 生命周期
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`aiohttp.ClientSession`
        """
        connector = aiohttp.TCPConnector(  # 注释：创建连接器
            limit=self.max_connections,
            limit_per_host=self.max_connections // 4,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=None)  # 注释：创建超时配置
        self.client = aiohttp.ClientSession(connector=connector, timeout=timeout)  # 注释：创建会话
        if self.verbose:  # 注释：可选日志
            logger.info(f"[router] aiohttp client initialized with max_connections={self.max_connections}")  # 注释：记录日志

    async def _on_shutdown(self):  # 注释：应用关闭回调
        """
        功能：安全关闭 aiohttp 客户端。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 关闭网络连接。（注释：资源释放）
        异常/边界条件：（注释：异常与边界）
          - client 为空时直接返回。（注释：空检查）
        最小示例：（注释：最小可理解示例）
          - 输入：FastAPI shutdown 触发
          - 输出：client 关闭
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::NaiveRouter._on_shutdown`
          - 典型调用路径：FastAPI shutdown -> `_on_shutdown`
          - 被谁调用：FastAPI 生命周期
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`aiohttp.ClientSession.close`
        """
        if self.client and not self.client.closed:  # 注释：仅在 client 存在且未关闭时执行
            await self.client.close()  # 注释：关闭会话
            if self.verbose:  # 注释：可选日志
                logger.info("[router] aiohttp client closed")  # 注释：记录日志

    async def _make_async_request(self, request: Request, endpoint: str):  # 注释：代理请求
        """
        功能：将请求转发至选定 worker，并处理重试。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - request (Request): FastAPI 请求对象。（注释：原请求）
          - endpoint (str): 路由路径。（注释：路径参数）
        返回：（注释：返回值说明）
          - dict|JSONResponse：下游响应或错误响应。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 发起网络请求并更新负载计数。（注释：网络/状态副作用）
        异常/边界条件：（注释：异常与边界）
          - worker_urls 为空返回 503。（注释：无 worker）
          - 多次失败后抛 RuntimeError。（注释：重试耗尽）
        最小示例：（注释：最小可理解示例）
          - 输入：GET /v1/xxx
          - 输出：转发后的 JSON
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::NaiveRouter._make_async_request`
          - 典型调用路径：FastAPI 路由 -> `_make_async_request`
          - 被谁调用：FastAPI 路由系统
          - 调用了谁（项目内）：`_select_worker`、`_release_worker`、`_read_async_response`
          - 调用了谁（外部依赖）：`aiohttp.ClientSession.request`
        """
        if not self.worker_urls:  # 注释：无 worker 可用
            return JSONResponse(status_code=503, content={"error": "No available workers"})  # 注释：返回 503

        worker_url = self._select_worker()  # 注释：选择 worker
        target_url = f"{worker_url}/{endpoint}"  # 注释：拼接目标 URL

        if self.verbose:  # 注释：可选日志
            logger.debug(f"[router] Forwarding request → {target_url}")  # 注释：记录转发日志

        # Copy request data  # 注释：复制请求数据
        body = await request.body()  # 注释：读取请求体
        headers = dict(request.headers)  # 注释：复制请求头

        for attempt in range(self.max_attempts):  # 注释：重试循环
            # Send request to worker  # 注释：发送请求
            try:
                async with self.client.request(request.method, target_url, data=body, headers=headers) as response:  # 注释：代理请求
                    response.raise_for_status()  # 注释：非 2xx 抛异常
                    output = await _read_async_response(response)  # 注释：读取响应
                    self._release_worker(worker_url)  # 注释：释放 worker 计数
                    return output  # 注释：返回响应
            except asyncio.TimeoutError:  # 注释：超时
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")  # 注释：记录超时
            except aiohttp.ClientConnectorError:  # 注释：连接错误
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")  # 注释：记录错误
            except aiohttp.ClientResponseError as e:  # 注释：HTTP 错误
                logger.error(f"HTTP error for {endpoint}: {e}")  # 注释：记录错误
                raise  # 注释：直接抛出
            except Exception as e:  # 注释：其他异常
                logger.error(f"Unexpected error for {endpoint}: {e}")  # 注释：记录错误
                if attempt == self.max_attempts - 1:  # 注释：最后一次失败则抛出
                    raise  # 注释：异常向上抛

            if attempt < self.max_attempts - 1:  # 注释：未到最后一次则退避
                await asyncio.sleep(self.retry_delay * (2**attempt))  # 注释：指数退避等待

        raise RuntimeError(  # 注释：重试耗尽抛异常
            f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts"
        )

    def _select_worker(self) -> str:  # 注释：选择 worker
        """
        功能：选择当前负载最小的 worker。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - str：选中的 worker URL。（注释：返回值）
        副作用：（注释：副作用说明）
          - 更新 request_counts。（注释：状态更新）
        异常/边界条件：（注释：异常与边界）
          - request_counts 为空会抛 ValueError。（注释：空列表）
        最小示例：（注释：最小可理解示例）
          - 输入：request_counts={"a":0,"b":1}
          - 输出："a"
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::NaiveRouter._select_worker`
          - 典型调用路径：`_make_async_request` -> `_select_worker`
          - 被谁调用：`_make_async_request`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`min`
        """
        url = min(self.request_counts, key=self.request_counts.get)  # 注释：取最小负载的 URL
        self.request_counts[url] += 1  # 注释：计数+1
        return url  # 注释：返回 URL

    def _release_worker(self, url: str) -> None:  # 注释：释放 worker 负载计数
        """
        功能：请求完成后降低 worker 负载计数。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - url (str): worker URL。（注释：目标 worker）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 修改 request_counts。（注释：状态更新）
        异常/边界条件：（注释：异常与边界）
          - url 不存在会 KeyError。（注释：键缺失）
        最小示例：（注释：最小可理解示例）
          - 输入：_release_worker("http://a")
          - 输出：request_counts["http://a"] 减 1
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/router/naive_router.py::NaiveRouter._release_worker`
          - 典型调用路径：`_make_async_request` -> `_release_worker`
          - 被谁调用：`_make_async_request`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`max`
        """
        self.request_counts[url] = max(0, self.request_counts[url] - 1)  # 注释：计数减 1（下限 0）
