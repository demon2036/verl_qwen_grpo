# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：分隔说明，保持逐行注释
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：获取许可证地址提示
#  # 注释：空行占位，保持逐行注释
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位，保持逐行注释
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无明示或暗示担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：提供 rollout HTTP 服务启动与端口选择等通用工具。  # 注释：模块用途
输入/输出：输入地址与 FastAPI app，输出空闲端口/任务句柄。  # 注释：输入输出概览
关键依赖：asyncio、socket、uvicorn、FastAPI。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- port, task = await run_unvicorn(app, server_args, "127.0.0.1")  # 注释：示例用法
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/rollout/vllm_rollout/vllm_async_server.py。  # 注释：上层入口举例
- 典型链路：rollout server 启动 -> get_free_port -> run_unvicorn。  # 注释：调用链路
"""  # 注释：模块 docstring 结束
# （分隔说明：标准库依赖）  # 注释：替代空行，保持逐行注释
import asyncio  # 注释：异步任务管理
import ipaddress  # 注释：IP 地址解析
import logging  # 注释：日志
import os  # 注释：系统退出
import socket  # 注释：端口绑定与套接字
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import uvicorn  # 注释：ASGI 服务器
from fastapi import FastAPI  # 注释：FastAPI 应用类型
# （分隔说明：logger）  # 注释：替代空行，保持逐行注释
logger = logging.getLogger(__file__)  # 注释：模块日志器
# （分隔说明：IPv6 判断）  # 注释：替代空行，保持逐行注释
def is_valid_ipv6_address(address: str) -> bool:  # 注释：检查 IPv6 地址合法性
    """判断字符串是否为合法 IPv6 地址。  # 注释：函数用途

    参数：  # 注释：参数说明标题
    - address (str)：IP 地址字符串。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：是否合法 IPv6。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：ValueError 被捕获并返回 False。  # 注释：异常说明
    最小示例：is_valid_ipv6_address("::1") -> True。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/utils.py::is_valid_ipv6_address。  # 注释：位置
    - 典型调用路径：get_free_port -> is_valid_ipv6_address。  # 注释：调用链
    - 被谁调用：get_free_port。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：ipaddress.IPv6Address。  # 注释：外部依赖
    """  # 注释：docstring 结束
    try:  # 注释：捕获非法地址
        ipaddress.IPv6Address(address)  # 注释：尝试解析 IPv6
        return True  # 注释：解析成功
    except ValueError:  # 注释：解析失败
        return False  # 注释：返回 False
# （分隔说明：端口获取）  # 注释：替代空行，保持逐行注释
def get_free_port(address: str) -> tuple[int, socket.socket]:  # 注释：获取空闲端口并返回 socket
    """根据地址类型绑定 0 端口获取空闲端口。  # 注释：函数用途

    参数：address (str)：监听地址。  # 注释：参数说明
    返回：  # 注释：返回值说明标题
    - (port, sock)：端口号与已绑定 socket。  # 注释：返回值语义
    副作用：创建并绑定 socket。  # 注释：副作用说明
    异常/边界条件：绑定失败抛 OSError。  # 注释：异常说明
    最小示例：port, sock = get_free_port("127.0.0.1")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/utils.py::get_free_port。  # 注释：位置
    - 典型调用路径：run_unvicorn -> get_free_port。  # 注释：调用链
    - 被谁调用：run_unvicorn。  # 注释：调用方说明
    - 调用了谁（项目内）：is_valid_ipv6_address。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：socket.socket。  # 注释：外部依赖
    """  # 注释：docstring 结束
    family = socket.AF_INET  # 注释：默认 IPv4
    if is_valid_ipv6_address(address):  # 注释：若为 IPv6
        family = socket.AF_INET6  # 注释：使用 IPv6 协议族
# （分隔说明：创建并绑定 socket）  # 注释：替代空行，保持逐行注释
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)  # 注释：创建 TCP socket
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 注释：允许地址复用
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # 注释：允许端口复用
    sock.bind((address, 0))  # 注释：绑定到任意空闲端口
# （分隔说明：读取端口）  # 注释：替代空行，保持逐行注释
    port = sock.getsockname()[1]  # 注释：获取分配的端口号
    return port, sock  # 注释：返回端口和 socket
# （分隔说明：启动 uvicorn）  # 注释：替代空行，保持逐行注释
async def run_unvicorn(app: FastAPI, server_args, server_address, max_retries=5) -> tuple[int, asyncio.Task]:  # 注释：启动 HTTP 服务
    """异步启动 uvicorn 服务，失败重试并返回端口与任务。  # 注释：函数用途

    参数：  # 注释：参数说明标题
    - app (FastAPI)：应用实例。  # 注释：参数含义
    - server_args：额外参数（透传给 app）。  # 注释：参数含义
    - server_address (str)：绑定地址。  # 注释：参数含义
    - max_retries (int)：最大重试次数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (port, task)：启动端口与主循环任务。  # 注释：返回值语义
    副作用：启动 HTTP 服务，占用端口。  # 注释：副作用说明
    异常/边界条件：多次失败后强制退出进程。  # 注释：异常说明
    最小示例：port, task = await run_unvicorn(app, args, "0.0.0.0")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/utils.py::run_unvicorn。  # 注释：位置
    - 典型调用路径：rollout server 启动 -> run_unvicorn。  # 注释：调用链
    - 被谁调用：vLLM/SGLang async server 模块。  # 注释：调用方说明
    - 调用了谁（项目内）：get_free_port。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：uvicorn.Config/Server。  # 注释：外部依赖
    """  # 注释：docstring 结束
    server_port, server_task = None, None  # 注释：初始化返回值
# （分隔说明：重试启动）  # 注释：替代空行，保持逐行注释
    for i in range(max_retries):  # 注释：循环重试
        try:  # 注释：捕获启动异常
            server_port, sock = get_free_port(server_address)  # 注释：获取空闲端口
            app.server_args = server_args  # 注释：挂载 server 参数到 app
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")  # 注释：构建 uvicorn 配置
            server = uvicorn.Server(config)  # 注释：创建服务器实例
            server.should_exit = True  # 注释：标记退出（由 main_loop 控制）
            await server.serve()  # 注释：启动服务（初始化）
            server_task = asyncio.create_task(server.main_loop())  # 注释：启动主循环任务
            break  # 注释：启动成功，退出重试
        except (OSError, SystemExit) as e:  # 注释：捕获端口占用或系统退出
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")  # 注释：记录错误
    else:  # 注释：多次失败仍未成功
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")  # 注释：记录最终失败
        os._exit(-1)  # 注释：强制退出进程
# （分隔说明：成功日志）  # 注释：替代空行，保持逐行注释
    logger.info(f"HTTP server started on port {server_port}")  # 注释：记录成功启动
    return server_port, server_task  # 注释：返回端口与任务
