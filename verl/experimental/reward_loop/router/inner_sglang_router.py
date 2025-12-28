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
模块用途：通过 sglang_router 启动内部路由进程，并进行健康检查。（注释：模块功能概述）
输入：worker_urls、超时配置。（注释：输入形态说明）
输出：router 地址与进程句柄。（注释：输出形态说明）
关键依赖：sglang_router、requests、ray。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - addr, proc = launch_router_process(worker_urls=[...])
调用路径概览：（注释：全局调用关系）
  - reward_model.py::_initialize_router（若启用 sglang）
"""

import logging  # 注释：日志记录
import multiprocessing  # 注释：启动独立进程
import os  # 注释：环境变量
import time  # 注释：等待与超时

import ray  # 注释：获取本机 IP
import requests  # 注释：HTTP 客户端（健康检查）
from sglang_router.launch_server import RouterArgs, launch_router  # 注释：sglang router 启动工具

from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address  # 注释：端口与 IPv6 工具

logger = logging.getLogger(__name__)  # 注释：模块级日志器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：日志级别


def launch_router_process(  # 注释：启动 sglang router 进程
    worker_urls: list[str],
    request_timeout: int = 180,
    max_wait_time: int = 300,
    timeout: int = 30,
) -> str:
    """
    功能：启动 sglang_router 并进行健康检查，返回地址与进程。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - worker_urls (list[str]): 下游 worker URL。（注释：worker 列表）
      - request_timeout (int): 单请求超时秒数。（注释：请求超时）
      - max_wait_time (int): 健康检查最大等待秒数。（注释：等待上限）
      - timeout (int): health check 请求超时。（注释：健康检查超时）
    返回：（注释：返回值说明）
      - (router_address, router_process)：路由地址与进程。（注释：返回结构）
    副作用：（注释：副作用说明）
      - 启动新进程并占用端口。（注释：进程副作用）
      - 进行健康检查请求。（注释：网络副作用）
    异常/边界条件：（注释：异常与边界）
      - 健康检查失败会终止进程并抛异常。（注释：启动失败）
    最小示例：（注释：最小可理解示例）
      - 输入：launch_router_process(["http://127.0.0.1:8000"])
      - 输出：("127.0.0.1:9000", Process(...))
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/router/inner_sglang_router.py::launch_router_process`
      - 典型调用路径：`RewardModelManager._initialize_router` -> `launch_router_process`
      - 被谁调用：`RewardModelManager`（当启用 sglang）
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：`launch_router`、`requests.Session`
    """
    router_ip = ray.util.get_node_ip_address().strip("[]")  # 注释：获取本机 IP
    router_port, _ = get_free_port(router_ip)  # 注释：分配空闲端口
    router_address = (  # 注释：构造地址字符串（兼容 IPv6）
        f"[{router_ip}]:{router_port}" if is_valid_ipv6_address(router_ip) else f"{router_ip}:{router_port}"
    )
    router_args = RouterArgs(  # 注释：构造 router 参数
        host=router_ip,
        port=router_port,
        worker_urls=worker_urls,
        balance_abs_threshold=0,
        log_level="warn",
        request_timeout_secs=request_timeout,
    )
    router_process = multiprocessing.Process(target=launch_router, args=(router_args,))  # 注释：创建进程
    router_process.daemon = True  # 注释：随主进程退出
    router_process.start()  # 注释：启动进程
    time.sleep(3)  # 注释：等待进程就绪
    assert router_process.is_alive()  # 注释：确认进程存活

    # health check  # 注释：健康检查
    start_time = time.time()  # 注释：开始时间
    url = f"http://{router_address}/health"  # 注释：健康检查 URL
    with requests.Session() as session:  # 注释：复用会话
        while time.time() - start_time < max_wait_time:  # 注释：等待直到超时
            try:  # 注释：发送健康检查请求
                response = session.get(url, timeout=timeout)  # 注释：GET 请求
                if response.status_code == 200:  # 注释：成功则退出
                    break  # 注释：结束循环
            except requests.RequestException as e:  # 注释：请求失败
                logger.debug(f"Health check failed: {e}")  # 注释：记录调试日志

            time.sleep(2)  # 注释：间隔等待
        else:  # 注释：超时未成功
            router_process.terminate()  # 注释：终止进程
            raise RuntimeError(f"Router health check failed after {max_wait_time} seconds.")  # 注释：抛异常

    logger.info(f"Router is running on {router_address}")  # 注释：记录 router 地址
    return router_address, router_process  # 注释：返回地址与进程
