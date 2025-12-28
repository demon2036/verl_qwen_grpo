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
模块用途：RewardModelManager 负责启动 RM 推理服务器与路由。（注释：模块功能概述）
输入：RewardModelConfig、RayResourcePool（可选）。（注释：输入形态说明）
输出：提供 router 地址与资源管理能力。（注释：输出形态说明）
关键依赖：get_rollout_replica_class、split_resource_pool。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - rm_manager = RewardModelManager(config.reward_model, pool)
  - addr = rm_manager.get_router_address()
调用路径概览：（注释：全局调用关系）
  - RewardLoopManager.__init__ -> RewardModelManager
"""

import asyncio  # 注释：异步调度
import logging  # 注释：日志记录
import os  # 注释：环境变量

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool  # 注释：Ray 资源池工具
from verl.workers.config import HFModelConfig, RewardModelConfig  # 注释：模型与配置类型
from verl.workers.rollout.replica import get_rollout_replica_class  # 注释：获取 rollout replica 类

logger = logging.getLogger(__file__)  # 注释：模块级日志器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：日志级别


class RewardModelManager:  # 注释：RewardModel 管理器
    """
    功能：启动 RM 推理服务并提供 router 地址。（注释：类职责）
    输入：RewardModelConfig、RayResourcePool（可选）。（注释：输入形态说明）
    输出：router 地址与 server handles。（注释：输出形态说明）
    关键依赖：rollout replica、router 进程。（注释：关键依赖）
    典型用法：（注释：最小使用示例）
      - rm = RewardModelManager(cfg, pool)
      - addr = rm.get_router_address()
    调用路径概览：（注释：全局调用关系）
      - RewardLoopManager.__init__ -> RewardModelManager
    """

    def __init__(
        self,
        config: RewardModelConfig,
        resource_pool: RayResourcePool = None,
    ):
        """
        功能：保存配置并初始化 RM 推理服务与 router。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config (RewardModelConfig): RM 配置。（注释：配置对象）
          - resource_pool (RayResourcePool|None): 资源池（可选）。（注释：资源池）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 启动 rollout replica 与 router 进程。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - skip_tokenizer_init 为 True 时触发断言。（注释：配置约束）
        最小示例：（注释：最小可理解示例）
          - 输入：RewardModelManager(config, pool)
          - 输出：RM 服务与 router 已启动
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager.__init__`
          - 典型调用路径：`RewardLoopManager.__init__` -> `RewardModelManager`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：`_initialize_llm_servers`、`_initialize_router`
          - 调用了谁（外部依赖）：无
        """
        self.config = config  # 注释：保存配置
        self.resource_pool = resource_pool  # 注释：保存资源池
        self._initialize_llm_servers()  # 注释：启动推理服务器
        self._initialize_router()  # 注释：启动路由器
        assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."  # 注释：校验配置
        if self.config.rollout.free_cache_engine:  # 注释：若配置释放缓存则休眠
            self.sleep()  # 注释：释放资源

    def _initialize_llm_servers(self):  # 注释：初始化 RM 推理服务
        """
        功能：创建 rollout replica 并启动推理服务。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无（使用 self.config）。（注释：配置来源）
        返回：（注释：返回值说明）
          - None。（注释：无显式返回）
        副作用：（注释：副作用说明）
          - 启动推理服务进程/actor。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - 资源池划分数量不匹配会触发断言。（注释：资源校验）
        最小示例：（注释：最小可理解示例）
          - 输入：rollout_world_size=2, world_size=4
          - 输出：启动 2 个 replica
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager._initialize_llm_servers`
          - 典型调用路径：`RewardModelManager.__init__` -> `_initialize_llm_servers`
          - 被谁调用：`RewardModelManager`
          - 调用了谁（项目内）：`get_rollout_replica_class`、`split_resource_pool`
          - 调用了谁（外部依赖）：无
        """
        rollout_world_size = self.config.rollout.tensor_model_parallel_size  # 注释：rollout 并行度
        world_size = (  # 注释：计算总 world size
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode  # 注释：资源池模式
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode  # 注释：独立模式
        )
        num_replicas = world_size // rollout_world_size  # 注释：replica 数量

        rollout_replica_class = get_rollout_replica_class(self.config.rollout.name)  # 注释：获取 replica 类
        rollout_config = self.config.rollout  # 注释：rollout 配置
        model_config = HFModelConfig(  # 注释：构造模型配置
            path=self.config.model.path,
            external_lib=self.config.model.external_lib,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.tokenizer = model_config.get_processor()  # 注释：初始化 tokenizer/processor
        self.rollout_replicas = [  # 注释：创建 replica 列表
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:  # 注释：资源池模式
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)  # 注释：划分资源池
            assert len(split_resource_pools) == len(self.rollout_replicas)  # 注释：数量匹配检查
            self._run_all(  # 注释：初始化 colocated 服务
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:  # 注释：独立模式
            self._run_all([server.init_standalone() for server in self.rollout_replicas])  # 注释：初始化 standalone 服务
        self.server_handles = [server._server_handle for server in self.rollout_replicas]  # 注释：保存 server handle
        self.server_addresses = [server._server_address for server in self.rollout_replicas]  # 注释：保存 server 地址

    def _initialize_router(self):  # 注释：启动 router 进程
        """
        功能：启动 router 进程并保存地址。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无（使用 server_addresses）。（注释：内部使用）
        返回：（注释：返回值说明）
          - None。（注释：无显式返回）
        副作用：（注释：副作用说明）
          - 启动路由进程。（注释：进程副作用）
        异常/边界条件：（注释：异常与边界）
          - router 启动失败会抛异常。（注释：启动失败）
        最小示例：（注释：最小可理解示例）
          - 输入：server_addresses=["ip:port", ...]
          - 输出：router_address="ip:port"
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager._initialize_router`
          - 典型调用路径：`RewardModelManager.__init__` -> `_initialize_router`
          - 被谁调用：`RewardModelManager`
          - 调用了谁（项目内）：`launch_router_process`
          - 调用了谁（外部依赖）：无
        """
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]  # 注释：构造 worker URL 列表

        # TODO (dyy): sglang router is not ready yet.  # 注释：SGLang router 尚未准备
        # if self.config.rollout.name == "sglang":  # 注释：SGLang 分支（预留）
        #     from .router.inner_sglang_router import launch_router_process  # 注释：SGLang router
        # else:  # 注释：默认分支
        #     from .router.naive_router import launch_router_process  # 注释：Naive router

        from .router.naive_router import launch_router_process  # 注释：使用 naive router

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)  # 注释：启动 router

    def get_router_address(self):  # 注释：获取 router 地址
        """
        功能：返回 router 地址字符串。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - str：router 地址。（注释：返回值）
        副作用：（注释：副作用说明）
          - 无。（注释：无副作用）
        异常/边界条件：（注释：异常与边界）
          - router 未初始化时可能为 None。（注释：未初始化）
        最小示例：（注释：最小可理解示例）
          - 输入：get_router_address()
          - 输出："ip:port"
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager.get_router_address`
          - 典型调用路径：`RewardLoopManager.__init__` -> `get_router_address`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：无
        """
        return self.router_address  # 注释：返回 router 地址

    def wake_up(self):  # 注释：唤醒所有 rollout replicas
        """
        功能：唤醒 rollout replicas（如解除休眠）。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 触发 replica 的 wake_up。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - replica.wake_up 异常会向上抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：wake_up()
          - 输出：replicas 全部唤醒
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager.wake_up`
          - 典型调用路径：`RewardLoopManager.compute_rm_score` -> `wake_up`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：`_run_all`
          - 调用了谁（外部依赖）：无
        """
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])  # 注释：并发唤醒

    def sleep(self):  # 注释：休眠所有 rollout replicas
        """
        功能：让 rollout replicas 进入休眠/释放缓存。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 触发 replica.sleep。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - replica.sleep 异常会向上抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：sleep()
          - 输出：replicas 休眠
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager.sleep`
          - 典型调用路径：`RewardLoopManager.compute_rm_score` -> `sleep`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：`_run_all`
          - 调用了谁（外部依赖）：无
        """
        self._run_all([replica.sleep() for replica in self.rollout_replicas])  # 注释：并发休眠

    def _run_all(self, tasks: list[asyncio.Task]):  # 注释：同步运行异步任务
        """
        功能：在新事件循环中并发运行任务列表。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - tasks (list[asyncio.Task]): 任务列表。（注释：任务输入）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 创建事件循环并运行。（注释：事件循环副作用）
        异常/边界条件：（注释：异常与边界）
          - 任务异常会向上抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：_run_all([task1, task2])
          - 输出：任务执行完成
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_model.py::RewardModelManager._run_all`
          - 典型调用路径：`wake_up`/`sleep` -> `_run_all`
          - 被谁调用：`RewardModelManager`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`asyncio.gather`、`asyncio.run`
        """
        async def run_all():  # 注释：内部协程，汇总执行
            """
            功能：并发执行任务列表。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - 无（使用外层 tasks）。（注释：闭包变量）
            返回：（注释：返回值说明）
              - None。（注释：无显式返回）
            副作用：（注释：副作用说明）
              - 调度协程任务。（注释：异步调度）
            异常/边界条件：（注释：异常与边界）
              - 任务异常会向上抛出。（注释：异常传播）
            最小示例：（注释：最小可理解示例）
              - 输入：tasks=[task1, task2]
              - 输出：任务执行完成
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`RewardModelManager._run_all.run_all`
              - 典型调用路径：`_run_all` -> `run_all`
              - 被谁调用：`RewardModelManager._run_all`
              - 调用了谁（项目内）：无
              - 调用了谁（外部依赖）：`asyncio.gather`
            """
            await asyncio.gather(*tasks)  # 注释：并发执行

        asyncio.run(run_all())  # 注释：运行事件循环
