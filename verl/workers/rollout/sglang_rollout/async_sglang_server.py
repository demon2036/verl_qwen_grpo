# Copyright 2023-2024 SGLang Team  # 注释：版权声明（SGLang）
# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明（Bytedance）
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
模块用途：在 Ray 集群中启动/管理 SGLang HTTP 服务器，并提供生成/缓存控制接口。  # 注释：模块用途
输入：RolloutConfig、HFModelConfig、Ray workers、节点/设备信息。  # 注释：输入说明
输出：TokenOutput、服务地址信息，以及服务端状态控制。  # 注释：输出说明
关键依赖：ray、sglang、torch、verl.workers.rollout.*。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- SGLangReplica.launch_servers -> SGLangHttpServer.launch_server -> generate/wake_up/sleep。  # 注释：示例
调用路径概览：  # 注释：调用路径标题
- RolloutReplica -> SGLangReplica -> SGLangHttpServer (Ray actor)。  # 注释：调用链
"""  # 注释：模块 docstring 结束

import asyncio  # 注释：标准库，异步调度
import dataclasses  # 注释：标准库，dataclass 反射
import json  # 注释：标准库，JSON 序列化
import logging  # 注释：标准库，日志
import os  # 注释：标准库，环境变量
from typing import Any, Optional  # 注释：类型注解

import ray  # 注释：第三方库，Ray
import sglang  # 注释：第三方库，SGLang
import sglang.srt.entrypoints.engine  # 注释：SGLang 引擎入口
import torch  # 注释：第三方库，Torch
from ray.actor import ActorHandle  # 注释：Ray actor 句柄类型
from sglang.srt.entrypoints.http_server import (  # 注释：SGLang HTTP server 入口
    ServerArgs,  # 注释：SGLang Server 参数
    _GlobalState,  # 注释：全局状态结构
    _launch_subprocesses,  # 注释：启动子进程
    app,  # 注释：FastAPI/ASGI 应用
    set_global_state,  # 注释：设置全局状态
)
from sglang.srt.managers.io_struct import (  # 注释：请求/响应结构
    GenerateReqInput,  # 注释：生成请求
    ReleaseMemoryOccupationReqInput,  # 注释：释放内存请求
    ResumeMemoryOccupationReqInput,  # 注释：恢复内存请求
)
from sglang.srt.managers.tokenizer_manager import ServerStatus  # 注释：服务状态枚举

from verl.single_controller.ray import RayClassWithInitArgs  # 注释：项目内 Ray 类包装
from verl.utils.config import omega_conf_to_dataclass  # 注释：OmegaConf 转 dataclass
from verl.workers.config import HFModelConfig, RolloutConfig  # 注释：配置类型
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput  # 注释：rollout 基类/结构
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config  # 注释：ServerAdapter 与环境补丁
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn  # 注释：工具函数

logger = logging.getLogger(__file__)  # 注释：获取模块 logger
logger.setLevel(logging.INFO)  # 注释：设置日志级别


@ray.remote(num_cpus=1)  # 注释：Ray actor 装饰器
class SGLangHttpServer:  # 注释：SGLang HTTP 服务器 actor
    """
    SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.

    功能：在单节点启动 SGLang HTTP Server，并提供生成/缓存/权重管理接口。  # 注释：类用途
    参数：config/model_config/rollout_mode/节点信息等。  # 注释：参数说明
    返回：Ray actor 实例。  # 注释：返回值说明
    副作用：设置环境变量、启动子进程与 HTTP 服务。  # 注释：副作用说明
    异常/边界条件：CUDA 不可用或参数非法会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - server = SGLangHttpServer.remote(...); await server.launch_server.remote()。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/async_sglang_server.py::SGLangHttpServer。  # 注释：类位置
    - 典型调用路径：SGLangReplica.launch_servers -> SGLangHttpServer.launch_server。  # 注释：调用链
    - 被谁调用：SGLangReplica。  # 注释：调用方说明
    - 调用了谁（项目内）：get_free_port、run_unvicorn。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：sglang、ray、torch。  # 注释：外部依赖
    """  # 注释：类 docstring 结束

    def __init__(  # 注释：初始化 HTTP server actor
        self,  # 注释：self
        config: RolloutConfig,  # 注释：rollout 配置
        model_config: HFModelConfig,  # 注释：模型配置
        rollout_mode: RolloutMode,  # 注释：rollout 模式
        workers: list[ActorHandle],  # 注释：关联的 worker 列表
        replica_rank: int,  # 注释：replica rank
        node_rank: int,  # 注释：node rank
        nnodes: int,  # 注释：节点数
        cuda_visible_devices: str,  # 注释：CUDA 可见设备
    ):  # 注释：参数结束
        """
        功能：初始化服务端配置、计算节点信息并准备 NCCL master 地址。  # 注释：函数用途
        参数：见 __init__ 签名。  # 注释：参数含义
        返回：None。  # 注释：返回值语义
        副作用：设置 CUDA_VISIBLE_DEVICES，创建 master socket。  # 注释：副作用说明
        异常/边界条件：CUDA 不可用则 assert 失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - SGLangHttpServer.remote(config, model_config, mode, workers, 0, 0, 1, "0")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.__init__。  # 注释：函数位置
        - 典型调用路径：SGLangReplica.launch_servers -> SGLangHttpServer(...)。  # 注释：调用链
        - 被谁调用：SGLangReplica。  # 注释：调用方说明
        - 调用了谁（项目内）：omega_conf_to_dataclass、get_free_port。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：torch.cuda.is_available。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")  # 注释：打印启动信息
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices  # 注释：设置可见 GPU
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"  # 注释：确保 GPU 可用

        self.config: RolloutConfig = omega_conf_to_dataclass(config)  # 注释：配置转 dataclass
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)  # 注释：模型配置转 dataclass
        self.config.max_model_len = self.config.prompt_length + self.config.response_length  # 注释：更新最大长度
        self.rollout_mode = rollout_mode  # 注释：保存 rollout 模式
        self.workers = workers  # 注释：保存 worker 列表

        self.replica_rank = replica_rank  # 注释：保存 replica_rank
        self.node_rank = node_rank  # 注释：保存 node_rank
        self.nnodes = nnodes  # 注释：保存节点数

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":  # 注释：非混合且 dummy 时
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")  # 注释：警告日志
            self.config.load_format = "auto"  # 注释：切换为 auto

        # used for http server  # 注释：原注释保留
        self._server_address = ray.util.get_node_ip_address().strip("[]")  # 注释：获取节点 IP
        self._server_port = None  # 注释：端口占位

        # used for NCCL process group  # 注释：原注释保留
        if self.node_rank == 0:  # 注释：master 节点
            self._master_address = self._server_address  # 注释：master 地址
            self._master_port, self._master_sock = get_free_port(self._server_address)  # 注释：获取可用端口
            logger.info(  # 注释：记录 master 信息
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "  # 注释：日志片段
                f"master address: {self._master_address}, port: {self._master_port}"  # 注释：日志片段
            )  # 注释：logger.info 结束
        else:  # 注释：非 master 节点
            self._master_address = None  # 注释：占位
            self._master_port = None  # 注释：占位

    def get_master_address(self):  # 注释：获取 master 地址与端口
        """
        功能：返回 NCCL 进程组初始化所需的 master 地址与端口。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：tuple(address, port)。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - addr, port = server.get_master_address()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.get_master_address。  # 注释：函数位置
        - 典型调用路径：SGLangReplica.launch_servers -> get_master_address。  # 注释：调用链
        - 被谁调用：SGLangReplica.launch_servers。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        return self._master_address, self._master_port  # 注释：返回 master 地址与端口

    def get_server_address(self):  # 注释：获取 HTTP server 地址
        """
        功能：返回 HTTP server 的地址与端口。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：tuple(address, port)。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：服务未启动则断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - addr, port = server.get_server_address()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.get_server_address。  # 注释：函数位置
        - 典型调用路径：ServerAdapter._init_server_adapter -> get_server_address。  # 注释：调用链
        - 被谁调用：ServerAdapter。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        assert self._server_port is not None, "http server is not launched, port is None"  # 注释：确保端口已设置
        return self._server_address, self._server_port  # 注释：返回地址与端口

    async def launch_server(self, master_address: str = None, master_port: int = None):  # 注释：启动 HTTP server
        """
        功能：在指定节点启动 SGLang HTTP server，并在 master 节点启动 API 服务。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - master_address/master_port：非 master 节点用于初始化 NCCL 的地址与端口。  # 注释：参数含义
        返回：None。  # 注释：返回值语义
        副作用：启动子进程、设置全局状态、启动 HTTP 服务。  # 注释：副作用说明
        异常/边界条件：quantization 非法或端口不可用会报错。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await server.launch_server(master_address, master_port)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.launch_server。  # 注释：函数位置
        - 典型调用路径：SGLangReplica.launch_servers -> SGLangHttpServer.launch_server。  # 注释：调用链
        - 被谁调用：SGLangReplica。  # 注释：调用方说明
        - 调用了谁（项目内）：_set_envs_and_config、run_unvicorn。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：sglang._launch_subprocesses、ServerArgs。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        if self.node_rank != 0:  # 注释：非 master 节点
            assert master_address and master_port, "non-master node should provide master address and port"  # 注释：校验地址
            self._master_address = master_address  # 注释：设置 master 地址
            self._master_port = master_port  # 注释：设置 master 端口

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}  # 注释：读取引擎额外参数
        attention_backend = engine_kwargs.pop("attention_backend", None)  # 注释：弹出 attention_backend
        quantization = self.config.get("quantization", None)  # 注释：读取量化配置
        if quantization is not None:  # 注释：量化配置存在
            if quantization == "fp8":  # 注释：FP8 量化
                assert sglang.__version__ >= "0.5.5", "sglang>=0.5.5 is required for FP8 quantization"  # 注释：版本检查
                FP8_BLOCK_QUANT_KWARGS = {  # 注释：FP8 量化配置
                    "activation_scheme": "dynamic",  # 注释：动态激活
                    "fmt": "e4m3",  # 注释：格式
                    "quant_method": "fp8",  # 注释：量化方法
                    "weight_block_size": [128, 128],  # 注释：块大小
                }  # 注释：配置结束
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)  # 注释：复制配置
            else:  # 注释：不支持的量化类型
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")  # 注释：抛出异常
        dist_init_addr = (  # 注释：构造 NCCL init 地址
            f"[{self._master_address}]:{self._master_port}"  # 注释：IPv6 地址格式
            if is_valid_ipv6_address(self._master_address)  # 注释：检查 IPv6
            else f"{self._master_address}:{self._master_port}"  # 注释：IPv4 地址格式
        )  # 注释：dist_init_addr 结束

        args = {  # 注释：构造 ServerArgs 参数字典
            "model_path": self.model_config.local_path,  # 注释：模型路径
            "dtype": self.config.dtype,  # 注释：模型 dtype
            "mem_fraction_static": self.config.gpu_memory_utilization,  # 注释：显存占用比例
            "disable_cuda_graph": self.config.enforce_eager,  # 注释：是否禁用 cuda graph
            "enable_memory_saver": True,  # 注释：启用内存节省
            "base_gpu_id": 0,  # 注释：基础 GPU id
            "gpu_id_step": 1,  # 注释：GPU id 步长
            "tp_size": self.config.tensor_model_parallel_size,  # 注释：TP 大小
            "dp_size": self.config.data_parallel_size,  # 注释：DP 大小
            "ep_size": self.config.expert_parallel_size,  # 注释：EP 大小
            "node_rank": self.node_rank,  # 注释：节点 rank
            "load_format": self.config.load_format,  # 注释：权重加载格式
            "dist_init_addr": dist_init_addr,  # 注释：NCCL 初始化地址
            "nnodes": self.nnodes,  # 注释：节点数
            "trust_remote_code": self.model_config.trust_remote_code,  # 注释：信任远程代码
            "max_running_requests": self.config.get("max_num_seqs", None),  # 注释：最大并发请求
            "log_level": "error",  # 注释：日志级别
            "mm_attention_backend": "fa3",  # 注释：多模态 attention 后端
            "attention_backend": attention_backend if attention_backend is not None else "fa3",  # 注释：attention 后端
            "skip_tokenizer_init": self.config.skip_tokenizer_init,  # 注释：是否跳过 tokenizer 初始化
            "skip_server_warmup": True,  # 注释：跳过预热
            "quantization": quantization,  # 注释：量化类型
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})  # 注释：量化配置 JSON
            if quantization == "fp8"  # 注释：条件
            else json.dumps({}),  # 注释：默认空配置
            **engine_kwargs,  # 注释：附加参数
        }  # 注释：args 结束

        if self.config.prometheus.enable:  # 注释：启用 Prometheus
            if self.config.prometheus.served_model_name:  # 注释：配置了模型名
                # Extract model name from path if it's a full path  # 注释：原注释保留
                served_model_name = self.config.prometheus.served_model_name  # 注释：读取模型名
                if "/" in served_model_name:  # 注释：若为路径
                    # If it's a full path, extract the last part as model name  # 注释：原注释保留
                    served_model_name = served_model_name.split("/")[-1]  # 注释：取最后一段
                args["served_model_name"] = served_model_name  # 注释：设置 served_model_name

            # start sglang metrics  # 注释：原注释保留
            args["enable_metrics"] = True  # 注释：启用 metrics

        # enable_weights_cpu_backup is supported in sglang>=0.5.3  # 注释：原注释保留
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:  # 注释：检查字段支持
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False  # 注释：仅 colocated 启用
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup  # 注释：设置参数

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.  # 注释：原注释保留
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py  # 注释：原注释保留
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config  # 注释：打补丁环境配置
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"  # 注释：允许子进程
        server_args = ServerArgs(**args)  # 注释：构造 ServerArgs
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(  # 注释：启动子进程
            server_args=server_args  # 注释：传入参数
        )  # 注释：子进程启动结束

        # In multi-node cases, non-zero rank nodes should not launch http server.  # 注释：原注释保留
        if self.node_rank > 0:  # 注释：非 master 节点
            return  # 注释：直接返回

        set_global_state(  # 注释：设置全局状态
            _GlobalState(  # 注释：构造全局状态
                tokenizer_manager=self.tokenizer_manager,  # 注释：tokenizer 管理器
                template_manager=self.template_manager,  # 注释：模板管理器
                scheduler_info=self.scheduler_info,  # 注释：调度信息
            )  # 注释：全局状态结束
        )  # 注释：set_global_state 结束
        app.is_single_tokenizer_mode = True  # 注释：标记单 tokenizer 模式

        # Set warmup_thread_{kw}args to avoid AttributeError in lifespan function  # 注释：原注释保留
        app.server_args = server_args  # 注释：保存 server_args
        app.warmup_thread_kwargs = {"server_args": server_args}  # 注释：保存 warmup kwargs
        app.warmup_thread_args = (server_args, None, None)  # 注释：保存 warmup args

        # Manually add Prometheus middleware before starting server  # 注释：原注释保留
        # This ensures /metrics endpoint is available immediately  # 注释：原注释保留
        if server_args.enable_metrics:  # 注释：若启用 metrics
            from sglang.srt.utils.common import add_prometheus_middleware  # 注释：导入 Prometheus 中间件

            add_prometheus_middleware(app)  # 注释：添加中间件

        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)  # 注释：启动 HTTP 服务
        self.tokenizer_manager.server_status = ServerStatus.Up  # 注释：更新服务状态

    async def wake_up(self):  # 注释：唤醒服务（加载权重/缓存）
        """
        功能：根据 rollout_mode 唤醒服务端权重与 KV cache。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：None。  # 注释：返回值语义
        副作用：可能向 workers/engine 发起远程调用。  # 注释：副作用说明
        异常/边界条件：远程调用失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await server.wake_up()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.wake_up。  # 注释：函数位置
        - 典型调用路径：SGLangReplica.wake_up -> SGLangHttpServer.wake_up。  # 注释：调用链
        - 被谁调用：rollout 管理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：workers.wake_up / tokenizer_manager。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：asyncio.gather。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        if self.rollout_mode == RolloutMode.HYBRID:  # 注释：混合模式
            # Call all workers to switch between trainer mode and rollout mode.  # 注释：原注释保留
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])  # 注释：唤醒所有 worker
        elif self.rollout_mode == RolloutMode.COLOCATED:  # 注释：同机模式
            # Directly call engine to wake up without sync weights.  # 注释：原注释保留
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])  # 注释：构造恢复请求
            await self.tokenizer_manager.resume_memory_occupation(obj, None)  # 注释：恢复内存占用
            await self.tokenizer_manager.flush_cache()  # 注释：刷新缓存
        elif self.rollout_mode == RolloutMode.STANDALONE:  # 注释：独立模式
            logger.info("skip wake_up in standalone mode")  # 注释：记录跳过日志

    async def sleep(self):  # 注释：休眠（释放权重/缓存）
        """
        功能：根据 rollout_mode 释放权重与 KV cache。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：None。  # 注释：返回值语义
        副作用：释放内存或通知 worker。  # 注释：副作用说明
        异常/边界条件：远程调用失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await server.sleep()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.sleep。  # 注释：函数位置
        - 典型调用路径：SGLangReplica.sleep -> SGLangHttpServer.sleep。  # 注释：调用链
        - 被谁调用：rollout 管理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：workers.sleep / tokenizer_manager。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：asyncio.gather。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        if self.rollout_mode == RolloutMode.HYBRID:  # 注释：混合模式
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])  # 注释：通知 worker 休眠
        elif self.rollout_mode == RolloutMode.COLOCATED:  # 注释：同机模式
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])  # 注释：构造释放请求
            await self.tokenizer_manager.release_memory_occupation(obj, None)  # 注释：释放内存占用
        elif self.rollout_mode == RolloutMode.STANDALONE:  # 注释：独立模式
            logger.info("skip sleep in standalone mode")  # 注释：记录跳过日志

    async def clear_kv_cache(self):  # 注释：清理 KV cache
        """
        功能：仅释放 KV cache（不释放权重）。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：None。  # 注释：返回值语义
        副作用：释放缓存占用。  # 注释：副作用说明
        异常/边界条件：远程调用失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await server.clear_kv_cache()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.clear_kv_cache。  # 注释：函数位置
        - 典型调用路径：rollout 管理逻辑 -> clear_kv_cache。  # 注释：调用链
        - 被谁调用：SGLangReplica/rollout 管理。  # 注释：调用方说明
        - 调用了谁（项目内）：tokenizer_manager.release_memory_occupation。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])  # 注释：构造释放请求
        await self.tokenizer_manager.release_memory_occupation(obj, None)  # 注释：释放缓存

    async def generate(  # 注释：生成接口
        self,  # 注释：self
        prompt_ids: torch.Tensor,  # 注释：输入 token ids
        sampling_params: dict[str, Any],  # 注释：采样参数
        request_id: str,  # 注释：请求 ID
        image_data: Optional[list[Any]] = None,  # 注释：可选多模态图像
    ) -> TokenOutput:  # 注释：返回 TokenOutput
        """
        Generate sequence with token-in-token-out.

        功能：调用 tokenizer_manager 进行生成，并返回 token ids/log_probs。  # 注释：函数用途
        参数：prompt_ids/sampling_params/request_id/image_data。  # 注释：参数含义
        返回：TokenOutput。  # 注释：返回值语义
        副作用：可能消耗服务端计算资源。  # 注释：副作用说明
        异常/边界条件：max_new_tokens 超限会断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await server.generate(prompt_ids, {"temperature":1.0}, request_id="x")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangHttpServer.generate。  # 注释：函数位置
        - 典型调用路径：ServerAdapter.generate -> SGLangHttpServer.generate。  # 注释：调用链
        - 被谁调用：ServerAdapter / rollout 逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：tokenizer_manager.generate_request。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：GenerateReqInput。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.  # 注释：原注释保留
        response_length = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)  # 注释：计算可用响应长度
        if "max_new_tokens" in sampling_params:  # 注释：显式设置 max_new_tokens
            max_new_tokens = sampling_params.pop("max_new_tokens")  # 注释：取出并移除
        elif "max_tokens" in sampling_params:  # 注释：兼容 vllm 风格参数
            # support vllm-style 'max_tokens' param  # 注释：原注释保留
            max_new_tokens = sampling_params.pop("max_tokens")  # 注释：取出并移除
        else:  # 注释：未设置最大 token
            max_new_tokens = response_length  # 注释：使用可用长度
        assert max_new_tokens <= response_length, (  # 注释：校验长度
            f"max_new_tokens {max_new_tokens} exceeds available response_length {response_length}"  # 注释：错误信息
        )  # 注释：assert 结束
        sampling_params["max_new_tokens"] = max_new_tokens  # 注释：写回参数
        return_logprob = sampling_params.pop("logprobs", False)  # 注释：是否返回 logprob

        request = GenerateReqInput(  # 注释：构造生成请求
            rid=request_id,  # 注释：请求 ID
            input_ids=prompt_ids,  # 注释：输入 token
            sampling_params=sampling_params,  # 注释：采样参数
            return_logprob=return_logprob,  # 注释：是否返回 logprob
            image_data=image_data,  # 注释：多模态图像
        )  # 注释：请求结束
        output = await self.tokenizer_manager.generate_request(request, None).__anext__()  # 注释：获取生成结果
        if return_logprob:  # 注释：需要 logprob
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]  # 注释：取出 logprobs
            log_probs, token_ids = zip(  # 注释：解包 logprob 与 token
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True  # 注释：列表推导
            )  # 注释：zip 结束
        else:  # 注释：不需要 logprob
            token_ids = output["output_ids"]  # 注释：取出 token ids
            log_probs = None  # 注释：logprob 置空
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)  # 注释：返回 TokenOutput


_rollout_worker_actor_cls = ray.remote(ServerAdapter)  # 注释：Ray 远程包装 ServerAdapter


class SGLangReplica(RolloutReplica):  # 注释：SGLang Replica
    """
    功能：管理 SGLang server actor 与 rollout worker 的生命周期。  # 注释：类用途
    参数：继承 RolloutReplica。  # 注释：参数说明
    返回：SGLangReplica 实例。  # 注释：返回值说明
    副作用：创建并持有 SGLangHttpServer actor 列表。  # 注释：副作用说明
    异常/边界条件：节点/worker 数不匹配将断言失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - replica = SGLangReplica(...); await replica.launch_servers()。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：...::SGLangReplica。  # 注释：类位置
    - 典型调用路径：rollout controller -> SGLangReplica.launch_servers。  # 注释：调用链
    - 被谁调用：rollout 管理组件。  # 注释：调用方说明
    - 调用了谁（项目内）：SGLangHttpServer、ServerAdapter。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：ray。  # 注释：外部依赖
    """  # 注释：类 docstring 结束
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:  # 注释：返回 worker actor 类
        """
        Get rollout worker actor class for colocated and standalone mode.

        功能：构造 RayClassWithInitArgs 用于创建 rollout worker actor。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：RayClassWithInitArgs。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - cls_args = replica.get_ray_class_with_init_args()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangReplica.get_ray_class_with_init_args。  # 注释：函数位置
        - 典型调用路径：rollout 初始化 -> get_ray_class_with_init_args。  # 注释：调用链
        - 被谁调用：RolloutReplica 初始化流程。  # 注释：调用方说明
        - 调用了谁（项目内）：RayClassWithInitArgs。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：ray.remote。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        worker_dict_cls = RayClassWithInitArgs(  # 注释：构造 RayClassWithInitArgs
            cls=_rollout_worker_actor_cls,  # 注释：远程类
            config=self.config,  # 注释：配置
            model_config=self.model_config,  # 注释：模型配置
            device_mesh=None,  # 注释：device_mesh 占位
        )  # 注释：构造结束
        return worker_dict_cls  # 注释：返回

    async def launch_servers(self):  # 注释：启动各节点 HTTP server
        """
        Launch http server in each node.

        功能：为每个节点创建 SGLangHttpServer actor 并启动服务。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：None。  # 注释：返回值语义
        副作用：创建 Ray actors、启动子进程与 HTTP 服务。  # 注释：副作用说明
        异常/边界条件：worker 数与 world_size 不一致会断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await replica.launch_servers()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::SGLangReplica.launch_servers。  # 注释：函数位置
        - 典型调用路径：rollout 初始化 -> launch_servers。  # 注释：调用链
        - 被谁调用：rollout controller。  # 注释：调用方说明
        - 调用了谁（项目内）：SGLangHttpServer、is_valid_ipv6_address。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：ray、asyncio。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        assert len(self.workers) == self.world_size, (  # 注释：确保 worker 数正确
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"  # 注释：错误信息
        )  # 注释：assert 结束

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers  # 注释：原注释保留
        worker_infos = await asyncio.gather(  # 注释：并发获取 worker 信息
            *[  # 注释：列表推导
                worker.__ray_call__.remote(  # 注释：调用 worker 远程方法
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])  # 注释：返回 node_id 与 GPU 列表
                )  # 注释：__ray_call__ 结束
                for worker in self.workers  # 注释：遍历 worker
            ]  # 注释：列表结束
        )  # 注释：gather 结束
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]  # 注释：提取 CUDA_VISIBLE_DEVICES
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]  # 注释：提取 node_id

        # create server actor in each node with node affinity and cuda visible devices  # 注释：原注释保留
        for node_rank in range(self.nnodes):  # 注释：遍历节点
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]  # 注释：该节点 worker 切片
            node_cuda_visible_devices = ",".join(  # 注释：拼接节点可见 GPU
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]  # 注释：切片
            )  # 注释：join 结束
            node_id = worker_node_ids[node_rank * self.gpus_per_node]  # 注释：节点 id
            name = (  # 注释：actor 名称
                f"sglang_server_{self.replica_rank}_{node_rank}"  # 注释：普通 actor 名
                if not self.is_reward_model  # 注释：非 reward 模型
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"  # 注释：reward actor 名
            )  # 注释：名称结束
            server = SGLangHttpServer.options(  # 注释：创建 actor
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(  # 注释：节点亲和调度
                    node_id=node_id,  # 注释：目标节点
                    soft=False,  # 注释：硬亲和
                ),  # 注释：scheduling_strategy 结束
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},  # 注释：运行时环境
                name=name,  # 注释：actor 名称
            ).remote(  # 注释：启动远程 actor
                config=self.config,  # 注释：配置
                model_config=self.model_config,  # 注释：模型配置
                rollout_mode=self.rollout_mode,  # 注释：rollout 模式
                workers=workers,  # 注释：节点 worker 列表
                replica_rank=self.replica_rank,  # 注释：replica rank
                node_rank=node_rank,  # 注释：node rank
                nnodes=self.nnodes,  # 注释：节点数
                cuda_visible_devices=node_cuda_visible_devices,  # 注释：可见 GPU
            )  # 注释：remote 结束
            self.servers.append(server)  # 注释：记录 server actor

        # launch http server in each node  # 注释：原注释保留
        master_address, master_port = await self.servers[0].get_master_address.remote()  # 注释：获取 master 地址
        await asyncio.gather(  # 注释：并发启动 server
            *[  # 注释：列表推导
                server.launch_server.remote(master_address=master_address, master_port=master_port)  # 注释：调用启动
                for server in self.servers  # 注释：遍历 server
            ]  # 注释：列表结束
        )  # 注释：gather 结束

        # get http server address from first server  # 注释：原注释保留
        server_address, server_port = await self.servers[0].get_server_address.remote()  # 注释：获取服务地址
        self._server_handle = self.servers[0]  # 注释：保存 server handle
        self._server_address = (  # 注释：格式化服务地址
            f"[{server_address}]:{server_port}"  # 注释：IPv6 格式
            if is_valid_ipv6_address(server_address)  # 注释：检查 IPv6
            else f"{server_address}:{server_port}"  # 注释：IPv4 格式
        )  # 注释：地址格式化结束
