# Copyright 2023-2024 SGLang Team  # 注释：版权声明（SGLang）
# Copyright 2025 ModelBest Inc. and/or its affiliates  # 注释：版权声明（ModelBest）
# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明（Bytedance）
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
模块用途：SGLang rollout 侧的 ServerAdapter，实现权重同步与缓存管理。  # 注释：模块用途
输入：RolloutConfig/HFModelConfig、DeviceMesh、权重生成器等。  # 注释：输入说明
输出：通过 AsyncHttpServerAdapter 调用 SGLang 服务（无显式返回）。  # 注释：输出说明
关键依赖：ray、sglang、torch、verl.workers.rollout.*。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- 由 ActorRolloutRefWorker 构建 ServerAdapter，并在 rollout 中调用 update_weights/resume/release。  # 注释：示例
调用路径概览：  # 注释：调用路径标题
- 训练入口 -> Ray worker -> ServerAdapter.update_weights/resume/release。  # 注释：调用链
"""  # 注释：模块 docstring 结束
from __future__ import annotations  # 注释：允许前向引用注解

import logging  # 注释：标准库，日志
import multiprocessing as mp  # 注释：标准库，多进程设置
import os  # 注释：标准库，环境变量
from typing import Generator  # 注释：类型注解

import ray  # 注释：第三方库，Ray actor/任务
import sglang.srt.entrypoints.engine  # 注释：第三方库，SGLang 引擎入口
import torch  # 注释：第三方库，张量与分布式
from sglang.srt.server_args import ServerArgs  # 注释：SGLang server 配置类型
from sglang.srt.utils import (  # 注释：SGLang 工具函数
    assert_pkg_version,  # 注释：版本检查
    is_cuda,  # 注释：CUDA 检查
    set_prometheus_multiproc_dir,  # 注释：Prometheus 多进程目录
    set_ulimit,  # 注释：设置 ulimit
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights  # 注释：SGLang 权重同步
from torch.distributed.device_mesh import DeviceMesh  # 注释：设备网格

from verl.workers.config import HFModelConfig, RolloutConfig  # 注释：项目内配置类型
from verl.workers.rollout.base import BaseRollout  # 注释：rollout 基类
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter  # 注释：HTTP 客户端适配器
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets  # 注释：分桶工具
from verl.workers.rollout.utils import is_valid_ipv6_address  # 注释：IPv6 地址检查

logger = logging.getLogger(__file__)  # 注释：获取模块 logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：设置日志级别


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723  # 注释：原注释保留（补丁说明）
def _set_envs_and_config(server_args: ServerArgs):  # 注释：设置环境变量并校验版本
    """
    功能：为 SGLang Server 设置环境变量、prometheus、ulimit，并进行版本检查。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - server_args (ServerArgs)：SGLang 服务器参数。  # 注释：参数含义
    返回：None。  # 注释：返回值语义
    副作用：修改 os.environ、设置 ulimit。  # 注释：副作用说明
    异常/边界条件：版本不满足时断言失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _set_envs_and_config(ServerArgs(...))  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/sglang_rollout.py::_set_envs_and_config。  # 注释：函数位置
    - 典型调用路径：SGLang 引擎启动 -> _set_envs_and_config。  # 注释：调用链
    - 被谁调用：sglang.srt.entrypoints.engine（被 monkey patch）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：sglang.utils、os.environ、mp.set_start_method。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    # Set global environments  # 注释：原注释保留（环境变量）
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 注释：降低 TF 日志
    os.environ["NCCL_CUMEM_ENABLE"] = "0"  # 注释：禁用 NCCL CUMEM
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))  # 注释：设置 NVLS 开关
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"  # 注释：避免记录 stream
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"  # 注释：限制连接数
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"  # 注释：CUDA 模块加载策略

    # Set prometheus env vars  # 注释：原注释保留
    if server_args.enable_metrics:  # 注释：启用 metrics 时
        set_prometheus_multiproc_dir()  # 注释：设置 Prometheus 多进程目录

    # Set ulimit  # 注释：原注释保留
    set_ulimit()  # 注释：设置 ulimit

    # Check flashinfer version  # 注释：原注释保留
    if server_args.attention_backend == "flashinfer":  # 注释：使用 flashinfer 后端
        assert_pkg_version(  # 注释：校验 flashinfer 版本
            "flashinfer_python",  # 注释：包名
            "0.2.5",  # 注释：最低版本
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",  # 注释：错误提示
        )  # 注释：assert_pkg_version 结束
    if is_cuda():  # 注释：CUDA 环境
        assert_pkg_version(  # 注释：校验 sgl-kernel 版本
            "sgl-kernel",  # 注释：包名
            "0.1.1",  # 注释：最低版本
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",  # 注释：错误提示
        )  # 注释：assert_pkg_version 结束

    # Set mp start method  # 注释：原注释保留
    mp.set_start_method("spawn", force=True)  # 注释：设置多进程启动方式


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config  # 注释：替换 SGLang 环境设置函数


# because chatCompletion is an async method, it makes the whole ray actor be an async actor  # 注释：原注释保留
# which can not call loop.run_until_complete. So we need to make the engine to be an async class  # 注释：原注释保留
class ServerAdapter(BaseRollout):  # 注释：SGLang ServerAdapter（HTTP 客户端）
    """
    SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.

    功能：在 rollout 侧与 SGLang HTTP server 交互，实现权重更新与缓存管理。  # 注释：类用途
    参数：config/model_config/device_mesh。  # 注释：参数说明
    返回：ServerAdapter 实例。  # 注释：返回值说明
    副作用：可能启动/连接 HTTP server actor。  # 注释：副作用说明
    异常/边界条件：环境变量缺失或 server 不可用会导致异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - adapter = ServerAdapter(config, model_config, device_mesh)  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/sglang_rollout.py::ServerAdapter。  # 注释：类位置
    - 典型调用路径：rollout worker -> ServerAdapter.update_weights/resume/release。  # 注释：调用链
    - 被谁调用：verl/workers/rollout/sglang_rollout/async_sglang_server.py。  # 注释：调用方说明
    - 调用了谁（项目内）：AsyncHttpServerAdapter、get_named_tensor_buckets。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：ray、sglang。  # 注释：外部依赖
    """  # 注释：类 docstring 结束

    def __init__(  # 注释：初始化 ServerAdapter
        self,  # 注释：self
        config: RolloutConfig,  # 注释：rollout 配置
        model_config: HFModelConfig,  # 注释：模型配置
        device_mesh: DeviceMesh,  # 注释：设备网格
    ):  # 注释：参数列表结束
        """
        功能：初始化 ServerAdapter 并计算各 rank/节点索引。  # 注释：函数用途
        参数：config/model_config/device_mesh。  # 注释：参数含义
        返回：None。  # 注释：返回值语义
        副作用：读取环境变量并设置成员变量。  # 注释：副作用说明
        异常/边界条件：若 FP8 版本不满足会断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - ServerAdapter(cfg, model_cfg, mesh)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::ServerAdapter.__init__。  # 注释：函数位置
        - 典型调用路径：worker 初始化 -> ServerAdapter(...)。  # 注释：调用链
        - 被谁调用：async_sglang_server.SGLangReplica。  # 注释：调用方说明
        - 调用了谁（项目内）：BaseRollout.__init__。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：os.environ。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        if config.get("quantization", None) == "fp8":  # 注释：若使用 FP8 量化
            import sglang  # 注释：导入 sglang 以检查版本

            assert sglang.__version__ >= "0.5.5", "sglang>=0.5.5 is required for FP8 quantization"  # 注释：版本检查
            FP8_BLOCK_QUANT_KWARGS = {  # 注释：FP8 量化配置
                "activation_scheme": "dynamic",  # 注释：激活方案
                "fmt": "e4m3",  # 注释：格式
                "quant_method": "fp8",  # 注释：量化方法
                "weight_block_size": [128, 128],  # 注释：块大小
            }  # 注释：FP8 配置结束
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)  # 注释：复制配置
            model_config.hf_config.quantization_config = fp8_block_quant_kwargs  # 注释：写入量化配置
        super().__init__(config, model_config, device_mesh)  # 注释：调用父类初始化
        self._engine: AsyncHttpServerAdapter = None  # 注释：HTTP 客户端占位

        rank = int(os.environ["RANK"])  # 注释：读取全局 rank
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])  # 注释：读取本地 world size
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size  # 注释：rollout 世界大小
        self.replica_rank = rank // rollout_world_size  # 注释：计算 replica_rank
        self.rollout_rank = rank % rollout_world_size  # 注释：计算 rollout_rank
        self.node_rank = self.rollout_rank // local_world_size  # 注释：计算 node_rank
        self.local_rank = self.rollout_rank % local_world_size  # 注释：计算 local_rank

    async def _init_server_adapter(self):  # 注释：懒加载 HTTP 适配器
        """
        功能：获取 Ray actor 并创建 AsyncHttpServerAdapter。  # 注释：函数用途
        参数：无。  # 注释：参数说明
        返回：None。  # 注释：返回值语义
        副作用：连接远端 server actor。  # 注释：副作用说明
        异常/边界条件：actor 不存在会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await self._init_server_adapter()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::ServerAdapter._init_server_adapter。  # 注释：函数位置
        - 典型调用路径：resume/release/update_weights -> _init_server_adapter。  # 注释：调用链
        - 被谁调用：本类内部。  # 注释：调用方说明
        - 调用了谁（项目内）：AsyncHttpServerAdapter、is_valid_ipv6_address。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：ray.get_actor。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        if self._engine is not None:  # 注释：已初始化则直接返回
            return  # 注释：提前退出

        # Lazy init http server adapter because http server is launched after hybrid engine.  # 注释：原注释保留
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")  # 注释：获取 server actor
        server_address, server_port = await self.server_actor.get_server_address.remote()  # 注释：获取 server 地址
        logger.debug(  # 注释：调试日志
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "  # 注释：日志片段
            f"server address: {server_address}, port: {server_port}"  # 注释：日志片段
        )  # 注释：logger.debug 结束
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address  # 注释：IPv6 地址处理
        self._engine = AsyncHttpServerAdapter(  # 注释：创建 HTTP 适配器
            model_path=self.model_config.local_path, host=host, port=server_port, launch_server=False  # 注释：连接参数
        )  # 注释：适配器创建结束

    async def resume(self, tags: list[str]):  # 注释：恢复权重或缓存
        """
        Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """  # 注释：保留英文说明
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:  # 注释：仅 local_rank0 执行
            await self._init_server_adapter()  # 注释：确保适配器已初始化
            await self._engine.resume_memory_occupation(tags=tags)  # 注释：调用恢复接口

    async def release(self):  # 注释：释放权重/缓存
        """
        Release weights and kv cache in GPU memory.
        """  # 注释：保留英文说明
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:  # 注释：仅 local_rank0 执行
            await self._init_server_adapter()  # 注释：确保适配器已初始化
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])  # 注释：释放缓存与权重

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):  # 注释：更新权重
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39

        功能：将权重按桶分批发送到 SGLang 服务并刷新缓存。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - weights：生成器，产出 (name, tensor) 对。  # 注释：参数含义
        - **kwargs：预留参数。  # 注释：参数含义
        返回：None。  # 注释：返回值语义
        副作用：通过 HTTP 适配器更新服务端权重。  # 注释：副作用说明
        异常/边界条件：服务不可用或权重格式不匹配将报错。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await update_weights(named_params_generator)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：...::ServerAdapter.update_weights。  # 注释：函数位置
        - 典型调用路径：rollout worker -> update_weights。  # 注释：调用链
        - 被谁调用：async_sglang_server.SGLangReplica。  # 注释：调用方说明
        - 调用了谁（项目内）：get_named_tensor_buckets、AsyncHttpServerAdapter。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：sgl_update_weights。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        if self.device_mesh["infer_tp"].get_local_rank() == 0:  # 注释：仅 local_rank0 初始化适配器
            await self._init_server_adapter()  # 注释：懒初始化

        update_weights_bucket_bytes = int(self.config.update_weights_bucket_megabytes) << 20  # 注释：计算桶大小字节数
        if self.config.get("quantization", None) == "fp8":  # 注释：FP8 量化路径
            from verl.utils.sglang.sglang_fp8_utils import quant_weights_by_name  # 注释：导入量化工具

            logger.info("Convert bf16 weights to fp8 format before loading")  # 注释：记录日志
            weights = quant_weights_by_name(  # 注释：执行量化
                weights,  # 注释：权重生成器
                self.model_config.hf_config.quantization_config,  # 注释：量化配置
                dtype=self.model_config.hf_config.dtype,  # 注释：权重 dtype
            )  # 注释：量化结束
        else:  # 注释：非 FP8 路径
            weights = weights  # 注释：保持原权重

        for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):  # 注释：分桶迭代
            await sgl_update_weights(  # 注释：调用 SGLang 权重更新
                engine=self._engine,  # 注释：HTTP 适配器
                params_batch=params_batch,  # 注释：一批参数
                device_mesh_key="infer_tp",  # 注释：设备网格 key
                device_mesh=self.device_mesh,  # 注释：设备网格
            )  # 注释：权重更新结束

        if self.device_mesh["infer_tp"].get_local_rank() == 0:  # 注释：仅 local_rank0 刷新缓存
            await self._engine.flush_cache()  # 注释：刷新服务端缓存
