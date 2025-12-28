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
模块用途：vLLM 后端的 Rollout 实现（异步/服务化），用于 RL 训练中的高效生成。（注释：模块级用途）
输入/输出：（注释：模块级 I/O）
  - 输入：RolloutConfig、HFModelConfig、DeviceMesh；运行期接收权重流与生成请求。（注释：输入说明）
  - 输出：生成后的 DataProto（或 ZeroMQ 地址/状态）。（注释：输出说明）
关键依赖：（注释：关键依赖）
  - vllm / ray / torch / zmq / filelock。（注释：外部依赖）
  - verl.workers.rollout.base.BaseRollout、verl.utils.vllm.*。（注释：项目内依赖）
典型用法（最小示例）：（注释：最小示例）
  - 由 ActorRolloutRefWorker._build_rollout() 构建并在 rollout 阶段调用 update_weights / generate_sequences。（注释：示例描述）
调用路径概览：（注释：调用路径）
  - `main_ppo.py` -> `RayPPOTrainer` -> `ActorRolloutRefWorker._build_rollout` -> `get_rollout_class` -> `vLLMAsyncRollout`。（注释：链路）
备注：（注释：补充说明）
  - FSDP 模式使用 DTensor/HF 权重加载；Megatron 模式需广播 pp stage 参数到各 tp rank。（注释：实现差异）
"""  # 注释：模块 docstring 结束

# 标准库导入（注释：用户/日志/路径/类型工具）
import getpass  # 注释：获取当前用户名用于锁文件命名
import logging  # 注释：日志记录
import os  # 注释：环境变量与进程信息
from dataclasses import asdict  # 注释：dataclass 转 dict
from types import MethodType  # 注释：动态绑定方法
from typing import Any, Generator  # 注释：类型提示

# 第三方依赖（注释：序列化/分布式/通信）
import cloudpickle as pickle  # 注释：跨进程序列化
import ray  # 注释：Ray 运行时
import torch  # 注释：张量与分布式
import torch.distributed  # 注释：分布式初始化检测
import zmq  # 注释：ZeroMQ 套接字
import zmq.asyncio  # 注释：异步 ZeroMQ
from filelock import FileLock  # 注释：文件锁避免端口冲突
from torch.distributed.device_mesh import DeviceMesh  # 注释：设备网格
from vllm.config import LoRAConfig  # 注释：vLLM LoRA 配置

# 项目内依赖（注释：事件循环工具）
from verl.utils.ray_utils import get_event_loop  # 注释：获取全局事件循环

# 第三方依赖（注释：vLLM Worker 兼容导入）
try:
    from vllm.worker.worker_base import WorkerWrapperBase  # 注释：vLLM 新路径
except ModuleNotFoundError:
    # 注释：旧版本 vLLM 兼容路径（参考 commit 说明）
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs  # 注释：版本比较工具

from verl import DataProto  # 注释：统一数据结构
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version  # 注释：vLLM 版本与休眠等级
from verl.utils.device import is_npu_available  # 注释：NPU 可用性
from verl.utils.distributed import initialize_global_process_group_ray  # 注释：Ray 分布式初始化
from verl.utils.ray_utils import ray_noset_visible_devices  # 注释：是否由 Ray 设置设备可见性
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge  # 注释：vLLM 补丁/LoRA 请求
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights  # 注释：FP8 辅助
from verl.workers.config import HFModelConfig, RolloutConfig  # 注释：配置类型
from verl.workers.rollout.base import BaseRollout  # 注释：Rollout 基类
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address  # 注释：端口/IPv6 工具
from verl.workers.rollout.vllm_rollout.utils import (  # 注释：vLLM LoRA 常量与工具
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)  # 注释：模块 logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：日志级别由环境变量控制

# 待办（TODO）：  # 注释：后续优化点
# 1. 支持 vLLM 中的 PP（流水线并行）  # 注释：并行能力补齐
# 2. tokenizer 传入可能不是必须（此处不做编码/解码）  # 注释：潜在简化
# 3. 简化初始化流程  # 注释：可维护性


# 若 vLLM 版本支持，注入必要补丁（注释：修复/兼容行为）
if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()  # 注释：执行 vLLM hijack


def _check_vllm_version_for_sleep_level():
    """
    检查 vLLM 版本是否满足 sleep_level 行为要求。（注释：函数用途）

    返回：（注释：返回值说明）
      - bool：True 表示版本 >= 0.11.0，支持安全 sleep_level 设置。（注释：返回含义）
    副作用：（注释：副作用说明）
      - 若无法读取版本，会打印 warning。（注释：日志副作用）
    调用路径依赖：（注释：调用关系说明）
      - `vLLMAsyncRollout.__init__` -> `_check_vllm_version_for_sleep_level`。（注释：调用链）
    """
    # 参考：https://github.com/vllm-project/vllm/issues/25171  # 注释：版本要求来源
    minver = "0.11.0"  # 注释：最低版本
    current_version = get_version("vllm")  # 注释：读取当前 vLLM 版本
    if not current_version:  # 注释：无法获取版本时
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False  # 注释：视为不满足
    return vs.parse(current_version) >= vs.parse(minver)  # 注释：比较版本


# 参考：https://github.com/vllm-project/vllm/issues/13175  # 注释：补丁背景
def _monkey_patch_compute_logits(model, vocab_size: int):
    """
    给 vLLM 模型注入 logits 裁剪逻辑，避免越界词表。（注释：函数用途）

    参数：（注释：参数说明）
      - model: vLLM 内部模型对象。（注释：输入含义）
      - vocab_size (int): tokenizer 词表大小。（注释：输入含义）
    副作用：（注释：副作用说明）
      - 覆盖 model.compute_logits 方法。（注释：行为修改）
    调用路径依赖：（注释：调用关系说明）
      - `vLLMAsyncRollout._load_model` -> `_monkey_patch_compute_logits`。（注释：调用链）
    """
    original_compute_logits = model.compute_logits  # 注释：保存原始方法

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)  # 注释：调用原始 logits
        logits[..., vocab_size:] = float("-inf")  # 注释：屏蔽超出词表的 logits
        return logits  # 注释：返回裁剪后的 logits

    model.compute_logits = MethodType(compute_logits, model)  # 注释：动态绑定新方法


class vLLMAsyncRollout(BaseRollout):
    """
    vLLM 异步 Rollout：包装 vLLM WorkerWrapperBase 的单进程推理引擎。（注释：类用途）

    输入：（注释：输入说明）
      - RolloutConfig / HFModelConfig / DeviceMesh。（注释：构造参数）
    输出：（注释：输出说明）
      - ZeroMQ 地址、权重更新结果、生成结果（DataProto）。（注释：运行期输出）
    关键依赖：（注释：依赖说明）
      - vLLM WorkerWrapperBase、ZeroMQ、Ray 分布式环境。（注释：依赖）
    调用路径依赖：（注释：调用关系说明）
      - `ActorRolloutRefWorker._build_rollout` -> `vLLMAsyncRollout`。（注释：构建路径）
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        """
        初始化异步 vLLM Rollout。（注释：方法用途）

        参数：（注释：参数说明）
          - config (RolloutConfig): rollout 配置。（注释：输入含义）
          - model_config (HFModelConfig): 模型/Tokenizer 配置。（注释：输入含义）
          - device_mesh (DeviceMesh): 设备网格。（注释：输入含义）
        副作用：（注释：副作用说明）
          - 创建 ZeroMQ socket 并启动后台 loop。（注释：通信副作用）
        调用路径依赖：（注释：调用关系说明）
          - `ActorRolloutRefWorker._build_rollout` -> `vLLMAsyncRollout.__init__`。（注释：调用链）
        """
        super().__init__(config, model_config, device_mesh)  # 注释：初始化基类
        self.tokenizer = self.model_config.tokenizer  # 注释：保存 tokenizer
        self.inference_engine: WorkerWrapperBase = None  # 注释：vLLM 引擎占位
        self.address = self._init_zeromq()  # 注释：初始化 ZeroMQ 并获得地址
        self.lora_config = (  # 注释：构造 LoRA 配置
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        # 依据版本与配置确定 sleep_level（注释：内存/兼容性权衡）
        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")  # 注释：风险提示
            self.sleep_level = 1  # 注释：降低 sleep_level
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL  # 注释：使用默认 sleep_level

    def _init_zeromq(self) -> str:
        """
        初始化 ZeroMQ REP socket，并启动异步监听循环。（注释：方法用途）

        返回：（注释：返回值说明）
          - str：ZeroMQ 地址（ipc 或 tcp）。（注释：返回含义）
        副作用：（注释：副作用说明）
          - 绑定端口并创建后台任务。（注释：资源副作用）
        """
        tensor_parallel_size = self.config.tensor_model_parallel_size  # 注释：TP 大小

        # 单机用 ipc，多机用 tcp（注释：通信方式）
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])  # 注释：本机 rank 数
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"  # 注释：通信类型

        # 文件锁防止多个 worker 绑定同一端口（注释：并发保护）
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()  # 注释：异步上下文
            self.socket = context.socket(zmq.REP)  # 注释：REP socket
            if socket_type == "ipc":  # 注释：IPC 模式
                pid = os.getpid()  # 注释：进程号
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"  # 注释：IPC 地址
            else:  # 注释：TCP 模式
                ip = ray.util.get_node_ip_address().strip("[]")  # 注释：节点 IP
                port, sock = get_free_port(ip)  # 注释：可用端口
                if is_valid_ipv6_address(ip):  # 注释：IPv6 情况
                    address = f"tcp://[{ip}]:{port}"  # 注释：IPv6 地址
                    self.socket.setsockopt(zmq.IPV6, 1)  # 注释：开启 IPv6
                else:
                    address = f"tcp://{ip}:{port}"  # 注释：IPv4 地址
            self.socket.bind(address)  # 注释：绑定 socket

        loop = get_event_loop()  # 注释：获取事件循环
        self.zmq_loop_task = loop.create_task(self._loop_forever())  # 注释：启动后台任务

        return address  # 注释：返回地址

    async def _loop_forever(self):
        """
        ZeroMQ 服务循环：接收请求、执行方法、返回结果。（注释：方法用途）
        """
        while True:  # 注释：持续服务
            try:
                message = await self.socket.recv()  # 注释：接收消息
                method, args, kwargs = pickle.loads(message)  # 注释：反序列化方法与参数
                result = await self._execute_method(method, *args, **kwargs)  # 注释：执行方法
                await self.socket.send(pickle.dumps(result))  # 注释：返回结果
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")  # 注释：记录异常
                await self.socket.send(pickle.dumps(e))  # 注释：返回异常
                break  # 注释：退出循环

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """
        初始化 vLLM worker 引擎。（注释：方法用途）

        参数：（注释：参数说明）
          - all_kwargs (list[dict]): vLLM worker 初始化参数列表。（注释：输入含义）
        副作用：（注释：副作用说明）
          - 初始化分布式进程组并构建 vLLM worker。（注释：副作用）
        """
        if not torch.distributed.is_initialized():  # 注释：分布式未初始化
            initialize_global_process_group_ray()  # 注释：Ray 环境初始化
        all_kwargs[0]["rank"] = int(os.environ["RANK"])  # 注释：注入 rank
        device_name = "NPU" if is_npu_available else "GPU"  # 注释：设备类型
        all_kwargs[0]["local_rank"] = (  # 注释：注入 local_rank
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]  # 注释：保存 vLLM 配置
        if self.lora_config:  # 注释：若启用 LoRA
            lora_dtype = getattr(torch, self.config.dtype)  # 注释：LoRA 精度
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)  # 注释：写入配置
        if self.config.quantization is not None:  # 注释：量化分支
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]  # 注释：支持的量化类型
            if self.config.quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(  # 注释：不支持的量化类型
                    f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {self.config.quantization}"
                )

            if self.config.quantization == "fp8":  # 注释：FP8 量化
                # Apply vllm fp8 patches  # 注释：应用 FP8 补丁
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.  # 注释：说明
                apply_vllm_fp8_patches()  # 注释：应用补丁

        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)  # 注释：构建 vLLM worker
        self.inference_engine.init_worker(all_kwargs)  # 注释：初始化 worker

    def _load_model(self, *args, **kwargs):
        """
        加载模型权重并对 logits 进行补丁。（注释：方法用途）
        """
        self.inference_engine.load_model(*args, **kwargs)  # 注释：委托 vLLM 加载
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))  # 注释：裁剪词表

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        """
        根据方法名分发到本地逻辑或 vLLM worker。（注释：方法用途）
        """
        if method == "init_worker":  # 注释：初始化 worker
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":  # 注释：加载模型
            return self._load_model(*args, **kwargs)
        else:  # 注释：其他方法转发给 vLLM
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """
        恢复权重或 KV cache 到 GPU 内存。（注释：方法用途）

        参数：（注释：参数说明）
          - tags (list[str]): ["weights"] 或 ["kv_cache"]。（注释：输入含义）
        """
        if self.config.free_cache_engine:  # 注释：仅在 free_cache_engine 开启时执行
            self.inference_engine.wake_up(tags=tags)  # 注释：唤醒 vLLM

    async def release(self):
        """
        释放权重与 KV cache 占用的 GPU 内存。（注释：方法用途）
        """
        if self.config.free_cache_engine:  # 注释：仅在 free_cache_engine 开启时执行
            self.inference_engine.sleep(level=self.sleep_level)  # 注释：进入 sleep

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """
        更新 rollout 模型权重（支持 LoRA 与全量权重）。（注释：方法用途）

        参数：（注释：参数说明）
          - weights: (name, tensor) 生成器。（注释：输入含义）
          - kwargs: peft_config/base_sync_done 等控制项。（注释：输入含义）
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)  # 注释：解析参数
        if peft_config and base_sync_done:  # 注释：LoRA 更新路径
            # 异步模式下先移除旧 LoRA（注释：避免冲突）
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)  # 注释：删除旧 LoRA
            weights = dict(weights)  # 注释：将生成器转为字典
            lora_request = TensorLoRARequest(  # 注释：构造 LoRA 请求
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)  # 注释：添加新 LoRA
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")  # 注释：日志
        else:  # 注释：全量权重更新路径
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader  # 注释：按需导入补丁

            model_runner = self.inference_engine.worker.model_runner  # 注释：取 model_runner
            model = model_runner.model  # 注释：取 model 实例
            patch_vllm_moe_model_weight_loader(model)  # 注释：修补 MoE 加载逻辑

            # FP8 量化权重加载逻辑（注释：兼容 sharding manager 退役）
            if is_fp8_model(model_runner.vllm_config):  # 注释：检测 FP8 模型
                logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")  # 注释：日志
                # 先将 bf16 权重转为 fp8 再加载（注释：权重转换）
                loaded_params = load_quanted_weights(weights, model_runner)  # 注释：加载 FP8 权重
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")  # 注释：日志
            else:  # 注释：非 FP8
                logger.info("Loading standard weights (non-FP8, async)")  # 注释：日志
                model.load_weights(weights)  # 注释：直接加载权重

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        同步模式生成序列（异步类中不支持）。（注释：方法用途）

        异常/边界条件：（注释：异常说明）
          - 直接抛 NotImplementedError。（注释：不支持）
        """
        raise NotImplementedError  # 注释：仅支持 server/async 形态

    # ==================== 服务端模式公开方法 ====================  # 注释：分隔标识

    def get_zeromq_address(self):
        """
        获取 ZeroMQ 服务地址。（注释：方法用途）

        返回：（注释：返回值说明）
          - str：ZeroMQ 地址。（注释：返回含义）
        """
        return self.address  # 注释：返回地址
