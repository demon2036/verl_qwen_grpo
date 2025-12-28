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
FSDP Workers 模块 - VERL 强化学习训练的分布式 Worker 实现（基于 FSDP 并行策略）

模块用途：
---------
本模块实现了基于 PyTorch FSDP（Fully Sharded Data Parallel）的 RL 训练 Worker 类，
支持 Actor、Rollout、Reference Policy、Critic、Reward Model 等多种角色的分布式训练与推理。
是 VERL 框架中 GRPO/PPO/RLOO 等 RL 算法在 FSDP 后端上的核心执行层。

主要功能：
1. **ActorRolloutRefWorker**：混合 Worker，可单独或组合扮演 Actor（训练）、Rollout（生成）、Ref（参考策略）角色
2. **CriticWorker**：Critic 模型的分布式训练与推理（仅用于 PPO，GRPO 不使用）
3. **RewardModelWorker**：奖励模型的分布式推理
4. **AsyncActorRolloutRefWorker**：异步版本的 ActorRolloutRefWorker，支持异步生成

输入/输出：
-----------
- 输入（初始化时）：
  * config: DictConfig（包含 model、actor、rollout、ref、critic 等子配置）
  * role: str（指定 worker 角色，如 "actor"、"rollout"、"actor_rollout_ref" 等）

- 输入（运行时）：
  * DataProto：封装了 batch 数据、meta_info、non_tensor_batch 的统一数据结构
  * prompts：生成任务的提示词

- 输出（运行时）：
  * DataProto：包含生成结果、log_prob、values、metrics 等信息

关键依赖：
---------
- PyTorch FSDP / FSDP2：模型分片与数据并行
- transformers：HuggingFace 模型加载（AutoModel、AutoModelForCausalLM 等）
- peft：LoRA 低秩适配（可选）
- Ray：分布式调度（通过 single_controller 抽象）
- vLLM / SGLang：高效推理后端（Rollout 阶段）
- verl.utils.fsdp_utils：FSDP 工具函数（wrap_policy、offload、checkpoint 等）
- verl.utils.ulysses：Ulysses 序列并行（Context Parallel）
- verl.workers.actor / critic：具体的 Actor/Critic 训练逻辑
- verl.workers.rollout：Rollout 后端抽象（vLLM/SGLang/HF）

典型用法（示例）：
------------------
# 1. 初始化混合 Worker（Actor + Rollout + Ref）
worker = ActorRolloutRefWorker(config=config, role="actor_rollout_ref")
worker.init_model()

# 2. 执行 Rollout 生成
prompts_data = DataProto(...)
output = worker.generate_sequences(prompts=prompts_data)

# 3. 计算 Ref log_prob
ref_log_prob_data = worker.compute_ref_log_prob(data=...)

# 4. 更新 Actor
metrics = worker.update_actor(data=train_data)

# 5. 保存 Checkpoint
worker.save_checkpoint(local_path="/path/to/ckpt", global_step=100)

调用路径概览：
--------------
- 入口脚本：verl/trainer/main_ppo.py
- 调度层：verl/trainer/ppo/ray_trainer.py（Ray 调度多个 worker）
- Worker 层（本模块）：
  * ActorRolloutRefWorker.init_model() -> _build_model_optimizer() / _build_rollout()
  * ActorRolloutRefWorker.generate_sequences() -> rollout.generate_sequences()
  * ActorRolloutRefWorker.update_actor() -> actor.update_policy()
  * ActorRolloutRefWorker.compute_ref_log_prob() -> ref_policy.compute_log_prob()
- Actor/Critic 逻辑：verl/workers/actor/dp_actor.py、verl/workers/critic/dp_critic.py
- Rollout 后端：verl/workers/rollout/vllm_rollout/vllm_rollout.py

被谁调用：
---------
- verl/trainer/ppo/ray_trainer.py::RayPPOTrainer（通过 Ray remote call）
- recipe/*/ray_trainer.py（各种算法变体的 Ray trainer）

调用了谁（项目内）：
--------------------
- verl.workers.actor.dp_actor::DataParallelPPOActor
- verl.workers.critic.dp_critic::DataParallelPPOCritic
- verl.workers.rollout.{vllm_rollout,sglang_rollout,hf_rollout}
- verl.workers.sharding_manager.fsdp_ulysses::FSDPUlyssesShardingManager
- verl.utils.fsdp_utils::{apply_fsdp2, offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu, ...}
- verl.utils.checkpoint.fsdp_checkpoint_manager::FSDPCheckpointManager
- verl.single_controller.base.worker::Worker
- verl.single_controller.base.decorator::{register, make_nd_compute_dataproto_dispatch_fn}

调用了谁（关键外部依赖）：
--------------------------
- torch.distributed.fsdp::FullyShardedDataParallel (FSDP)
- torch.distributed.device_mesh::init_device_mesh（Device Mesh 管理）
- transformers::{AutoModel, AutoModelForCausalLM, AutoConfig}
- peft::{get_peft_model, LoraConfig, PeftModel}
- ray（通过 verl.utils.ray_utils::get_event_loop）

注意事项：
---------
1. **Hybrid Engine**：actor_rollout 模式下，worker 在训练（trainer_mode）和生成（rollout_mode）之间动态切换
2. **Offload**：支持模型参数和优化器状态的 CPU offload，降低 GPU 显存峰值
3. **LoRA**：支持 LoRA 低秩适配，仅训练适配器参数
4. **Ulysses SP**：支持 Ulysses 序列并行（Context Parallel），扩展长序列训练能力
5. **FSDP2**：支持 PyTorch 2.4+ 的 FSDP2（fully_shard API）
6. **Profiler**：集成 NVTX/torch profiler/NPU profiler，支持性能分析
"""  # 注释：模块级 docstring，包含用途、输入输出、依赖、用法、调用路径等全部信息

# 标准库导入（注释：时间/日志/路径/类型等基础工具）
import datetime  # 注释：超时/时间相关
import json  # 注释：序列化/反序列化
import logging  # 注释：日志记录
import os  # 注释：环境变量与路径
import warnings  # 注释：警告控制
from dataclasses import asdict  # 注释：dataclass 转 dict
from typing import Any, Optional  # 注释：类型提示

# 第三方依赖（注释：数值、分布式、配置、性能与权重保存）
import numpy as np  # 注释：数组处理
import psutil  # 注释：系统资源监控
import torch  # 注释：张量与模型
import torch.distributed  # 注释：分布式初始化
import torch.distributed as dist  # 注释：分布式通信简写
from codetiming import Timer  # 注释：性能计时
from omegaconf import DictConfig, OmegaConf, open_dict  # 注释：配置管理
from peft import LoraConfig, TaskType, get_peft_model  # 注释：LoRA 适配
from safetensors.torch import save_file  # 注释：safetensors 保存
from torch.distributed.device_mesh import init_device_mesh  # 注释：设备网格
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 注释：FSDP 封装
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType  # 注释：FSDP 状态字典

try:
    # for torch 2.5+  （注释：见英文说明）
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

# 项目内依赖（注释：VERL 工具与子模块）
import verl.utils.torch_functional as verl_F  # 注释：张量功能函数
from verl import DataProto  # 注释：统一数据结构
from verl.models.transformers.monkey_patch import apply_monkey_patch  # 注释：模型补丁
from verl.single_controller.base import Worker  # 注释：Worker 抽象基类
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register  # 注释：分发装饰器
from verl.utils import hf_processor, hf_tokenizer  # 注释：HF tokenizer/processor 加载
from verl.utils.activation_offload import enable_activation_offloading  # 注释：激活 offload
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager  # 注释：FSDP checkpoint 管理
from verl.utils.config import omega_conf_to_dataclass  # 注释：配置转 dataclass
from verl.utils.device import (  # 注释：设备工具
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter  # 注释：FLOPs 统计
from verl.utils.fs import copy_to_local  # 注释：远端文件拷贝
from verl.utils.fsdp_utils import (  # 注释：FSDP 工具函数
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs  # 注释：外部库导入
from verl.utils.memory_utils import aggressive_empty_cache  # 注释：显存清理
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys  # 注释：模型辅助
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer  # 注释：性能分析
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max  # 注释：性能汇总
from verl.utils.py_functional import convert_to_regular_types  # 注释：类型转换
from verl.utils.ray_utils import get_event_loop  # 注释：事件循环
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig  # 注释：配置类型
from verl.workers.config.optimizer import build_optimizer  # 注释：优化器构建
from verl.workers.rollout import get_rollout_class  # 注释：rollout 类选择
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager  # 注释：Ulysses 分片管理

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# 注释：获取设备名称（cuda/npu/cpu）
device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    """
    创建 Device Mesh（设备网格），用于 FSDP/Hybrid Shard 并行策略。

    Device Mesh 是 PyTorch 分布式的核心抽象，定义了设备的拓扑结构，
    支持多维并行（如 FSDP + DDP、FSDP + TP 等）。

    参数：
    ------
    world_size : int
        全局进程总数（所有 GPU/NPU 的数量）
    fsdp_size : int
        FSDP 分片维度的大小。
        * fsdp_size < 0 或 >= world_size：仅使用 FSDP（1D mesh）
        * fsdp_size > 0 且 < world_size：使用 Hybrid Shard（2D mesh），即 DDP + FSDP

    返回：
    ------
    device_mesh : DeviceMesh
        设备网格对象，包含 mesh_dim_names（维度名称）和 mesh_shape（维度大小）

    逻辑说明：
    ---------
    1. 如果 fsdp_size 无效（< 0 或 >= world_size）：
       - 创建 1D mesh，shape=(world_size,)，dim_names=["fsdp"]
       - 表示所有设备都在同一个 FSDP 组内，完全分片（FULL_SHARD）

    2. 如果 fsdp_size 有效（0 < fsdp_size < world_size）：
       - 创建 2D mesh，shape=(world_size // fsdp_size, fsdp_size)，dim_names=["ddp", "fsdp"]
       - 表示 Hybrid Shard：DDP 组内数据并行，FSDP 组内模型分片
       - 例如：world_size=16, fsdp_size=4 -> mesh_shape=(4, 4)，4 个 DDP 组，每组 4 个 FSDP rank

    示例：
    ------
    # 场景 1：纯 FSDP（8 GPU 全分片）
    mesh = create_device_mesh(world_size=8, fsdp_size=-1)
    # -> mesh_shape=(8,), dim_names=["fsdp"]  （注释：见英文说明）

    # 场景 2：Hybrid Shard（8 GPU = 2 DDP 组 × 4 FSDP rank）
    mesh = create_device_mesh(world_size=8, fsdp_size=4)
    # -> mesh_shape=(2, 4), dim_names=["ddp", "fsdp"]  （注释：见英文说明）

    调用路径：
    ---------
    - ActorRolloutRefWorker.__init__() -> create_device_mesh()
    - CriticWorker.__init__() -> create_device_mesh()
    - RewardModelWorker.__init__() -> create_device_mesh()
    """
    # 注释：判断是否使用 Hybrid Shard
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 注释：fsdp_size 无效，创建 1D mesh（纯 FSDP）
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        # 注释：fsdp_size 有效，创建 2D mesh（DDP + FSDP）
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    """
    根据 Device Mesh 和 Zero3 配置，获取 FSDP ShardingStrategy（分片策略）。

    FSDP 支持多种分片策略，影响模型参数、梯度、优化器状态在 GPU 间的分布方式，
    从而权衡显存占用与通信开销。

    参数：
    ------
    device_mesh : DeviceMesh
        设备网格对象，ndim 决定是 FSDP（1D）还是 Hybrid Shard（2D）
    zero3_enable : bool, default=True
        是否启用 ZeRO-3（完全分片参数、梯度、优化器状态）
        * True：使用 FULL_SHARD（纯 FSDP）或 HYBRID_SHARD（Hybrid Shard）
        * False：使用 SHARD_GRAD_OP（仅分片梯度和优化器状态，参数不分片，类似 ZeRO-2）

    返回：
    ------
    sharding_strategy : ShardingStrategy
        PyTorch FSDP 的分片策略枚举值

    逻辑说明：
    ---------
    1. 如果 zero3_enable=True（默认）：
       - 1D mesh（ndim=1）：返回 FULL_SHARD（ZeRO-3）
       - 2D mesh（ndim=2）：返回 HYBRID_SHARD（ZeRO-3 + DDP）

    2. 如果 zero3_enable=False：
       - 1D mesh（ndim=1）：返回 SHARD_GRAD_OP（ZeRO-2，参数不分片，仅分片梯度和优化器状态）
       - 2D mesh（ndim=2）：返回 _HYBRID_SHARD_ZERO2（Hybrid Shard + ZeRO-2）

    3. 其他 ndim（> 2）：抛出 NotImplementedError

    **ShardingStrategy 说明**：
    - FULL_SHARD（ZeRO-3）：参数、梯度、优化器状态均分片，显存占用最小，通信开销最大
    - HYBRID_SHARD（ZeRO-3 + DDP）：FSDP 组内 FULL_SHARD，DDP 组间数据并行，平衡显存与通信
    - SHARD_GRAD_OP（ZeRO-2）：参数不分片（每 GPU 复制完整参数），仅分片梯度和优化器状态
    - _HYBRID_SHARD_ZERO2：Hybrid Shard + ZeRO-2，参数 DDP 复制，梯度和优化器状态 FSDP 分片

    示例：
    ------
    # 场景 1：8 GPU，纯 FSDP，ZeRO-3
    mesh = create_device_mesh(world_size=8, fsdp_size=-1)  # 1D mesh
    strategy = get_sharding_strategy(mesh, zero3_enable=True)
    # -> strategy = ShardingStrategy.FULL_SHARD  （注释：见英文说明）

    # 场景 2：8 GPU，Hybrid Shard（2 DDP × 4 FSDP），ZeRO-3
    mesh = create_device_mesh(world_size=8, fsdp_size=4)  # 2D mesh
    strategy = get_sharding_strategy(mesh, zero3_enable=True)
    # -> strategy = ShardingStrategy.HYBRID_SHARD  （注释：见英文说明）

    # 场景 3：8 GPU，纯 FSDP，ZeRO-2（参数不分片）
    mesh = create_device_mesh(world_size=8, fsdp_size=-1)  # 1D mesh
    strategy = get_sharding_strategy(mesh, zero3_enable=False)
    # -> strategy = ShardingStrategy.SHARD_GRAD_OP  （注释：见英文说明）

    调用路径：
    ---------
    - ActorRolloutRefWorker._build_model_optimizer() -> get_sharding_strategy()
    - CriticWorker._build_critic_model_optimizer() -> get_sharding_strategy()
    - RewardModelWorker._build_model() -> get_sharding_strategy()
    """
    from torch.distributed.fsdp import ShardingStrategy

    # 注释：根据 zero3_enable 选择对应的分片策略
    if zero3_enable:
        # 注释：ZeRO-3 模式（参数、梯度、优化器状态均分片）
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        # 注释：ZeRO-2 模式（仅分片梯度和优化器状态，参数不分片）
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    # 注释：根据 device_mesh 维度选择 FSDP 或 Hybrid Shard
    if device_mesh.ndim == 1:
        # 注释：1D mesh，纯 FSDP
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        # 注释：2D mesh，Hybrid Shard（DDP + FSDP）
        sharding_strategy = hsdp_strategy
    else:
        # 注释：不支持 3D 及以上的 mesh（如 TP + FSDP + DDP）
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def get_vl_model_vision_tower(vl_model_instance):
    """
    从多模态（Vision-Language）模型实例中提取 Vision Tower（视觉编码器）。

    Vision Tower 是多模态模型（如 Qwen2-VL、GLM-4V、LLaVA 等）中负责图像特征提取的模块，
    通常基于 Vision Transformer（ViT）或 CLIP 视觉编码器。在训练时，可以选择冻结 Vision Tower
    以节省显存和计算开销（仅训练语言模型部分）。

    参数：
    ------
    vl_model_instance : PreTrainedModel
        多模态模型实例（如 Qwen2VLForConditionalGeneration、GLM4VForCausalLM 等）

    返回：
    ------
    vision_tower : nn.Module or None
        视觉编码器模块（如 Qwen2VisionTransformer）
        * 如果模型有 Vision Tower：返回该模块
        * 如果模型不是多模态或无 Vision Tower：返回 None

    逻辑说明：
    ---------
    1. 兼容 transformers >= 4.52.0：Vision Tower 位于 vl_model_instance.model.visual
    2. 兼容 transformers < 4.52.0：Vision Tower 位于 vl_model_instance.visual
    3. 如果两者都不存在：返回 None（纯文本模型或不支持的模型）

    **为什么需要 Vision Tower**：
    - 冻结 Vision Tower（freeze_vision_tower=True）：减少显存和计算开销，适用于微调语言部分
    - 训练 Vision Tower：适用于从头训练多模态模型或联合微调视觉和语言部分

    示例：
    ------
    # 场景 1：Qwen2-VL 模型（transformers >= 4.52.0）
    model = Qwen2VLForConditionalGeneration.from_pretrained(...)
    vision_tower = get_vl_model_vision_tower(model)
    # -> vision_tower = model.model.visual (Qwen2VisionTransformer)  （注释：见英文说明）

    # 场景 2：Qwen2-VL 模型（transformers < 4.52.0）
    model = Qwen2VLForConditionalGeneration.from_pretrained(...)
    vision_tower = get_vl_model_vision_tower(model)
    # -> vision_tower = model.visual (Qwen2VisionTransformer)  （注释：见英文说明）

    # 场景 3：纯文本模型（如 Qwen2-7B）
    model = Qwen2ForCausalLM.from_pretrained(...)
    vision_tower = get_vl_model_vision_tower(model)
    # -> vision_tower = None  （注释：见英文说明）

    调用路径：
    ---------
    - ActorRolloutRefWorker._build_model_optimizer() -> get_vl_model_vision_tower()
      当 config.actor.freeze_vision_tower=True 时，冻结返回的 Vision Tower
    - CriticWorker._build_critic_model_optimizer() -> get_vl_model_vision_tower()
      当 config.critic.freeze_vision_tower=True 时，冻结返回的 Vision Tower
    """
    # 注释：兼容 transformers >= 4.52.0（Vision Tower 位于 model.visual）
    if hasattr(vl_model_instance, "model") and hasattr(vl_model_instance.model, "visual"):
        # transformers >= 4.52.0  （注释：见英文说明）
        return vl_model_instance.model.visual
    # 注释：兼容 transformers < 4.52.0（Vision Tower 位于 visual）
    elif hasattr(vl_model_instance, "visual"):
        # transformers < 4.52.0  （注释：见英文说明）
        return vl_model_instance.visual
    # 注释：纯文本模型或不支持的模型，返回 None
    return None


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """
    ActorRolloutRefWorker 类 - 混合角色 FSDP Worker（可扮演 Actor/Rollout/Ref 中的一个或多个）

    类用途：
    -------
    本类是 VERL 框架中 FSDP 后端的核心 Worker 类，支持在强化学习训练中扮演以下角色（单独或组合）：
    1. **Actor**：策略模型训练（PPO/GRPO 的 policy update）
    2. **Rollout**：策略模型推理（生成 response）
    3. **Ref**：参考策略（计算 ref_log_prob，用于 PPO/GRPO 的 KL 惩罚）

    **关键设计**：
    - **Hybrid Engine**：actor_rollout 模式下，worker 在 trainer_mode（训练）和 rollout_mode（生成）之间动态切换，
      共享同一个模型实例，通过 context switch 实现显存复用
    - **LoRA 支持**：actor 使用 LoRA 适配器训练，ref 使用 base model（无 LoRA），节省显存
    - **Offload 支持**：支持模型参数和优化器状态的 CPU offload，降低 GPU 显存峰值
    - **Ulysses SP**：支持 Ulysses 序列并行（Context Parallel），扩展长序列训练能力

    **角色组合**：
    - "actor"：仅训练，依赖外部 rollout worker 生成数据
    - "rollout"：仅生成，从 actor checkpoint 加载模型
    - "ref"：仅计算 ref_log_prob
    - "actor_rollout"：训练 + 生成（Hybrid Engine，最常用）
    - "actor_rollout_ref"：训练 + 生成 + 参考策略（LoRA 模式下，ref = actor without LoRA）

    输入：
    ------
    - config : DictConfig
        包含以下子配置：
        * model：模型配置（path、lora_rank、override_config 等）
        * actor：Actor 配置（ppo_mini_batch_size、ppo_epochs、fsdp_config、optim 等）
        * rollout：Rollout 配置（name="vllm/sglang/hf"、tensor_model_parallel_size、temperature 等）
        * ref：Ref 配置（log_prob_micro_batch_size、fsdp_config 等）
    - role : str
        Worker 角色，可选值：["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

    输出：
    ------
    运行时方法的输出（均为 DataProto）：
    - init_model()：无返回值，初始化模型和优化器
    - generate_sequences(prompts)：返回 DataProto（包含 responses、response_length、log_probs、timing 等）
    - compute_log_prob(data)：返回 DataProto（包含 old_log_probs / ref_log_prob、entropys）
    - update_actor(data)：返回 DataProto（包含 metrics，如 loss、entropy、approx_kl、lr 等）
    - save_checkpoint(local_path, ...)：保存 checkpoint 到本地/HDFS
    - load_checkpoint(local_path, ...)：从本地/HDFS 加载 checkpoint

    关键依赖：
    ---------
    - verl.workers.actor.dp_actor::DataParallelPPOActor（Actor 训练逻辑）
    - verl.workers.rollout.vllm_rollout / sglang_rollout / hf_rollout（Rollout 后端）
    - verl.utils.fsdp_utils::{apply_fsdp2, offload_fsdp_model_to_cpu, ...}（FSDP 工具）
    - verl.workers.sharding_manager.fsdp_ulysses::FSDPUlyssesShardingManager（Ulysses SP）
    - verl.utils.checkpoint.fsdp_checkpoint_manager::FSDPCheckpointManager（Checkpoint 管理）

    典型用法：
    ----------
    # 场景 1：Hybrid Engine（Actor + Rollout，最常用）
    worker = ActorRolloutRefWorker(config=config, role="actor_rollout")
    worker.init_model()

    # 训练循环
    for step in range(num_steps):
        # 1. Rollout 生成
        prompts = DataProto(...)
        output = worker.generate_sequences(prompts)
        # -> 自动 context switch：trainer_mode -> rollout_mode -> trainer_mode

        # 2. 计算 old_log_prob
        old_log_prob_data = worker.compute_log_prob(data)

        # 3. Actor 训练
        metrics = worker.update_actor(train_data)

        # 4. 定期保存 checkpoint
        if step % 100 == 0:
            worker.save_checkpoint(local_path=f"/ckpt/step_{step}", global_step=step)

    # 场景 2：LoRA + Ref（Actor + Rollout + Ref）
    worker = ActorRolloutRefWorker(config=config, role="actor_rollout_ref")
    worker.init_model()
    # Actor 使用 LoRA 适配器，Ref 使用 base model（通过 disable_adapter() 实现）

    # 计算 ref_log_prob
    ref_data = worker.compute_ref_log_prob(data)
    # -> 内部调用 actor.compute_log_prob() 并禁用 LoRA

    调用路径：
    ---------
    - verl/trainer/ppo/ray_trainer.py::RayPPOTrainer（Ray 调度，remote call）
      * rollout_worker.init_model.remote()
      * rollout_worker.generate_sequences.remote(prompts)
      * actor_worker.update_actor.remote(data)
      * ref_worker.compute_ref_log_prob.remote(data)

    被调用：
    -------
    - self._build_model_optimizer()：构建 Actor/Ref 模型和优化器
    - self._build_rollout()：构建 Rollout 后端（vLLM/SGLang/HF）
    - self.actor.update_policy()：执行 PPO/GRPO 的 policy update
    - self.rollout.generate_sequences()：调用推理后端生成 response
    - self.rollout.update_weights()：同步 Actor 权重到 Rollout

    注意事项：
    ---------
    1. **Hybrid Engine Context Switch**：
       - rollout_mode()：offload FSDP 模型到 CPU，将权重同步到 Rollout，启用 Rollout
       - trainer_mode()：offload Rollout（释放 KV cache），恢复 FSDP 模型到 GPU
    2. **LoRA Ref**：当 _is_lora=True 且 role 包含 "ref" 时，ref 使用 actor without LoRA（通过 disable_adapter() 实现）
    3. **Offload**：param_offload / optimizer_offload 可降低 GPU 显存峰值，但增加 CPU-GPU 传输开销
    4. **Ulysses SP**：ulysses_sequence_parallel_size > 1 时，启用 Ulysses SP，attention 计算在 SP 维度上并行
    5. **FSDP2**：strategy="fsdp2" 时，使用 PyTorch 2.4+ 的 fully_shard API，支持 offload_policy
    """  # 注释：ActorRolloutRefWorker 类的详细 docstring

    def __init__(self, config: DictConfig, role: str, **kwargs):
        """
        初始化 Actor/Rollout/Ref 混合 Worker，并建立分布式与设备网格环境。（注释：方法用途）

        参数：（注释：参数说明）
          - config (DictConfig): actor/rollout/ref/model 等配置集合。（注释：输入含义）
          - role (str): 角色标识（actor/rollout/ref/actor_rollout/actor_rollout_ref）。（注释：输入含义）
          - **kwargs: 预留扩展参数（当前未使用）。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：构造方法无返回）
        副作用：（注释：副作用说明）
          - 初始化分布式进程组、DeviceMesh、Profiler，并规范化 batch 相关配置。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - role 不合法会触发断言/ValueError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> worker = ActorRolloutRefWorker(cfg, role="actor_rollout")  # 初始化（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.__init__`。（注释：位置）
          - 典型调用路径：`main_ppo.py` -> `RayPPOTrainer.init_workers` -> Ray remote -> `__init__`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`（Ray worker 创建）。（注释：调用方）
          - 调用了谁（项目内）：`create_device_mesh` / `FSDPUlyssesShardingManager` / `omega_conf_to_dataclass`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch.distributed.init_process_group`。（注释：外部依赖）
        """
        Worker.__init__(self)  # 注释：初始化 Worker 基类

        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for FSDP  （注释：见英文说明）
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model  （注释：见英文说明）
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel  （注释：见英文说明）
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch  （注释：见英文说明）
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self.config.model.get("lora_adapter_path") is not None or self._lora_rank > 0

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]
        self.use_orig_params = self.config.actor.fsdp_config.get("use_orig_params", False)

        # TODO(haibin.lin):  （注释：见英文说明）
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,  （注释：见英文说明）
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.  （注释：见英文说明）
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)  （注释：见英文说明）
        # as they provides DictConfig-like interface  （注释：见英文说明）
        # The benefit of creating the dataclass config is to perform validation during __post_init__  （注释：见英文说明）
        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)  （注释：见英文说明）
            # This is for extendability in AsyncRL cases  （注释：见英文说明）
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig  （注释：见英文说明）
        # profiler_config is a ProfilerConfig dataclass  （注释：见英文说明）
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get("param_offload", False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get("optimizer_offload", False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload  （注释：见英文说明）
            self._is_offload_param = self.config.ref.fsdp_config.get("param_offload", False)

        # normalize config  （注释：见英文说明）
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            assert self.config.actor.ppo_mini_batch_size > 0, (
                f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after "
                f"normalization"
            )
            # micro bsz  （注释：见英文说明）
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size

            if self.config.actor.ppo_micro_batch_size_per_gpu is not None:
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, (
                    f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than "
                    f"ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                )

        # normalize rollout config  （注释：见英文说明）
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config  （注释：见英文说明）
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config: FSDPEngineConfig,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
    ):
        """
        构建 FSDP 模型与优化器（Actor 或 Ref），并返回相关对象。（注释：方法用途）

        参数：（注释：参数说明）
          - model_path: 模型权重路径或 HuggingFace Hub 名称。（注释：输入含义）
          - fsdp_config (FSDPEngineConfig): FSDP 引擎配置。（注释：输入含义）
          - optim_config: 优化器配置（仅 actor 需要）。（注释：输入含义）
          - override_model_config: 覆盖模型配置字段。（注释：输入含义）
          - use_remove_padding/use_fused_kernels/enable_gradient_checkpointing: 性能与显存选项。（注释：输入含义）
          - trust_remote_code/use_liger: HF 远程代码与 Liger kernel 开关。（注释：输入含义）
          - role (str): "actor" 或 "ref"。（注释：输入含义）
          - enable_activation_offload: 激活 offload 开关。（注释：输入含义）
        返回：（注释：返回值说明）
          - actor_module_fsdp: FSDP 包装后的模型。（注释：输出含义）
          - actor_optimizer: 构建好的优化器（ref 为 None）。（注释：输出含义）
          - actor_lr_scheduler: 学习率调度器（可为 None）。（注释：输出含义）
          - actor_model_config: 更新后的模型配置。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 初始化 tokenizer/processor，并可能修改其 chat_template。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - role 非法会触发断言；LR scheduler 类型不支持会抛错。（注释：边界）
        最小示例：（注释：最小示例）
          >>> model, opt, sched, cfg = self._build_model_optimizer(path, fsdp_cfg, optim_cfg, {})  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker._build_model_optimizer`。（注释：位置）
          - 典型调用路径：`ActorRolloutRefWorker.init_model` -> `_build_model_optimizer`。（注释：链路）
          - 被谁调用：`ActorRolloutRefWorker.init_model`。（注释：调用方）
          - 调用了谁（项目内）：`hf_tokenizer` / `apply_fsdp2` / `build_optimizer`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers.AutoModel*`。（注释：外部依赖）
        """
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForVision2Seq,
        )

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect  （注释：见英文说明）
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly  （注释：见英文说明）
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs  （注释：见英文说明）
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53  （注释：见英文说明）
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids  （注释：见英文说明）
        # Maybe support Ulysses in VisionAttention in the future and remove this patch  （注释：见英文说明）
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for kimi-vl  （注释：见英文说明）
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang  （注释：见英文说明）
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

            # Apply Liger kernel to the model if use_liger is set to True  （注释：见英文说明）
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2  （注释：见英文说明）
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to actor module")
            actor_module.enable_input_require_grads()

            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to {role} from: {lora_adapter_path}")

                # Copy adapter to local if needed  （注释：见英文说明）
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string  （注释：见英文说明）
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM

            else:
                # Convert config to regular Python types before creating PEFT model  （注释：见英文说明）
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "exclude_modules": convert_to_regular_types(self.config.model.exclude_modules),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well  （注释：见英文说明）
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = PrecisionType.to_dtype(fsdp_config.dtype)
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self._is_lora,
        )

        # if self._is_rollout and self.config.rollout.name == "hf":  （注释：见英文说明）
        #     # TODO(zhangchi.usc1992, shengguangming) fix me.  （注释：见英文说明）
        #     Current, auto_wrap_policy causes HFRollout to hang in Gemma  （注释：见英文说明）
        #     auto_wrap_policy = None  （注释：见英文说明）

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        fsdp_enable_zero3 = fsdp_config.reshard_after_forward
        sharding_strategy = get_sharding_strategy(fsdp_mesh, fsdp_enable_zero3)

        # TODO: add transformer policy  （注释：见英文说明）
        # We force reference policy to use CPUOffload to save memory.  （注释：见英文说明）
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation  （注释：见英文说明）
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                use_orig_params=self.use_orig_params,
                forward_prefetch=fsdp_config.get("forward_prefetch", False),
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = None if role == "actor" else CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config  （注释：见英文说明）
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

            actor_optimizer = build_optimizer(actor_module_fsdp.parameters(), optim_config)

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            lr_scheduler_type = optim_config.get("lr_scheduler_type", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            if self.rank == 0:
                print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if lr_scheduler_type == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif lr_scheduler_type == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self, trust_remote_code=False):
        """
        构建 rollout 推理后端（vLLM/SGLang/HF），并保存到 self.rollout。（注释：方法用途）

        参数：（注释：参数说明）
          - trust_remote_code (bool): 是否允许 HF 远程代码。（注释：输入含义）
        返回：（注释：返回值说明）
          - None（结果存入 self.rollout）。  # 注释：无显式返回
        副作用：（注释：副作用说明）
          - 初始化推理引擎与 tokenizer，并设置生成配置。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - rollout.name 不支持会抛 ValueError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> self._build_rollout(trust_remote_code=False)  # 构建 vLLM/HF rollout（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker._build_rollout`。（注释：位置）
          - 典型调用路径：`ActorRolloutRefWorker.init_model` -> `_build_rollout`。（注释：链路）
          - 被谁调用：`ActorRolloutRefWorker.init_model`。（注释：调用方）
          - 调用了谁（项目内）：`get_rollout_class` / `create_device_mesh`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`vllm`/`sglang` 推理引擎。（注释：外部依赖）
        """
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config  （注释：见英文说明）
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh  （注释：见英文说明）
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        self.rollout_device_mesh = rollout_device_mesh

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states  （注释：见英文说明）
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model  （注释：见英文说明）
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params  （注释：见英文说明）
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # used for LoRA  （注释：见英文说明）
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode  （注释：见英文说明）
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.  （注释：见英文说明）
        # For sync mode, we directly switch to trainer mode here.  （注释：见英文说明）
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.  （注释：见英文说明）
        if rollout_config.mode == "sync" and self._is_actor:
            loop = get_event_loop()
            loop.run_until_complete(self.trainer_mode())

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        # Special handling for LoRA with sleep_level=2:  （注释：见英文说明）
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.  （注释：见英文说明）
        # separately collect and update LoRA weights and base model weights through their respective interfaces.  （注释：见英文说明）
        # Here: params contains LoRA weights, base_model_params contains base model weights.  （注释：见英文说明）
        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2:
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        # important: need to manually set the random states of each tp to be identical.  （注释：见英文说明）
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        self.actor_module_fsdp.train()

        # add empty cache after each compute  （注释：见英文说明）
        aggressive_empty_cache(force_sync=True)

        set_expandable_segments(True)

        # restore random states  （注释：见英文说明）
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化 Actor/Ref 模型、优化器与 rollout 引擎。（注释：方法用途）

        返回：（注释：返回值说明）
          - None。（注释：初始化流程无返回）
        副作用：（注释：副作用说明）
          - 构建模型与优化器，可能加载 checkpoint，并初始化 rollout 后端。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 配置缺失或模型加载失败会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> worker.init_model()  # 构建 actor/rollout/ref（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.init_model`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.init_workers` -> `init_model`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`（remote 调用）。  # 注释：调用方
          - 调用了谁（项目内）：`_build_model_optimizer` / `_build_rollout` / `FSDPCheckpointManager`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers` / `torch`。（注释：外部依赖）
        """
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems  （注释：见英文说明）
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout  （注释：见英文说明）
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module  （注释：见英文说明）
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,  （注释：见英文说明）
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.  （注释：见英文说明）

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """
        执行 Actor 的 PPO/GRPO 更新并返回指标。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 训练 batch（含 advantages、old_log_prob 等）。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto：包含 loss/kl/entropy 等 metrics。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 更新 Actor 模型参数与优化器状态。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 若角色非 actor 会触发断言。（注释：边界）
        最小示例：（注释：最小示例）
          >>> metrics = worker.update_actor(batch)  # 返回指标（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.update_actor`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `update_actor`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`（remote 调用）。  # 注释：调用方
          - 调用了谁（项目内）：`self.actor.update_policy`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 反向传播。（注释：外部依赖）
        """
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy

            # perform training  （注释：见英文说明）
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr.item() if torch.is_tensor(lr) else lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics  （注释：见英文说明）
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        """
        使用 rollout 后端生成序列（responses/log_probs 等）。（注释：方法用途）

        参数：（注释：参数说明）
          - prompts (DataProto): 输入提示词 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto：包含 responses、log_probs、timing 等。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 可能触发 context switch（trainer_mode <-> rollout_mode）。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 未初始化 rollout 会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> out = worker.generate_sequences(prompts)  # 生成输出（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.generate_sequences`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `generate_sequences`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`（remote 调用）。  # 注释：调用方
          - 调用了谁（项目内）：`self.rollout.generate_sequences`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：vLLM/SGLang 推理引擎。（注释：外部依赖）
        """
        # Support all hardwares  （注释：见英文说明）
        assert self._is_rollout
        prompts = prompts.to(get_device_id())

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks  （注释：见英文说明）
        # to make sure meta_info["timing"] is the same  （注释：见英文说明）
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache  （注释：见英文说明）
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        """
        计算 Actor 的 log_prob（必要时禁用 LoRA）。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 含 prompts/responses 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto：包含 log_prob/entropy 等字段。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 可能临时禁用/恢复 LoRA 适配器。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 若角色非 actor，会触发断言。（注释：边界）
        最小示例：（注释：最小示例）
          >>> out = worker.compute_log_prob(batch)  # 计算 log_prob（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.compute_log_prob`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `compute_log_prob`。（注释：链路）
          - 被谁调用：`RayPPOTrainer` / `compute_ref_log_prob`。（注释：调用方）
          - 调用了谁（项目内）：`self.actor.compute_log_prob`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 前向计算。（注释：外部依赖）
        """
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob  （注释：见英文说明）
        # which is mostly used for ref log_prob calculation  （注释：见英文说明）
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares  （注释：见英文说明）
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        # we should always recompute old_log_probs when it is HybridEngine  （注释：见英文说明）
        config_source = self.config.ref if is_lora else self.config.rollout
        data.meta_info["micro_batch_size"] = config_source.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = config_source.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = config_source.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob  （注释：见英文说明）
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=not is_lora)
            tensors = {"ref_log_prob": output} if is_lora else {"old_log_probs": output}
            if not is_lora:
                tensors["entropys"] = entropys
            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes  （注释：见英文说明）
        # unshard the root FSDP module  （注释：见英文说明）
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        """
        计算参考策略（ref）log_prob。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 含 prompts/responses 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto：包含 ref_log_prob 等字段。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 可能通过禁用 LoRA 复用 actor 权重。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 若 ref 未初始化且无法复用会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> out = worker.compute_ref_log_prob(batch)  # 计算 ref log_prob（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.compute_ref_log_prob`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `compute_ref_log_prob`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`compute_log_prob` / `self.ref.compute_log_prob`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 前向计算。（注释：外部依赖）
        """
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref  （注释：见英文说明）
            data.meta_info["is_lora"] = True
            return self.compute_log_prob(data)
        assert self._is_ref
        # else:  （注释：见英文说明）
        # otherwise, the class have a standalone ref model  （注释：见英文说明）

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on ref.compute_log_prob
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes  （注释：见英文说明）
        # unshard the root FSDP module  （注释：见英文说明）
        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        保存 Actor/Rollout 相关 checkpoint（本地与可选 HDFS）。（注释：方法用途）

        参数：（注释：参数说明）
          - local_path: 本地保存目录。（注释：输入含义）
          - hdfs_path: 可选 HDFS 保存路径。（注释：输入含义）
          - global_step (int): 当前步数，用于命名/记录。（注释：输入含义）
          - max_ckpt_to_keep: 最多保留的 ckpt 数量。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 写入磁盘/HDFS；可能删除旧 checkpoint。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 路径无权限或磁盘不足会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> worker.save_checkpoint("/tmp/ckpt", global_step=100)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.save_checkpoint`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `save_checkpoint`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`FSDPCheckpointManager.save_checkpoint`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：文件系统 API。（注释：外部依赖）
        """
        from verl.utils.logger import log_with_rank

        # only support save and load ckpt for actor  （注释：见英文说明）
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        dist.barrier()

        if self._is_lora and hasattr(getattr(self, "actor_module", self.actor_module_fsdp), "peft_config"):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_model = getattr(self, "actor_module", self.actor_module_fsdp)
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(peft_model.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if fsdp_version(self.actor_module_fsdp) > 0:
                    self.actor_module_fsdp = self.actor_module_fsdp.to(get_device_name())
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(lora_params, os.path.join(lora_save_path, "adapter_model.safetensors"))
                        with open(os.path.join(lora_save_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                log_with_rank(
                    f"Save LoRA Adapter Error ({e})", rank=dist.get_rank(), logger=logger, log_only_rank_0=True
                )

            dist.barrier()
            log_with_rank(
                f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}",
                rank=dist.get_rank(),
                logger=logger,
                log_only_rank_0=True,
            )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        """
        加载 Actor/Rollout checkpoint（可从 HDFS 拉取到本地再加载）。（注释：方法用途）

        参数：（注释：参数说明）
          - local_path: 本地 checkpoint 路径。（注释：输入含义）
          - hdfs_path: 可选 HDFS 路径。（注释：输入含义）
          - del_local_after_load (bool): 加载后是否删除本地文件。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 读取文件并恢复模型/优化器状态。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 角色不支持加载会触发断言。（注释：边界）
        最小示例：（注释：最小示例）
          >>> worker.load_checkpoint("/tmp/ckpt")  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::ActorRolloutRefWorker.load_checkpoint`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `load_checkpoint`（resume）。  # 注释：链路
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`FSDPCheckpointManager.load_checkpoint`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：文件系统 API。（注释：外部依赖）
        """
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )

        # No checkpoint to load, just offload the model and optimizer to CPU  （注释：见英文说明）
        if local_path is None:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(self.actor_optimizer)
            return

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system  （注释：见英文说明）
        # This method is kept for backward compatibility but delegates to profiler  （注释：见英文说明）
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality  （注释：见英文说明）
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception:
                # silently ignore if profiler doesn't support memory snapshots  （注释：见英文说明）
                pass


class CriticWorker(Worker, DistProfilerExtension):
    """
    CriticWorker 类 - Critic 模型的 FSDP Worker（仅用于 PPO，GRPO 不使用）

    类用途：
    -------
    本类实现 Critic 模型（Value Function）的分布式训练与推理，用于 PPO 算法中的优势函数估计。
    **注意**：GRPO 算法不需要 Critic，因此 GRPO 训练时不会实例化此 worker。

    Critic 模型的作用：
    - 估计状态价值 V(s)，用于计算 GAE（Generalized Advantage Estimation）
    - 通过最小化 Value Loss（通常是 MSE）来训练
    - 与 Actor 模型并行训练，但独立优化

    输入：
    ------
    - config : FSDPCriticConfig
        Critic 配置，包含：
        * model：模型配置（path、tokenizer_path、lora_rank、fsdp_config 等）
        * ppo_mini_batch_size：PPO mini-batch 大小
        * ppo_epochs：PPO 训练轮数
        * optim：优化器配置（lr、weight_decay、lr_scheduler_type 等）
        * forward_micro_batch_size：前向推理的 micro-batch 大小
        * ulysses_sequence_parallel_size：Ulysses SP 大小（默认 1）

    输出：
    ------
    运行时方法的输出（均为 DataProto）：
    - init_model()：无返回值，初始化 Critic 模型和优化器
    - compute_values(data)：返回 DataProto（包含 values，shape=(batch_size, response_length)）
    - update_critic(data)：返回 DataProto（包含 metrics，如 critic_loss、lr 等）
    - save_checkpoint(...)：保存 checkpoint 到本地/HDFS
    - load_checkpoint(...)：从本地/HDFS 加载 checkpoint

    关键依赖：
    ---------
    - verl.workers.critic.dp_critic::DataParallelPPOCritic（Critic 训练逻辑）
    - verl.utils.model::load_valuehead_model（加载 Critic 模型，带 value head）
    - verl.utils.fsdp_utils（FSDP 工具）
    - verl.workers.sharding_manager.fsdp_ulysses::FSDPUlyssesShardingManager（Ulysses SP）

    典型用法：
    ----------
    # PPO 训练时（GRPO 不使用）
    critic_worker = CriticWorker(config=critic_config)
    critic_worker.init_model()

    # 1. 计算 values（用于 GAE）
    values_data = critic_worker.compute_values(data)
    # -> 返回 DataProto(tensors={"values": ...})

    # 2. 更新 Critic
    metrics = critic_worker.update_critic(train_data)
    # -> 返回 DataProto(meta_info={"metrics": {"critic_loss": ..., "lr": ...}})

    调用路径：
    ---------
    - verl/trainer/ppo/ray_trainer.py::RayPPOTrainer（Ray 调度，remote call）
      * critic_worker.init_model.remote()
      * critic_worker.compute_values.remote(data)
      * critic_worker.update_critic.remote(data)

    被调用：
    -------
    - self._build_critic_model_optimizer()：构建 Critic 模型和优化器
    - self.critic.compute_values()：执行前向推理，计算 values
    - self.critic.update_critic()：执行 Critic 的训练更新

    注意事项：
    ---------
    1. **Critic vs Actor**：Critic 估计 V(s)，Actor 输出 π(a|s)，两者独立训练
    2. **GRPO 不使用**：GRPO 使用 group-relative 优势，不需要 Critic
    3. **Offload 支持**：支持参数和优化器的 CPU offload（param_offload / optimizer_offload）
    4. **LoRA 支持**：支持 LoRA 微调（通过 lora_rank / lora_adapter_path 配置）
    5. **Value Head**：Critic 模型是 AutoModelForTokenClassification + value head（输出标量 value）
    """  # 注释：CriticWorker 类的详细 docstring
    def __init__(self, config: FSDPCriticConfig):
        """
        初始化 Critic Worker（用于 PPO 价值函数训练）。（注释：方法用途）

        参数：（注释：参数说明）
          - config (FSDPCriticConfig): Critic 相关配置。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：构造方法无返回）
        副作用：（注释：副作用说明）
          - 初始化分布式环境与设备网格。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 配置缺失会触发断言或 KeyError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> critic = CriticWorker(cfg)  # 初始化（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker.__init__`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.init_workers` -> `CriticWorker`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`create_device_mesh`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch.distributed.init_process_group`。（注释：外部依赖）
        """
        Worker.__init__(self)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
        self.config: FSDPCriticConfig = config

        # build device mesh for Ulysses Sequence Parallel  （注释：见英文说明）
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # create training dispatch  （注释：见英文说明）
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "critic", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("critic", dp_rank=self.rank, is_collect=True)

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params  （注释：见英文说明）
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config  （注释：见英文说明）
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.forward_micro_batch_size //= (
                torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size
            )
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size

        if self.config.ppo_micro_batch_size_per_gpu is not None:
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, (
                f"normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than "
                f"ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}"
            )
        self._is_lora = (
            self.config.model.get("lora_adapter_path") is not None or self.config.model.get("lora_rank", 0) > 0
        )
        self.use_orig_params = self.config.model.fsdp_config.get("use_orig_params", False)

    def _build_critic_model_optimizer(self, config):
        """
        构建 Critic 模型与优化器。（注释：方法用途）

        参数：（注释：参数说明）
          - config: Critic 配置（包含 model/optim/fsdp 等）。  # 注释：输入含义
        返回：（注释：返回值说明）
          - critic_module_fsdp: FSDP 包装后的 critic 模型。（注释：输出含义）
          - critic_optimizer: 优化器。（注释：输出含义）
          - critic_lr_scheduler: 学习率调度器（可为 None）。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 初始化 tokenizer/processor 与模型权重。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 不支持的模型类型会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> model, opt, sched = self._build_critic_model_optimizer(cfg)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker._build_critic_model_optimizer`。（注释：位置）
          - 典型调用路径：`CriticWorker.init_model` -> `_build_critic_model_optimizer`。（注释：链路）
          - 被谁调用：`CriticWorker.init_model`。（注释：调用方）
          - 调用了谁（项目内）：`build_optimizer` / `apply_fsdp2`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers.AutoModelForCausalLM`。（注释：外部依赖）
        """
        # the following line is necessary  （注释：见英文说明）
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        use_shm = config.model.get("use_shm", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info  （注释：见英文说明）
        # using random initialized model from any architecture. May not be the same as Actor.  （注释：见英文说明）

        tokenizer_path = copy_to_local(config.model.tokenizer_path, use_shm=use_shm)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template
        override_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        # override model kwargs  （注释：见英文说明）
        attn_implementation = override_config.get("attn_implementation", "flash_attention_2")
        critic_model_config = AutoConfig.from_pretrained(
            local_path,
            attn_implementation=attn_implementation,
            trust_remote_code=config.model.get("trust_remote_code", False),
        )
        # TODO: VL models use VisionAttention, which directly uses flash_attention in transformers>=4.53  （注释：见英文说明）
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids  （注释：见英文说明）
        # Maybe support Ulysses in VisionAttention in the future and remove this patch  （注释：见英文说明）
        if self.ulysses_sequence_parallel_size > 1 and hasattr(critic_model_config, "vision_config"):
            critic_model_config.vision_config._attn_implementation = "eager"

        critic_model_config.num_labels = 1
        # patch for kimi-vl  （注释：见英文说明）
        if getattr(critic_model_config, "model_type", None) == "kimi_vl":
            critic_model_config.text_config.topk_method = "greedy"

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            critic_model_config.summary_dropout_prob = 0.0

            critic_module = load_valuehead_model(
                local_path,
                torch_dtype,
                critic_model_config,
                config.model.get("trust_remote_code", False),
            )

            use_remove_padding = config.model.get("use_remove_padding", False)

            apply_monkey_patch(
                model=critic_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            # some parameters may not in torch_dtype  （注释：见英文说明）
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()

            # Check if we should load a pre-trained LoRA adapter  （注释：见英文说明）
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                print(f"Loading pre-trained LoRA adapter to critic from: {lora_adapter_path}")

                # Copy adapter to local if needed  （注释：见英文说明）
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.get("use_shm", False))

                critic_module = PeftModel.from_pretrained(critic_module, local_adapter_path, is_trainable=True)
                peft_config = critic_module.peft_config["default"]
                # Ensure task_type is TaskType enum, not string  （注释：见英文说明）
                # Use TOKEN_CLS for Critic since it's loaded as AutoModelForTokenClassification  （注释：见英文说明）
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.TOKEN_CLS

            else:
                # Convert config to regular Python types before creating PEFT model  （注释：见英文说明）
                # Use TOKEN_CLS for Critic since it's loaded as AutoModelForTokenClassification  （注释：见英文说明）
                lora_config = {
                    "task_type": TaskType.TOKEN_CLS,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=critic_module,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self._is_lora,
        )

        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.model.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(critic_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[critic model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[critic model] No vision tower found.")

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation  （注释：见英文说明）
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=self.use_orig_params,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        if config.model.get("enable_activation_offload", False):
            enable_gradient_checkpointing = config.model.get("enable_gradient_checkpointing", False)
            enable_activation_offloading(critic_module, config.strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = build_optimizer(critic_module.parameters(), config.optim)

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))

        lr_scheduler_type = config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        if self.rank == 0:
            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if lr_scheduler_type == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps
            )
        elif lr_scheduler_type == "cosine":
            min_lr_ratio = config.optim.get("min_lr_ratio", 0.0)
            num_cycles = config.optim.get("num_cycles", 0.5)
            critic_lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=critic_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps,
                min_lr_ratio=min_lr_ratio,
                num_cycles=num_cycles,
            )
        else:
            raise NotImplementedError(f"LR scheduler type {lr_scheduler_type} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化 Critic 模型与优化器，并加载必要补丁。（注释：方法用途）

        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 构建模型/优化器，并应用 monkey patch。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 模型加载失败会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> critic.init_model()  # 初始化 critic（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker.init_model`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.init_workers` -> `CriticWorker.init_model`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`_build_critic_model_optimizer` / `apply_monkey_patch`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers` / `torch`。（注释：外部依赖）
        """
        # This is used to import external_lib into the huggingface systems  （注释：见英文说明）
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        """
        前向计算 Critic value 预测。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 含 prompts/responses 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto：包含 values 等字段。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 若启用 offload，可能触发模型上卡/下卡。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 未初始化 critic 会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> out = critic.compute_values(batch)  # 计算 value（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker.compute_values`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `compute_values`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`self.critic.compute_values`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 前向计算。（注释：外部依赖）
        """
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation  （注释：见英文说明）
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.compute_values
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        """
        执行 Critic 更新并返回训练指标。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 训练 batch（含 returns/values 等）。  # 注释：输入含义
        返回：（注释：返回值说明）
          - DataProto：包含 critic loss 等 metrics。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 更新 critic 参数与优化器状态。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 未初始化 critic 会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> out = critic.update_critic(batch)  # 更新 critic（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker.update_critic`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `update_critic`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`self.critic.update_critic`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 反向传播。（注释：外部依赖）
        """
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())

        # perform forward computation  （注释：见英文说明）
        with self.ulysses_sharding_manager:
            data = data.to("cpu")  # data will to device with each micro batch on critic.update_critic
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr
            self.critic_lr_scheduler.step()

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        """
        保存 Critic checkpoint（本地与可选 HDFS）。（注释：方法用途）

        参数：（注释：参数说明）
          - local_path: 本地保存目录。（注释：输入含义）
          - hdfs_path: 可选 HDFS 保存路径。（注释：输入含义）
          - global_step (int): 当前步数。（注释：输入含义）
          - max_ckpt_to_keep: 最多保留的 ckpt 数量。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 写入磁盘/HDFS，可能清理旧 ckpt。（注释：副作用）
        最小示例：（注释：最小示例）
          >>> critic.save_checkpoint("/tmp/ckpt", global_step=10)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker.save_checkpoint`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `save_checkpoint`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`FSDPCheckpointManager.save_checkpoint`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：文件系统 API。（注释：外部依赖）
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=True):
        """
        加载 Critic checkpoint（可从 HDFS 拉取）。（注释：方法用途）

        参数：（注释：参数说明）
          - local_path: 本地 checkpoint 路径。（注释：输入含义）
          - hdfs_path: 可选 HDFS 路径。（注释：输入含义）
          - del_local_after_load (bool): 加载后是否删除本地文件。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 恢复模型与优化器状态。（注释：副作用）
        最小示例：（注释：最小示例）
          >>> critic.load_checkpoint("/tmp/ckpt")  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::CriticWorker.load_checkpoint`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `load_checkpoint`（resume）。  # 注释：链路
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`FSDPCheckpointManager.load_checkpoint`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：文件系统 API。（注释：外部依赖）
        """
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py  （注释：见英文说明）
class RewardModelWorker(Worker, DistProfilerExtension):
    """
    RewardModelWorker 类 - Reward Model 的 FSDP Worker（用于 PPO/GRPO 的奖励模型推理）

    类用途：
    -------
    本类实现 Reward Model（奖励模型）的分布式推理，用于 PPO 算法中的奖励计算。
    奖励模型通常是一个分类器（继承自 AutoModelForTokenClassification），输出每个 token 的标量奖励值，
    最终取序列结束位置的奖励作为整个 response 的奖励分数。

    **注意**：
    - **GRPO 算法不使用奖励模型**（GRPO 直接用 ground-truth 答案的正确性作为奖励）
    - **PPO 算法需要奖励模型**（PPO 用训练好的 Reward Model 对生成的 response 打分）
    - 本实现仅支持继承自 `AutoModelForTokenClassification` 的奖励模型
    - 奖励模型通常预先训练好（如在 preference learning 阶段），在 PPO 训练时仅做推理（freeze）

    输入配置：
    ----------
    config : OmegaConf / DictConfig
        包含以下关键配置：
        - config.model.path : str
            奖励模型的权重路径（HuggingFace 模型目录 或 HDFS 路径）
        - config.model.input_tokenizer : str, optional
            输入序列的 tokenizer 路径（若奖励模型的 tokenizer 与生成模型不同则需指定）
        - config.model.fsdp_config.fsdp_size : int
            FSDP 分片大小（<0 表示 FSDP-only，>0 表示 Hybrid Shard）
        - config.ulysses_sequence_parallel_size : int, optional
            Ulysses 序列并行大小（默认 1，即不使用序列并行）
        - config.model.use_remove_padding : bool, optional
            是否使用 remove_padding 优化（需配合 flash-attn）
        - config.strategy : str
            FSDP 策略，"fsdp" 或 "fsdp2"（fsdp2 需 PyTorch >= 2.4）
        - config.micro_batch_size : int, optional
            每个 GPU 的 micro batch size（用于推理分批）

    输出 DataProto：
    ---------------
    compute_rm_scores(data: DataProto) -> DataProto
        输入 data.batch 包含：
            - input_ids : torch.Tensor, shape [batch_size, seq_len]
                输入序列的 token IDs
            - attention_mask : torch.Tensor, shape [batch_size, seq_len]
                注意力掩码
            - position_ids : torch.Tensor, shape [batch_size, seq_len]
                位置 ID（可选，用于旋转位置编码）
        输出 DataProto.batch 包含：
            - rm_scores : torch.Tensor, shape [batch_size]
                每个序列的奖励分数（通常取序列结束位置的 logit）

    关键依赖：
    ---------
    - PyTorch FSDP / FSDP2 (fully_shard API)
    - transformers.AutoModelForTokenClassification
    - verl.utils.model.load_valuehead_model（可选，用于加载带 value head 的奖励模型）
    - verl.single_controller.base.Worker（Worker 基类）
    - flash-attn（可选，用于 flash attention 加速）

    典型用法：
    ----------
    示例 1：PPO 训练中的奖励模型推理
    ```python
    # 1. 定义奖励模型配置
    rm_config = OmegaConf.create({
        "model": {
            "path": "~/reward_model/skywork-reward-gemma-2-27b-v0.2",  # 奖励模型路径
            "fsdp_config": {"fsdp_size": 4},  # 4-way FSDP
            "use_remove_padding": True,  # 使用 remove_padding 优化
        },
        "strategy": "fsdp2",  # 使用 FSDP2
        "micro_batch_size": 8,  # 推理 micro batch size
    })

    # 2. 创建 Reward Model Worker（在 Ray 集群中）
    rm_worker_group = RayWorkerGroup(
        resources_config={"reward_model": 4},  # 分配 4 个 GPU
        worker_cls=RewardModelWorker,
        worker_cls_args=(rm_config,)
    )

    # 3. 初始化奖励模型
    rm_worker_group.init_model()

    # 4. 推理奖励分数（在 PPO rollout 后）
    data = DataProto(batch={
        "input_ids": rollout_data.batch["input_ids"],  # [batch_size, seq_len]
        "attention_mask": rollout_data.batch["attention_mask"],  # [batch_size, seq_len]
    })
    output = rm_worker_group.compute_rm_scores(data)
    rm_scores = output.batch["rm_scores"]  # [batch_size]，每个 response 的奖励分数

    # 5. 在 PPO 算法中，rm_scores 会用于计算 advantages（优势函数）
    # advantages = rm_scores + gamma * V(s') - V(s)  （注释：见英文说明）
    ```

    示例 2：多模态奖励模型（如评估图像生成质量）
    ```python
    rm_config = OmegaConf.create({
        "model": {
            "path": "~/reward_model/vision-reward-model",
            "input_tokenizer": "~/qwen2-vl-7b-instruct",  # 输入 tokenizer
        },
        # ...  （注释：见英文说明）
    })
    # 推理时需同时传入 pixel_values
    data = DataProto(batch={
        "input_ids": ...,
        "pixel_values": ...,  # [batch_size, num_images, 3, H, W]
    })
    output = rm_worker_group.compute_rm_scores(data)
    ```

    调用路径概览：
    --------------
    训练脚本 (verl/trainer/main_ppo.py)
      -> PPOTrainer.fit()
        -> ray_trainer.PPORayTrainer.fit_epoch()
          -> RewardModelWorker.compute_rm_scores(rollout_data)  # 奖励模型推理
            -> DataParallelRewardModel.compute_rm_scores()
              -> FSDP forward（分布式推理）

    所在位置：
    ----------
    - 路径：`verl/workers/fsdp_workers.py`
    - 类名：`RewardModelWorker`

    被谁调用：
    ----------
    - `verl/trainer/ppo/ray_trainer.py` 的 PPORayTrainer（PPO 训练时调用奖励模型）
    - `verl/experimental/reward_loop/reward_loop.py`（实验性的 reward loop 功能）

    调用了谁（项目内）：
    ------------------
    - `verl/workers/reward_model/base.py::DataParallelRewardModel`（奖励模型的推理逻辑）
    - `verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager`（Ulysses 序列并行）
    - `verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager`（checkpoint 管理）

    调用了谁（关键外部依赖）：
    ----------------------
    - `torch.distributed.fsdp.FullyShardedDataParallel`（FSDP 分布式推理）
    - `transformers.AutoModelForTokenClassification`（奖励模型加载）
    - `flash_attn`（可选，flash attention 加速）

    注意事项：
    ----------
    1. **仅用于 PPO，不用于 GRPO**
       - GRPO 算法直接用 ground-truth 答案的正确性（如 GSM8K 的数学答案）作为奖励
       - PPO 算法需要 Reward Model 对任意生成的 response 打分

    2. **奖励模型的类型限制**
       - 本实现仅支持 `AutoModelForTokenClassification`（分类器）
       - 奖励模型输出 shape 为 [batch_size, seq_len, 1]，取最后一个有效 token 的 logit 作为奖励

    3. **奖励模型通常是冻结的（freeze）**
       - 奖励模型在 PPO 训练时不更新参数（仅推理）
       - 奖励模型通常预先在 preference dataset 上训练好（如 RLHF 的 reward modeling 阶段）

    4. **CPU Offload 支持**
       - 奖励模型支持 CPU offload（config.strategy="fsdp" + CPUOffload）
       - 在显存紧张时可将模型参数卸载到 CPU（推理时再 load 到 GPU）

    5. **与 input_tokenizer 的区别**
       - 若奖励模型的 tokenizer 与生成模型不同，需指定 `config.model.input_tokenizer`
       - 推理时会先用 input_tokenizer 解码，再用 reward_model_tokenizer 重新编码

    6. **Ulysses 序列并行支持**
       - 支持 Ulysses Sequence Parallel（长序列并行）
       - 需设置 `config.ulysses_sequence_parallel_size > 1`
    """

    def __init__(self, config):
        """
        初始化 Reward Model Worker（用于规则/模型奖励计算）。  # 注释：方法用途

        参数：（注释：参数说明）
          - config: reward_model 配置（含模型路径、批大小等）。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：构造方法无返回）
        副作用：（注释：副作用说明）
          - 初始化分布式进程组与设备网格。（注释：副作用）
        最小示例：（注释：最小示例）
          >>> rm = RewardModelWorker(cfg)  # 初始化（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker.__init__`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.init_workers` -> `RewardModelWorker`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`create_device_mesh`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch.distributed.init_process_group`。（注释：外部依赖）
        """
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )

        import torch.distributed

        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # build device mesh for Ulysses Sequence Parallel  （注释：见英文说明）
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # create training dispatch  （注释：见英文说明）
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "reward", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("reward", dp_rank=self.rank, is_collect=True)

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)

        # normalize config  （注释：见英文说明）
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        """
        构建 Reward Model（FSDP 包装）。  # 注释：方法用途

        参数：（注释：参数说明）
          - config: reward_model 配置（model/fsdp/optim 等）。（注释：输入含义）
        返回：（注释：返回值说明）
          - model_fsdp: FSDP 包装后的奖励模型。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 初始化 tokenizer/processor 并加载权重。（注释：副作用）
        最小示例：（注释：最小示例）
          >>> model = self._build_model(cfg)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker._build_model`。（注释：位置）
          - 典型调用路径：`RewardModelWorker.init_model` -> `_build_model`。（注释：链路）
          - 被谁调用：`RewardModelWorker.init_model`。（注释：调用方）
          - 调用了谁（项目内）：`apply_fsdp2` / `hf_tokenizer`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers.AutoModelForCausalLM`。（注释：外部依赖）
        """
        # the following line is necessary  （注释：见英文说明）
        from torch.distributed.fsdp import CPUOffload
        from transformers import AutoConfig, AutoModelForTokenClassification

        use_shm = config.model.get("use_shm", False)
        # download the checkpoint from hdfs  （注释：见英文说明）
        local_path = copy_to_local(config.model.path, use_shm=use_shm)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
            self.input_tokenizer = hf_tokenizer(
                input_tokenizer_local_path, trust_remote_code=config.model.get("trust_remote_code", False)
            )
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get("trust_remote_code", False))

        trust_remote_code = config.model.get("trust_remote_code", False)
        override_config = OmegaConf.to_container(OmegaConf.create(config.model.get("override_config", {})))
        model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=override_config.get("attn_implementation", "flash_attention_2"),
        )
        model_config.num_labels = 1

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect  （注释：见英文说明）
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_config.classifier_dropout = 0.0
            reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=trust_remote_code,
            )

            apply_monkey_patch(
                model=reward_module,
                use_remove_padding=config.model.get("use_remove_padding", False),
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
            )

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        if config.strategy == "fsdp":
            reward_module = FSDP(
                reward_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=sharding_strategy,  # zero3
                sync_module_states=True,
                cpu_offload=CPUOffload(offload_params=True),
                forward_prefetch=self.config.model.fsdp_config.forward_prefetch,
                device_mesh=self.device_mesh,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            cpu_offload = CPUOffloadPolicy(pin_memory=True)
            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "offload_policy": cpu_offload,
                "reshard_after_forward": config.model.fsdp_config.reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=self.device_mesh.shape[-1]),
            }
            full_state = reward_module.state_dict()
            apply_fsdp2(reward_module, fsdp_kwargs, config.model.fsdp_config)
            fsdp2_load_full_state_dict(reward_module, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"Unknown strategy: {config.strategy}")
        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        初始化 Reward Model 与相关 tokenizer/processor。（注释：方法用途）

        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 构建模型并应用 monkey patch。（注释：副作用）
        最小示例：（注释：最小示例）
          >>> rm.init_model()  # 初始化奖励模型（示例）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker.init_model`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.init_workers` -> `RewardModelWorker.init_model`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
          - 调用了谁（项目内）：`_build_model` / `apply_monkey_patch`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers` / `torch`。（注释：外部依赖）
        """
        # This is used to import external_lib into the huggingface systems  （注释：见英文说明）
        import_external_libs(self.config.model.get("external_lib", None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        """
        对单个 micro-batch 执行 RM 前向并返回分数。（注释：方法用途）

        参数：（注释：参数说明）
          - micro_batch: micro-batch 输入（含 input_ids/attention_mask 等）。  # 注释：输入含义
        返回：（注释：返回值说明）
          - torch.Tensor：该 micro-batch 的 reward 分数。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 无（纯前向计算）。  # 注释：纯计算
        最小示例：（注释：最小示例）
          >>> score = self._forward_micro_batch(mb)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker._forward_micro_batch`。（注释：位置）
          - 典型调用路径：`compute_rm_score` -> `_forward_micro_batch`。（注释：链路）
          - 被谁调用：`RewardModelWorker.compute_rm_score`。（注释：调用方）
          - 调用了谁（项目内）：`self.reward_module`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 前向计算。（注释：外部依赖）
        """
        from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
        from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs

        with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary  （注释：见英文说明）
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1  （注释：见英文说明）
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen  （注释：见英文说明）
                output = self.reward_module(
                    input_ids=input_ids_rmpad, attention_mask=None, position_ids=position_ids_rmpad, use_cache=False
                )
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1  （注释：见英文说明）
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outputs_and_unpad(
                        reward_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back  （注释：见英文说明）
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token  （注释：见英文说明）
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        """
        将样本级 reward 扩展到 token 级别，并对齐 response mask。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 输入 batch（含 attention_mask）。  # 注释：输入含义
          - scores (torch.Tensor): 每样本分数（形状 B）。  # 注释：输入含义
        返回：（注释：返回值说明）
          - torch.Tensor：token 级 reward（形状 B x T）。  # 注释：输出含义
        最小示例：（注释：最小示例）
          >>> token_scores = self._expand_to_token_level(data, scores)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker._expand_to_token_level`。（注释：位置）
          - 典型调用路径：`compute_rm_score` -> `_expand_to_token_level`。（注释：链路）
          - 被谁调用：`RewardModelWorker.compute_rm_score`。（注释：调用方）
          - 调用了谁（项目内）：`compute_position_id_with_mask`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 张量广播。（注释：外部依赖）
        """
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward  （注释：见英文说明）
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        response_length = data.batch["responses"].shape[-1]
        if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            position_ids = position_ids[:, 0, :]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part  （注释：见英文说明）
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        """
        根据数据源切换 tokenizer 的 chat_template，并重新构造输入。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 含 prompts 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto：替换后的 model_inputs。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 临时修改 tokenizer/processor 的 chat_template。（注释：副作用）
        最小示例：（注释：最小示例）
          >>> new_data = self._switch_chat_template(data)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker._switch_chat_template`。（注释：位置）
          - 典型调用路径：`compute_rm_score` -> `_switch_chat_template`。（注释：链路）
          - 被谁调用：`RewardModelWorker.compute_rm_score`。（注释：调用方）
          - 调用了谁（项目内）：`self.processor` / `self.tokenizer`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`transformers` tokenizer。（注释：外部依赖）
        """
        src_max_length = data.batch["attention_mask"].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
                raise TypeError(
                    f"raw_prompt must be a list or numpy array, got {type(data.non_tensor_batch['raw_prompt'][i])}"
                )

            # extract raw prompt  （注释：见英文说明）
            chat: list = list(data.non_tensor_batch["raw_prompt"][i])

            # extract response  （注释：见英文说明）
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode  （注释：见英文说明）
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos  （注释：见英文说明）
            response = response.replace(src_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(
                chat, add_generation_prompt=False, tokenize=False
            )
            if self.rank == 0 and i == 0:
                # for debugging purpose  （注释：见英文说明）
                print(f"Switch template. chat: {prompt_with_chat_template}")

            # the maximum length is actually determined by the reward model itself  （注释：见英文说明）
            max_length = self.config.get("max_length", src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get("truncation", "right"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown", role="compute_rm_score")
    def compute_rm_score(self, data: DataProto):
        """
        计算奖励模型分数，并返回 token 级 reward/额外信息。（注释：方法用途）

        参数：（注释：参数说明）
          - data (DataProto): 含 prompts/responses 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto 或 tuple（在 reward_loop 场景可能返回额外数据）。  # 注释：输出含义
        副作用：（注释：副作用说明）
          - 可能进行 chat_template 切换与数据重编码。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 数据缺失必要字段会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> out = rm.compute_rm_score(batch)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::RewardModelWorker.compute_rm_score`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer.fit` -> `compute_reward` -> `compute_rm_score`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`（remote 调用）。  # 注释：调用方
          - 调用了谁（项目内）：`_forward_micro_batch` / `_expand_to_token_level` / `_switch_chat_template`。（注释：项目内依赖）
          - 调用了谁（关键外部依赖）：`torch` 前向计算。（注释：外部依赖）
        """
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        # Support all hardwares  （注释：见英文说明）
        data = data.to(get_device_id())
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)
        else:
            rm_input_ids = data.batch["input_ids"]
            rm_attention_mask = data.batch["attention_mask"]
            rm_position_ids = data.batch["position_ids"]
            rm_inputs = {
                "input_ids": rm_input_ids,
                "attention_mask": rm_attention_mask,
                "position_ids": rm_position_ids,
            }
            rm_data = DataProto.from_dict(rm_inputs)

        # Support all hardwares  （注释：见英文说明）
        rm_data = rm_data.to(get_device_id())

        # perform forward computation  （注释：见英文说明）
        with self.ulysses_sharding_manager:
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL  （注释：见英文说明）
            output = DataProto.from_dict(tensors={"rm_scores": token_level_scores})

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes  （注释：见英文说明）
        # unshard the root FSDP module  （注释：见英文说明）
        if self.world_size > 1 and fsdp_version(self.reward_module) == 1:
            self.reward_module._handle.reshard(True)

        output = output.to("cpu")
        return output


# ================================= Async related workers =================================  （注释：见英文说明）
class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    AsyncActorRolloutRefWorker 类 - 异步版 ActorRolloutRefWorker（支持异步 rollout 生成）

    类用途：
    -------
    本类继承自 `ActorRolloutRefWorker`，在其基础上添加了**异步 rollout 生成**的支持。
    主要用于实现**异步策略更新**（async policy optimization），即：
    - **Rollout 生成**（generate responses）与**训练更新**（update policy）**并行执行**
    - 当 Rollout Worker 在生成新 responses 时，Actor Worker 可同时对旧 batch 进行训练
    - 提高 GPU 利用率（特别是在 Rollout 生成较慢时，如使用 vLLM/SGLang 推理）

    **关键差异**：
    - `ActorRolloutRefWorker`：同步模式（generate 完成后才能 train）
    - `AsyncActorRolloutRefWorker`：异步模式（generate 与 train 可并行）

    典型应用场景：
    -------------
    1. **Fully Async Policy Optimization**（完全异步策略优化）
       - 使用两组 Worker：Rollout Workers（专门生成）+ Actor Workers（专门训练）
       - Rollout Workers 持续生成新数据，Actor Workers 持续训练
       - 通过消息队列（Message Queue）或 Ray 的 async actor 机制协调

    2. **One-Step Off-Policy**（一步离线策略）
       - Rollout Worker 生成 N 个 responses 后，切换为 Ref/Actor 模式参与训练
       - 训练时使用"旧策略"生成的数据（off-policy），但只滞后 1 步（减少 off-policy 偏差）

    输入配置：
    ----------
    继承自 `ActorRolloutRefWorker` 的所有配置（详见 ActorRolloutRefWorker 的 docstring）

    额外支持的异步特性：
    --------------------
    - async wake_up() / async sleep()：异步模式切换（Rollout ↔ Trainer 模式）
    - async generate()：异步生成单个 response（逐个 token 流式生成，用于 SGLang）
    - async chat_completion()：异步 chat 接口（兼容 OpenAI API 格式，用于 SGLang）
    - get_zeromq_address()：获取 vLLM 的 ZeroMQ 地址（用于异步通信）

    输出 DataProto：
    ---------------
    与 `ActorRolloutRefWorker` 相同，但支持异步调用：
    - async generate_sequences(data: DataProto) -> DataProto
        异步生成 responses（内部调用 vLLM/SGLang 的异步 API）

    关键依赖：
    ---------
    - 继承自 `ActorRolloutRefWorker`（包含其所有依赖）
    - Python asyncio（异步 IO 库）
    - vLLM Async Server（可选，用于异步 vLLM 推理）
    - SGLang Async Server（可选，用于异步 SGLang 推理）
    - Ray async actor（可选，用于异步 Ray 调度）

    典型用法：
    ----------
    示例 1：Fully Async Policy Optimization（完全异步策略优化）
    ```python
    # 1. 创建异步 Rollout Workers（专门生成）
    async_rollout_worker_group = RayWorkerGroup(
        resources_config={"rollout": 8},  # 8 个 GPU 专门做 rollout
        worker_cls=AsyncActorRolloutRefWorker,
        worker_cls_args=(config,),
        detached=True,  # detached 模式，可独立运行
    )

    # 2. 创建异步 Actor Workers（专门训练）
    async_actor_worker_group = RayWorkerGroup(
        resources_config={"actor": 8},  # 8 个 GPU 专门做训练
        worker_cls=AsyncActorRolloutRefWorker,
        worker_cls_args=(config,),
    )

    # 3. Rollout Workers 切换为 rollout 模式（异步生成）
    await async_rollout_worker_group.wake_up()  # 异步调用

    # 4. Actor Workers 保持训练模式
    async_actor_worker_group.init_model()  # 初始化 Actor 模型

    # 5. 开始异步训练循环
    async def training_loop():
        # Rollout Workers 持续生成新数据（异步）
        rollout_task = asyncio.create_task(
            async_rollout_worker_group.generate_sequences(prompt_data)
        )

        # Actor Workers 同时训练旧数据（异步）
        train_task = asyncio.create_task(
            async_actor_worker_group.update_policy(old_rollout_data)
        )

        # 等待两个任务完成（并行执行）
        new_rollout_data, train_metrics = await asyncio.gather(
            rollout_task, train_task
        )

        return new_rollout_data, train_metrics

    # 运行异步训练循环
    await training_loop()
    ```

    示例 2：One-Step Off-Policy（一步离线策略）
    ```python
    # 1. 创建 Async Worker（既做 rollout 又做 training）
    worker_group = RayWorkerGroup(
        resources_config={"default": 8},
        worker_cls=AsyncActorRolloutRefWorker,
        worker_cls_args=(config,),
    )

    # 2. 训练循环
    for epoch in range(num_epochs):
        # 2.1 切换为 rollout 模式，生成 N 个 responses
        await worker_group.wake_up()  # 切换为 rollout 模式
        rollout_data = await worker_group.generate_sequences(prompt_data)

        # 2.2 切换为 training 模式，训练（使用刚生成的数据）
        await worker_group.sleep()  # 切换为 training 模式
        metrics = await worker_group.update_policy(rollout_data)

        # 2.3 重复（每次都用"旧策略"生成的数据训练，但只滞后 1 步）
    ```

    示例 3：异步 SGLang 推理（单个 request 流式生成）
    ```python
    # 1. 创建 Async Worker
    worker_group = RayWorkerGroup(
        resources_config={"rollout": 1},
        worker_cls=AsyncActorRolloutRefWorker,
        worker_cls_args=(config,),
    )

    # 2. 切换为 rollout 模式
    await worker_group.wake_up()

    # 3. 异步生成单个 response（逐 token 流式返回）
    prompt_ids = tokenizer.encode("What is the capital of France?")
    sampling_params = {"temperature": 0.7, "max_tokens": 100}
    request_id = "req_001"

    # 调用异步 generate（内部调用 SGLang 的异步 API）
    response_ids = await worker_group.generate(
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
        request_id=request_id,
    )
    response_text = tokenizer.decode(response_ids)
    print(f"Generated: {response_text}")
    ```

    调用路径概览：
    --------------
    Fully Async Policy：
    训练脚本 (recipe/fully_async_policy/fully_async_main.py)
      -> FullyAsyncPPOTrainer.fit()
        -> AsyncActorRolloutRefWorker.generate_sequences() [异步]  # Rollout Workers
        -> AsyncActorRolloutRefWorker.update_policy() [异步]      # Actor Workers

    One-Step Off-Policy：
    训练脚本 (recipe/one_step_off_policy/main_ppo.py)
      -> OneStepOffPolicyTrainer.fit()
        -> AsyncActorRolloutRefWorker.wake_up() [切换为 rollout]
        -> AsyncActorRolloutRefWorker.generate_sequences() [异步生成]
        -> AsyncActorRolloutRefWorker.sleep() [切换为 training]
        -> AsyncActorRolloutRefWorker.update_policy() [异步训练]

    所在位置：
    ----------
    - 路径：`verl/workers/fsdp_workers.py`
    - 类名：`AsyncActorRolloutRefWorker`
    - 继承自：`ActorRolloutRefWorker`

    被谁调用：
    ----------
    - `recipe/fully_async_policy/fully_async_trainer.py`（Fully Async Policy 训练）
    - `recipe/one_step_off_policy/ray_trainer.py`（One-Step Off-Policy 训练）
    - `verl/workers/rollout/vllm_rollout/vllm_async_server.py`（vLLM 异步推理）
    - `verl/workers/rollout/sglang_rollout/async_sglang_server.py`（SGLang 异步推理）

    调用了谁（项目内）：
    ------------------
    - 继承 `ActorRolloutRefWorker` 的所有方法（详见 ActorRolloutRefWorker 的 docstring）
    - `verl/workers/rollout/vllm_rollout/vllm_rollout.py::VLLMRollout.generate()` [异步]
    - `verl/workers/rollout/sglang_rollout/sglang_rollout.py::SGLangRollout.generate()` [异步]
    - `verl/workers/rollout/sglang_rollout/sglang_rollout.py::SGLangRollout.chat_completion()` [异步]

    调用了谁（关键外部依赖）：
    ----------------------
    - Python `asyncio`（异步 IO 协程）
    - vLLM AsyncLLMEngine（异步推理引擎）
    - SGLang AsyncServer（异步推理服务器）
    - Ray async actor（异步 Ray 调度）

    注意事项：
    ----------
    1. **异步 vs 同步的选择**
       - **同步 (ActorRolloutRefWorker)**：适合 Rollout 生成很快的场景（如小模型、短序列）
       - **异步 (AsyncActorRolloutRefWorker)**：适合 Rollout 生成较慢的场景（如大模型、长序列）
       - 异步模式可提高 GPU 利用率（Rollout 与 Training 并行），但增加代码复杂度

    2. **wake_up() / sleep() 的作用**
       - `wake_up()`：异步切换为 **rollout 模式**（释放 Actor 模型的显存，加载 Rollout 模型）
       - `sleep()`：异步切换为 **training 模式**（释放 Rollout 模型的显存，加载 Actor 模型）
       - 用于 One-Step Off-Policy 等需要在同一 Worker 上切换角色的场景

    3. **Fully Async Policy 的 off-policy 问题**
       - 在完全异步模式下，训练时使用的数据是"旧策略"生成的（off-policy）
       - 需要通过 Importance Sampling 或 PPO Clip 来纠正分布偏差
       - 通常滞后不超过 1-2 步（否则 off-policy 偏差过大，训练不稳定）

    4. **One-Step Off-Policy 的优势**
       - 只滞后 1 步（on-policy 近似），减少 off-policy 偏差
       - 同时保留部分异步优势（切换模式时可并行其他操作）

    5. **vLLM vs SGLang 的异步支持**
       - **vLLM**：支持 AsyncLLMEngine（异步批量生成）
       - **SGLang**：支持 AsyncServer（异步流式生成，单个 request 级别）
       - 本类同时支持两种后端的异步接口

    6. **Ray async actor 的使用**
       - Ray 的 async actor 可以并行处理多个异步方法调用
       - 配合 `asyncio.gather()` 实现 Rollout 与 Training 的真正并行

    7. **异步模式的性能权衡**
       - **优势**：提高 GPU 利用率（Rollout 与 Training 并行）
       - **劣势**：增加代码复杂度（需要管理异步状态、消息队列等）
       - **建议**：仅在 Rollout 成为瓶颈时使用（如大模型、长序列、复杂推理）
    """
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================  （注释：见英文说明）

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        """
        获取异步 rollout 服务的 ZeroMQ 地址。（注释：方法用途）

        返回：（注释：返回值说明）
          - str：ZeroMQ 地址。（注释：输出含义）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/fsdp_workers.py::AsyncActorRolloutRefWorker.get_zeromq_address`。（注释：位置）
          - 典型调用路径：`RayPPOTrainer` -> `get_zeromq_address`。（注释：链路）
          - 被谁调用：`verl/trainer/ppo/ray_trainer.py`（异步 rollout 场景）。  # 注释：调用方
          - 调用了谁（项目内）：`self.rollout.get_zeromq_address`。（注释：项目内依赖）
        """
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================  （注释：见英文说明）

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
