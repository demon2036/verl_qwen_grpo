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
模块用途：FSDP 后端的轻量 SFT Trainer，负责模型构建、分布式训练、验证与保存。（注释：模块功能概述）
输入：Hydra 配置（sft_trainer.yaml）、训练/验证 parquet、模型权重与 tokenizer。（注释：输入说明）
输出：checkpoint（本地/HDFS）、训练/验证指标日志。（注释：输出说明）
关键依赖：torch、hydra、transformers、peft、torch.distributed.fsdp、verl.utils.*。（注释：依赖说明）
典型用法：
  torchrun -m verl.trainer.fsdp_sft_trainer data.train_files=... trainer.total_epochs=1。（注释：最小运行示例）
调用路径概览：run_qwen_05_peft.sh -> torchrun -m verl.trainer.fsdp_sft_trainer -> main -> run_sft -> FSDPSFTTrainer.fit。（注释：调用路径）
"""  # 注释：模块级 docstring 结束

import os  # 标准库：环境变量与路径（注释：os 用途）

os.environ["NCCL_DEBUG"] = "WARN"  # 设置 NCCL 日志级别（注释：分布式日志控制）
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 允许 tokenizer 并行（注释：性能优化）

import logging  # 标准库：日志（注释：logging）
import re  # 标准库：正则（注释：extract_step 使用）
import time  # 标准库：计时（注释：训练耗时统计）
from contextlib import nullcontext  # 标准库：空上下文管理器（注释：条件上下文）

import hydra  # 第三方：Hydra 配置（注释：入口装饰器）
import torch  # 第三方：PyTorch（注释：张量与训练）
import torch.distributed  # 第三方：分布式通信（注释：all_reduce 等）
from omegaconf import DictConfig, OmegaConf  # 第三方：配置对象（注释：OmegaConf）
from peft import LoraConfig, TaskType, get_peft_model  # 第三方：PEFT LoRA（注释：LoRA 配置）
from tensordict import TensorDict  # 第三方：TensorDict（注释：训练 batch 容器）
from torch import nn  # 第三方：神经网络模块（注释：loss）
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh  # 第三方：设备网格（注释：FSDP Mesh）
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy  # 第三方：FSDP 组件（注释：FSDP 配置）
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 第三方：FSDP 类（注释：FSDP 包装）
from torch.utils.data import Dataset, DistributedSampler  # 第三方：Dataset 与采样器（注释：数据分片）
from torchdata.stateful_dataloader import StatefulDataLoader  # 第三方：可恢复 DataLoader（注释：断点续训）
from tqdm import tqdm  # 第三方：进度条（注释：训练展示）
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel  # 第三方：HF 模型（注释：模型加载）

import verl.utils.hdfs_io as hdfs_io  # 项目内：HDFS IO（注释：远端存储）
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input  # 项目内：变长序列工具（注释：remove padding）
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename  # 项目内：checkpoint 工具（注释：最新路径）
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager  # 项目内：FSDP checkpoint 管理（注释：保存/加载）
from verl.utils.dataset import SFTDataset  # 项目内：单轮 SFT 数据集（注释：数据集类）
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset  # 项目内：多轮 SFT 数据集（注释：多轮数据）
from verl.utils.device import (  # 项目内：设备工具（注释：设备检测）
    auto_set_ascend_device_name,
    get_device_id,
    get_device_name,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group  # 项目内：分布式初始化/销毁（注释：进程组管理）
from verl.utils.fs import copy_to_local  # 项目内：文件拷贝（注释：本地缓存）
from verl.utils.fsdp_utils import (  # 项目内：FSDP 辅助函数（注释：FSDP2 支持）
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank  # 项目内：rank 日志（注释：分布式输出）
from verl.utils.profiler import log_gpu_memory_usage  # 项目内：显存记录（注释：profiling）
from verl.utils.py_functional import convert_to_regular_types  # 项目内：类型转换（注释：PEFT 需要）
from verl.utils.torch_dtypes import PrecisionType  # 项目内：dtype 枚举（注释：精度转换）
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup  # 项目内：LR 调度（注释：scheduler）
from verl.utils.tracking import Tracking  # 项目内：日志追踪（注释：指标记录）
from verl.utils.ulysses import (  # 项目内：Ulysses 序列并行（注释：SP 工具）
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config.optimizer import build_optimizer  # 项目内：构建优化器（注释：优化器工厂）
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager  # 项目内：SP 分片管理（注释：Ulysses 管理器）

logger = logging.getLogger(__file__)  # 获取 logger（注释：日志实例）
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))  # 读取环境变量设置日志级别（注释：日志级别）


def extract_step(path):
    """
    函数用途：从 checkpoint 路径中解析 global_step。（注释：函数目标）
    参数：
      path (str): checkpoint 路径，形如 ".../global_step_123"。（注释：输入路径）
    返回：
      step (int|None): 成功则返回步数，否则 None。（注释：返回说明）
    副作用：无。（注释：无副作用）
    异常/边界：路径不包含 global_step_ 时返回 None。（注释：边界情况）
    最小示例：
      输入："/ckpt/global_step_42"。（注释：示例输入）
      中间：正则匹配到 "42"。（注释：中间结果）
      输出：42。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/fsdp_sft_trainer.py::extract_step`。（注释：位置）
      典型调用路径：`FSDPSFTTrainer.load_checkpoint` -> `extract_step`。（注释：调用链）
      被谁调用：`load_checkpoint` / `_find_latest_checkpoint`。（注释：外部调用）
      调用了谁（项目内）：无。（注释：内部依赖）
      调用了谁（外部依赖）：`re.search`。（注释：第三方依赖）
    """  # 注释：extract_step docstring 结束
    match = re.search(r"global_step_(\d+)", path)  # 正则提取数字（注释：匹配 step）
    if match:
        return int(match.group(1))  # 转为 int 并返回（注释：解析成功）
    return None  # 无匹配返回 None（注释：解析失败）


class FSDPSFTTrainer:
    """
    类用途：使用 FSDP/FSDP2 训练单轮 SFT 模型，支持 LoRA 与序列并行。（注释：类职责）
    参数：
      config (DictConfig): Hydra 配置对象。（注释：配置输入）
      device_mesh (DeviceMesh): FSDP 设备网格。（注释：设备网格）
      ulysses_device_mesh (DeviceMesh): Ulysses SP 设备网格。（注释：序列并行）
      tokenizer: tokenizer 实例。（注释：分词器）
      train_dataset/val_dataset (Dataset): 训练/验证数据集。（注释：数据集）
    副作用：构建模型、初始化优化器、保存/加载 checkpoint。（注释：副作用）
    调用路径依赖：
      所在位置：`verl/trainer/fsdp_sft_trainer.py::FSDPSFTTrainer`。（注释：位置）
      典型调用路径：`run_sft` -> `FSDPSFTTrainer.__init__` -> `fit`。（注释：调用链）
      被谁调用：`run_sft`。（注释：外部调用）
      调用了谁（项目内）：`FSDPCheckpointManager`、`build_optimizer` 等。（注释：内部依赖）
      调用了谁（外部依赖）：`torch.distributed.fsdp`、`transformers`。（注释：第三方依赖）
    """  # 注释：类 docstring 结束

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        """
        函数用途：初始化 Trainer，构建 dataloader、模型/优化器与 checkpoint 管理器。（注释：函数目标）
        参数：
          config (DictConfig): Hydra 配置。（注释：配置参数）
          device_mesh (DeviceMesh): FSDP Mesh。（注释：FSDP 网格）
          ulysses_device_mesh (DeviceMesh): SP Mesh。（注释：序列并行网格）
          tokenizer: tokenizer 实例。（注释：分词器）
          train_dataset/val_dataset (Dataset): 数据集。（注释：数据集）
        返回：无。（注释：返回说明）
        副作用：可能加载 checkpoint 并打印配置。（注释：副作用）
        异常/边界：chat_template 非空时抛 ValueError。（注释：边界情况）
        最小示例：
          输入：train_batch_size=256，dp_size=8。（注释：示例输入）
          中间：_normalize_config_bsz 将每卡 batch 设为 32。（注释：中间结果）
          输出：Trainer 实例可调用 fit。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::FSDPSFTTrainer.__init__`。（注释：位置）
          典型调用路径：`run_sft` -> `FSDPSFTTrainer(...)`。（注释：调用链）
          被谁调用：`run_sft`。（注释：外部调用）
          调用了谁（项目内）：`_build_dataloader`、`_build_model_optimizer`。（注释：内部依赖）
          调用了谁（外部依赖）：无。（注释：第三方依赖）
        """  # 注释：__init__ docstring 结束
        self.config = config  # 保存配置对象（注释：配置持久化）
        self.device_mesh = device_mesh  # 保存 FSDP 设备网格（注释：设备网格）
        self.ulysses_device_mesh = ulysses_device_mesh  # 保存 SP 设备网格（注释：SP 网格）
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)  # 创建 Ulysses 分片管理器（注释：SP 管理）
        self.tokenizer = tokenizer  # 保存 tokenizer（注释：分词器）
        if self.config.data.chat_template is not None:  # 当前不支持 config chat_template（注释：功能限制）
            raise ValueError("Apply Chat template from config is not supported yet.")  # 抛出异常（注释：非法配置）

        # normalize dp size（注释：规范 batch size）
        self._normalize_config_bsz()  # 将全局 batch 分摊到 DP（注释：batch 归一化）

        # Set sequence parallel size（注释：设置序列并行大小）
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)  # 读取 SP 大小（注释：默认 1）
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)  # 读取 remove padding 开关（注释：功能开关）
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")  # 打印 SP 大小（注释：日志输出）
            print(f"Using remove padding: {self.use_remove_padding}")  # 打印 remove padding（注释：日志输出）

        self._build_dataloader(train_dataset, val_dataset)  # 构建 dataloader（注释：数据加载）

        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0  # 判断是否启用 LoRA（注释：LoRA 开关）

        # Initialize resume-related variables（注释：初始化断点变量）
        self.resume_global_step = 0  # 断点步数（注释：默认 0）

        # build model（注释：构建模型与优化器）
        self._build_model_optimizer()  # 构建模型/优化器/LR 调度（注释：核心构建）

        # Initialize checkpoint manager（注释：初始化 checkpoint 管理）
        self._init_checkpoint_manager()  # 初始化 checkpoint 管理器（注释：保存/加载）

        self.load_checkpoint()  # 尝试加载 checkpoint（注释：断点恢复）

        if self.device_mesh.get_rank() == 0:
            print(self.config)  # 打印配置（注释：调试输出）

        self.device_name = self.config.trainer.device  # 保存设备名称（注释：设备标识）

    def _normalize_config_bsz(self):
        """
        函数用途：按 DP 大小归一化全局 batch，确保可整除。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：无（更新 config.data.train_batch_size）。（注释：返回说明）
        副作用：修改配置中的 train_batch_size。（注释：副作用）
        异常/边界：全局 batch 无法被 DP 整除会触发断言。（注释：边界情况）
        最小示例：
          输入：train_batch_size=256，dp_size=8。（注释：示例输入）
          中间：256 % 8 == 0。（注释：中间检查）
          输出：train_batch_size 变为 32（每 DP）。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_normalize_config_bsz`。（注释：位置）
          典型调用路径：`FSDPSFTTrainer.__init__` -> `_normalize_config_bsz`。（注释：调用链）
          被谁调用：`FSDPSFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：无。（注释：内部依赖）
          调用了谁（外部依赖）：`assert`（内置）。（注释：第三方依赖）
        """  # 注释：_normalize_config_bsz docstring 结束
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)  # 计算 DP 大小（注释：DP 规模）
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")  # 打印 DP 大小（注释：日志输出）

        assert self.config.data.train_batch_size % dp_size == 0, (  # 保证可整除（注释：断言检查）
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size  # 归一化到每 DP（注释：batch 缩放）

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0  # 确保 micro-batch 整除（注释：断言检查）

    def _build_dataloader(self, train_dataset, val_dataset):
        """
        函数用途：构建训练/验证 DataLoader 与 DistributedSampler。（注释：函数目标）
        参数：
          train_dataset/val_dataset (Dataset): 数据集实例。（注释：参数说明）
        返回：无（设置 train_dataloader/val_dataloader）。（注释：返回说明）
        副作用：创建 DistributedSampler 会改变样本顺序。（注释：副作用）
        异常/边界：SP 模式下使用不同 rank/size。（注释：边界情况）
        最小示例：
          输入：world_size=8，sp_size=2。（注释：示例输入）
          中间：dp_size=4，rank=local dp rank。（注释：中间结果）
          输出：每个 DP rank 处理不同数据。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_build_dataloader`。（注释：位置）
          典型调用路径：`FSDPSFTTrainer.__init__` -> `_build_dataloader`。（注释：调用链）
          被谁调用：`FSDPSFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`get_device_name`。（注释：内部依赖）
          调用了谁（外部依赖）：`DistributedSampler`、`StatefulDataLoader`。（注释：第三方依赖）
        """  # 注释：_build_dataloader docstring 结束
        # build dataset（注释：保留原注释）
        config = self.config  # 读取配置（注释：局部引用）
        self.train_dataset, self.val_dataset = train_dataset, val_dataset  # 保存数据集引用（注释：成员赋值）

        # build dataloader（注释：构建 DataLoader）
        # Use data parallel rank and size instead of global rank and world size（注释：DP 范围分片）

        # If doing SP, we need to use the local rank and size（注释：SP 模式处理）
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")  # SP 下取本地 DP rank（注释：SP rank）
            world_size = self.ulysses_device_mesh.size(0)  # SP 下 DP 总数（注释：DP size）
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")  # 打印 SP rank/size（注释：日志输出）
                print("Each SP rank gets different data, but the same data WITHIN the same rank")  # 解释数据分发（注释：数据分片说明）
        else:
            rank = self.device_mesh.get_rank()  # 非 SP 情况用全局 rank（注释：rank 选择）
            world_size = self.device_mesh.size()  # 全局 world size（注释：world size）
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")  # 打印 FSDP rank/size（注释：日志输出）

        # Set pin_memory_device when pin_memory is enabled.（注释：pin_memory 设备配置）
        device_name = get_device_name()  # 读取设备名（注释：设备信息）

        self.train_sampler = DistributedSampler(  # 构建训练采样器（注释：分布式采样）
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(  # 构建训练 DataLoader（注释：支持恢复）
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        self.val_sampler = DistributedSampler(  # 构建验证采样器（注释：分布式采样）
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(  # 构建验证 DataLoader（注释：支持恢复）
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_model_optimizer(self):
        """
        函数用途：构建 HF 模型、应用 LoRA/patch、包裹 FSDP，并创建优化器与调度器。（注释：函数目标）
        参数：无（读取 self.config）。（注释：参数说明）
        返回：无（设置 self.fsdp_model/self.optimizer/self.lr_scheduler）。（注释：返回说明）
        副作用：分配显存、修改模型权重、可能加载 LoRA 适配器。（注释：副作用）
        异常/边界：fsdp_strategy 非法时抛 NotImplementedError。（注释：边界情况）
        最小示例：
          输入：fsdp_strategy="fsdp2"，lora_rank=32。（注释：示例输入）
          中间：apply_fsdp2 + get_peft_model。（注释：中间步骤）
          输出：self.fsdp_model 可用于训练。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_build_model_optimizer`。（注释：位置）
          典型调用路径：`FSDPSFTTrainer.__init__` -> `_build_model_optimizer`。（注释：调用链）
          被谁调用：`FSDPSFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`build_optimizer`、`get_fsdp_wrap_policy`。（注释：内部依赖）
          调用了谁（外部依赖）：`transformers.AutoModelForCausalLM`、`FSDP`。（注释：第三方依赖）
        """  # 注释：_build_model_optimizer docstring 结束
        # TODO (zhangchi.usc1992):（注释：原 TODO 保留）
        # 1. support pretrain from random weights（注释：TODO：支持随机初始化）
        # 2. support init directly from sharded weights（注释：TODO：支持分片权重初始化）
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)  # 将权重拷贝到本地（注释：本地缓存）

        if self.config.model.get("external_lib", None) is not None:  # 若需要外部库（注释：外部依赖）
            # This is used to import external_lib into the huggingface systems（注释：注入 HF 环境）
            import importlib  # 标准库：动态导入（注释：importlib）

            importlib.import_module(self.config.model.external_lib)  # 动态导入外部库（注释：外部依赖导入）

        log_gpu_memory_usage("Before model allocation", logger=logger)  # 记录模型分配前显存（注释：显存监控）

        trust_remote_code = self.config.model.trust_remote_code  # 是否信任远程代码（注释：安全开关）
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")  # 读取 dtype（注释：精度配置）
        torch_dtype = PrecisionType.to_dtype(torch_dtype)  # 转换为 torch dtype（注释：dtype 转换）
        # load config first（注释：先加载配置）
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)  # 加载模型配置（注释：HF AutoConfig）
        self.model_config = config  # 保存模型配置（注释：配置缓存）
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(  # 扩展最大位置（注释：长度上限）
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"  # SP 需 remove padding（注释：约束）

        # This may be very large（注释：权重可能很大）
        init_context = get_init_weight_context_manager(  # 选择权重初始化上下文（注释：初始化策略）
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():  # 在初始化上下文中构建模型（注释：延迟分配）
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # 加载 HF 模型（注释：模型加载）
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:  # 若开启 remove padding/SP（注释：条件分支）
                from verl.models.transformers.monkey_patch import apply_monkey_patch  # 延迟导入 patch（注释：monkey patch）

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)  # 应用 patch（注释：变长注意力）

            # Apply Liger kernel if use_liger is enabled（注释：可选 Liger 加速）
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance  # 导入 Liger patch（注释：Liger kernel）

                _apply_liger_kernel_to_instance(model=self.model)  # 应用 Liger patch（注释：性能优化）

            if self.lora:  # 若启用 LoRA（注释：LoRA 分支）
                self.model.enable_input_require_grads()  # 允许输入梯度（注释：LoRA 需要）

                lora_adapter_path = self.config.model.get("lora_adapter_path")  # LoRA 适配器路径（注释：可选预训练）
                if lora_adapter_path is not None:
                    from peft import PeftModel  # 延迟导入 PeftModel（注释：PEFT）

                    print(f"Loading pre-trained LoRA adapter for sft from: {lora_adapter_path}")  # 打印加载信息（注释：日志）

                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.use_shm)  # 拷贝适配器到本地（注释：本地缓存）

                    self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)  # 加载 LoRA 适配器（注释：PEFT 加载）
                    peft_config = self.model.peft_config["default"]  # 读取默认配置（注释：PEFT 配置）
                    # Ensure task_type is TaskType enum, not string（注释：兼容性修正）
                    if isinstance(peft_config.task_type, str):
                        peft_config.task_type = TaskType.CAUSAL_LM  # 修正 task_type（注释：枚举化）
                else:
                    # Convert config to regular Python types before creating PEFT model（注释：转换配置类型）
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))  # 构建 LoRA 模型（注释：PEFT 初始化）
                self.model = self.model.to(torch_dtype)  # 将模型转到指定 dtype（注释：精度转换）

        if self.config.model.enable_gradient_checkpointing:  # 若启用梯度检查点（注释：显存优化）
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # 开启 checkpointing（注释：reentrant 设置）

        log_gpu_memory_usage("After model allocation", logger=logger)  # 记录模型分配后显存（注释：显存监控）

        mixed_precision = MixedPrecision(  # 设置 FSDP 混合精度策略（注释：混合精度）
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(  # 获取自动 wrap 策略（注释：FSDP wrap）
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )

        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)  # 打印 wrap 策略（注释：调试输出）

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None  # 未启用 CPU offload（注释：禁用 offload）
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)  # 启用 offload（注释：参数 offload）

        fsdp_strategy = self.config.model.strategy  # 读取 FSDP 策略（注释：fsdp/fsdp2）
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(  # 使用经典 FSDP 包装（注释：FSDP1）
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"  # FSDP2 版本要求（注释：断言）
            mp_policy = MixedPrecisionPolicy(  # 设置 FSDP2 精度策略（注释：FSDP2 精度）
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {  # FSDP2 配置参数（注释：FSDP2 参数）
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()  # 保存完整 state_dict（注释：全量权重）
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)  # 应用 FSDP2 包装（注释：fully_shard）
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)  # 加载权重（注释：FSDP2 加载）
            self.fsdp_model = self.model  # FSDP2 下直接使用模型（注释：模型引用）
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")  # 不支持的策略（注释：异常）

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)  # 记录 FSDP 包装后显存（注释：显存监控）

        self.optimizer = build_optimizer(self.fsdp_model.parameters(), self.config.optim)  # 构建优化器（注释：优化器初始化）

        log_gpu_memory_usage("After initialize optimizer", logger=logger)  # 记录优化器初始化后显存（注释：显存监控）

        self.steps_per_epoch = len(self.train_dataloader)  # 计算每 epoch 步数（注释：步数统计）
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs  # 计算总步数（注释：总步数）

        if self.device_mesh.get_rank() == 0:
            print(  # 打印训练步数信息（注释：日志输出）
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)  # 计算 warmup 步数（注释：warmup）

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(  # 使用 cosine 调度（注释：LR 调度）
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(  # 使用 WSD 调度（注释：LR 调度）
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")  # 未知调度器（注释：异常）

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """
        函数用途：计算 loss 并可选反向传播，支持 remove padding 与序列并行。（注释：函数目标）
        参数：
          batch (TensorDict): 一个 micro-batch 数据。（注释：输入 batch）
          do_backward (bool): 是否执行 backward。（注释：反向传播开关）
          n_micro_batches (int): micro-batch 数，用于 loss 归一化。（注释：归一化因子）
        返回：
          loss (Tensor): 当前 micro-batch 的归一化 loss。（注释：返回说明）
        副作用：当 do_backward=True 时会累积梯度。（注释：副作用）
        异常/边界：SP 模式要求 remove_padding=True。（注释：边界情况）
        最小示例：
          输入：input_ids=[[1,2,3,4]]，attention_mask=[[1,1,1,0]]。（注释：示例输入）
          中间：shift_labels=[2,3,4]，loss_mask 过滤 padding。（注释：中间结果）
          输出：loss 为标量并可 backward。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_compute_loss_and_backward`。（注释：位置）
          典型调用路径：`training_step` -> `_compute_loss_and_backward`。（注释：调用链）
          被谁调用：`training_step`、`validation_step`。（注释：外部调用）
          调用了谁（项目内）：`unpad_input`、`ulysses_pad_and_slice_inputs`。（注释：内部依赖）
          调用了谁（外部依赖）：`torch.autocast`、`nn.CrossEntropyLoss`。（注释：第三方依赖）
        """  # 注释：_compute_loss_and_backward docstring 结束
        """Compute loss with optional sequence parallelism and remove padding features"""  # 保留原始英文注释（注释：兼容原注释）
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1  # 判断是否启用 SP（注释：SP 开关）

        # Move inputs to GPU and prepare loss mask（注释：准备输入张量）
        input_ids = batch["input_ids"].to(self.device_name)  # input_ids（注释：设备迁移）
        attention_mask = batch["attention_mask"].to(self.device_name)  # attention_mask（注释：设备迁移）
        position_ids = batch["position_ids"].to(self.device_name)  # position_ids（注释：设备迁移）
        loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to(self.device_name)  # loss_mask 展平（注释：mask 处理）
        loss_fct = nn.CrossEntropyLoss(reduction="none")  # 逐 token loss（注释：loss 函数）

        # Context manager for sequence parallel if needed（注释：选择上下文管理器）
        context = self.sharding_manager if use_sp else nullcontext()  # SP 使用 sharding_manager（注释：上下文选择）
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):  # autocast 混合精度（注释：自动混合精度）
            if not use_sp:
                # Standard forward pass without sequence parallel（注释：非 SP 路径）
                labels = input_ids[:, 1:].contiguous()  # shift labels（注释：标签右移）
                output = self.fsdp_model(  # 前向计算（注释：模型前向）
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits  # 取出 logits（注释：模型输出）

                shift_logits = logits[..., :-1, :].contiguous()  # 右移 logits（注释：对齐标签）
                shift_labels = labels.contiguous()  # labels contiguous（注释：内存连续）
                # Flatten the tokens（注释：展平 token）
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)  # 展平 logits（注释：二维形状）
                shift_labels = shift_labels.view(-1)  # 展平 labels（注释：一维形状）
                # Enable model parallelism（注释：对齐设备）
                shift_labels = shift_labels.to(shift_logits.device)  # labels 迁移（注释：设备对齐）
                loss = loss_fct(shift_logits, shift_labels)  # 计算 token loss（注释：逐 token loss）
                loss = loss * loss_mask.to(loss.device)  # 应用 loss_mask（注释：mask 过滤）
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks（注释：SP 前提假设）
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence（注释：SP 假设说明）
                # 1. All SP ranks will receive the *SAME* batch（注释：同组相同数据）
                # 2. Different SP groups will receive *DIFFERENT* batches（注释：不同组不同数据）
                # This is implemented by the DistributedSampler（注释：由采样器保证）

                batch_size, seqlen = input_ids.shape  # 取出 batch 与序列长度（注释：形状获取）
                # Remove padding（注释：移除 padding）
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)（注释：变长压缩）
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)（注释：转置形状）

                # Unpad position_ids to align rotary（注释：position_ids 对齐）
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)  # 转置以匹配 input_ids（注释：形状对齐）

                # Pad and slice inputs for sequence parallelism（注释：为 SP pad & slice）
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss（注释：准备 loss 的 labels）
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)（注释：右移标签）
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)（注释：去掉维度）

                # Forward pass（注释：前向计算）
                output = self.fsdp_model(  # SP 前向（注释：模型前向）
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen（注释：变长不需要 mask）
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate（注释：本地 loss 后聚合）
                logits_rmpad = output.logits.squeeze(0)  # 移除 batch 维（注释：logits 形状）
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)  # 标签迁移（注释：设备对齐）
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)  # 逐 token loss（注释：loss 计算）
                # Gather and unpad for sequence parallelism（注释：SP 聚合与反 pad）
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)  # 聚合 loss（注释：gather）

                # This is the loss collected from all ulysses ranks（注释：收集全量 loss）
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss（注释：去除最后 token）
                full_loss = full_loss.reshape(-1)  # 展平（注释：形状统一）
                loss_mask = loss_mask.to(full_loss.device)  # mask 迁移（注释：设备对齐）
                loss = full_loss * loss_mask  # 应用 mask（注释：过滤 padding）

            valid_token_this_rank = torch.sum(loss_mask)  # 统计有效 token（注释：token 计数）

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)  # 跨 DP 聚合 token 数（注释：all_reduce）
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()  # 计算 DP 大小（注释：DP 规模）
            else:
                dp_size = 1  # 不平衡时不缩放（注释：默认 1）

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size  # 平均 loss 并缩放（注释：loss 归一化）

            loss = loss / n_micro_batches  # normalize loss（注释：micro-batch 归一化）

            if do_backward:
                loss.backward()  # 反向传播（注释：梯度累积）
            return loss  # 返回 loss（注释：函数返回）

    def training_step(self, batch: TensorDict):
        """
        函数用途：执行单个训练 step（含 micro-batch 切分与优化器更新）。（注释：函数目标）
        参数：
          batch (TensorDict): 一个全局 batch。（注释：输入 batch）
        返回：
          metrics (dict): 训练指标（loss/lr/time）。（注释：返回说明）
        副作用：更新模型参数、lr_scheduler 前进。（注释：副作用）
        异常/边界：grad_norm 非有限时跳过更新。（注释：边界情况）
        最小示例：
          输入：micro_batch_size_per_gpu=4，batch size=8。（注释：示例输入）
          中间：切成 2 个 micro-batch。（注释：中间结果）
          输出：返回 train/loss、train/lr、train/time。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::training_step`。（注释：位置）
          典型调用路径：`fit` -> `training_step`。（注释：调用链）
          被谁调用：`fit`。（注释：外部调用）
          调用了谁（项目内）：`_compute_loss_and_backward`。（注释：内部依赖）
          调用了谁（外部依赖）：`torch.distributed.all_reduce`。（注释：第三方依赖）
        """  # 注释：training_step docstring 结束
        start_time = time.time()  # 记录开始时间（注释：计时开始）

        self.fsdp_model.train()  # 进入训练模式（注释：train 模式）

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)  # 记录清零前显存（注释：显存监控）

        self.optimizer.zero_grad()  # 清空梯度（注释：梯度清零）

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)  # 记录清零后显存（注释：显存监控）

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)  # 切分 micro-batch（注释：micro 切分）
        n_micro_batches = len(micro_batches)  # micro-batch 数量（注释：计数）
        step_loss = 0  # 累计 loss（注释：loss 累加）
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, n_micro_batches=n_micro_batches)  # 计算并 backward（注释：loss 计算）
            step_loss += loss.item()  # 累计 loss（注释：loss 累计）

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)  # FSDP1 梯度裁剪（注释：grad_norm）
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)  # FSDP2 裁剪（注释：grad_norm）
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")  # 不支持策略（注释：异常）

        log_gpu_memory_usage("Before optimizer step", logger=logger)  # 记录更新前显存（注释：显存监控）

        # if grad_norm is not finite, skip the update（注释：梯度异常处理）
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")  # 打印警告（注释：异常日志）
            self.optimizer.zero_grad()  # 清空梯度（注释：避免坏梯度）
        else:
            self.optimizer.step()  # 更新参数（注释：优化器 step）

        log_gpu_memory_usage("After optimizer step", logger=logger)  # 记录更新后显存（注释：显存监控）

        self.lr_scheduler.step()  # 更新学习率（注释：scheduler step）

        # reduce loss across dp ranks（注释：跨 DP 聚合 loss）
        lr = self.lr_scheduler.get_last_lr()[0]  # 读取当前 lr（注释：学习率）

        log_gpu_memory_usage("After offload weights", logger=logger)  # 记录 offload 后显存（注释：显存监控）

        step_loss = torch.tensor(step_loss).to(self.device_name)  # 转为张量（注释：便于 all_reduce）

        # compute time spent per step（注释：计算 step 时间）
        end_time = time.time()  # 记录结束时间（注释：计时结束）
        spend_time_per_step = end_time - start_time  # 计算耗时（注释：耗时统计）

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)  # CUDA 下求平均（注释：all_reduce）
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)  # NPU 下求和（注释：all_reduce）
            step_loss /= self.device_mesh.size(0)  # 手动除以 world size（注释：平均）
        return {  # 返回指标（注释：metrics）
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
        }

    def validation_step(self, batch: TensorDict):
        """
        函数用途：执行单个验证 step。（注释：函数目标）
        参数：
          batch (TensorDict): 验证 batch。（注释：输入 batch）
        返回：
          loss (Tensor): 验证 loss（已聚合）。（注释：返回说明）
        副作用：进行分布式 all_reduce。（注释：副作用）
        异常/边界：无。（注释：边界情况）
        最小示例：
          输入：验证 batch（注释：示例输入）
          输出：标量 loss（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::validation_step`。（注释：位置）
          典型调用路径：`fit` -> `validation_step`。（注释：调用链）
          被谁调用：`fit`。（注释：外部调用）
          调用了谁（项目内）：`_compute_loss_and_backward`。（注释：内部依赖）
          调用了谁（外部依赖）：`torch.no_grad`、`torch.distributed.all_reduce`。（注释：第三方依赖）
        """  # 注释：validation_step docstring 结束
        self.fsdp_model.eval()  # 进入 eval 模式（注释：评估模式）
        with torch.no_grad():  # 关闭梯度（注释：验证不反传）
            loss = self._compute_loss_and_backward(batch, do_backward=False)  # 计算 loss（注释：验证 loss）
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)  # CUDA 平均（注释：all_reduce）
            elif is_npu_available:
                torch.distributed.all_reduce(loss)  # NPU 求和（注释：all_reduce）
                loss /= self.device_mesh.size(0)  # 手动平均（注释：除以 world size）
        return loss  # 返回验证 loss（注释：函数返回）

    def save_checkpoint(self, step):
        """
        函数用途：使用 FSDPCheckpointManager 保存 checkpoint，并保存 dataloader 状态。（注释：函数目标）
        参数：
          step (int): 当前 global_step。（注释：参数说明）
        返回：无。（注释：返回说明）
        副作用：写入磁盘/HDFS、更新 tracker 文件。（注释：副作用）
        异常/边界：目录不存在时会创建。（注释：边界情况）
        最小示例：
          输入：step=100。（注释：示例输入）
          中间：保存到 checkpoints/.../global_step_100。（注释：中间路径）
          输出：生成 data.pt 与 tracker 文件。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::save_checkpoint`。（注释：位置）
          典型调用路径：`fit` -> `save_checkpoint`。（注释：调用链）
          被谁调用：`fit`。（注释：外部调用）
          调用了谁（项目内）：`FSDPCheckpointManager.save_checkpoint`。（注释：内部依赖）
          调用了谁（外部依赖）：`torch.save`。（注释：第三方依赖）
        """  # 注释：save_checkpoint docstring 结束
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""  # 保留原英文说明（注释：兼容原注释）
        from verl.utils.fs import local_mkdir_safe  # 延迟导入目录创建（注释：路径工具）

        # Determine checkpoint path（注释：确定保存路径）
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")  # 拼接路径（注释：保存目录）

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")  # 打印保存路径（注释：日志）

        # Get max checkpoints to keep（注释：最大保留数量）
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)  # 读取配置（注释：保留上限）

        # Use checkpoint manager to save（注释：调用 checkpoint manager）
        self.checkpoint_manager.save_checkpoint(  # 保存模型/优化器等（注释：保存调用）
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state（注释：保存 dataloader 状态）
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)  # 确保目录存在（注释：目录创建）
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")  # data.pt 路径（注释：状态文件）

            # Use StatefulDataLoader's built-in state dict functionality（注释：StatefulDataLoader 状态）
            dataloader_state_dict = self.train_dataloader.state_dict()  # 获取 dataloader 状态（注释：状态字典）
            torch.save(dataloader_state_dict, dataloader_local_path)  # 保存状态（注释：写入文件）
            print(f"Saved dataloader state to: {dataloader_local_path}")  # 打印保存信息（注释：日志）

            # Update latest checkpoint tracker (atomic write)（注释：更新 tracker 文件）
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)  # tracker 文件路径（注释：路径）
            temp_tracker_file = tracker_file + ".tmp"  # 临时文件路径（注释：原子写）
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))  # 写入 step（注释：写入内容）
            os.rename(temp_tracker_file, tracker_file)  # 原子替换（注释：重命名）
            print(f"Updated checkpoint tracker: {tracker_file}")  # 打印更新信息（注释：日志）

        # Copy to HDFS if configured（注释：可选同步到 HDFS）
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)  # 确保 HDFS 目录（注释：远端目录）
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)  # 复制到 HDFS（注释：远端拷贝）

        torch.distributed.barrier()  # 同步所有 rank（注释：同步）

    def _init_checkpoint_manager(self):
        """
        函数用途：初始化 FSDPCheckpointManager。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：无（设置 self.checkpoint_manager）。（注释：返回说明）
        副作用：创建 checkpoint 管理器实例。（注释：副作用）
        异常/边界：未配置 checkpoint 时使用默认 save/load 内容。（注释：边界情况）
        最小示例：
          输入：checkpoint.save_contents=["model"]。（注释：示例输入）
          中间：构建 DictConfig。（注释：中间步骤）
          输出：self.checkpoint_manager 可 save/load。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_init_checkpoint_manager`。（注释：位置）
          典型调用路径：`FSDPSFTTrainer.__init__` -> `_init_checkpoint_manager`。（注释：调用链）
          被谁调用：`FSDPSFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`FSDPCheckpointManager`。（注释：内部依赖）
          调用了谁（外部依赖）：`DictConfig`。（注释：第三方依赖）
        """  # 注释：_init_checkpoint_manager docstring 结束
        """Initialize checkpoint manager with proper configuration"""  # 保留原始英文说明（注释：兼容原注释）
        # Get checkpoint configuration from config, with defaults（注释：读取 checkpoint 配置）
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})  # 读取配置（注释：可选）

        # Set default values if not specified（注释：设置默认值）
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])  # 默认保存内容（注释：保存项）
        load_contents = checkpoint_config.get("load_contents", save_contents)  # 默认加载内容（注释：加载项）

        # Create checkpoint config dict（注释：构建配置字典）
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility（注释：转为 DictConfig）
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)  # 转换类型（注释：兼容性）

        # Initialize checkpoint manager（注释：实例化管理器）
        self.checkpoint_manager = FSDPCheckpointManager(  # 创建 FSDP checkpoint 管理器（注释：管理器实例）
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        """
        函数用途：根据 resume 配置加载模型与 dataloader 状态。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：
          resume_step (int): 加载到的 global_step。（注释：返回说明）
        副作用：更新模型权重与 dataloader 状态。（注释：副作用）
        异常/边界：路径不合法时返回 0。（注释：边界情况）
        最小示例：
          输入：resume_mode="auto" 且存在 global_step_100。（注释：示例输入）
          中间：extract_step=100。（注释：中间结果）
          输出：resume_global_step=100。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::load_checkpoint`。（注释：位置）
          典型调用路径：`FSDPSFTTrainer.__init__` -> `load_checkpoint`。（注释：调用链）
          被谁调用：`FSDPSFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`_determine_resume_path`、`_load_dataloader_state`。（注释：内部依赖）
          调用了谁（外部依赖）：无。（注释：第三方依赖）
        """  # 注释：load_checkpoint docstring 结束
        # Determine resume path based on configuration（注释：确定恢复路径）
        checkpoint_path = self._determine_resume_path()  # 根据配置选择路径（注释：路径选择）

        if checkpoint_path is None:
            return 0  # 没有恢复路径（注释：从头开始）

        # extract resume step from checkpoint path（注释：解析步数）
        resume_step = extract_step(checkpoint_path)  # 从路径解析 step（注释：解析）
        if resume_step is None:
            log_with_rank(  # 记录警告（注释：日志）
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0  # 无法解析时从 0 开始（注释：回退）
        self.resume_global_step = resume_step  # 记录 resume step（注释：状态更新）

        # Use checkpoint manager to load model state（注释：加载模型权重）
        self.checkpoint_manager.load_checkpoint(checkpoint_path)  # 加载 checkpoint（注释：权重恢复）
        log_with_rank(  # 记录加载成功（注释：日志）
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader（注释：加载 dataloader 状态）
        self._load_dataloader_state(checkpoint_path)  # 恢复 dataloader（注释：恢复数据迭代）

        return resume_step  # 返回 resume step（注释：函数返回）

    def _load_dataloader_state(self, checkpoint_path: str):
        """
        函数用途：从 checkpoint 目录加载 dataloader 状态。（注释：函数目标）
        参数：
          checkpoint_path (str): checkpoint 目录路径。（注释：参数说明）
        返回：无。（注释：返回说明）
        副作用：更新 train_dataloader 状态。（注释：副作用）
        异常/边界：data.pt 不存在时打印警告。（注释：边界情况）
        最小示例：
          输入：checkpoint_path=".../global_step_10"。（注释：示例输入）
          中间：加载 data.pt。（注释：中间步骤）
          输出：dataloader 可从断点继续。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_load_dataloader_state`。（注释：位置）
          典型调用路径：`load_checkpoint` -> `_load_dataloader_state`。（注释：调用链）
          被谁调用：`load_checkpoint`。（注释：外部调用）
          调用了谁（项目内）：`self.train_dataloader.load_state_dict`。（注释：内部依赖）
          调用了谁（外部依赖）：`torch.load`。（注释：第三方依赖）
        """  # 注释：_load_dataloader_state docstring 结束
        """Load dataloader state from checkpoint"""  # 保留原英文说明（注释：兼容原注释）
        dataloader_path = os.path.join(checkpoint_path, "data.pt")  # 拼接 data.pt 路径（注释：状态文件）

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality（注释：StatefulDataLoader 状态）
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)  # 从 CPU 加载（注释：加载状态）
            self.train_dataloader.load_state_dict(dataloader_state_dict)  # 恢复 dataloader（注释：状态恢复）

            log_with_rank(  # 记录加载成功（注释：日志）
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(  # 记录警告（注释：日志）
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """
        函数用途：根据 resume_mode 决定恢复路径。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：
          path (str|None): 恢复路径或 None。（注释：返回说明）
        副作用：无。（注释：无副作用）
        异常/边界：resume_mode 非法时抛 ValueError。（注释：边界情况）
        最小示例：
          输入：resume_mode="auto"，目录下存在 global_step_20。（注释：示例输入）
          中间：_find_latest_checkpoint 返回路径。（注释：中间结果）
          输出：返回该路径。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_determine_resume_path`。（注释：位置）
          典型调用路径：`load_checkpoint` -> `_determine_resume_path`。（注释：调用链）
          被谁调用：`load_checkpoint`。（注释：外部调用）
          调用了谁（项目内）：`_find_latest_checkpoint`。（注释：内部依赖）
          调用了谁（外部依赖）：`os.path.exists`。（注释：第三方依赖）
        """  # 注释：_determine_resume_path docstring 结束
        """Determine the path to resume from based on resume_mode configuration"""  # 保留原英文说明（注释：兼容原注释）
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")  # 读取 resume_mode（注释：配置读取）
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)  # 读取 resume_from_path（注释：配置读取）

        if resume_mode == "disable":
            return None  # 禁用恢复（注释：从头训练）
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (  # 确保路径存在（注释：断言）
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"  # 确保含 global_step_（注释：路径校验）
                return resume_from_path  # 优先使用指定路径（注释：优先级）
            # Try to find the latest checkpoint in the default directory（注释：自动查找最新）
            return self._find_latest_checkpoint()  # 查找最新 checkpoint（注释：路径查找）
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (  # resume_path 必须存在（注释：断言）
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"  # 校验路径（注释：路径校验）
            return resume_from_path  # 返回指定路径（注释：直接返回）
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")  # 非法模式（注释：异常）

    def _find_latest_checkpoint(self):
        """
        函数用途：在默认目录中查找最新 checkpoint。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：
          latest_checkpoint (str|None): 最新 checkpoint 路径或 None。（注释：返回说明）
        副作用：rank0 打印日志。（注释：副作用）
        异常/边界：目录不存在返回 None。（注释：边界情况）
        最小示例：
          输入：目录包含 global_step_1/2。（注释：示例输入）
          中间：find_latest_ckpt_path 返回 global_step_2。（注释：中间结果）
          输出：返回 global_step_2。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::_find_latest_checkpoint`。（注释：位置）
          典型调用路径：`_determine_resume_path` -> `_find_latest_checkpoint`。（注释：调用链）
          被谁调用：`_determine_resume_path`。（注释：外部调用）
          调用了谁（项目内）：`find_latest_ckpt_path`。（注释：内部依赖）
          调用了谁（外部依赖）：`os.path.exists`。（注释：第三方依赖）
        """  # 注释：_find_latest_checkpoint docstring 结束
        """Find the latest checkpoint in the default local directory"""  # 保留原英文说明（注释：兼容原注释）
        checkpoint_dir = self.config.trainer.default_local_dir  # 默认 ckpt 目录（注释：目录路径）

        if not os.path.exists(checkpoint_dir):
            return None  # 目录不存在（注释：无 checkpoint）

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)  # 查找最新 ckpt（注释：路径查找）

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)  # 解析 step（注释：解析）
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")  # 打印日志（注释：输出）

        return latest_checkpoint  # 返回路径（注释：函数返回）

    def fit(self):
        """
        函数用途：执行训练主循环，包含日志、验证与保存。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：无（训练完成后 return）。（注释：返回说明）
        副作用：写日志、保存 checkpoint、分布式同步。（注释：副作用）
        异常/边界：test_freq/save_freq<=0 时跳过。（注释：边界情况）
        最小示例：
          输入：total_epochs=1，steps_per_epoch=100。（注释：示例输入）
          中间：global_step 从 resume_global_step 递增。（注释：中间状态）
          输出：训练完成并返回。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/fsdp_sft_trainer.py::fit`。（注释：位置）
          典型调用路径：`run_sft` -> `FSDPSFTTrainer.fit`。（注释：调用链）
          被谁调用：`run_sft`。（注释：外部调用）
          调用了谁（项目内）：`training_step`、`validation_step`、`save_checkpoint`。（注释：内部依赖）
          调用了谁（外部依赖）：`tqdm`、`torch.distributed`。（注释：第三方依赖）
        """  # 注释：fit docstring 结束
        rank = self.device_mesh.get_rank()  # 获取当前 rank（注释：rank 记录）

        # TODO: add a unified tracking（注释：待办事项）
        if rank == 0:
            tracking = Tracking(  # 创建 Tracking（注释：日志追踪）
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step（注释：断点起步）
        last_valid_metric = None  # 记录最后验证指标（注释：指标缓存）
        # compute the total training steps.（注释：计算总步数）
        # the total training steps in SFT is mainly for early exit（注释：用于提前退出）
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs  # 默认总步数（注释：默认计算）

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps  # 使用配置覆盖（注释：手动覆盖）

        self.total_training_steps = total_training_steps  # 保存总步数（注释：状态更新）
        log_with_rank(  # 打印总步数（注释：日志）
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps（注释：StatefulDataLoader 自动恢复）
        # The dataloader will automatically resume from where it left off（注释：无需手动维护）
        if global_step > 0:
            log_with_rank(  # 打印恢复信息（注释：断点提示）
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()（注释：计算起始 epoch）
        start_epoch = global_step // self.steps_per_epoch  # 由 step 反推 epoch（注释：起始轮次）

        train_time = 0  # 累计训练时间（注释：时间统计）
        for epoch in range(start_epoch, self.config.trainer.total_epochs):  # 迭代 epoch（注释：epoch 循环）
            self.train_sampler.set_epoch(epoch=epoch)  # 设置 sampler epoch（注释：采样一致性）

            for step_in_epoch, data in enumerate(  # 迭代 batch（注释：batch 循环）
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1  # 更新全局步数（注释：step++）
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)  # 构造 TensorDict（注释：数据封装）
                metric = self.training_step(data)  # 执行训练 step（注释：训练调用）
                train_time += metric["train/time(s)"]  # 累计时间（注释：时间累计）
                if rank == 0:
                    tracking.log(data=metric, step=global_step)  # 记录训练指标（注释：日志记录）

                is_last_step = global_step >= self.total_training_steps  # 是否最后一步（注释：终止条件）
                is_valid_step = global_step % self.config.trainer.test_freq == 0  # 是否验证步（注释：验证条件）
                is_save_step = global_step % self.config.trainer.save_freq == 0  # 是否保存步（注释：保存条件）

                # early exit or validation step（注释：验证或提前退出）
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation（注释：执行验证）
                    val_losses = []  # 验证 loss 列表（注释：容器）
                    for val_data in self.val_dataloader:  # 遍历验证集（注释：验证循环）
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(  # 构造 TensorDict（注释：数据封装）
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)  # 验证 step（注释：验证调用）
                        val_losses.append(val_loss)  # 收集 loss（注释：loss 汇总）
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))  # 求平均（注释：loss 平均）
                        metric = {"val/loss": val_loss.detach().item()}  # 组装指标（注释：指标字典）
                        tracking.log(data=metric, step=global_step)  # 记录验证指标（注释：日志记录）
                        last_valid_metric = metric  # 更新最后指标（注释：缓存）
                    torch.distributed.barrier()  # 同步 rank（注释：同步）

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)  # 保存 checkpoint（注释：保存）

                if is_last_step:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")  # 打印耗时（注释：训练耗时）
                        print(f"Final validation metrics: {last_valid_metric}")  # 打印最终指标（注释：最终结果）
                    return  # 训练结束（注释：退出 fit）


def run_sft(config):
    """
    函数用途：初始化分布式 Mesh，构建 tokenizer/数据集，并启动 FSDP SFT 训练。（注释：函数目标）
    参数：
      config (DictConfig): Hydra 配置。（注释：参数说明）
    返回：无。（注释：返回说明）
    副作用：初始化/销毁进程组，分配显存。（注释：副作用）
    异常/边界：world_size 与 SP size 不匹配可能报错。（注释：边界情况）
    最小示例：
      输入：world_size=8，ulysses_sequence_parallel_size=1。（注释：示例输入）
      中间：device_mesh 形状为 (8,)。（注释：中间结果）
      输出：训练完成并销毁进程组。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/fsdp_sft_trainer.py::run_sft`。（注释：位置）
      典型调用路径：`main` -> `run_sft`。（注释：调用链）
      被谁调用：`main`。（注释：外部调用）
      调用了谁（项目内）：`create_sft_dataset`、`FSDPSFTTrainer.fit`。（注释：内部依赖）
      调用了谁（外部依赖）：`init_device_mesh`。（注释：第三方依赖）
    """  # 注释：run_sft docstring 结束
    device_name = get_device_name()  # 获取设备名称（注释：设备信息）
    local_rank, rank, world_size = initialize_global_process_group()  # 初始化进程组（注释：分布式初始化）

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))  # 创建 FSDP 设备网格（注释：FSDP mesh）
    dp_size = world_size // config.ulysses_sequence_parallel_size  # 计算 DP 大小（注释：DP 规模）
    ulysses_device_mesh = init_device_mesh(  # 创建 Ulysses SP 网格（注释：SP mesh）
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first（注释：先构建 tokenizer 与数据集）
    from verl.utils import hf_tokenizer  # 延迟导入 tokenizer 工具（注释：HF tokenizer）

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)  # 本地缓存模型权重（注释：权重拷贝）
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)  # 构建 tokenizer（注释：分词器）
    train_dataset = create_sft_dataset(  # 构建训练集（注释：训练数据）
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(  # 构建验证集（注释：验证数据）
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )

    trainer = FSDPSFTTrainer(  # 创建 Trainer（注释：实例化）
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()  # 启动训练（注释：训练入口）

    destroy_global_process_group()  # 销毁进程组（注释：分布式清理）


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    """
    函数用途：Hydra 入口，自动设置设备并启动 run_sft。（注释：函数目标）
    参数：
      config (DictConfig): Hydra 配置。（注释：参数说明）
    返回：无。（注释：返回说明）
    副作用：可能修改 config.trainer.device。（注释：副作用）
    异常/边界：配置缺失会导致下游错误。（注释：边界情况）
    最小示例：
      输入：python -m verl.trainer.fsdp_sft_trainer。（注释：示例输入）
      输出：进入 FSDP SFT 训练流程。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/fsdp_sft_trainer.py::main`。（注释：位置）
      典型调用路径：`torchrun -m verl.trainer.fsdp_sft_trainer` -> `main`。（注释：调用链）
      被谁调用：命令行入口。（注释：外部调用）
      调用了谁（项目内）：`auto_set_ascend_device_name`、`run_sft`。（注释：内部依赖）
      调用了谁（外部依赖）：`hydra.main`。（注释：第三方依赖）
    """  # 注释：main docstring 结束
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.（注释：Ascend 自动设置）
    auto_set_ascend_device_name(config)  # 自动设置设备（注释：设备选择）

    run_sft(config)  # 启动训练（注释：调用 run_sft）


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """
    函数用途：根据配置创建单轮或多轮 SFT 数据集。（注释：函数目标）
    参数：
      data_paths (str|List[str]): parquet 路径。（注释：输入路径）
      data_config (DictConfig): data 配置。（注释：数据配置）
      tokenizer: tokenizer 实例。（注释：分词器）
      max_samples (int): 最大采样数。（注释：采样上限）
    返回：
      dataset: 数据集实例。（注释：返回说明）
    副作用：读取 parquet。（注释：副作用）
    异常/边界：multiturn.enable=true 时使用 MultiTurnSFTDataset。（注释：边界情况）
    最小示例：
      输入：multiturn.enable=false。（注释：示例输入）
      中间：选择 SFTDataset。（注释：中间选择）
      输出：dataset 为单轮数据集。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/fsdp_sft_trainer.py::create_sft_dataset`。（注释：位置）
      典型调用路径：`run_sft` -> `create_sft_dataset`。（注释：调用链）
      被谁调用：`run_sft`。（注释：外部调用）
      调用了谁（项目内）：`SFTDataset`、`MultiTurnSFTDataset`。（注释：内部依赖）
      调用了谁（外部依赖）：无。（注释：第三方依赖）
    """  # 注释：create_sft_dataset docstring 结束
    """Create a dataset."""  # 保留原始英文注释（注释：兼容原注释）
    # build dataset（注释：保留原注释）
    # First check if a custom dataset class is specified（注释：优先自定义类）
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object  # 延迟导入外部对象加载器（注释：动态加载）

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)  # 动态加载类（注释：外部类）
    # Then check if multi-turn dataset should be used（注释：检查多轮开关）
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset  # 多轮数据集（注释：多轮选择）
    # Default to single-turn dataset（注释：默认单轮）
    else:
        dataset_cls = SFTDataset  # 单轮数据集（注释：默认选择）

    # Create datasets based on the selected class（注释：实例化数据集）
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)  # 构建数据集（注释：数据集实例）
    return dataset  # 返回数据集（注释：函数返回）


if __name__ == "__main__":
    main()  # 作为脚本执行时进入主入口（注释：命令行入口）
