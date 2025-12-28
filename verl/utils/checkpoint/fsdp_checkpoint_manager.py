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
模块用途：实现 FSDP 场景下的 checkpoint 保存/加载。（注释：模块职责）
输入/输出：
  - 输入：FSDP 包装后的模型、优化器、调度器与路径配置。（注释：输入说明）
  - 输出：分片模型/优化器状态、RNG/调度器状态、HF 配置与 tokenizer。（注释：输出说明）
关键依赖：torch.distributed.fsdp、accelerate、transformers。（注释：依赖说明）
典型用法（最小示例）：
  - `manager = FSDPCheckpointManager(model, optimizer, ...)`。（注释：实例化）
  - `manager.save_checkpoint(local_path, global_step=100)`。（注释：保存）
调用路径概览：
  - `verl/trainer/fsdp_sft_trainer.py`/`verl/workers/fsdp_workers.py` -> `FSDPCheckpointManager`。（注释：典型入口）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import json  # 注释：保存 FSDP 配置到 JSON
import logging  # 注释：日志输出
import os  # 注释：路径操作
import warnings  # 注释：弃用提醒
from dataclasses import asdict, dataclass  # 注释：数据类与序列化
from typing import Optional  # 注释：类型标注

# ===== 第三方依赖导入 =====
import torch  # 注释：PyTorch 核心
import torch.distributed  # 注释：分布式通信
from accelerate import init_empty_weights  # 注释：HF 模型空权重初始化
from omegaconf import DictConfig  # 注释：配置类型
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 注释：FSDP 包装器
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType  # 注释：FSDP 状态配置
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin  # 注释：HF 组件
from transformers.dynamic_module_utils import custom_object_save  # 注释：保存自定义对象到 HF 目录

# ===== 项目内依赖导入 =====
from verl.utils.device import is_cuda_available  # 注释：CUDA 可用性
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe  # 注释：文件系统工具
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx  # 注释：FSDP 工具
from verl.utils.logger import log_with_rank  # 注释：按 rank 打印日志

from .checkpoint_manager import BaseCheckpointManager  # 注释：通用 checkpoint 基类

# Setup logging
logger = logging.getLogger(__file__)  # 注释：以文件名创建 logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))  # 注释：设置日志级别


@dataclass
class FSDPConfig:
    """
    FSDP checkpoint 配置快照。（注释：类用途）

    参数：
      - FSDP_version (int)：FSDP 版本号（1/2）。（注释：参数说明）
      - world_size (int)：分布式世界大小。（注释：参数说明）
    返回：配置对象。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPConfig`。（注释：定位）
      - 典型调用路径：`FSDPCheckpointManager.save_checkpoint` -> `FSDPConfig`。（注释：典型路径）
      - 被谁调用：本文件内 `save_checkpoint`。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：dataclasses。（注释：外部依赖）
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    FSDP 分片 checkpoint 管理器，支持 SPMD 保存与恢复。（注释：类用途）

    功能：
      - 保存/加载 per-rank 分片模型与优化器状态。（注释：功能说明）
      - 保存完整 lr_scheduler 与 RNG 状态。（注释：功能说明）
      - 保存 HF tokenizer/processor 与模型配置。（注释：功能说明）
    参数：
      - model (FSDP)：FSDP 包装模型。（注释：输入说明）
      - optimizer：优化器。（注释：输入说明）
      - lr_scheduler：学习率调度器。（注释：输入说明）
      - processing_class：HF tokenizer 或 processor。（注释：输入说明）
      - checkpoint_config：保存/加载内容配置。（注释：输入说明）
    返回：实例对象。（注释：返回说明）
    副作用：创建日志记录、读取 rank/world_size。（注释：副作用）
    最小示例：
      - 输入：FSDPCheckpointManager(model, optimizer, ...)。（注释：示例输入）
      - 输出：可调用 save/load 的 manager。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `FSDPCheckpointManager`。（注释：典型路径）
      - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`BaseCheckpointManager`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.fsdp`。（注释：外部依赖）
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        """
        初始化 FSDPCheckpointManager。（注释：方法用途）

        参数：同类说明，并兼容旧的 `tokenizer` 字段。（注释：输入说明）
        返回：无。（注释：构造函数）
        副作用：可能触发弃用警告；调用 BaseCheckpointManager 初始化。（注释：副作用）
        异常/边界条件：若传入 tokenizer 则自动迁移。（注释：边界条件）
        最小示例：
          - 输入：processing_class=None, tokenizer=tok。（注释：示例输入）
          - 输出：processing_class=tok。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager.__init__`。（注释：定位）
          - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `FSDPCheckpointManager(...)`。（注释：典型路径）
          - 被谁调用：训练器/worker 初始化。（注释：调用方）
          - 调用了谁（项目内）：`BaseCheckpointManager.__init__`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`warnings.warn`。（注释：外部依赖）
        """
        if processing_class is None and "tokenizer" in kwargs:  # 注释：兼容旧字段
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")  # 注释：迁移 tokenizer 到 processing_class

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )  # 注释：调用基类初始化

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        加载当前 rank 的 FSDP checkpoint。（注释：方法用途）

        内容包括：模型/优化器分片 + 额外状态（调度器/RNG）。（注释：加载内容）

        参数：
          - local_path (str)：本地 checkpoint 目录。（注释：输入说明）
          - hdfs_path (str, optional)：保留参数（兼容 API），当前未使用。（注释：输入说明）
          - del_local_after_load (bool)：加载后是否删除本地文件。（注释：输入说明）
        返回：无。（注释：仅副作用）
        副作用：加载模型/优化器/调度器/RNG 状态。（注释：副作用）
        异常/边界条件：local_path 为空则直接返回。（注释：边界条件）
        最小示例：
          - 输入：local_path="/ckpt/global_step_10"。（注释：示例输入）
          - 输出：当前 rank 状态恢复。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager.load_checkpoint`。（注释：定位）
          - 典型调用路径：训练恢复 -> `load_checkpoint`。（注释：典型路径）
          - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`verl/workers/fsdp_workers.py`。（注释：调用方）
          - 调用了谁（项目内）：`get_fsdp_state_ctx`、`copy_to_local`、`BaseCheckpointManager.load_rng_state`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`torch.load`、`torch.distributed.barrier`。（注释：外部依赖）
        """
        if local_path is None:  # 注释：空路径直接返回
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:  # 注释：需加载模型时校验
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:  # 注释：需加载优化器时校验
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # every rank download its own checkpoint
        state_dict_cfg = (  # 注释：构造模型分片状态配置
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (  # 注释：构造优化器分片状态配置
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):  # 注释：进入 FSDP state dict 上下文
            if self.should_load_model:  # 注释：加载模型分片
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)  # 注释：下载/复制到本地
                model_state_dict = torch.load(local_model_path, weights_only=False)  # 注释：读取模型状态
                self.model.load_state_dict(model_state_dict)  # 注释：加载到模型
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:  # 注释：加载优化器分片
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)  # 注释：下载/复制到本地
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)  # 注释：读取优化器状态
                self.optimizer.load_state_dict(optimizer_state_dict)  # 注释：加载优化器状态
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:  # 注释：加载额外状态
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)  # 注释：复制到本地
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)  # 注释：读取额外状态
            # recover random state
            if "rng" in extra_state_dict:  # 注释：兼容旧 ckpt 可能缺失 rng
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])  # 注释：恢复 RNG 状态
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]  # 注释：读取调度器状态
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:  # 注释：仅当存在调度器时恢复
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)  # 注释：加载调度器状态
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:  # 注释：rank0 按需删除本地文件
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None  # 注释：删除模型文件
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None  # 注释：删除优化器文件
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None  # 注释：删除额外状态文件
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )  # 注释：删除失败时记录并忽略

        # wait for everyone to load checkpoints
        torch.distributed.barrier()  # 注释：同步所有 rank

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        保存当前 rank 的 FSDP checkpoint。（注释：方法用途）

        保存内容：
          - 模型/优化器分片。（注释：保存内容）
          - 额外状态（调度器/RNG）。（注释：保存内容）
          - rank0 保存 HF 配置/tokenizer。（注释：保存内容）
          - 可选保存完整 HF 模型。（注释：保存内容）

        参数：
          - local_path (str)：保存目录。（注释：输入说明）
          - hdfs_path (str, optional)：保留参数（API 兼容）。（注释：输入说明）
          - global_step (int)：当前训练步。（注释：输入说明）
          - max_ckpt_to_keep (int, optional)：最大保留数量。（注释：输入说明）
        返回：无。（注释：仅副作用）
        副作用：写入 checkpoint 文件，可能删除旧文件。（注释：副作用）
        异常/边界条件：local_path 为空则直接返回。（注释：边界条件）
        最小示例：
          - 输入：local_path="/ckpt/global_step_10"。（注释：示例输入）
          - 输出：生成 per-rank ckpt 文件。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager.save_checkpoint`。（注释：定位）
          - 典型调用路径：训练循环 -> `save_checkpoint`。（注释：典型路径）
          - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`verl/workers/fsdp_workers.py`。（注释：调用方）
          - 调用了谁（项目内）：`get_fsdp_state_ctx`、`get_fsdp_full_state_dict`、`BaseCheckpointManager.get_rng_state`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`torch.save`、`transformers.*`。（注释：外部依赖）
        """
        if local_path is None:  # 注释：空路径直接返回
            return

        # record the previous global step
        self.previous_global_step = global_step  # 注释：记录上次保存的步数

        # remove previous local_path, only rank 0 should do this
        if (
            self.rank == 0
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1  # 注释：保留窗口起点
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])  # 注释：删除旧目录
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]  # 注释：更新历史列表

        local_path = local_mkdir_safe(local_path)  # 注释：创建保存目录
        torch.distributed.barrier()  # 注释：同步目录创建

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:  # 注释：需要保存模型则校验
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:  # 注释：需要保存优化器则校验
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)  # 注释：模型分片配置
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)  # 注释：优化器分片配置
        with warnings.catch_warnings():  # 注释：屏蔽不必要的警告
            warnings.simplefilter("ignore")  # 注释：忽略警告
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):  # 注释：FSDP 上下文
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:  # 注释：保存模型分片
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:  # 注释：保存优化器分片
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:  # 注释：保存额外状态
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:  # 注释：rank0 保存 HF 配置/tokenizer
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:  # 注释：FSDP1 解包
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")  # 注释：HF 目录
            local_mkdir_safe(hf_config_tokenizer_path)  # 注释：创建目录
            model_config = unwrap_model.config  # 注释：模型配置
            generation_config = None  # 注释：生成配置占位
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:  # 注释：可生成模型
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:  # 注释：清理 auto_map 中的 None
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)  # 注释：保存 config
            if self.processing_class is not None:  # 注释：保存 tokenizer/processor
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):  # 注释：保存自定义模型代码
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")  # 注释：FSDP 配置路径
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )  # 注释：构造 FSDP 配置快照
            with open(fsdp_config_path, "w") as f:  # 注释：写入 JSON
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        torch.distributed.barrier()  # 注释：同步所有 rank 完成本地写入

        if self.should_save_hf_model:  # 注释：可选保存完整 HF 模型
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)  # 注释：获取完整 state_dict

            if self.rank == 0:  # 注释：仅 rank0 保存 HF 模型
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    # Handle different transformers versions for Vision2Seq models
                    import transformers
                    from packaging import version

                    if version.parse(transformers.__version__) >= version.parse("4.54.0"):
                        # transformers >= 4.54.0 uses AutoModelForImageTextToText
                        from transformers import AutoModelForImageTextToText

                        auto_model_cls = AutoModelForImageTextToText
                    else:
                        # transformers < 4.54.0 uses AutoModelForVision2Seq
                        from transformers import AutoModelForVision2Seq

                        auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():  # 注释：空权重初始化模型
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")  # 注释：将参数移动到 CPU 空张量

                if save_model.can_generate():  # 注释：配置生成参数
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)  # 注释：保存 HF 模型
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict  # 注释：释放内存
                del save_model  # 注释：释放模型

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()  # 注释：同步 rank0 保存完成

        self.previous_saved_paths.append(local_path)  # 注释：记录保存历史
