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
模块用途：引擎无关版 SFT Trainer，使用 TrainingWorker 抽象完成数据加载、训练、验证与保存。（注释：说明模块职责）
输入：Hydra 配置（sft_trainer_engine.yaml）、训练/验证 parquet、模型/引擎/优化器配置。（注释：说明主要输入）
输出：checkpoint（本地/HDFS）与日志指标（console/wandb）。（注释：说明输出产物）
关键依赖：torch、hydra、tensordict、verl.workers.engine_workers、verl.utils.dataset。（注释：说明依赖）
典型用法：python -m verl.trainer.sft_trainer data.train_files=... trainer.total_epochs=1。（注释：最小运行示例）
调用路径概览：命令行 -> main(...) -> run_sft(...) -> SFTTrainer.fit(...) -> TrainingWorker.train_batch(...)。（注释：入口链路）
"""  # 注释：模块级说明结束

import os  # 标准库：环境变量与路径（注释：os 用途）
from functools import partial  # 标准库：函数偏应用（注释：partial 用途）

from tensordict.tensorclass import NonTensorData  # 第三方：封装非张量数据（注释：NonTensorData）

os.environ["NCCL_DEBUG"] = "WARN"  # 环境变量：降低 NCCL 日志（注释：分布式调试）
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # 环境变量：开启 tokenizer 并行（注释：性能优化）

import logging  # 标准库：日志（注释：logging 用途）

import hydra  # 第三方：配置管理（注释：Hydra 入口）
import torch  # 第三方：张量与分布式（注释：PyTorch）
import torch.distributed  # 第三方：分布式通信（注释：torch.distributed）
from omegaconf import OmegaConf  # 第三方：OmegaConf（注释：配置转 dict）
from torch.utils.data import DistributedSampler  # 第三方：分布式采样器（注释：数据分片）
from torchdata.stateful_dataloader import StatefulDataLoader  # 第三方：可恢复 DataLoader（注释：断点续训）
from tqdm import tqdm  # 第三方：进度条（注释：训练可视化）

from verl.utils import tensordict_utils as tu  # 项目内：tensordict 工具（注释：数据封装）
from verl.utils.checkpoint import CheckpointHandler  # 项目内：统一 checkpoint 管理（注释：断点保存）
from verl.utils.dataset.dataset_utils import SFTTensorCollator  # 项目内：SFT collate 函数（注释：batch 组装）
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset  # 项目内：多轮数据集（注释：数据类）
from verl.utils.device import get_device_name  # 项目内：设备名（注释：cuda/npu）
from verl.utils.distributed import destroy_global_process_group  # 项目内：销毁分布式（注释：清理）
from verl.utils.logger import log_with_rank  # 项目内：分布式日志（注释：rank 控制）
from verl.utils.tracking import Tracking  # 项目内：日志跟踪器（注释：指标记录）
from verl.workers.engine_workers import TrainingWorker  # 项目内：训练引擎抽象（注释：引擎封装）

logger = logging.getLogger(__file__)  # 获取当前文件 logger（注释：日志实例）
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))  # 读取环境变量设置级别（注释：日志级别）


class SFTTrainer:
    """
    类用途：引擎无关 SFT Trainer，负责数据/引擎/ckpt 组装并驱动训练循环。（注释：类职责）
    参数：
      config (DictConfig): Hydra 配置对象，包含 data/model/engine/optim/trainer。（注释：参数说明）
    关键属性：
      engine: TrainingWorker 内部引擎实例。（注释：训练引擎）
      train_dataloader/val_dataloader: 训练/验证数据加载器。（注释：数据迭代）
    副作用：初始化分布式训练、加载/保存 checkpoint、写日志。（注释：副作用）
    调用路径依赖：
      所在位置：`verl/trainer/sft_trainer.py::SFTTrainer`。（注释：位置）
      典型调用路径：`python -m verl.trainer.sft_trainer` -> `main` -> `run_sft` -> `SFTTrainer`。（注释：调用链）
      被谁调用：`run_sft(...)`。（注释：外部调用）
      调用了谁（项目内）：`TrainingWorker`、`CheckpointHandler`、`Tracking`。（注释：内部依赖）
      调用了谁（外部依赖）：`torch.distributed`、`StatefulDataLoader`。（注释：第三方依赖）
    """  # 注释：类 docstring 结束

    def __init__(
        self,
        config,
    ):
        """
        函数用途：初始化 Trainer，构建配置、数据集、引擎与 DataLoader，并加载断点。（注释：函数目标）
        参数：
          config (DictConfig): Hydra 配置对象。（注释：参数说明）
        返回：无，初始化实例状态。（注释：返回说明）
        副作用：读取分布式 rank、初始化引擎、可能加载 checkpoint。（注释：副作用）
        异常/边界：若配置缺字段或分布式未初始化将报错。（注释：异常情况）
        最小示例：
          输入：config.trainer.total_epochs=1，data.train_files=...（注释：示例输入）
          中间：_build_dataset 构建 DataLoader。（注释：关键中间结果）
          输出：trainer 实例可调用 fit()。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::SFTTrainer.__init__`。（注释：位置）
          典型调用路径：`main` -> `run_sft` -> `SFTTrainer(...)`。（注释：调用链）
          被谁调用：`run_sft(...)`。（注释：外部调用）
          调用了谁（项目内）：`_build_config`、`_build_dataset`、`_build_engine` 等。（注释：内部调用）
          调用了谁（外部依赖）：`torch.distributed.get_rank`。（注释：第三方依赖）
        """  # 注释：__init__ docstring 结束
        self.config = config  # 保存配置对象（注释：持久化配置）

        self.rank = torch.distributed.get_rank()  # 获取当前进程 rank（注释：分布式 rank）

        self._build_config()  # 构建 dataclass 配置（注释：配置转换）
        self._build_dataset()  # 构建训练/验证数据集（注释：数据准备）

        self._build_engine()  # 初始化训练引擎（注释：引擎构建）

        self._build_dataloader()  # 构建 DataLoader（注释：数据加载器）

        self._init_engine()  # 初始化引擎的训练步数等（注释：引擎初始化）

        self._build_ckpt_handler()  # 构建 checkpoint 管理器（注释：断点保存）

        # Initialize resume-related variables（注释：初始化断点信息）
        self.resume_global_step = self.ckpt_handler.load_checkpoint()  # 读取断点并返回步数（注释：断点恢复）

        self.device_name = self.config.trainer.device  # 记录设备名称（注释：设备标识）

        if self.rank == 0:  # 仅 rank0 打印配置（注释：主进程日志）
            print(self.config)  # 输出完整配置（注释：调试输出）

    def _build_ckpt_handler(self):
        """
        函数用途：根据 trainer 配置创建 CheckpointHandler。（注释：函数目标）
        参数：无（读取 self.config）。（注释：参数说明）
        返回：无（设置 self.ckpt_handler）。（注释：返回说明）
        副作用：创建 checkpoint 管理器对象。（注释：副作用）
        异常/边界：配置字段缺失会使用默认值。（注释：边界条件）
        最小示例：
          输入：resume_mode="auto"，default_local_dir="./ckpt"。（注释：示例输入）
          中间：构建 CheckpointHandler(...).（注释：中间结果）
          输出：self.ckpt_handler 可调用 load/save。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_build_ckpt_handler`。（注释：位置）
          典型调用路径：`SFTTrainer.__init__` -> `_build_ckpt_handler`。（注释：调用链）
          被谁调用：`SFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`CheckpointHandler`。（注释：内部依赖）
          调用了谁（外部依赖）：无。（注释：第三方依赖）
        """  # 注释：_build_ckpt_handler docstring 结束
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")  # 读取 resume 模式（注释：resume 配置）
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)  # 读取 resume 路径（注释：resume 路径）
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)  # 最大 ckpt 数（注释：清理策略）
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)  # HDFS 保存目录（注释：远端保存）

        self.ckpt_handler = CheckpointHandler(  # 实例化 checkpoint 管理器（注释：创建对象）
            engine=self.engine,  # 绑定训练引擎（注释：引擎依赖）
            train_dataloader=self.train_dataloader,  # 绑定 dataloader（注释：状态保存）
            default_local_dir=self.config.trainer.default_local_dir,  # 本地保存目录（注释：输出路径）
            max_ckpt_to_keep=max_ckpt_to_keep,  # 最大保留数量（注释：清理策略）
            default_hdfs_dir=default_hdfs_dir,  # HDFS 保存目录（注释：远端保存）
            resume_mode=resume_mode,  # 断点恢复模式（注释：resume 模式）
            resume_from_path=resume_from_path,  # 指定恢复路径（注释：resume 路径）
        )

    def _build_config(self):
        """
        函数用途：将 OmegaConf 配置转为 dataclass 配置对象。（注释：函数目标）
        参数：无（读取 self.config）。（注释：参数说明）
        返回：无（设置 model/engine/optim/checkpoint 配置）。（注释：返回说明）
        副作用：更新 self.*_config 字段。（注释：副作用）
        异常/边界：配置字段缺失将抛出属性错误。（注释：异常情况）
        最小示例：
          输入：config.model.partial_pretrain="Qwen/..."。（注释：示例输入）
          中间：omega_conf_to_dataclass 转换。（注释：中间步骤）
          输出：self.model_config.partial_pretrain 可访问。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_build_config`。（注释：位置）
          典型调用路径：`SFTTrainer.__init__` -> `_build_config`。（注释：调用链）
          被谁调用：`SFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`verl.utils.config.omega_conf_to_dataclass`。（注释：内部依赖）
          调用了谁（外部依赖）：无。（注释：第三方依赖）
        """  # 注释：_build_config docstring 结束
        from verl.utils.config import omega_conf_to_dataclass  # 延迟导入避免循环（注释：延迟导入）

        self.model_config = omega_conf_to_dataclass(self.config.model)  # 转换模型配置（注释：模型配置）
        self.engine_config = omega_conf_to_dataclass(self.config.engine)  # 转换引擎配置（注释：引擎配置）
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)  # 转换优化器配置（注释：优化器配置）
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)  # 转换 checkpoint 配置（注释：断点配置）

    def _build_engine(self):
        """
        函数用途：构建 TrainingWorker 并配置损失函数。（注释：函数目标）
        参数：无（读取 self.*_config）。（注释：参数说明）
        返回：无（设置 self.engine / self.training_client）。（注释：返回说明）
        副作用：初始化训练引擎对象。（注释：副作用）
        异常/边界：引擎配置不合法将抛出异常。（注释：异常情况）
        最小示例：
          输入：model_type="language_model"。（注释：示例输入）
          中间：TrainingWorkerConfig 构建。（注释：中间步骤）
          输出：self.engine 可执行 train_batch。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_build_engine`。（注释：位置）
          典型调用路径：`SFTTrainer.__init__` -> `_build_engine`。（注释：调用链）
          被谁调用：`SFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`TrainingWorker`、`sft_loss`。（注释：内部依赖）
          调用了谁（外部依赖）：`functools.partial`。（注释：第三方依赖）
        """  # 注释：_build_engine docstring 结束
        from verl.workers.engine_workers import TrainingWorkerConfig  # 延迟导入配置类（注释：延迟导入）
        from verl.workers.utils.losses import sft_loss  # 延迟导入损失（注释：损失函数）

        self.loss_fn = partial(sft_loss, config=None)  # 固定 config 参数（注释：偏应用）

        config = TrainingWorkerConfig(  # 构建 TrainingWorker 配置（注释：引擎配置）
            model_type="language_model",  # 模型类型（注释：语言模型）
            model_config=self.model_config,  # 模型配置（注释：模型参数）
            engine_config=self.engine_config,  # 引擎配置（注释：并行/设备）
            optimizer_config=self.optimizer_config,  # 优化器配置（注释：学习率等）
            checkpoint_config=self.checkpoint_config,  # checkpoint 配置（注释：保存策略）
        )

        self.training_client = TrainingWorker(config=config)  # 创建训练 worker（注释：引擎实例）
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)  # 设置损失函数（注释：训练损失）
        # Note that in SPMD world, this abstraction has to break（注释：SPMD 注意事项）
        self.engine = self.training_client.engine  # 获取内部引擎（注释：引擎引用）

    def _init_engine(self):
        """
        函数用途：初始化训练步数、保存/验证频率并 reset 引擎。（注释：函数目标）
        参数：无（读取 self.config 与 dataloader）。（注释：参数说明）
        返回：无（更新训练相关属性）。（注释：返回说明）
        副作用：修改 optimizer_config.total_training_steps。（注释：副作用）
        异常/边界：train_dataloader 为空将导致步数计算异常。（注释：边界情况）
        最小示例：
          输入：len(train_dataloader)=100，total_epochs=2。（注释：示例输入）
          中间：total_training_steps=200。（注释：中间计算）
          输出：save_freq/test_freq 可能被替换为 steps_per_epoch。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_init_engine`。（注释：位置）
          典型调用路径：`SFTTrainer.__init__` -> `_init_engine`。（注释：调用链）
          被谁调用：`SFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`self.training_client.reset`。（注释：内部依赖）
          调用了谁（外部依赖）：无。（注释：第三方依赖）
        """  # 注释：_init_engine docstring 结束
        # patch optimizer config（注释：同步训练步数到优化器）
        if self.config.trainer.total_training_steps is not None:  # 若手动指定总步数（注释：手动覆盖）
            self.total_training_steps = self.config.trainer.total_training_steps  # 使用配置值（注释：优先配置）
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs  # 由 epoch 推断（注释：自动推断）
        self.optimizer_config.total_training_steps = self.total_training_steps  # 写回优化器配置（注释：同步步数）

        self.steps_per_epoch = len(self.train_dataloader)  # 每个 epoch 步数（注释：步数统计）

        # manage save and test frequency（注释：保存/验证频率处理）
        self.save_freq = self.config.trainer.save_freq  # 读取保存频率（注释：配置读取）
        if self.save_freq == "after_each_epoch":  # 若按 epoch 保存（注释：特殊字符串）
            self.save_freq = self.steps_per_epoch  # 转换为步数（注释：步数化）

        self.test_freq = self.config.trainer.test_freq  # 读取验证频率（注释：配置读取）
        if self.test_freq == "after_each_epoch":  # 若按 epoch 验证（注释：特殊字符串）
            self.test_freq = self.steps_per_epoch  # 转换为步数（注释：步数化）

        self.training_client.reset()  # 重置引擎状态（注释：确保干净状态）

    def _build_dataset(self):
        """
        函数用途：根据配置创建训练/验证数据集。（注释：函数目标）
        参数：无（读取 self.config 与 self.model_config）。（注释：参数说明）
        返回：无（设置 self.train_dataset/self.val_dataset）。（注释：返回说明）
        副作用：可能加载 tokenizer/processor 相关配置。（注释：副作用）
        异常/边界：val_files 为空则 val_dataset=None。（注释：边界情况）
        最小示例：
          输入：train_files=["train.parquet"], val_files=null。（注释：示例输入）
          中间：create_sft_dataset 被调用。（注释：中间步骤）
          输出：train_dataset 有效，val_dataset=None。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_build_dataset`。（注释：位置）
          典型调用路径：`SFTTrainer.__init__` -> `_build_dataset`。（注释：调用链）
          被谁调用：`SFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`create_sft_dataset`。（注释：内部依赖）
          调用了谁（外部依赖）：无。（注释：第三方依赖）
        """  # 注释：_build_dataset docstring 结束
        config = self.config  # 读取配置（注释：局部引用）
        tokenizer = self.model_config.tokenizer  # 读取 tokenizer（注释：模型配置）
        processor = self.model_config.processor  # 读取 processor（注释：多模态处理）
        train_dataset = create_sft_dataset(  # 构建训练集（注释：训练数据）
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:  # 若有验证集（注释：验证数据存在）
            val_dataset = create_sft_dataset(  # 构建验证集（注释：验证数据）
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None  # 无验证集（注释：关闭验证）

        self.train_dataset, self.val_dataset = train_dataset, val_dataset  # 保存数据集引用（注释：成员赋值）

    def _build_dataloader(self):
        """
        函数用途：构建分布式 DataLoader 与采样器。（注释：函数目标）
        参数：无（读取 self.train_dataset/self.val_dataset）。（注释：参数说明）
        返回：无（设置 train_dataloader/val_dataloader）。（注释：返回说明）
        副作用：创建 DistributedSampler 影响样本顺序。（注释：副作用）
        异常/边界：val_dataset 为空则 val_dataloader=None。（注释：边界情况）
        最小示例：
          输入：dp_size=2，global_batch=256。（注释：示例输入）
          中间：train_batch_size_per_dp=128。（注释：中间计算）
          输出：每个 DP rank 处理 128 样本。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_build_dataloader`。（注释：位置）
          典型调用路径：`SFTTrainer.__init__` -> `_build_dataloader`。（注释：调用链）
          被谁调用：`SFTTrainer.__init__`。（注释：外部调用）
          调用了谁（项目内）：`SFTTensorCollator`。（注释：内部依赖）
          调用了谁（外部依赖）：`DistributedSampler`、`StatefulDataLoader`。（注释：第三方依赖）
        """  # 注释：_build_dataloader docstring 结束
        # build dataset（注释：保留注释说明）
        config = self.config  # 读取配置（注释：局部引用）
        # build dataloader（注释：保留注释说明）
        # Use data parallel rank and size instead of global rank and world size（注释：DP 范围内分片）

        # Set pin_memory_device when pin_memory is enabled.（注释：pin_memory 设备）
        device_name = get_device_name()  # 获取设备名（注释：cuda/npu）

        dp_rank = self.engine.get_data_parallel_rank()  # DP rank（注释：数据并行 rank）
        dp_size = self.engine.get_data_parallel_size()  # DP size（注释：数据并行规模）

        self.train_sampler = DistributedSampler(  # 创建训练采样器（注释：分布式采样）
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size  # 全局 batch（注释：配置读取）
        self.train_batch_size_per_dp = self.global_batch_size // dp_size  # 每 DP batch（注释：均分 batch）
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)  # collate 函数（注释：padding 策略）

        self.train_dataloader = StatefulDataLoader(  # 构建训练 DataLoader（注释：支持恢复）
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:  # 若存在验证集（注释：验证开关）
            self.val_sampler = DistributedSampler(  # 构建验证采样器（注释：分布式采样）
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(  # 构建验证 DataLoader（注释：支持恢复）
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=8,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None  # 无验证集（注释：置空）

    def _get_batch_seqlens(self, data):
        """
        函数用途：统计并聚合 batch 的序列长度（跨 DP all_gather）。（注释：函数目标）
        参数：
          data (TensorDict): batch 数据，包含 input_ids/attention_mask。（注释：参数说明）
        返回：
          batch_seqlens (List[int]): 全局 batch 的序列长度列表。（注释：返回说明）
        副作用：进行一次 all_gather 通信。（注释：副作用）
        异常/边界：nested tensor 与普通 tensor 分支不同。（注释：边界情况）
        最小示例：
          输入：attention_mask=[[1,1,0],[1,1,1]]。（注释：示例输入）
          中间：sum -> [2,3]，all_gather 合并 DP。（注释：中间计算）
          输出：[2,3,...]（全局）。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::_get_batch_seqlens`。（注释：位置）
          典型调用路径：`SFTTrainer.fit` -> `_get_batch_seqlens`。（注释：调用链）
          被谁调用：`fit`。（注释：外部调用）
          调用了谁（项目内）：`self.engine.get_data_parallel_group`。（注释：内部依赖）
          调用了谁（外部依赖）：`torch.distributed.all_gather_into_tensor`。（注释：第三方依赖）
        """  # 注释：_get_batch_seqlens docstring 结束
        # mean over dp group（注释：注释保留说明）
        is_nested = data["input_ids"].is_nested  # 判断是否为 nested tensor（注释：分支判断）
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()  # 通过 offsets 计算长度（注释：nested 路径）
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)  # 通过 mask 求长度（注释：普通路径）
        batch_seqlens = batch_seqlens.to(self.device_name)  # 移到当前设备（注释：设备迁移）

        output_tensor = torch.empty(  # 创建全局容器（注释：gather 缓冲区）
            (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )

        torch.distributed.all_gather_into_tensor(  # 跨 DP 收集长度（注释：通信聚合）
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=self.engine.get_data_parallel_group(),
        )

        batch_seqlens = output_tensor.tolist()  # 转为 Python list（注释：便于后续统计）
        return batch_seqlens  # 返回长度列表（注释：函数返回）

    def fit(self):
        """
        函数用途：执行训练主循环，包含日志、验证与保存。（注释：函数目标）
        参数：无。（注释：参数说明）
        返回：无（训练结束返回）。（注释：返回说明）
        副作用：写日志、保存 checkpoint、进行分布式同步。（注释：副作用）
        异常/边界：val_dataloader=None 时跳过验证。（注释：边界情况）
        最小示例：
          输入：total_epochs=1，steps_per_epoch=100。（注释：示例输入）
          中间：global_step 从 resume_global_step 递增。（注释：中间状态）
          输出：训练完成并保存 checkpoint。（注释：示例输出）
        调用路径依赖：
          所在位置：`verl/trainer/sft_trainer.py::SFTTrainer.fit`。（注释：位置）
          典型调用路径：`run_sft` -> `SFTTrainer.fit`。（注释：调用链）
          被谁调用：`run_sft`。（注释：外部调用）
          调用了谁（项目内）：`TrainingWorker.train_batch`、`CheckpointHandler.save_checkpoint`。（注释：内部依赖）
          调用了谁（外部依赖）：`tqdm`、`torch.distributed`。（注释：第三方依赖）
        """  # 注释：fit docstring 结束
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0  # 仅主 rank 记录日志（注释：日志开关）

        # TODO: add a unified tracking（注释：待办事项）
        if is_logging:
            tracking = Tracking(  # 初始化日志跟踪器（注释：Tracking 实例）
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step（注释：断点起步）
        last_valid_metric = None  # 保存最后一次验证指标（注释：指标缓存）

        log_with_rank(  # 打印总步数信息（注释：训练统计）
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps（注释：StatefulDataLoader 自动恢复）
        # The dataloader will automatically resume from where it left off（注释：无需手动维护）
        if global_step > 0:
            log_with_rank(  # 提示从断点恢复（注释：断点提示）
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()（注释：计算起始 epoch）
        start_epoch = global_step // self.steps_per_epoch  # 用步数反推 epoch（注释：起始轮次）

        meta_info = {  # 非张量元信息（注释：训练元信息）
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0  # 训练耗时累计（注释：时间统计）
        total_tokens = 0  # 累计 token 数（注释：吞吐统计）
        for epoch in range(start_epoch, self.config.trainer.total_epochs):  # 遍历 epoch（注释：训练轮次）
            self.train_sampler.set_epoch(epoch=epoch)  # 设置 sampler 的 epoch（注释：打乱一致性）

            for step_in_epoch, data in enumerate(  # 遍历 batch（注释：训练迭代）
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1  # 更新全局步数（注释：step++）

                # construct tensordict（注释：构造 TensorDict）
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)  # 合并元信息（注释：拼接 meta）
                batch_seqlens = self._get_batch_seqlens(data=data)  # 统计序列长度（注释：长度统计）
                # this is necessary. Otherwise, it is interpreted as NonTensorStack（注释：避免 NonTensorStack）
                batch_seqlens = NonTensorData(batch_seqlens)  # 包装成 NonTensorData（注释：类型封装）

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens)  # 写入 non-tensor 元信息（注释：附加信息）

                # train for on batch（注释：执行单 batch 训练）
                output = self.training_client.train_batch(data=data)  # 调用引擎训练（注释：训练一步）

                if self.engine.is_mp_src_rank_with_outputs():  # 仅主 rank 处理指标（注释：指标聚合）
                    metrics = tu.get(output, "metrics")  # 获取 metrics（注释：读取指标）

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics（注释：待优化事项）
                    metrics["train/loss"] = metrics.pop("loss")  # 重命名 loss（注释：指标命名）
                    metrics["train/grad_norm"] = metrics.pop("grad_norm")  # 重命名 grad_norm（注释：指标命名）
                    metrics["train/lr"] = metrics.pop("lr")  # 重命名 lr（注释：指标命名）
                    metrics["train/mfu"] = metrics.pop("mfu")  # 重命名 mfu（注释：指标命名）
                    metrics["train/global_tokens"] = torch.sum(  # 统计全局 token（注释：token 统计）
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]  # 累计 token（注释：吞吐统计）
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9  # 转换为十亿单位（注释：可读性）

                    if self.engine.get_data_parallel_rank() == 0:  # DP rank0 打日志（注释：日志过滤）
                        tracking.log(data=metrics, step=global_step)  # 写日志（注释：指标记录）

                is_last_step = global_step >= self.total_training_steps  # 是否最后一步（注释：终止条件）
                is_valid_step = global_step % self.test_freq == 0  # 是否验证步（注释：验证条件）
                is_save_step = global_step % self.save_freq == 0  # 是否保存步（注释：保存条件）

                # early exit or validation step（注释：验证或提前结束）
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation（注释：执行验证）
                    val_losses = []  # 收集验证 loss（注释：列表容器）
                    for val_data in self.val_dataloader:  # 遍历验证集（注释：验证迭代）
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)  # 构造 TensorDict（注释：合并 meta）
                        output = self.training_client.infer_batch(val_data)  # 推理（注释：验证前向）

                        if self.engine.is_mp_src_rank_with_outputs():  # 仅主 rank 收集指标（注释：指标过滤）
                            metrics = tu.get(output, "metrics")  # 读取指标（注释：metrics）
                            val_losses.append(metrics["loss"])  # 收集 loss（注释：loss 收集）

                    if self.engine.is_mp_src_rank_with_outputs():  # 主 rank 做聚合（注释：聚合计算）
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))  # 平均 loss（注释：均值）
                        # average over data parallel group（注释：DP 组平均）
                        torch.distributed.all_reduce(  # 跨 DP 求平均（注释：all_reduce）
                            val_loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()
                        )

                    if is_logging:  # 仅日志 rank 记录（注释：日志控制）
                        metric = {"val/loss": val_loss.detach().item()}  # 构造验证指标（注释：指标字典）
                        tracking.log(data=metric, step=global_step)  # 写日志（注释：记录验证）
                        last_valid_metric = metric  # 记录最后指标（注释：缓存）
                    torch.distributed.barrier()  # 同步（注释：rank 同步）

                if is_last_step or (self.save_freq > 0 and is_save_step):  # 保存条件（注释：保存开关）
                    self.ckpt_handler.save_checkpoint(step=global_step)  # 保存 checkpoint（注释：保存状态）

                if is_last_step:  # 最后一步退出（注释：训练结束）
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")  # 打印耗时（注释：训练耗时）
                        print(f"Final validation metrics: {last_valid_metric}")  # 打印最终指标（注释：最终结果）
                    return  # 结束训练（注释：退出 fit）


def run_sft(config):
    """
    函数用途：初始化分布式并运行 SFTTrainer。（注释：函数目标）
    参数：
      config (DictConfig): Hydra 配置。（注释：参数说明）
    返回：无。（注释：返回说明）
    副作用：初始化/销毁分布式进程组。（注释：副作用）
    异常/边界：分布式初始化失败会抛出异常。（注释：异常情况）
    最小示例：
      输入：config.trainer.total_epochs=1。（注释：示例输入）
      中间：SFTTrainer.fit 执行训练。（注释：中间步骤）
      输出：训练结束并释放资源。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/sft_trainer.py::run_sft`。（注释：位置）
      典型调用路径：`main` -> `run_sft`。（注释：调用链）
      被谁调用：`main`。（注释：外部调用）
      调用了谁（项目内）：`SFTTrainer.fit`、`destroy_global_process_group`。（注释：内部依赖）
      调用了谁（外部依赖）：`torch.distributed`。（注释：第三方依赖）
    """  # 注释：run_sft docstring 结束
    from verl.utils.distributed import initialize_global_process_group  # 延迟导入（注释：分布式初始化）

    initialize_global_process_group()  # 初始化进程组（注释：分布式初始化）
    trainer = SFTTrainer(config=config)  # 创建 Trainer（注释：实例化）
    trainer.fit()  # 启动训练（注释：训练入口）
    destroy_global_process_group()  # 销毁进程组（注释：分布式清理）


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    """
    函数用途：Hydra 入口，接收配置并启动 run_sft。（注释：函数目标）
    参数：
      config (DictConfig): Hydra 注入配置。（注释：参数说明）
    返回：无。（注释：返回说明）
    副作用：调用 run_sft 引发训练流程。（注释：副作用）
    异常/边界：配置缺失会导致下游错误。（注释：边界情况）
    最小示例：
      输入：python -m verl.trainer.sft_trainer。（注释：示例输入）
      输出：进入 SFT 训练流程。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/sft_trainer.py::main`。（注释：位置）
      典型调用路径：`python -m verl.trainer.sft_trainer` -> `main`。（注释：调用链）
      被谁调用：命令行入口。（注释：外部调用）
      调用了谁（项目内）：`run_sft`。（注释：内部依赖）
      调用了谁（外部依赖）：`hydra.main`。（注释：第三方依赖）
    """  # 注释：main docstring 结束
    run_sft(config)  # 执行训练入口（注释：调用 run_sft）


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """
    函数用途：构建 SFT 数据集实例（默认多轮实现）。（注释：函数目标）
    参数：
      data_paths (str|List[str]): parquet 路径或列表。（注释：输入路径）
      data_config (DictConfig): data 配置段。（注释：数据配置）
      tokenizer: tokenizer 实例。（注释：分词器）
      processor: 可能的多模态 processor。（注释：处理器）
      max_samples (int): 最大采样数，-1 表示全量。（注释：采样上限）
    返回：
      dataset: 构建后的数据集对象。（注释：返回说明）
    副作用：读取 parquet 文件。（注释：副作用）
    异常/边界：data_config.custom_cls 缺失时走默认类。（注释：边界情况）
    最小示例：
      输入：data_paths="train.parquet"，max_samples=100。（注释：示例输入）
      中间：选择 MultiTurnSFTDataset。（注释：中间选择）
      输出：dataset 可被 DataLoader 迭代。（注释：示例输出）
    调用路径依赖：
      所在位置：`verl/trainer/sft_trainer.py::create_sft_dataset`。（注释：位置）
      典型调用路径：`SFTTrainer._build_dataset` -> `create_sft_dataset`。（注释：调用链）
      被谁调用：`_build_dataset`。（注释：外部调用）
      调用了谁（项目内）：`MultiTurnSFTDataset`。（注释：内部依赖）
      调用了谁（外部依赖）：无。（注释：第三方依赖）
    """  # 注释：create_sft_dataset docstring 结束
    """Create a dataset."""  # 保留原始英文说明（注释：兼容原注释）
    # build dataset（注释：保留原始注释）
    # First check if a custom dataset class is specified（注释：优先自定义类）
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object  # 延迟导入外部对象加载器（注释：动态加载）

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)  # 动态加载类（注释：外部类）
    else:
        # Default to multi-turn dataset（注释：默认多轮数据集）
        dataset_cls = MultiTurnSFTDataset  # 默认数据集类（注释：默认选择）

    # Create datasets based on the selected class（注释：实例化数据集）
    dataset = dataset_cls(  # 构建数据集对象（注释：数据集实例）
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset  # 返回数据集（注释：函数返回）


if __name__ == "__main__":
    main()  # 作为脚本执行时进入主入口（注释：命令行入口）
