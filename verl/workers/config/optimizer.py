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
模块用途：定义优化器配置结构并提供构建优化器的工厂函数。（注释：模块职责）
输入/输出：
  - 输入：Hydra/OmegaConf 配置、模型参数列表。（注释：输入说明）
  - 输出：结构化配置对象与优化器实例。（注释：输出说明）
关键依赖：dataclasses、omegaconf、torch.optim（动态导入）。（注释：依赖说明）
典型用法（最小示例）：
  - `opt = build_optimizer(model.parameters(), config)`。（注释：最常见用法）
调用路径概览：
  - 训练入口/worker 配置 -> `OptimizerConfig` -> `build_optimizer`。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import warnings  # 注释：弃用配置提醒
from dataclasses import dataclass  # 注释：数据类
from typing import Optional  # 注释：类型标注

# ===== 第三方依赖导入 =====
from omegaconf import MISSING  # 注释：OmegaConf 占位标记

# ===== 项目内依赖导入 =====
from verl.base_config import BaseConfig  # 注释：配置基类

__all__ = [  # 注释：导出 API 列表
    "OptimizerConfig",
    "FSDPOptimizerConfig",
    "McoreOptimizerConfig",
    "build_optimizer",
    "VeOmniOptimizerConfig",
]


@dataclass
class OptimizerConfig(BaseConfig):
    """
    基础优化器配置。（注释：类用途）

    参数：
      - lr (float)：学习率。（注释：参数说明）
      - lr_warmup_steps_ratio (float)：warmup 比例。（注释：参数说明）
      - total_training_steps (int)：总训练步数（运行时注入）。（注释：参数说明）
      - weight_decay (float)：权重衰减系数。（注释：参数说明）
      - lr_warmup_steps (Optional[int])：warmup 步数；若未设置则使用比例。（注释：参数说明）
    返回：配置对象。（注释：返回说明）
    副作用：无。（注释：仅数据结构）
    异常/边界条件：lr 必须指定；grad_clip 已弃用。（注释：边界条件）
    最小示例：
      - 输入：OptimizerConfig(lr=1e-4, weight_decay=0.01)。（注释：示例输入）
      - 输出：配置对象。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/workers/config/optimizer.py::OptimizerConfig`。（注释：定位）
      - 典型调用路径：Hydra 配置加载 -> `OptimizerConfig` -> `build_optimizer`。（注释：典型路径）
      - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`BaseConfig`。（注释：内部依赖）
      - 调用了谁（外部依赖）：dataclasses。（注释：外部依赖）
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        """
        初始化后校验与弃用字段转换。（注释：方法用途）

        参数：无。（注释：dataclass 回调）
        返回：无。（注释：仅修改自身字段）
        副作用：可能触发 DeprecationWarning。（注释：副作用）
        异常/边界条件：lr 不能为 MISSING。（注释：边界条件）
        最小示例：
          - 输入：grad_clip=1.0。（注释：示例输入）
          - 输出：clip_grad 被覆盖。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/workers/config/optimizer.py::OptimizerConfig.__post_init__`。（注释：定位）
          - 典型调用路径：Hydra 创建配置对象时自动调用。（注释：典型路径）
          - 被谁调用：dataclass 生命周期。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：`warnings.warn`。（注释：外部依赖）
        """
        assert self.lr != MISSING  # 注释：确保学习率已设置
        if self.grad_clip is not None:  # 注释：兼容旧字段 grad_clip
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip  # 注释：将旧字段迁移到新字段


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """
    VeOmni 引擎的优化器配置。（注释：类用途）

    参数：
      - optimizer (str)：优化器名，默认 "adamw"。（注释：参数说明）
      - lr/lr_min/lr_start：学习率相关配置。（注释：参数说明）
      - lr_decay_ratio (float)：学习率衰减比例。（注释：参数说明）
      - lr_scheduler_type (str)："constant"/"cosine"。（注释：参数说明）
    返回：配置对象。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/workers/config/optimizer.py::VeOmniOptimizerConfig`。（注释：定位）
      - 典型调用路径：VeOmni 引擎配置加载 -> `VeOmniOptimizerConfig`。（注释：典型路径）
      - 被谁调用：`verl/workers/engine/veomni/transformer_impl.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`OptimizerConfig`。（注释：内部依赖）
      - 调用了谁（外部依赖）：dataclasses。（注释：外部依赖）
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """
    FSDP 场景的优化器配置。（注释：类用途）

    参数：
      - optimizer (str)：优化器类名。（注释：参数说明）
      - optimizer_impl (str)：优化器模块路径。（注释：参数说明）
      - min_lr_ratio/lr_scheduler_type/num_cycles：学习率调度配置。（注释：参数说明）
    返回：配置对象。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/workers/config/optimizer.py::FSDPOptimizerConfig`。（注释：定位）
      - 典型调用路径：FSDP 配置加载 -> `FSDPOptimizerConfig` -> `build_optimizer`。（注释：典型路径）
      - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/trainer/fsdp_sft_trainer.py`。（注释：调用方）
      - 调用了谁（项目内）：`OptimizerConfig`。（注释：内部依赖）
      - 调用了谁（外部依赖）：dataclasses。（注释：外部依赖）
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        """
        初始化后处理：弃用字段转换与调度类型校验。（注释：方法用途）

        参数：无。（注释：dataclass 回调）
        返回：无。（注释：仅修改自身字段）
        副作用：可能触发 DeprecationWarning。（注释：副作用）
        异常/边界条件：lr_scheduler_type 必须为 constant/cosine。（注释：边界条件）
        最小示例：
          - 输入：warmup_style="cosine"。（注释：示例输入）
          - 输出：lr_scheduler_type 被设置为 "cosine"。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/workers/config/optimizer.py::FSDPOptimizerConfig.__post_init__`。（注释：定位）
          - 典型调用路径：Hydra 创建配置对象时自动调用。（注释：典型路径）
          - 被谁调用：dataclass 生命周期。（注释：调用方）
          - 调用了谁（项目内）：`OptimizerConfig.__post_init__`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`warnings.warn`。（注释：外部依赖）
        """
        if self.warmup_style is not None:  # 注释：兼容旧字段
            assert self.warmup_style in ["constant", "cosine"]  # 注释：校验取值
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style  # 注释：迁移旧字段
        assert self.lr_scheduler_type in ["constant", "cosine"]  # 注释：最终校验
        return super().__post_init__()  # 注释：调用父类校验


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """
    Megatron-Core（mcore）优化器配置。（注释：类用途）

    参数：包含学习率衰减、权重衰减、WSD 等调度字段。（注释：参数说明）
    返回：配置对象。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/workers/config/optimizer.py::McoreOptimizerConfig`。（注释：定位）
      - 典型调用路径：Megatron worker 配置加载 -> `McoreOptimizerConfig`。（注释：典型路径）
      - 被谁调用：`verl/workers/megatron_workers.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`OptimizerConfig`。（注释：内部依赖）
      - 调用了谁（外部依赖）：dataclasses。（注释：外部依赖）
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """
    根据配置动态构建优化器实例。（注释：函数用途）

    参数：
      - parameters：模型参数迭代器/列表。（注释：输入说明）
      - config (FSDPOptimizerConfig)：优化器配置。（注释：输入说明）
    返回：
      - optimizer：优化器实例。（注释：返回说明）
    副作用：动态导入优化器模块。（注释：副作用）
    异常/边界条件：
      - 模块导入失败抛 ImportError。（注释：边界条件）
      - 类名不存在抛 AttributeError。（注释：边界条件）
    最小示例：
      - 输入：optimizer_impl="torch.optim", optimizer="AdamW"。（注释：示例输入）
      - 输出：torch.optim.AdamW 实例。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/workers/config/optimizer.py::build_optimizer`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `build_optimizer`。（注释：典型路径）
      - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py`。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`importlib.import_module`。（注释：外部依赖）
    """
    import importlib

    optimizer_args = {  # 注释：构造基础优化器参数
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    optimizer_name_lower = config.optimizer.lower()  # 注释：小写化用于判断类型
    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:  # 注释：Adam 系列添加 betas
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:  # 注释：合并额外参数
        optimizer_args.update(config.override_optimizer_config)

    try:
        module = importlib.import_module(config.optimizer_impl)  # 注释：动态导入优化器模块
        optimizer_cls = getattr(module, config.optimizer)  # 注释：获取优化器类
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e  # 注释：包装导入异常
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e  # 注释：类名不存在

    return optimizer_cls(parameters, **optimizer_args)  # 注释：实例化并返回优化器
