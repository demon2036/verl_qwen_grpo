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
模块用途：
  - 定义 Reward Model 相关的配置数据结构（dataclass 形式）。
  - 支持奖励模型启用、资源池参数、以及沙箱融合（sandbox fusion）配置。

输入：
  - 通常由 Hydra/OmegaConf 将 YAML 配置解析后转换为本模块的 dataclass。

输出：
  - RewardModelConfig / SandboxFusionConfig 实例，供 worker/trainer 使用。

关键依赖：
  - `verl.base_config.BaseConfig`：基础配置基类与后处理逻辑。
  - `verl.workers.config.model.HFModelConfig`：模型相关配置。
  - `verl.workers.config.rollout.RolloutConfig`：rollout 相关配置。

典型用法（最小示例）：
  - `cfg = RewardModelConfig(enable=True, n_gpus_per_node=1)`
  - `cfg.sandbox_fusion.url = "http://..."`  # 启用沙箱融合。

调用路径概览：
  - `verl/trainer/config/*.yaml`
    -> `verl/trainer/config/config.py`（OmegaConf -> dataclass）
    -> `RewardModelConfig`（本模块）
    -> `verl/workers/reward_model/*`（实例化 reward model worker）。
"""

import logging  # 日志模块
import os  # 读取环境变量
from dataclasses import dataclass, field  # dataclass 定义
from typing import Optional  # Optional 类型提示

from verl.base_config import BaseConfig  # 配置基类，提供通用校验/后处理

from .model import HFModelConfig  # 模型配置（HF 模型）
from .rollout import RolloutConfig  # rollout 配置

__all__ = ["SandboxFusionConfig", "RewardModelConfig"]  # 对外导出符号

logger = logging.getLogger(__name__)  # 获取模块级 logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 通过环境变量控制日志等级


@dataclass
class SandboxFusionConfig(BaseConfig):
    """
    功能：
      - 描述“云/本地沙箱融合执行”的参数。
      - 主要用于奖励函数在隔离环境中执行的资源限制与并发控制。

    参数：
      - url (Optional[str]): 沙箱执行的 HTTP/本地入口 URL。
      - max_concurrent (int): 允许的最大并发请求数。
      - memory_limit_mb (int): 单个沙箱进程的内存上限（MB）。

    返回：
      - SandboxFusionConfig 实例。

    副作用：
      - 无（纯配置对象）。

    异常/边界条件：
      - max_concurrent <= 0 或 memory_limit_mb <= 0 时配置意义不合法（需由上层校验）。

    最小示例（伪输入输出）：
      - 输入：SandboxFusionConfig(url="http://localhost:8080", max_concurrent=8)
      - 输出：包含 url/max_concurrent/memory_limit_mb 的配置对象。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/config/reward_model.py`
        - 类：`SandboxFusionConfig`
      典型调用路径：
        - `verl/trainer/config/reward_model/*.yaml`
          -> `RewardModelConfig.sandbox_fusion`
      被谁调用：
        - `verl/workers/reward_model/*`（奖励模型/沙箱执行逻辑）
      调用了谁（项目内）：
        - `BaseConfig.__post_init__`（若需要）
      调用了谁（关键外部依赖）：
        - 无
    """

    url: Optional[str] = None  # 沙箱执行入口 URL（None 表示不启用融合）
    max_concurrent: int = 64  # 允许的最大并发请求数
    memory_limit_mb: int = 1024  # 单进程内存上限（MB）


@dataclass
class RewardModelConfig(BaseConfig):
    """
    功能：
      - 描述 Reward Model worker 的总体配置，包括资源池、模型、rollout 与沙箱融合参数。

    参数：
      - reward_manager (Optional[str]): 旧字段（已废弃），保留兼容。
      - enable (bool): 是否启用 reward model worker。
      - enable_resource_pool (bool): 是否启用独立资源池。
      - n_gpus_per_node (int): 每节点 GPU 数量。
      - nnodes (int): 节点数。
      - rollout (RolloutConfig): rollout 相关配置。
      - model (HFModelConfig): HuggingFace 模型配置。
      - sandbox_fusion (SandboxFusionConfig): 沙箱融合配置。

    返回：
      - RewardModelConfig 实例。

    副作用：
      - __post_init__ 中可能输出 deprecation 警告日志。

    异常/边界条件：
      - 当 enable=True 但 n_gpus_per_node/nnodes 为 0 时，上层可能无法分配资源。

    最小示例（伪输入输出）：
      - 输入：RewardModelConfig(enable=True, n_gpus_per_node=1, nnodes=1)
      - 输出：包含 rollout/model/sandbox_fusion 子配置的对象。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/config/reward_model.py`
        - 类：`RewardModelConfig`
      典型调用路径：
        - `verl/trainer/config/reward_model/*.yaml`
          -> `verl/trainer/config/config.py`（解析为 dataclass）
          -> `RewardModelConfig`
      被谁调用：
        - `verl/workers/reward_model/*`（reward model worker 初始化）
      调用了谁（项目内）：
        - `RolloutConfig` / `HFModelConfig` / `SandboxFusionConfig`
      调用了谁（关键外部依赖）：
        - 无
    """

    _mutable_fields = BaseConfig._mutable_fields  # 允许被 OmegaConf 修改的字段集合

    reward_manager: Optional[str] = None  # 旧字段（已废弃，保留兼容）

    enable: bool = False  # 是否启用 reward model worker
    enable_resource_pool: bool = False  # 是否单独开资源池
    n_gpus_per_node: int = 0  # 每节点 GPU 数
    nnodes: int = 0  # 节点数

    # reward model args（子配置）
    rollout: RolloutConfig = field(default_factory=RolloutConfig)  # rollout 配置
    model: HFModelConfig = field(default_factory=HFModelConfig)  # 模型配置
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)  # 沙箱融合配置

    def __post_init__(self):
        """
        功能：
          - 调用 BaseConfig 的通用后处理逻辑。
          - 检测废弃字段 reward_manager，并打印警告。

        参数：
          - self: RewardModelConfig 实例。

        返回：
          - None。

        副作用：
          - 可能记录 warning 日志。

        异常/边界条件：
          - 无显式异常；若 logger 配置异常可能抛出日志相关错误。

        最小示例（伪输入输出）：
          - 输入：RewardModelConfig(reward_manager="naive")
          - 输出：打印 deprecation 警告，不影响配置对象可用性。

        调用路径依赖：
          所在位置：
            - 路径：`verl/workers/config/reward_model.py`
            - 方法：`RewardModelConfig.__post_init__(self)`
          典型调用路径：
            - `RewardModelConfig(...)` 构造时自动触发
          被谁调用：
            - dataclass 构造流程自动调用
          调用了谁（项目内）：
            - `BaseConfig.__post_init__`
          调用了谁（关键外部依赖）：
            - `logging.Logger.warning`
        """
        super().__post_init__()  # 先执行基类后处理
        if self.reward_manager is not None:
            # 旧字段兼容提示
            logger.warning(
                f"`reward_model.reward_manager` is deprecated, but got value {self.reward_manager}. "
                "Please use `reward_manager.name instead. "
                "See `verl/trainer/config/config.py:RewardManagerConfig` for more details."
            )
