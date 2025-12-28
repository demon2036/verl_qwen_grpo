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
  - 定义 Actor 相关配置（PPO/GRPO/SFT）及其子配置（PolicyLoss/RouterReplay）。
  - 提供 FSDP/Megatron（Mcore）两种后端的 ActorConfig 变体。

输入：
  - Hydra/OmegaConf YAML 配置解析后的字典内容。

输出：
  - ActorConfig/FSDPActorConfig/McoreActorConfig 等 dataclass 实例。

关键依赖：
  - `verl.trainer.config.CheckpointConfig`：checkpoint 配置。
  - `verl.workers.config.optimizer.OptimizerConfig`：优化器配置。
  - `verl.workers.config.engine.*`：FSDP/Mcore 引擎配置。

典型用法（最小示例）：
  - `cfg = FSDPActorConfig(rollout_n=1, ppo_mini_batch_size=64)`
  - `cfg.validate(n_gpus=8, train_batch_size=512)`  # 运行期校验。

调用路径概览：
  - `verl/trainer/config/actor/*.yaml`
    -> `verl/trainer/config/config.py`
    -> `ActorConfig`（本模块）
    -> `verl/workers/actor/*`（Actor worker 初始化与训练）。
"""

from dataclasses import dataclass, field  # dataclass 定义
from typing import Any, Optional  # 类型提示

from omegaconf import MISSING  # OmegaConf 必填占位符

from verl.base_config import BaseConfig  # 基础配置基类
from verl.trainer.config import CheckpointConfig  # checkpoint 配置
from verl.utils.profiler.config import ProfilerConfig  # profiler 配置

from .engine import FSDPEngineConfig, McoreEngineConfig  # 引擎配置
from .model import HFModelConfig  # 模型配置
from .optimizer import OptimizerConfig  # 优化器配置

__all__ = ["PolicyLossConfig", "RouterReplayConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]  # 导出列表


@dataclass
class RouterReplayConfig(BaseConfig):
    """
    功能：
      - 控制 MoE（Mixture of Experts）模型的路由记录/回放行为。
      - 支持 deterministic 训练：记录路由或回放路由。

    参数：
      - mode (str): 路由回放模式，支持 "disabled" / "R2" / "R3"。
      - record_file (Optional[str]): 路由记录文件路径（记录时使用）。
      - replay_file (Optional[str]): 路由回放文件路径（回放时使用）。

    返回：
      - RouterReplayConfig 实例。

    副作用：
      - __post_init__ 会校验 mode 合法性。

    异常/边界条件：
      - mode 不在合法集合时抛出 ValueError。

    最小示例（伪输入输出）：
      - 输入：RouterReplayConfig(mode="R2", record_file="route.json")
      - 输出：合法配置对象。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/config/actor.py`
        - 类：`RouterReplayConfig`
      典型调用路径：
        - `verl/trainer/config/actor/*.yaml`
          -> `ActorConfig.router_replay`
      被谁调用：
        - `verl/workers/actor/*`（MoE 路由相关逻辑）
      调用了谁（项目内）：
        - `BaseConfig.__post_init__`（若需要）
      调用了谁（关键外部依赖）：
        - 无
    """

    mode: str = "disabled"  # 路由回放模式
    record_file: Optional[str] = None  # 路由记录输出路径
    replay_file: Optional[str] = None  # 路由回放输入路径

    def __post_init__(self):
        """
        功能：
          - 校验 mode 的合法性。

        参数：
          - self: RouterReplayConfig 实例。

        返回：
          - None。

        副作用：
          - 可能抛出 ValueError。

        异常/边界条件：
          - mode 不合法时抛错。

        最小示例（伪输入输出）：
          - 输入：mode="invalid"
          - 输出：ValueError。

        调用路径依赖：
          所在位置：
            - 路径：`verl/workers/config/actor.py`
            - 方法：`RouterReplayConfig.__post_init__(self)`
          典型调用路径：
            - dataclass 构造时自动调用
          被谁调用：
            - dataclass 构造流程
          调用了谁（项目内）：
            - 无
          调用了谁（关键外部依赖）：
            - 无
        """
        valid_modes = ["disabled", "R2", "R3"]  # 允许的模式
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """
    功能：
      - 配置策略损失（policy loss）的计算方式与相关超参数。
      - 支持 vanilla / clip-cov / kl-cov / gpg 等模式。

    参数：
      - loss_mode (str): 损失模式。
      - clip_cov_ratio (float): clip-cov 需要 clip 的 token 比例。
      - clip_cov_lb/clip_cov_ub (float): clip-cov 的上下界。
      - kl_cov_ratio (float): kl-cov 施加 KL 的 token 比例。
      - ppo_kl_coef (float): KL 惩罚系数。

    返回：
      - PolicyLossConfig 实例。

    副作用：
      - 无。

    异常/边界条件：
      - loss_mode 非法时，上层使用处可能抛出错误。

    最小示例（伪输入输出）：
      - 输入：PolicyLossConfig(loss_mode="vanilla")
      - 输出：配置对象，loss_mode="vanilla"。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/config/actor.py`
        - 类：`PolicyLossConfig`
      典型调用路径：
        - `verl/trainer/config/actor/*.yaml`
          -> `ActorConfig.policy_loss`
      被谁调用：
        - `verl/workers/utils/losses.py::ppo_loss`（读取 loss_mode）
      调用了谁（项目内）：
        - 无
      调用了谁（关键外部依赖）：
        - 无
    """

    loss_mode: str = "vanilla"  # 策略损失模式
    clip_cov_ratio: float = 0.0002  # clip-cov token 比例
    clip_cov_lb: float = 1.0  # clip 下界
    clip_cov_ub: float = 5.0  # clip 上界
    kl_cov_ratio: float = 0.0002  # kl-cov token 比例
    ppo_kl_coef: float = 0.1  # KL 惩罚系数


@dataclass
class ActorConfig(BaseConfig):
    """
    功能：
      - 定义 Actor（策略模型）训练所需的超参数与子配置。
      - 支持 PPO/GRPO 的 batch 组织、loss 聚合、entropy/kl 正则等配置。

    参数（关键字段）：
      - strategy (str): 训练策略（如 "fsdp"/"megatron"），必须指定。
      - ppo_mini_batch_size (int): PPO mini-batch 大小。
      - ppo_micro_batch_size / ppo_micro_batch_size_per_gpu: micro-batch 配置（二选一）。
      - use_dynamic_bsz (bool): 是否启用动态 batch size。
      - clip_ratio / clip_ratio_low / clip_ratio_high: PPO clip 配置。
      - policy_loss (PolicyLossConfig): policy loss 细节配置。
      - loss_agg_mode: loss 聚合模式。
      - entropy_coeff / use_kl_loss / kl_loss_*: 正则项配置。
      - checkpoint/optim/engine/model_config: 子配置。

    返回：
      - ActorConfig 实例。

    副作用：
      - __post_init__ 会对配置进行校验。

    异常/边界条件：
      - strategy/rollout_n 未设置时会断言失败。
      - 同时设置 ppo_micro_batch_size 与 ppo_micro_batch_size_per_gpu 会抛错。

    最小示例（伪输入输出）：
      - 输入：ActorConfig(strategy="fsdp", rollout_n=1, ppo_mini_batch_size=64, ppo_micro_batch_size_per_gpu=4)
      - 输出：合法配置对象。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/config/actor.py`
        - 类：`ActorConfig`
      典型调用路径：
        - `verl/trainer/config/actor/*.yaml`
          -> `verl/trainer/config/config.py`
          -> `ActorConfig`
      被谁调用：
        - `verl/workers/actor/dp_actor.py`（Actor worker 初始化）
      调用了谁（项目内）：
        - `CheckpointConfig` / `OptimizerConfig` / `ProfilerConfig`
      调用了谁（关键外部依赖）：
        - 无
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
