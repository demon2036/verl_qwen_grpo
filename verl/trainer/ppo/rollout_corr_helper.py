# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
模块用途：提供 rollout 校正（IS/RS/指标诊断）工具，缓解 rollout 与训练策略不一致带来的 off-policy 问题。（注释：模块级用途概述）
输入/输出：（注释：模块级 I/O 概览）
  - 输入：old_log_prob、rollout_log_prob、response_mask 等张量。（注释：输入来源）
  - 输出：IS 权重、拒绝采样 mask、off-policy 指标字典。（注释：输出形态）
关键依赖：（注释：列出关键依赖）
  - torch 张量运算、verl.utils.torch_functional 掩码统计。（注释：关键依赖）
  - RolloutCorrectionConfig / PolicyLossConfig。（注释：配置依赖）
典型用法（最小示例）：（注释：最小用法）
  >>> weights, mask, metrics = compute_rollout_correction_and_rejection_mask(old, roll, resp_mask)  # 示例
调用路径概览：（注释：入口链路）
  - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> `compute_rollout_correction_and_add_to_batch`。（注释：训练流程）

核心能力：（注释：功能概览）
  1) 重要性采样（IS）：token/sequence 粒度的权重计算与截断。（注释：IS 功能）
  2) 拒绝采样（RS）：按阈值过滤 outlier。（注释：RS 功能）
  3) 灾难性 token veto：低权重 token 直接拒绝整条序列。（注释：安全机制）
  4) off-policy 诊断指标：KL/PPL/χ² 等。（注释：指标统计）

参考资料：（注释：保留参考链接便于深入）
  - When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch（RL 失稳分析）
  - Off-policy RL（IS 理论基础）
"""  # 注释：模块 docstring 结束

# 类型注解（注释：用于函数签名）
from typing import Any, Optional

# 第三方依赖（注释：张量运算）
import torch

# 项目内依赖（注释：掩码统计工具）
import verl.utils.torch_functional as verl_F
# 项目内依赖（注释：DataProto 容器）
from verl.protocol import DataProto
# 项目内依赖（注释：rollout 校正配置）
from verl.trainer.config.algorithm import RolloutCorrectionConfig
# 项目内依赖（注释：策略损失配置）
from verl.workers.config.actor import PolicyLossConfig

# 安全指数范围，避免 exp 上下溢（注释：数值稳定性）
# exp(20) ≈ 4.85e8，exp(-20) ≈ 2e-9（注释：阈值意义）
SAFETY_BOUND = 20.0  # 注释：全局安全边界


def compute_rollout_rejection_mask(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str = "token",
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算拒绝采样（RS）mask，用于过滤 outlier token/sequence。（注释：函数用途）

    参数：（注释：参数说明）
      - log_ratio (torch.Tensor): log(π_train/π_rollout)，形状 (B, T)。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask，形状 (B, T)。（注释：输入含义）
      - rollout_rs (str): 聚合级别："token"/"sequence"/"geometric"。（注释：输入含义）
      - rollout_rs_threshold (float): 上阈值（必须提供）。（注释：输入含义）
      - rollout_rs_threshold_lower (float|None): 下阈值，默认 1/上阈值。（注释：输入含义）
    返回：（注释：返回值说明）
      - modified_response_mask (torch.Tensor): 应用 RS 的 mask。（注释：输出含义）
      - metrics (dict[str,float]): RS 统计指标。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无（仅计算并返回）。（注释：无状态修改）
    异常/边界条件：（注释：异常说明）
      - rollout_rs 无效或未提供阈值时抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> mask, metrics = compute_rollout_rejection_mask(log_ratio, resp_mask, rollout_rs="token", rollout_rs_threshold=2.0)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_rollout_rejection_mask(...)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_rollout_correction_and_rejection_mask` -> `compute_rollout_rejection_mask`。（注释：内部调用）
      被谁调用
      --------
      - 本文件 `compute_rollout_correction_and_rejection_mask`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `compute_rs_metrics` / `verl_F.masked_*`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.exp` / `torch.clamp`。（注释：外部依赖）
    """
    # 校验输入参数（注释：检查 rs 模式与阈值）
    valid_rs_levels = {"token", "sequence", "geometric"}
    if rollout_rs not in valid_rs_levels:
        raise ValueError(f"Invalid rollout_rs: {rollout_rs}. Must be one of {valid_rs_levels}.")
    if rollout_rs_threshold is None:
        raise ValueError("rollout_rs_threshold must be provided for rejection sampling.")

    # 设置下阈值（注释：默认取上阈值倒数）
    upper_threshold = rollout_rs_threshold
    lower_threshold = rollout_rs_threshold_lower if rollout_rs_threshold_lower is not None else 1.0 / upper_threshold

    # 计算 IS 权重（注释：按不同聚合级别）
    if rollout_rs == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_rs == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to (batch_size, seq_length)

    elif rollout_rs == "geometric":
        # Sequence-level geometric mean: exp(mean(log ratios))
        log_ratio_mean: torch.Tensor = verl_F.masked_mean(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_mean

        log_ratio_mean_safe: torch.Tensor = torch.clamp(log_ratio_mean, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_mean_safe).expand_as(log_ratio)

    else:
        raise ValueError(f"Unsupported rollout_rs: {rollout_rs}")

    # 生成 outlier mask（注释：阈值区间内为 1）
    mask: torch.Tensor = (rollout_is_weights >= lower_threshold) & (rollout_is_weights <= upper_threshold)
    mask = mask.float()

    # 计算 RS 指标（注释：统计分布与阈值比例）
    metrics: dict[str, float] = compute_rs_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=upper_threshold,
        rollout_rs_threshold_lower=lower_threshold,
    )

    # 统计 token/sequence 拒绝比例（注释：用于监控）
    # rollout_rs_masked_fraction: fraction of tokens rejected (unified for all modes)
    metrics["rollout_rs_masked_fraction"] = verl_F.masked_mean(1 - mask, response_mask).item()

    # rollout_rs_seq_masked_fraction: fraction of sequences rejected (mode-dependent)
    if rollout_rs == "token":
        # Token-level aggregation: sequence is rejected if any token is rejected
        seq_has_masked: torch.Tensor = verl_F.masked_sum(1 - mask, response_mask, axis=-1) > 0
        metrics["rollout_rs_seq_masked_fraction"] = seq_has_masked.float().mean().item()
    else:
        # Sequence-level aggregation: check first token's mask (all tokens in sequence have same mask)
        metrics["rollout_rs_seq_masked_fraction"] = (1 - mask[:, 0]).mean().item()

    # 将 RS mask 应用到 response_mask（注释：剔除 outlier）
    modified_response_mask: torch.Tensor = response_mask * mask

    return modified_response_mask, metrics


def compute_rs_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_rs: str,
    rollout_rs_threshold: float,
    rollout_rs_threshold_lower: float,
) -> dict[str, float]:
    """
    计算拒绝采样相关统计指标。（注释：函数用途）

    参数：（注释：参数说明）
      - rollout_is_weights (torch.Tensor): IS 权重（已截断/裁剪），形状 (B,T)。（注释：输入含义）
      - log_ratio_for_metrics (torch.Tensor): 未裁剪的 log_ratio。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask。（注释：输入含义）
      - rollout_rs (str): 聚合级别（token/sequence/geometric）。（注释：输入含义）
      - rollout_rs_threshold (float): 上阈值。（注释：输入含义）
      - rollout_rs_threshold_lower (float): 下阈值。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, float]: RS 指标字典。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - response_mask 全为 0 会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> metrics = compute_rs_metrics(weights, log_ratio, mask, "token", 2.0, 0.5)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_rs_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_rollout_rejection_mask` -> `compute_rs_metrics`。（注释：内部调用）
      被谁调用
      --------
      - 本文件 `compute_rollout_rejection_mask`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `verl_F.masked_mean` / `torch.clamp`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.log` / `torch.exp`。（注释：外部依赖）
    """
    # 响应 mask 必须包含有效 token（注释：基础校验）
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}  # 注释：指标容器
    device: torch.device = rollout_is_weights.device  # 注释：设备信息

    # 预计算 log 阈值（注释：用于稳定的阈值比较）
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_rs_threshold_lower, device=device))

    # 根据聚合级别统计（注释：token/sequence 分支）
    if rollout_rs in ["sequence", "geometric"]:
        # Sequence-level aggregation: use log-space for accurate max/min/threshold checks
        # True max/min (unclamped) converted with safety bounds
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_rs_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_rs_min"] = torch.exp(log_min).item()

        # Mean uses clamped weights to avoid overflow
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        # Both sequence and geometric modes operate at sequence level (batch_size, 1)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_rs_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from clamped weights
        metrics["rollout_rs_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_rs_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_rs_threshold_lower
        metrics["rollout_rs_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_rs_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens first)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_rs_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_rs_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # 计算标准差（注释：使用裁剪权重提升稳定性）
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        # Clamp weights to threshold range to avoid squaring extreme values
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_rs_threshold_lower, max=rollout_rs_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        # Variance = E[X²] - (E[X])² (masked to valid tokens)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_rs_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_rs_std"] = 0.0

    # 计算有效样本量 ESS（注释：衡量权重方差）
    # ESS = 1 / E[(w_i / E[w_i])²] (using clamped weights for stability)
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_rs_threshold_lower, max=rollout_rs_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_rs_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # 序列级指标（注释：对每条序列的平均权重统计）
    if rollout_is_weights.dim() > 1:
        # Mean weight per sequence (masked to valid tokens)
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_rs_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_rs_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_rs_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_rs_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_rs_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_rs_seq_fraction_high"] = (seq_mean_weights > rollout_rs_threshold).float().mean().item()
        metrics["rollout_rs_seq_fraction_low"] = (seq_mean_weights < rollout_rs_threshold_lower).float().mean().item()

    return metrics  # 注释：返回 RS 指标


def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算重要性采样（IS）权重并进行截断/归一化。（注释：函数用途）

    参数：（注释：参数说明）
      - log_ratio (torch.Tensor): log(π_train/π_rollout)，形状 (B,T)。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask，形状 (B,T)。（注释：输入含义）
      - rollout_is (str): IS 聚合级别："token" 或 "sequence"。（注释：输入含义）
      - rollout_is_threshold (float): 权重截断上阈值。（注释：输入含义）
      - rollout_is_batch_normalize (bool): 是否按 batch 归一化到均值 1。（注释：输入含义）
    返回：（注释：返回值说明）
      - rollout_is_weights (torch.Tensor): 截断后的 IS 权重。（注释：输出含义）
      - metrics (dict[str,float]): 权重统计指标。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - rollout_is 无效或阈值<=0 会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> weights, metrics = compute_rollout_correction_weights(log_ratio, mask, "token", 2.0)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_rollout_correction_weights(...)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_rollout_correction_and_rejection_mask` -> `compute_rollout_correction_weights`。（注释：内部调用）
      被谁调用
      --------
      - 本文件 `compute_rollout_correction_and_rejection_mask`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `compute_is_metrics` / `verl_F.masked_mean`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.exp` / `torch.clamp`。（注释：外部依赖）
    """
    # 参数校验（注释：rollout_is 必须合法）
    valid_is_levels = {"token", "sequence"}
    if rollout_is not in valid_is_levels:
        raise ValueError(f"Invalid rollout_is: {rollout_is}. Must be one of {valid_is_levels}.")
    if rollout_is_threshold <= 0:
        raise ValueError(f"rollout_is_threshold must be positive, got {rollout_is_threshold}.")

    # 计算 IS 权重（注释：按不同聚合级别）
    if rollout_is == "token":
        # Per-token IS weight: exp(log(π_train/π_rollout)) with safety clamp
        log_ratio_for_metrics: torch.Tensor = log_ratio
        log_ratio_safe: torch.Tensor = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights: torch.Tensor = torch.exp(log_ratio_safe)

    elif rollout_is == "sequence":
        # Sequence-level IS weight: product of token ratios (exp(sum(log ratios)))
        log_ratio_sum: torch.Tensor = verl_F.masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(
            -1
        )  # Shape: (batch_size, 1)
        log_ratio_for_metrics = log_ratio_sum

        log_ratio_sum_safe: torch.Tensor = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rollout_is_weights = torch.exp(log_ratio_sum_safe).expand_as(log_ratio)  # Broadcast to sequence length

    else:
        raise ValueError(f"Unsupported rollout_is: {rollout_is}")

    # 对 padding token 置零（注释：避免无效 token 影响统计）
    rollout_is_weights = rollout_is_weights * response_mask

    # 计算 IS 权重指标（注释：在截断前统计比例）
    metrics: dict[str, float] = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
    )

    # 截断极端权重（注释：TIS）
    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold)

    # 阻断梯度（注释：IS 权重不应反向传播）
    # IS weights change the measure, not the objective. See §3.2.2 in docs/algo/rollout_corr_math.md
    rollout_is_weights = rollout_is_weights.detach()

    # 可选：按 batch 归一化（注释：均值归一化）
    if rollout_is_batch_normalize:
        # Compute mean based on aggregation level
        mask_float = response_mask.to(dtype=rollout_is_weights.dtype)
        if rollout_is == "token":
            # Token-level: normalize over all token weights
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(rollout_is_weights, mask_float)
            else:
                weights_mean = verl_F.masked_mean(rollout_is_weights, response_mask)
        elif rollout_is == "sequence":
            # Sequence-level: normalize over sequence weights (one weight per sequence)
            # For each sequence, compute mean over valid tokens (they all have the same weight)
            # then average across sequences
            seq_weights = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)  # (batch_size,)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=rollout_is_weights.dtype)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                weights_mean = verl_F.distributed_masked_mean(seq_weights, seq_mask)
            else:
                weights_mean = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)
        else:
            raise ValueError(f"Unsupported rollout_is: {rollout_is}")

        # 归一化到均值 1（注释：避免除零）
        if weights_mean > 1e-8:
            rollout_is_weights = rollout_is_weights / weights_mean
            metrics["rollout_is_batch_norm_factor"] = weights_mean.item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return rollout_is_weights, metrics  # 注释：返回权重与指标


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
    """
    计算 IS 权重的统计指标（截断后）。（注释：函数用途）

    参数：（注释：参数说明）
      - rollout_is_weights (torch.Tensor): 截断后的 IS 权重，形状 (B,T)。（注释：输入含义）
      - log_ratio_for_metrics (torch.Tensor): 未裁剪的 log_ratio。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask。（注释：输入含义）
      - rollout_is (str): IS 聚合级别。（注释：输入含义）
      - rollout_is_threshold (float): 上阈值。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str,float]: IS 指标字典。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - response_mask 全为 0 会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> metrics = compute_is_metrics(weights, log_ratio, mask, "token", 2.0)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_is_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_rollout_correction_weights` -> `compute_is_metrics`。（注释：内部调用）
      被谁调用
      --------
      - 本文件 `compute_rollout_correction_weights`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `verl_F.masked_mean` / `torch.clamp`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.log` / `torch.exp`。（注释：外部依赖）
    """
    # 基础校验（注释：必须包含有效 token）
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}  # 注释：指标容器
    device: torch.device = rollout_is_weights.device  # 注释：设备信息
    # 下阈值为上阈值倒数（注释：对称区间）
    rollout_is_threshold_lower: float = 1.0 / rollout_is_threshold

    # 预计算 log 阈值（注释：准确阈值比较）
    log_threshold_upper: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold, device=device))
    log_threshold_lower: torch.Tensor = torch.log(torch.tensor(rollout_is_threshold_lower, device=device))

    # 按聚合级别计算指标（注释：sequence/token 分支）
    if rollout_is == "sequence":
        # Sequence-level aggregation: use log-space for unclamped stats
        log_max: torch.Tensor = log_ratio_for_metrics.max()
        log_min: torch.Tensor = log_ratio_for_metrics.min()
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_is_min"] = torch.exp(log_min).item()

        # Mean uses truncated weights to avoid overflow
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of weights exceeding thresholds (log-space for accuracy)
        exceeds_upper: torch.Tensor = log_ratio_for_metrics > log_threshold_upper
        below_lower: torch.Tensor = log_ratio_for_metrics < log_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = exceeds_upper.float().mean().item()
        metrics["rollout_is_ratio_fraction_low"] = below_lower.float().mean().item()

    else:  # token-level
        # Token-level aggregation: compute directly from truncated weights
        metrics["rollout_is_mean"] = verl_F.masked_mean(rollout_is_weights, response_mask).item()

        # Fraction of tokens exceeding thresholds
        rollout_is_above_threshold: torch.Tensor = rollout_is_weights > rollout_is_threshold
        rollout_is_below_threshold: torch.Tensor = rollout_is_weights < rollout_is_threshold_lower
        metrics["rollout_is_ratio_fraction_high"] = verl_F.masked_mean(
            rollout_is_above_threshold.float(), response_mask
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = verl_F.masked_mean(
            rollout_is_below_threshold.float(), response_mask
        ).item()

        # Max/min (mask out padding tokens)
        mask_bool: torch.Tensor = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    # Compute standard deviation (using clamped weights for stability)
    mask_count: torch.Tensor = response_mask.sum()
    if mask_count > 1:
        weights_for_std: torch.Tensor = rollout_is_weights.clamp(
            min=rollout_is_threshold_lower, max=rollout_is_threshold
        )
        mean_clamped: torch.Tensor = verl_F.masked_mean(weights_for_std, response_mask)
        rollout_is_var: torch.Tensor = (
            verl_F.masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        )
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_is_std"] = 0.0

    # Compute Effective Sample Size (ESS) for truncated weights
    weights_for_ess: torch.Tensor = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
    mean_for_ess: torch.Tensor = verl_F.masked_mean(weights_for_ess, response_mask)
    is_weights_normalized: torch.Tensor = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
    metrics["rollout_is_eff_sample_size"] = (
        1.0 / verl_F.masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    # Add sequence-level metrics if weights have batch dimension
    if rollout_is_weights.dim() > 1:
        seq_mean_weights: torch.Tensor = verl_F.masked_mean(rollout_is_weights, response_mask, axis=-1)

        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_is_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min().item()

        # Sequence deviation from ideal weight (1.0)
        seq_deviation: torch.Tensor = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max().item()

        # Fraction of sequences with extreme weights
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean().item()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean().item()

    return metrics


def compute_rollout_correction_and_rejection_mask(
    old_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: Optional[float] = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = 2.0,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    rollout_is_batch_normalize: bool = False,
) -> tuple[Optional[DataProto], torch.Tensor, dict[str, float]]:
    """
    统一接口：计算 IS 权重 + RS mask + off-policy 指标。（注释：函数用途）

    参数：（注释：参数说明）
      - old_log_prob (torch.Tensor): 训练策略 log prob，形状 (B,T)。（注释：输入含义）
      - rollout_log_prob (torch.Tensor): rollout 策略 log prob，形状 (B,T)。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask。（注释：输入含义）
      - rollout_is (str|None): IS 聚合级别，None 表示关闭。（注释：输入含义）
      - rollout_is_threshold (float|None): IS 截断阈值。（注释：输入含义）
      - rollout_rs (str|None): RS 聚合级别，None 表示关闭。（注释：输入含义）
      - rollout_rs_threshold (float|None): RS 上阈值。（注释：输入含义）
      - rollout_rs_threshold_lower (float|None): RS 下阈值。（注释：输入含义）
      - rollout_token_veto_threshold (float|None): 灾难性 token veto 阈值。（注释：输入含义）
      - rollout_is_batch_normalize (bool): 是否 batch 归一化。（注释：输入含义）
    返回：（注释：返回值说明）
      - rollout_is_weights_proto (DataProto|None): IS 权重 DataProto。（注释：输出含义）
      - modified_response_mask (torch.Tensor): 更新后的 mask。（注释：输出含义）
      - metrics (dict[str,float]): 以 rollout_corr/ 前缀标记的指标。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无（不修改输入张量）。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - mask/shape 不匹配会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> weights, mask, metrics = compute_rollout_correction_and_rejection_mask(old, roll, resp_mask, rollout_is="token")  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_rollout_correction_and_rejection_mask(...)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_rollout_correction_and_add_to_batch` -> 本函数。（注释：训练调用）
      被谁调用
      --------
      - 本文件 `compute_rollout_correction_and_add_to_batch`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `compute_rollout_correction_weights` / `compute_rollout_rejection_mask` / `compute_offpolicy_metrics`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.log` / `torch.exp`。（注释：外部依赖）
    """
    # 校验输入 mask 与形状（注释：保障一致性）
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")
    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            f"old_log_prob shape {old_log_prob.shape} does not match rollout_log_prob shape {rollout_log_prob.shape}."
        )
    if old_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"log_prob shape {old_log_prob.shape} does not match response_mask shape {response_mask.shape}."
        )

    # Step 1: 计算 log_ratio（注释：log(π_train/π_rollout)）
    log_ratio: torch.Tensor = old_log_prob - rollout_log_prob
    device: torch.device = log_ratio.device
    metrics: dict[str, float] = {}

    # Step 2: 计算 IS 权重（注释：可选）
    rollout_is_weights: Optional[torch.Tensor] = None
    if rollout_is is not None and rollout_is_threshold is not None:
        rollout_is_weights, is_metrics = compute_rollout_correction_weights(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_is=rollout_is,
            rollout_is_threshold=rollout_is_threshold,
            rollout_is_batch_normalize=rollout_is_batch_normalize,
        )
        metrics.update(is_metrics)

    # Step 3: 计算 RS mask（注释：可选）
    modified_response_mask: torch.Tensor = response_mask.clone()
    if rollout_rs is not None:
        if rollout_rs_threshold is None:
            raise ValueError(
                "rollout_rs_threshold must be explicitly provided when rollout_rs is enabled. "
                "Set rollout_rs_threshold to the desired threshold value."
            )
        modified_response_mask, rs_metrics = compute_rollout_rejection_mask(
            log_ratio=log_ratio,
            response_mask=response_mask,
            rollout_rs=rollout_rs,
            rollout_rs_threshold=rollout_rs_threshold,
            rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        )
        metrics.update(rs_metrics)

    # Step 4: 灾难性 token veto（注释：序列级拒绝）
    if rollout_token_veto_threshold is not None:
        if rollout_token_veto_threshold <= 0:
            raise ValueError(f"rollout_token_veto_threshold must be positive, got {rollout_token_veto_threshold}.")

        # Compute log threshold for numerical stability
        log_veto_threshold: torch.Tensor = torch.log(torch.tensor(rollout_token_veto_threshold, device=device))
        # Identify catastrophic tokens (log ratio below threshold + valid mask)
        catastrophic_tokens: torch.Tensor = (log_ratio < log_veto_threshold) & response_mask.bool()
        # Check if sequence contains any catastrophic token
        has_catastrophic: torch.Tensor = catastrophic_tokens.any(dim=-1, keepdim=True)
        # Create veto mask (0=reject sequence, 1=keep)
        veto_mask: torch.Tensor = (~has_catastrophic).float()

        # Track veto metrics
        metrics["rollout_is_veto_fraction"] = has_catastrophic.float().mean().item()
        metrics["rollout_is_catastrophic_token_fraction"] = verl_F.masked_mean(
            catastrophic_tokens.float(), response_mask
        ).item()

        # Apply veto to response mask (overrides previous rejection)
        modified_response_mask = modified_response_mask * veto_mask
    else:
        # Add placeholder metrics if veto is disabled
        metrics["rollout_is_veto_fraction"] = 0.0
        metrics["rollout_is_catastrophic_token_fraction"] = 0.0

    # Step 5: 计算 off-policy 指标（注释：KL/PPL/χ²）
    offpolicy_metrics: dict[str, float] = compute_offpolicy_metrics(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )
    metrics.update(offpolicy_metrics)

    # Step 6: 指标前缀统一（注释：便于日志分组）
    metrics_scalar: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_scalar[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_scalar[f"rollout_corr/{key}"] = value

    # Step 7: 将 IS 权重封装为 DataProto（注释：与接口一致）
    rollout_is_weights_proto: Optional[DataProto] = None
    if rollout_is_weights is not None:
        rollout_is_weights_proto = DataProto.from_dict(tensors={"rollout_is_weights": rollout_is_weights})

    return rollout_is_weights_proto, modified_response_mask, metrics_scalar  # 注释：返回结果


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """
    计算 off-policy 诊断指标（KL、PPL、χ² 等）。（注释：函数用途）

    参数：（注释：参数说明）
      - old_log_prob (torch.Tensor): 训练策略 log prob。（注释：输入含义）
      - rollout_log_prob (torch.Tensor|None): rollout 策略 log prob。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, Any]: off-policy 指标字典（无前缀）。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - response_mask 全为 0 会触发断言失败。（注释：边界）
    最小示例：（注释：最小示例）
      >>> metrics = compute_offpolicy_metrics(old, roll, mask)  # 返回 KL/PPL 等（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_offpolicy_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_rollout_correction_and_rejection_mask` -> `compute_offpolicy_metrics`。（注释：内部调用）
      被谁调用
      --------
      - 本文件 `compute_rollout_correction_and_rejection_mask` / `compute_rollout_corr_metrics_from_logprobs`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `verl_F.masked_mean` / `verl_F.masked_sum`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.exp` / `torch.clamp`。（注释：外部依赖）
    """
    # 至少包含一个有效 token（注释：基础校验）
    assert response_mask.any(), "Expected at least one valid token in response_mask"

    metrics = {}  # 注释：指标容器

    # 1) 训练策略 PPL（注释：始终可计算）
    mean_log_prob_training = verl_F.masked_mean(old_log_prob, response_mask, axis=-1)  # (batch_size,)
    training_ppl = torch.exp(-mean_log_prob_training).mean()  # 注释：序列级 PPL 后取均值
    metrics["training_ppl"] = training_ppl.detach().item()  # 注释：记录训练 PPL

    # 同时记录 log_ppl（注释：避免指数尺度）
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    # 2) rollout 相关指标（注释：仅当 rollout_log_prob 可用）
    if rollout_log_prob is not None:
        # 2a. KL(π_rollout || π_training)
        metrics["kl"] = verl_F.masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()

        # 2b. k3_kl 稳定估计
        log_ratio = old_log_prob - rollout_log_prob
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = verl_F.masked_mean(k3_kl_matrix, response_mask).detach().item()

        # 2c. rollout PPL
        mean_log_prob_rollout = verl_F.masked_mean(rollout_log_prob, response_mask, axis=-1)  # (batch_size,)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()
        metrics["rollout_ppl"] = rollout_ppl.detach().item()
        metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

        # 2d. log_ppl_diff 统计
        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()

        # 2e. ppl_ratio（注释：exp(log_ppl_diff)）
        ppl_ratio = torch.exp(log_ppl_diff).mean()
        metrics["ppl_ratio"] = ppl_ratio.detach().item()

        # 2f. χ² divergence（注释：衡量 IS 权重方差）
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)
        rho_squared_token = rho_token.square()
        chi2_token = verl_F.masked_mean(rho_squared_token, response_mask) - 1.0
        metrics["chi2_token"] = chi2_token.detach().item()

        # Sequence-level χ²
        log_ratio_sum = verl_F.masked_sum(log_ratio, response_mask, axis=-1)
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)
        chi2_seq = rho_squared_seq.mean() - 1.0
        metrics["chi2_seq"] = chi2_seq.detach().item()

    return metrics  # 注释：返回 off-policy 指标


def compute_rollout_correction_and_add_to_batch(
    batch: DataProto, rollout_corr_config: RolloutCorrectionConfig
) -> tuple[DataProto, dict]:
    """
    计算 rollout 校正并写回 batch（mask + IS 权重 + 指标）。（注释：函数用途）

    参数：（注释：参数说明）
      - batch (DataProto): 含 old_log_probs/rollout_log_probs/response_mask 的批数据。（注释：输入含义）
      - rollout_corr_config (RolloutCorrectionConfig): 校正配置。（注释：输入含义）
    返回：（注释：返回值说明）
      - updated_batch (DataProto): 更新后的 batch（response_mask 必更新，IS 权重可选）。（注释：输出含义）
      - metrics (dict): 以 rollout_corr/ 前缀标记的指标。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 原 batch 的 response_mask 会被更新。（注释：原地修改）
    异常/边界条件：（注释：异常说明）
      - 若 batch 缺失关键字段会在下游函数报错。（注释：边界）
    最小示例：（注释：最小示例）
      >>> batch, metrics = compute_rollout_correction_and_add_to_batch(batch, cfg.algorithm.rollout_correction)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_rollout_correction_and_add_to_batch(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> 本函数。（注释：训练流程）
      被谁调用
      --------
      - `RayPPOTrainer.fit`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `compute_rollout_correction_and_rejection_mask`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - 无。（注释：外部依赖）
    """
    # 从配置读取校正参数（注释：新的 API）
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # 计算 IS 权重与修改后的 response_mask（注释：核心逻辑）
    rollout_is_weights, modified_response_mask, rollout_corr_metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=batch.batch["response_mask"],
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
    )

    # 必须更新 response_mask（注释：RS/veto 始终生效）
    batch.batch["response_mask"] = modified_response_mask

    # 如果计算了 IS 权重则合并到 batch（注释：可选）
    if rollout_is_weights is not None:
        batch = batch.union(rollout_is_weights)

    return batch, rollout_corr_metrics  # 注释：返回更新后的 batch 与指标


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    """
    用当前策略 log_prob 与 rollout log_prob 计算 off-policy 指标。（注释：函数用途）

    参数：（注释：参数说明）
      - log_prob (torch.Tensor): 当前策略 log prob。（注释：输入含义）
      - rollout_log_prob (torch.Tensor): rollout 策略 log prob。（注释：输入含义）
      - response_mask (torch.Tensor): 有效 token mask。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, float]: 带 rollout_corr/ 前缀的指标。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - response_mask 全为 0 时下游断言失败。（注释：边界）
    最小示例：（注释：最小示例）
      >>> metrics = compute_rollout_corr_metrics_from_logprobs(log_prob, roll_prob, mask)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`compute_rollout_corr_metrics_from_logprobs(...)`（注释：函数名）
      典型调用路径
      ------------
      - Actor worker 内部日志统计 -> 本函数。（注释：训练过程）
      被谁调用
      --------
      - `verl/workers/actor/dp_actor.py`（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `compute_offpolicy_metrics`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - 无。（注释：外部依赖）
    """
    # 计算 off-policy 指标（注释：复用统一函数）
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    # 添加 rollout_corr/ 前缀（注释：便于日志分组）
    metrics_with_prefix = {}
    for key, value in offpolicy_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_with_prefix[f"rollout_corr/{key}"] = value

    return metrics_with_prefix  # 注释：返回指标


def apply_bypass_mode(
    batch: DataProto,
    rollout_corr_config: Optional[RolloutCorrectionConfig] = None,
    policy_loss_config: PolicyLossConfig = None,
) -> None:
    """
    启用 bypass 模式：用 rollout_log_probs 直接作为 old_log_probs。（注释：函数用途）

    参数：（注释：参数说明）
      - batch (DataProto): 含 rollout_log_probs 的 batch。（注释：输入含义）
      - rollout_corr_config (RolloutCorrectionConfig|None): 校正配置。（注释：输入含义）
      - policy_loss_config (PolicyLossConfig): actor loss 配置。（注释：输入含义）
    返回：（注释：返回值说明）
      - None。（注释：原地修改 batch 与 policy_loss_config）
    副作用：（注释：副作用说明）
      - 修改 batch.batch["old_log_probs"] 与 policy_loss_config 字段。（注释：原地修改）
    异常/边界条件：（注释：异常说明）
      - batch 缺少 rollout_log_probs 会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> apply_bypass_mode(batch, cfg.algorithm.rollout_correction, cfg.actor_rollout_ref.actor.policy_loss)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/rollout_corr_helper.py`（注释：文件路径）
      - 函数：`apply_bypass_mode(...)`（注释：函数名）
      典型调用路径
      ------------
      - `RayPPOTrainer.fit` -> `apply_bypass_mode`。（注释：训练循环）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `omegaconf.open_dict`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - 无。（注释：外部依赖）
    """
    from omegaconf import open_dict  # 注释：允许修改只读配置

    # 必须存在 rollout_log_probs（注释：bypass 前置条件）
    if "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "bypass_mode=True requires rollout_log_probs in batch. "
            "Ensure rollout worker is configured to calculate_log_probs=true."
        )

    # 用 rollout_log_probs 替换 old_log_probs（注释：避免额外前向）
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    with open_dict(policy_loss_config):
        # 将 rollout_correction 配置传入 loss 配置（注释：用于计算指标）
        policy_loss_config["rollout_correction"] = rollout_corr_config
        # 强制使用 bypass_mode loss（注释：兼容 ppo_clip/reinforce）
        policy_loss_config["loss_mode"] = "bypass_mode"
