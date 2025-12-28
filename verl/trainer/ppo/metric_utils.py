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
模块用途：提供 PPO/GRPO 训练中的指标统计与验证汇总工具。（注释：模块级用途概述）
输入/输出：（注释：模块级 I/O 概览）
  - 输入：DataProto 批数据、计时信息、验证样本信息等。（注释：输入来源）
  - 输出：可直接上报的指标字典（mean/max/min/throughput 等）。（注释：输出形态）
关键依赖：（注释：列出关键依赖）
  - torch / numpy。（注释：张量与数值统计）
  - verl.DataProto。（注释：批数据容器）
典型用法（最小示例）：（注释：给出最小可用片段）
  >>> metrics = compute_data_metrics(batch, use_critic=True)  # 统计训练指标（示例）
  >>> timing = compute_timing_metrics(batch, timing_raw={"gen": 0.2, "step": 1.0})  # 计时指标（示例）
调用路径概览：（注释：说明入口链路）
  - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> 本模块函数。（注释：训练循环调用）
"""  # 注释：模块 docstring 结束

# 标准库导入（注释：用于分组聚合）
from collections import defaultdict
# 标准库导入（注释：用于构造偏函数）
from functools import partial
# 类型注解（注释：提高可读性）
from typing import Any, Callable

# 第三方依赖（注释：数值统计）
import numpy as np
# 第三方依赖（注释：张量计算）
import torch

# 项目内依赖（注释：批数据结构）
from verl import DataProto
# 项目内依赖（注释：弃用装饰器）
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    将“指标名 -> 值列表”的字典压缩为“指标名 -> 均值”的字典。（注释：函数用途）

    参数：（注释：参数说明）
      - metrics (dict[str, list[Any]]): 指标列表字典。（注释：输入类型与含义）
    返回：（注释：返回值说明）
      - dict[str, Any]: 每个列表取均值后的指标字典。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无（仅计算并返回）。（注释：无状态修改）
    异常/边界条件：（注释：异常说明）
      - 若列表为空，外部 reduce_metrics 可能抛异常或返回 NaN。（注释：边界）
    最小示例：（注释：最小示例）
      >>> reduce_metrics({"loss": [1.0, 2.0], "acc": [0.5, 1.0]})  # {"loss": 1.5, "acc": 0.75}（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`reduce_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> `reduce_metrics`。（注释：训练循环调用）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `verl/utils/metric.py::reduce_metrics`（注释：真正实现）
      调用了谁（关键外部依赖）
      ----------------------
      - 无（项目内封装）。（注释：外部依赖）
    """
    from verl.utils.metric import reduce_metrics  # 注释：调用最新实现

    return reduce_metrics(metrics)  # 注释：返回均值化指标


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    计算 prompt/response 的 mask 与长度统计。（注释：函数用途）

    参数：（注释：参数说明）
      - batch (DataProto): 含有 responses 与 attention_mask 的批数据。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, Any]: 包含 response_mask / prompt_length / response_length。（注释：输出结构）
    副作用：（注释：副作用说明）
      - 无（仅读取 batch）。（注释：无状态修改）
    异常/边界条件：（注释：异常说明）
      - 若 batch 中缺失 "responses"/"attention_mask" 会抛 KeyError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> info = _compute_response_info(batch)  # 返回长度与 mask（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`_compute_response_info(batch)`（注释：函数名）
      典型调用路径
      ------------
      - `compute_data_metrics` / `compute_timing_metrics` -> `_compute_response_info`。（注释：内部调用）
      被谁调用
      --------
      - 本文件内部函数（无外部引用）。（注释：使用范围）
      调用了谁（项目内）
      ----------------
      - 无。（注释：仅张量操作）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.Tensor` 操作。（注释：外部依赖）
    """
    response_length = batch.batch["responses"].shape[-1]  # 注释：响应序列长度（统一长度）

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]  # 注释：prompt 区域 mask
    response_mask = batch.batch["attention_mask"][:, -response_length:]  # 注释：response 区域 mask

    prompt_length = prompt_mask.sum(-1).float()  # 注释：每条样本的 prompt token 数
    response_length = response_mask.sum(-1).float()  # (batch_size,)  # 注释：每条样本的 response token 数

    return dict(
        response_mask=response_mask,  # 注释：返回 response mask
        prompt_length=prompt_length,  # 注释：返回 prompt 长度
        response_length=response_length,  # 注释：返回 response 长度
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    从一个 batch 计算训练指标（分数、奖励、优势、长度等）。（注释：函数用途）

    参数：（注释：参数说明）
      - batch (DataProto): 含 token_level_scores/advantages/returns 等字段的批数据。（注释：输入含义）
      - use_critic (bool): 是否统计 critic 相关指标（values、解释方差）。(默认 True)（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, Any]: 指标字典（mean/max/min/ratio 等）。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无（只读 batch）。（注释：无状态修改）
    异常/边界条件：（注释：异常说明）
      - 若全为 aborted（response_length=0），会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> metrics = compute_data_metrics(batch, use_critic=False)  # 输出含 critic/score 等指标（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`compute_data_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> `compute_data_metrics`。（注释：训练循环）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `_compute_response_info`（本文件内部）。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch` 张量统计函数。（注释：外部依赖）
    """
    # 1) 汇总序列级分数与奖励（注释：先从 token 级聚合）
    sequence_score = batch.batch["token_level_scores"].sum(-1)  # 注释：序列分数（按 token 累加）
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)  # 注释：序列奖励（按 token 累加）

    # 2) 取出优势与回报（注释：后续统计用）
    advantages = batch.batch["advantages"]  # 注释：优势张量
    returns = batch.batch["returns"]  # 注释：回报张量

    # 3) 计算 prompt/response 的最大长度（注释：用于长度统计）
    max_response_length = batch.batch["responses"].shape[-1]  # 注释：响应最大长度

    # 4) 计算 prompt/response mask（注释：区分 prompt 与 response）
    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()  # 注释：prompt mask
    response_mask = batch.batch["response_mask"].bool()  # 注释：response mask

    max_prompt_length = prompt_mask.size(-1)  # 注释：prompt 最大长度

    # 5) 计算各样本 prompt/response 长度（注释：用于均值/裁剪率）
    response_info = _compute_response_info(batch)  # 注释：内部辅助函数
    prompt_length = response_info["prompt_length"]  # 注释：prompt 长度
    response_length = response_info["response_length"]  # 注释：response 长度

    # 6) 标记 aborted（response_length=0）样本（注释：用于过滤）
    aborted_mask = (response_length == 0).bool()  # 注释：aborted 样本
    non_aborted_mask = ~aborted_mask  # 注释：有效样本

    # 7) 过滤 aborted 样本（注释：避免 0 干扰统计）
    non_aborted_sequence_score = sequence_score[non_aborted_mask]  # 注释：有效分数
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]  # 注释：有效奖励

    # 8) 统计分数与奖励（注释：均值/最大/最小）
    score_mean = torch.mean(non_aborted_sequence_score).detach().item()  # 注释：分数均值
    score_max = torch.max(non_aborted_sequence_score).detach().item()  # 注释：分数最大值
    score_min = torch.min(non_aborted_sequence_score).detach().item()  # 注释：分数最小值

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()  # 注释：奖励均值
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()  # 注释：奖励最大值
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()  # 注释：奖励最小值

    # 9) 过滤 response 区域用于统计优势/回报（注释：只统计有效 token）
    valid_adv = torch.masked_select(advantages, response_mask)  # 注释：有效优势
    valid_returns = torch.masked_select(returns, response_mask)  # 注释：有效回报

    # 10) critic 相关统计（注释：仅在 use_critic=True 时）
    if use_critic:
        values = batch.batch["values"]  # 注释：value 预测
        valid_values = torch.masked_select(values, response_mask)  # 注释：有效 value
        return_diff_var = torch.var(valid_returns - valid_values)  # 注释：回报-价值方差
        return_var = torch.var(valid_returns)  # 注释：回报方差

    # 11) aborted 比例与非 aborted 长度统计（注释：防止 0 影响）
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()  # 注释：aborted 占比

    non_aborted_response_length = response_length[non_aborted_mask]  # 注释：有效 response 长度
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()  # 注释：均值
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()  # 注释：最大
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()  # 注释：最小
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )  # 注释：达到最大长度的比例
    else:
        raise ValueError("All samples are aborted, this should not happen.")  # 注释：异常保护

    # 12) 组装指标字典（注释：统一输出）
    metrics = {
        # score（注释：序列分数统计）
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward（注释：序列奖励统计）
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv（注释：优势统计）
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns（注释：回报统计）
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values（注释：critic 价值统计）
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var（注释：解释方差）
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length（注释：response 长度统计）
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (non-aborted only)（注释：仅有效样本）
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio（注释：response_length==0 占比）
        "response/aborted_ratio": aborted_ratio,
        # prompt length（注释：prompt 长度统计）
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }  # 注释：指标字典构建结束

    # 13) 多轮对话统计（注释：仅在数据中提供时）
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]  # 注释：多轮数量
        metrics["num_turns/min"] = num_turns.min()  # 注释：最小轮数
        metrics["num_turns/max"] = num_turns.max()  # 注释：最大轮数
        metrics["num_turns/mean"] = num_turns.mean()  # 注释：平均轮数

    # 14) 工具调用次数统计（注释：仅在数据中提供时）
    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]  # 注释：工具调用次数
        metrics["tool_call_counts/min"] = tool_call_counts.min()  # 注释：最小值
        metrics["tool_call_counts/max"] = tool_call_counts.max()  # 注释：最大值
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()  # 注释：均值

    return metrics  # 注释：返回指标字典


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    计算训练各阶段的耗时与每 token 耗时。（注释：函数用途）

    参数：（注释：参数说明）
      - batch (DataProto): 含 attention_mask 的批数据。（注释：输入含义）
      - timing_raw (dict[str, float]): 阶段名 -> 秒级耗时。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, Any]: `timing_s/*` 与 `timing_per_token_ms/*` 指标字典。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - 若某阶段 token 数为 0，除法会产生 inf/异常。（注释：边界）
    最小示例：（注释：最小示例）
      >>> compute_timing_metrics(batch, {"gen": 0.2, "step": 1.0})  # 返回 timing_s/gen 等（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`compute_timing_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> `compute_timing_metrics`。（注释：训练循环）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `_compute_response_info`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.sum`。（注释：外部依赖）
    """
    # 1) 统计 prompt/response token 数（注释：用于归一化）
    response_info = _compute_response_info(batch)  # 注释：得到长度统计
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()  # 注释：prompt token 数
    num_response_tokens = torch.sum(response_info["response_length"]).item()  # 注释：response token 数
    num_overall_tokens = num_prompt_tokens + num_response_tokens  # 注释：总 token 数

    # 2) 各阶段对应的归一化 token 数（注释：gen 只看 response，其他看全量）
    num_tokens_of_section = {
        "gen": num_response_tokens,  # 注释：生成阶段仅统计 response token
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    # 3) 返回 raw 秒级耗时与每 token 毫秒耗时（注释：输出指标）
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},  # 注释：原始秒级耗时
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },  # 注释：按 token 归一化耗时
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    计算训练吞吐量相关指标（token/s/GPU）。（注释：函数用途）

    参数：（注释：参数说明）
      - batch (DataProto): meta_info 中需包含 global_token_num。（注释：输入含义）
      - timing_raw (dict[str, float]): 各阶段耗时，需包含 "step"。（注释：输入含义）
      - n_gpus (int): GPU 数量。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, Any]: perf/total_num_tokens、perf/time_per_step、perf/throughput。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - n_gpus=0 或 time=0 会导致除零。（注释：边界）
    最小示例：（注释：最小示例）
      >>> compute_throughout_metrics(batch, {"step": 2.0}, n_gpus=8)  # 返回吞吐量（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`compute_throughout_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` -> `compute_throughout_metrics`。（注释：训练循环）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - 无。（注释：无项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - Python 内置 sum / 除法。（注释：外部依赖）
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])  # 注释：全局 token 总数
    time = timing_raw["step"]  # 注释：单步耗时（秒）
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,  # 注释：token 总数
        "perf/time_per_step": time,  # 注释：单步耗时
        "perf/throughput": total_num_tokens / (time * n_gpus),  # 注释：吞吐量
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    通过 bootstrap 重采样估计指标的均值与标准差。（注释：函数用途）

    参数：（注释：参数说明）
      - data (list[Any]): 原始样本列表。（注释：输入含义）
      - subset_size (int): 每次采样的子集大小。（注释：输入含义）
      - reduce_fns (list[Callable]): 对子集计算指标的函数列表。（注释：输入含义）
      - n_bootstrap (int): 重采样次数。（注释：输入含义）
      - seed (int): 随机种子。（注释：输入含义）
    返回：（注释：返回值说明）
      - list[tuple[float, float]]: 每个 reduce_fn 的 (mean, std)。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 设置 numpy 全局随机种子。（注释：影响外部随机性）
    异常/边界条件：（注释：异常说明）
      - subset_size > len(data) 仍允许（replace=True），但需注意代表性。（注释：边界）
    最小示例：（注释：最小示例）
      >>> bootstrap_metric([1,2,3], 2, [np.mean], n_bootstrap=3)  # 返回均值与 std（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`bootstrap_metric(...)`（注释：函数名）
      典型调用路径
      ------------
      - `process_validation_metrics` -> `bootstrap_metric`。（注释：验证统计）
      被谁调用
      --------
      - 本文件 `process_validation_metrics`。（注释：内部调用）
      调用了谁（项目内）
      ----------------
      - 无。（注释：无项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `numpy.random.choice` / `numpy.mean` / `numpy.std`。（注释：外部依赖）
    """
    np.random.seed(seed)  # 注释：固定随机种子，保证可复现

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]  # 注释：为每个指标收集结果
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)  # 注释：重采样索引
        bootstrap_data = [data[i] for i in bootstrap_idxs]  # 注释：构造子集
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))  # 注释：计算子集指标
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]  # 注释：返回均值与标准差


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    多数投票：返回“票数最多的类别”的对应值。（注释：函数用途）

    参数：（注释：参数说明）
      - data (list[dict[str, Any]]): 每条记录含 vote_key 与 val_key。（注释：输入含义）
      - vote_key (str): 用于投票计数的键。（注释：输入含义）
      - val_key (str): 返回值所在的键。（注释：输入含义）
    返回：（注释：返回值说明）
      - float: 多数票类别对应的第一个 val。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - data 为空时 max 会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> calc_maj_val([{"pred":"A","val":1.0},{"pred":"A","val":2.0}], "pred", "val")  # 1.0（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`calc_maj_val(...)`（注释：函数名）
      典型调用路径
      ------------
      - `process_validation_metrics` -> `bootstrap_metric` -> `calc_maj_val`。（注释：多数投票统计）
      被谁调用
      --------
      - 本文件 `process_validation_metrics`。（注释：内部调用）
      调用了谁（项目内）
      ----------------
      - `defaultdict`（本文件导入）。（注释：项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - Python `max`。（注释：外部依赖）
    """
    vote2vals = defaultdict(list)  # 注释：票 -> 值列表
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])  # 注释：聚合同一投票类别的值

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}  # 注释：每类票数
    maj_vote = max(vote2cnt, key=vote2cnt.get)  # 注释：多数票类别

    maj_val = vote2vals[maj_vote][0]  # 注释：取多数票类别的第一个值

    return maj_val  # 注释：返回多数票对应值


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    按数据源与样本 UID 聚合验证指标，并生成多种统计量。（注释：函数用途）

    参数：（注释：参数说明）
      - data_sources (list[str]): 每条样本的数据源标识。（注释：输入含义）
      - sample_uids (list[str]): 每条样本的 UID。（注释：输入含义）
      - infos_dict (dict[str, list[Any]]): 指标名 -> 值列表（与样本一一对应）。（注释：输入含义）
      - seed (int): bootstrap 随机种子。（注释：输入含义）
    返回：（注释：返回值说明）
      - dict[str, dict[str, dict[str, float]]]: data_source -> var -> metric_name -> value。（注释：输出结构）
    副作用：（注释：副作用说明）
      - 调用 `bootstrap_metric` 会设置 numpy 随机种子。（注释：副作用）
    异常/边界条件：（注释：异常说明）
      - infos_dict 中若出现空列表或非数值类型，统计会被跳过或报错。（注释：边界）
    最小示例：（注释：最小示例）
      >>> process_validation_metrics(["s1","s1"], ["u1","u1"], {"score":[1.0,2.0]})  # 返回统计字典（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/metric_utils.py`（注释：文件路径）
      - 函数：`process_validation_metrics(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::_validate` -> `process_validation_metrics`。（注释：验证流程）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `bootstrap_metric` / `calc_maj_val`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `numpy.mean` / `numpy.std`。（注释：外部依赖）
    """
    # 1) 按 data_source -> uid -> var 聚合原始值（注释：分组整理）
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # 注释：三层字典
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]  # 注释：样本 UID
        var2vals = data_src2uid2var2vals[data_source][uid]  # 注释：定位到该 UID 的容器
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])  # 注释：追加当前样本值

    # 2) 对每个 uid 的指标计算统计量（注释：先做局部统计）
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # 注释：统计结果容器
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue  # 注释：字符串指标不参与数值统计

                metric = {}  # 注释：该变量的统计字典
                n_resps = len(var_vals)  # 注释：该 UID 的样本数
                metric[f"mean@{n_resps}"] = np.mean(var_vals)  # 注释：均值

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)  # 注释：标准差

                    # 构造 bootstrap 子集规模（2,4,8,...,n_resps）（注释：层级统计）
                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )  # 注释：best/worst bootstrap 统计
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std  # 注释：best 统计
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std  # 注释：worst 统计
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]  # 注释：构造多数投票数据
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )  # 注释：多数投票 bootstrap 统计
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std  # 注释：记录

                data_src2uid2var2metric[data_source][uid][var_name] = metric  # 注释：保存该 uid 统计

    # 3) 按数据源聚合 uid 维度的统计（注释：跨 uid 汇总）
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # 注释：中间容器
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    # 4) 对每个指标在 uid 维度取均值（注释：最终汇总）
    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # 注释：最终结果容器
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val  # 注释：返回最终统计结构
