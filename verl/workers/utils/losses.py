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
模块用途：
  - 定义 Actor/Critic 在 SFT/PPO/GRPO 场景下常用的损失计算函数。
  - 提供对 “无 padding / NestedTensor” 输出的统一切片与聚合逻辑。

输入：
  - 模型输出（log_probs/values/entropy 等张量或 NestedTensor）。
  - TensorDict batch（包含 response_mask、old_log_probs、advantages 等）。

输出：
  - 标量损失与可记录的 metrics 字典。

关键依赖：
  - `verl.trainer.ppo.core_algos`：策略/价值损失与 KL 相关实现。
  - `verl.utils.tensordict_utils`：pad_mode 等元信息读取。
  - `verl.utils.torch_functional`：masked_mean/masked_sum。

典型用法（最小示例）：
  - `loss, metrics = ppo_loss(actor_cfg, model_output, batch_td)`  # Actor 损失。
  - `vf_loss, vf_metrics = value_loss(critic_cfg, model_output, batch_td)`  # Critic 损失。

调用路径概览：
  - `verl/trainer/main_ppo.py`
    -> `verl/trainer/ppo/ray_trainer.py`
    -> `verl/workers/*`（Actor/Critic worker）
    -> `verl/workers/utils/losses.py`（本模块）
"""


import torch  # 张量运算与 NestedTensor 支持
import torch.nn.functional as F  # padding 等函数
from tensordict import TensorDict  # 承载 batch 数据

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty  # 损失聚合核心
from verl.utils import tensordict_utils as tu  # TensorDict 元信息访问
from verl.utils.dataset.dataset_utils import DatasetPadMode  # padding 模式枚举
from verl.utils.torch_functional import masked_mean, masked_sum  # 带 mask 的统计
from verl.workers.config import ActorConfig, CriticConfig  # 配置类型定义


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """
    功能：
      - 计算 SFT（监督微调）阶段的 token-level 负对数似然损失。
      - 支持 NO_PADDING（NestedTensor）与常规右 padding 两种数据格式。

    参数：
      - config (ActorConfig): Actor 侧配置，当前主要用于损失聚合语义说明。
      - model_output (dict): 模型输出，必须包含 "log_probs"。
      - data (TensorDict): batch 数据，必须包含 loss_mask/response_mask 等字段。
      - dp_group: 数据并行组（未直接使用，保留接口兼容）。

    返回：
      - (loss, metrics): loss 为标量张量；metrics 为空字典（保留接口）。

    副作用：
      - 无（不修改 data/config）。

    异常/边界条件：
      - 若缺少 "log_probs"/"loss_mask"/"response_mask" 等字段会报错。
      - 当 pad_mode 为 NO_PADDING 时，要求 log_prob/loss_mask 为 NestedTensor。

    最小示例（伪输入输出）：
      - 输入：log_probs=[-0.1,-0.2], loss_mask=[1,1], batch_num_tokens=2, dp_size=1；
      - 输出：loss = -sum(log_probs*mask)/2 = 0.15。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/utils/losses.py`
        - 函数：`sft_loss(config, model_output, data, dp_group=None)`
      典型调用路径：
        - `verl/trainer/fsdp_sft_trainer.py`
          -> `verl/workers/*`（Actor）
          -> `sft_loss(...)`
      被谁调用：
        - 仓库内未检索到直接引用（可能由外部或后续扩展调用）。
      调用了谁（项目内）：
        - `verl.utils.torch_functional.masked_sum`
        - `verl.utils.tensordict_utils.get_non_tensor_data`
      调用了谁（关键外部依赖）：
        - `torch.roll` / `torch` 张量运算
    """
    # --- 读取 pad 模式与批次统计 ---
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)  # padding 方式
    dp_size = data["dp_size"]  # 数据并行规模
    batch_num_tokens = data["batch_num_tokens"]  # 全局 token 数（已聚合）

    # --- 取出 log_probs ---
    log_prob = model_output["log_probs"]  # 形状与输入对齐的对数概率

    if pad_mode == DatasetPadMode.NO_PADDING:
        # --- NestedTensor 路径：按 jagged 展开 ---
        loss_mask = data["loss_mask"]  # loss mask（NestedTensor）

        log_prob_flatten = log_prob.values()  # 展开为扁平值
        loss_mask_flatten = loss_mask.values()  # 展开为扁平 mask

        # 左移一位对齐 log_prob（log_prob 对应下一个 token）
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss 以全局 token 数平均，并乘 dp_size 以保证并行缩放一致
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        # --- 常规 padding 路径：直接按 response_mask 计算 ---
        response_mask = data["response_mask"].to(bool)  # response 区域掩码
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}  # metrics 为空以保持接口一致


def _slice_response_from_unpad_output(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """
    功能：
      - 从 “unpad 展开后的模型输出” 中切出 response 区间，并对齐到最大 response 长度。

    参数：
      - tensor (torch.Tensor): 模型输出（NestedTensor 或普通张量）。
      - data (TensorDict): 必须包含 "prompts"/"responses"/"attention_mask"。

    返回：
      - torch.Tensor: 形状 [bsz, max_response_len] 的 response 片段。

    副作用：
      - 无。

    异常/边界条件：
      - prompt/response 的 offsets 与 values 总长度不一致会触发断言。
      - attention_mask 维度异常时切片失败。

    最小示例（伪输入输出）：
      - 输入：prompt_len=[2,1], response_len=[3,2]，values 长度=8；
      - 输出：每个样本截取 response 区间并 pad 到 max_response_len=3。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/utils/losses.py`
        - 函数：`_slice_response_from_unpad_output(tensor, data)`
      典型调用路径：
        - `ppo_loss(...)` / `value_loss(...)` 内部调用。
      被谁调用：
        - `ppo_loss`、`value_loss`（本文件内）。
      调用了谁（项目内）：
        - 无（本函数仅调用 torch API）。
      调用了谁（关键外部依赖）：
        - `torch.stack` / `torch.nn.functional.pad`
    """
    # --- 统一获得扁平 values ---
    values = tensor.values() if tensor.is_nested else tensor  # NestedTensor -> values
    # --- 读取 prompts/response 与 mask ---
    prompt_ids = data["prompts"]  # prompt token 序列
    response_ids = data["responses"]  # response token 序列
    attention_mask = data["attention_mask"]  # attention mask

    if prompt_ids.is_nested:
        # --- NestedTensor 路径：直接用 offsets 计算长度 ---
        prompt_lens = prompt_ids.offsets().diff()  # 每条 prompt 长度
        response_lens = response_ids.offsets().diff()  # 每条 response 长度
        max_response_len = response_ids.offsets().max().item()  # 最大 response 长度
    else:
        # --- 常规 padding 路径：用 attention_mask 统计 ---
        assert not attention_mask.is_nested  # 保障 mask 也是定长
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)  # prompt 长度
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)  # response 长度
        max_response_len = response_ids.shape[1]  # response 定长维度

    # --- 计算每条样本在扁平 values 中的结束位置 ---
    sequence_lens = prompt_lens + response_lens  # 总长度
    sequence_offsets = sequence_lens.cumsum(dim=0)  # 前缀和作为结束偏移
    assert sequence_offsets[-1].item() == values.shape[0]  # 保证总长度对齐

    # --- 逐条切出 response 区间，并 pad 到最大长度 ---
    response_list = []  # 收集每条样本的 response 片段
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len  # 需要补的右侧长度
        # 左移一位对齐 log_prob/values（预测第 t+1 token）
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (0, pad_size)))

    output = torch.stack(response_list, dim=0)  # [bsz, max_response_len]
    return output  # 返回定长 response 张量


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """
    功能：
      - 计算 PPO/GRPO actor 的策略损失（含可选 entropy/kl 正则）。
      - 兼容 NestedTensor 与常规 padding 数据。

    参数：
      - config (ActorConfig): Actor 配置（含 loss_mode、kl/entropy 系数等）。
      - model_output (dict): 模型输出，包含 "log_probs" 与可选 "entropy"。
      - data (TensorDict): batch 数据，包含 old_log_probs/advantages/response_mask 等。
      - dp_group: 数据并行组（预留）。

    返回：
      - (policy_loss, metrics): 标量损失与统计指标字典。

    副作用：
      - 会写入 `config.global_batch_info`（用于 loss 聚合的全局统计）。

    异常/边界条件：
      - 若缺少 old_log_probs/advantages/response_mask，会在计算中出错。
      - kl_loss 依赖 data["ref_log_prob"]，缺失会报错。

    最小示例（伪输入输出）：
      - 输入：advantages=[1,1]，log_prob=[-0.1,-0.2]，old_log_prob=[-0.1,-0.1]；
      - 输出：pg_loss 为负的优势加权损失，metrics 包含 actor/pg_loss。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/utils/losses.py`
        - 函数：`ppo_loss(config, model_output, data, dp_group=None)`
      典型调用路径：
        - `verl/trainer/ppo/ray_trainer.py`
          -> `verl/workers/actor/*`（Actor 训练）
          -> `ppo_loss(...)`
      被谁调用：
        - 仓库内未检索到直接调用（可能由 Actor worker 在运行时动态引用）。
      调用了谁（项目内）：
        - `get_policy_loss_fn` / `agg_loss` / `kl_penalty`
        - `_slice_response_from_unpad_output`
      调用了谁（关键外部依赖）：
        - `torch` 张量运算
    """
    # --- 从模型输出切出 response 部分 ---
    log_prob = _slice_response_from_unpad_output(model_output["log_probs"], data)  # log_probs 对齐 response
    entropy = model_output.get("entropy", None)  # 可选 entropy
    if entropy is not None:
        entropy = _slice_response_from_unpad_output(entropy, data)  # 对齐 response

    # --- 写入全局 batch 信息（用于 loss 聚合） ---
    config.global_batch_info["dp_size"] = data["dp_size"]  # 数据并行大小
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]  # token 数
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]  # 全局 batch 大小
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor  # loss 缩放系数

    metrics = {}  # 统计指标容器

    # --- 准备策略梯度所需张量 ---
    response_mask = data["response_mask"].to(bool)  # response 区域 mask
    old_log_prob = data["old_log_probs"]  # rollout 时的旧 log_prob
    advantages = data["advantages"]  # 优势函数
    rollout_is_weights = data.get("rollout_is_weights", None)  # 可选 IS 权重

    loss_agg_mode = config.loss_agg_mode  # loss 聚合方式
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")  # 策略损失模式

    # --- 计算策略损失 ---
    policy_loss_fn = get_policy_loss_fn(loss_mode)  # 根据配置选择损失函数
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    metrics.update(pg_metrics)  # 合并策略损失相关指标
    metrics["actor/pg_loss"] = pg_loss.detach().item()  # 记录标量值
    policy_loss = pg_loss  # 初始损失为 pg_loss

    # --- 可选：加入 entropy 正则 ---
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff  # entropy 系数
        policy_loss -= entropy_coeff * entropy_loss  # 减去 entropy（鼓励探索）

    # --- 可选：加入 KL 正则 ---
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]  # 参考策略 log_prob
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)  # KL 张量
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef  # 加入 KL 约束
        metrics["kl_loss"] = kl_loss.detach().item()  # 记录 KL 损失
        metrics["kl_coef"] = config.kl_loss_coef  # 记录 KL 系数

    return policy_loss, metrics  # 返回损失与指标


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """
    功能：
      - 计算 Critic 的价值函数损失（含 clip 机制）。

    参数：
      - config (CriticConfig): Critic 配置，包含 cliprange_value 等。
      - model_output (dict): 模型输出，包含 "values"。
      - data (TensorDict): batch 数据，包含 values/returns/response_mask 等。
      - dp_group: 数据并行组（预留）。

    返回：
      - (vf_loss, metrics): 价值损失与指标字典（含 vf_clipfrac/vpred_mean）。

    副作用：
      - 无。

    异常/边界条件：
      - 若缺少 values/returns/response_mask，会在计算中出错。

    最小示例（伪输入输出）：
      - 输入：vpreds=[0.0,0.5], returns=[1.0,0.0]；
      - 输出：vf_loss 为 MSE/clip 后的标量，metrics 包含 vf_clipfrac。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/utils/losses.py`
        - 函数：`value_loss(config, model_output, data, dp_group=None)`
      典型调用路径：
        - `verl/trainer/ppo/ray_trainer.py`
          -> `verl/workers/critic/*`（Critic 训练）
          -> `value_loss(...)`
      被谁调用：
        - 仓库内未检索到直接调用（可能由 Critic worker 在运行时动态引用）。
      调用了谁（项目内）：
        - `compute_value_loss`
        - `_slice_response_from_unpad_output`
      调用了谁（关键外部依赖）：
        - `torch` 张量运算
    """
    # --- 切出 response 对应的 value 预测 ---
    vpreds = _slice_response_from_unpad_output(model_output["values"], data)  # (bsz, response_length)

    # --- 读取监督信号 ---
    values = data["values"]  # 旧 value 预测（用于 clip）
    returns = data["returns"]  # target return
    response_mask = data["response_mask"].to(bool)  # response mask

    # --- 计算 value loss（含 clip） ---
    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}  # 指标容器

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),  # 价值损失
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),  # clip 比例
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),  # 预测均值
        }
    )

    return vf_loss, metrics  # 返回损失与指标
