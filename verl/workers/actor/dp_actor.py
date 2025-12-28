# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
DataParallelPPOActor 模块 - FSDP Data Parallel 的 PPO Actor/Ref 实现

模块用途：
---------
本模块实现了基于 FSDP（Fully Sharded Data Parallel）的 PPO Actor 和 Reference Policy 的核心逻辑。
主要功能包括：
1. **compute_log_prob**：计算策略的 log probability（用于 Ref 模式）
2. **update_policy**：执行 PPO 策略更新（用于 Actor 模式）
3. **_forward_micro_batch**：前向计算，支持 remove_padding、Ulysses 序列并行、多模态输入等优化

**关键特性**：
- **单进程设计**：与 fsdp_workers.py 中的 ActorRolloutRefWorker 不同，本类仅实现 Actor/Ref 的核心逻辑，
  由 Worker 层负责调度和数据分发
- **Actor / Ref 双模式**：通过 `actor_optimizer` 是否为 None 区分两种模式
  * Ref 模式：仅计算 log_prob（冻结参数，用于计算 KL 惩罚）
  * Actor 模式：训练策略（更新参数）
- **PPO 核心算法**：实现 PPO 的 policy loss、entropy bonus、KL penalty、梯度裁剪等
- **多种优化**：支持 remove_padding（flash-attn 变长序列）、fused kernels（融合算子）、
  Ulysses 序列并行（长序列）、dynamic batch（动态 batch size）、torch.compile 等

输入/输出：
----------
输入（DataProto）：
    - compute_log_prob 方法：
        * input_ids : torch.Tensor, [batch_size, seq_len]
            拼接后的 prompt + response
        * attention_mask : torch.Tensor, [batch_size, seq_len]
        * position_ids : torch.Tensor, [batch_size, seq_len]
        * responses : torch.Tensor, [batch_size, response_len]
        * meta_info["temperature"] : float
            采样温度（影响 logits 的缩放）
    - update_policy 方法：
        * 上述字段 + advantages : torch.Tensor, [batch_size, response_len]
            优势函数（A = Q - V 或 GAE）
        * old_log_probs : torch.Tensor, [batch_size, response_len]
            旧策略的 log_prob（用于 PPO ratio）
        * response_mask : torch.Tensor, [batch_size, response_len]
            有效 token 掩码（padding 位置为 0）

输出：
    - compute_log_prob：
        * log_probs : torch.Tensor, [batch_size, response_len]
            每个 token 的 log probability
        * entropys : torch.Tensor, [batch_size, response_len]（可选）
            每个 token 的熵（entropy）
    - update_policy：
        * metrics : dict
            训练指标（pg_loss, entropy, kl_loss, grad_norm 等）

关键依赖：
---------
- PyTorch FSDP / FSDP2
- verl.trainer.ppo.core_algos（PPO 核心算法：compute_policy_loss_vanilla/gpg/clip_cov）
- verl.utils.torch_functional（logprobs_from_logits, entropy_from_logits）
- verl.utils.attention_utils（remove_padding/pad_input，用于变长序列优化）
- verl.utils.ulysses（Ulysses 序列并行）
- flash-attn（可选，用于变长序列的 flash attention）

典型用法：
----------
示例 1：Ref 模式（计算 ref_log_prob）
```python
# 1. 创建 Ref 模型（optimizer=None）
ref_actor = DataParallelPPOActor(
    config=ref_config,
    actor_module=ref_model,  # FSDP wrapped model
    actor_optimizer=None,  # Ref 模式：不训练，仅推理
)

# 2. 计算 ref_log_prob（用于 KL penalty）
data = DataProto(batch={
    "input_ids": ...,  # [batch_size, prompt_len + response_len]
    "attention_mask": ...,
    "position_ids": ...,
    "responses": ...,  # [batch_size, response_len]
}, meta_info={"temperature": 1.0, "micro_batch_size": 8})

ref_log_probs, ref_entropys = ref_actor.compute_log_prob(data, calculate_entropy=True)
# ref_log_probs: [batch_size, response_len]
```

示例 2：Actor 模式（PPO 训练）
```python
# 1. 创建 Actor 模型（optimizer != None）
actor = DataParallelPPOActor(
    config=actor_config,
    actor_module=actor_model,  # FSDP wrapped model
    actor_optimizer=actor_optimizer,  # Adam/AdamW
)

# 2. 准备训练数据（包含 advantages、old_log_probs）
data = DataProto(batch={
    "input_ids": ...,
    "attention_mask": ...,
    "position_ids": ...,
    "responses": ...,
    "response_mask": ...,  # 有效 token 掩码
    "advantages": ...,  # [batch_size, response_len]，优势函数
    "old_log_probs": ...,  # [batch_size, response_len]，旧策略 log_prob
    "ref_log_prob": ...,  # [batch_size, response_len]，可选，用于 KL loss
}, meta_info={"temperature": 1.0})

# 3. 执行 PPO 更新（内部自动分 mini-batch 和 micro-batch）
metrics = actor.update_policy(data)
# metrics: {"actor/pg_loss": ..., "actor/entropy": ..., "actor/grad_norm": ...}
```

示例 3：启用 remove_padding 和 Ulysses 序列并行
```python
actor_config = ActorConfig(
    use_remove_padding=True,  # 启用 remove_padding（需 flash-attn）
    ulysses_sequence_parallel_size=4,  # 4-way Ulysses SP
    ppo_mini_batch_size=64,
    ppo_micro_batch_size_per_gpu=8,
    ppo_epochs=4,
    grad_clip=1.0,
)
actor = DataParallelPPOActor(config=actor_config, actor_module=actor_model, actor_optimizer=optimizer)
metrics = actor.update_policy(data)
```

调用路径概览：
--------------
训练脚本 (verl/trainer/main_ppo.py / main_grpo.sh)
  -> PPOTrainer.fit() / GRPOTrainer.fit()
    -> ray_trainer.PPORayTrainer.fit_epoch()
      -> ActorRolloutRefWorker.update_policy(data)  # Worker 层
        -> DataParallelPPOActor.update_policy(data)  # 本模块（Actor 核心逻辑）
          -> _forward_micro_batch() [多次调用，处理每个 micro batch]
          -> compute_policy_loss_vanilla/gpg/clip_cov()  # PPO loss
          -> _optimizer_step()  # 梯度裁剪 + optimizer.step()

所在位置：
----------
- 路径：`verl/workers/actor/dp_actor.py`
- 类名：`DataParallelPPOActor`
- 继承自：`BasePPOActor`

被谁调用：
----------
- `verl/workers/fsdp_workers.py::ActorRolloutRefWorker`（Actor/Ref Worker，调用本类的方法）
- `verl/workers/megatron_workers.py::MegatronActorRolloutRefWorker`（Megatron 版本，类似调用）
- `verl/trainer/ppo/ray_trainer.py::PPORayTrainer`（间接调用，通过 Worker 层）

调用了谁（项目内）：
------------------
- `verl.trainer.ppo.core_algos::compute_policy_loss_vanilla`（PPO vanilla loss）
- `verl.trainer.ppo.core_algos::compute_policy_loss_gpg`（GPG loss）
- `verl.trainer.ppo.core_algos::compute_policy_loss_clip_cov`（Clip + CoV loss）
- `verl.trainer.ppo.core_algos::kl_penalty`（KL 惩罚）
- `verl.trainer.ppo.core_algos::agg_loss`（loss 聚合）
- `verl.utils.torch_functional::logprobs_from_logits`（计算 log_prob）
- `verl.utils.torch_functional::entropy_from_logits`（计算 entropy）
- `verl.utils.attention_utils::unpad_input/pad_input`（remove_padding 优化）
- `verl.utils.ulysses::ulysses_pad_and_slice_inputs`（Ulysses 序列并行）
- `verl.utils.seqlen_balancing::prepare_dynamic_batch`（动态 batch）

调用了谁（关键外部依赖）：
----------------------
- `torch.nn.Module.forward`（模型前向计算）
- `torch.optim.Optimizer.step`（优化器步骤）
- `torch.nn.utils.clip_grad_norm_`（梯度裁剪）
- `torch.autocast`（自动混合精度）
- `torch.compile`（可选，torch 编译加速）
- `flash_attn`（可选，flash attention 变长序列）

注意事项：
----------
1. **Actor vs Ref 的区分**
   - **Ref 模式**：optimizer=None，仅调用 `compute_log_prob()`（冻结参数）
   - **Actor 模式**：optimizer!=None，调用 `update_policy()`（训练）

2. **temperature 参数**
   - **必须**在 `data.meta_info["temperature"]` 中提供
   - temperature 影响 logits 的缩放：logits /= temperature
   - temperature=1.0 表示不缩放（标准 PPO）
   - temperature<1.0 表示更尖锐的分布（更确定性）

3. **remove_padding 的作用**
   - 去除 padding token，仅计算有效 token（减少计算量）
   - 需要 flash-attn 支持（flash_attn_varlen_func）
   - 可显著加速长序列训练（特别是 prompt 长度不一致时）

4. **Ulysses 序列并行**
   - 将序列在长度维度切分到多个 GPU（适合超长序列，如 32k/64k）
   - 需要 `ulysses_sequence_parallel_size > 1`
   - 内部会在计算 log_prob 后执行 all-gather

5. **dynamic batch 的作用**
   - 动态调整 micro batch 的大小，使每个 micro batch 的 token 数接近 `max_token_len`
   - 可提高 GPU 利用率（避免某些 micro batch 过大/过小）
   - 需要 `use_dynamic_bsz=True`

6. **fused kernels 的优化**
   - 融合 log_prob 和 entropy 的计算（减少中间张量和 kernel launch 开销）
   - 需要 `use_fused_kernels=True` 和对应的融合算子支持
   - 可提高训练速度（~10-20%）

7. **torch.compile 的作用**
   - 对 `entropy_from_logits` 函数进行 JIT 编译（TorchDynamo）
   - 可提高 entropy 计算速度（~20-30%）
   - 默认启用，可通过 `config.use_torch_compile=False` 关闭

8. **PPO loss 的多种模式**
   - **vanilla**：标准 PPO loss（clip ratio）
   - **gpg**：GPG loss（Generalized Policy Gradient）
   - **clip_cov**：Clip + Coefficient of Variation loss
   - 通过 `config.policy_loss.loss_mode` 指定

9. **梯度裁剪的必要性**
   - **必须**设置 `config.grad_clip`（如 1.0）
   - PPO 训练容易出现梯度爆炸（ratio 可能很大）
   - 裁剪后可稳定训练

10. **on-policy 的特殊优化**
    - 若 `ppo_epochs=1` 且 `mini_batches=1`，则判定为 on-policy
    - on-policy 时，`old_log_prob = log_prob.detach()`（避免重复计算）
    - 可加速训练（GRPO 算法通常是 on-policy）
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
