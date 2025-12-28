# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行说明
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：提示许可证链接
#  # 注释：保留注释符号，保证该行有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证地址
#  # 注释：保留注释符号，保证该行有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
Apply monkey-patch function to models  # 注释：保留英文简述
模块用途：为不同模型注入/替换 forward，实现 Ulysses 序列并行与 fused kernel 选择。  # 注释：模块用途
输入/输出：输入 PreTrainedModel 与配置，输出为被 monkey patch 的模型（就地修改）。  # 注释：模块输入输出概览
关键依赖：transformers flash attention utils、verl.utils.ulysses、transformers 版本检测。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- apply_monkey_patch(model, ulysses_sp_size=2, use_remove_padding=True)  # 注释：应用 monkey patch
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/fsdp_workers.py、verl/trainer/fsdp_sft_trainer.py。  # 注释：上层入口举例
- 典型链路：worker/训练入口 -> apply_monkey_patch -> patch_vlm_for_ulysses_input_slicing/_ulysses_flash_attention_forward。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import sys  # 注释：访问模块对象
from types import SimpleNamespace  # 注释：简单对象容器
from typing import Optional  # 注释：类型注解
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import torch  # 注释：张量操作
from transformers.modeling_flash_attention_utils import _flash_attention_forward  # 注释：flash attention 核心函数
from transformers.modeling_utils import PreTrainedModel  # 注释：模型基类
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl.utils.import_utils import is_trl_available  # 注释：检测 trl 是否可用
from verl.utils.transformers_compat import is_transformers_version_in_range  # 注释：transformers 版本判断
from verl.utils.ulysses import (  # 注释：Ulysses 序列并行工具
    gather_heads_scatter_seq,  # 注释：all2all 工具（head -> seq）
    gather_seq_scatter_heads,  # 注释：all2all 工具（seq -> head）
    get_ulysses_sequence_parallel_group,  # 注释：获取通信 group
    get_ulysses_sequence_parallel_world_size,  # 注释：获取序列并行 size
    slice_input_tensor,  # 注释：切分输入张量
)  # 注释：导入结束
# （分隔说明：工具函数 repeat_kv）  # 注释：替代空行，保持逐行注释
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:  # 注释：重复 KV 头以匹配 attention 头数
    """
    This is the equivalent of torch.repeat_interleave(x, dim=2, repeats=n_rep). The hidden states go from (batch,
    seqlen, num_key_value_heads, head_dim) to (batch, seqlen, num_attention_heads, head_dim)  # 注释：保留英文说明

    功能：重复 KV 头维度以匹配 attention 头数量。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - hidden_states (Tensor)：形状 (B, S, n_kv, D)。  # 注释：参数含义
    - n_rep (int)：重复次数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - Tensor：形状 (B, S, n_kv*n_rep, D)。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：n_rep=1 时直接返回原张量。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - repeat_kv(x, 2) -> 头数翻倍。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/monkey_patch.py::repeat_kv。  # 注释：函数位置
    - 典型调用路径：_ulysses_flash_attention_forward -> repeat_kv。  # 注释：典型调用链
    - 被谁调用：本文件 _ulysses_flash_attention_forward。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.Tensor.expand/reshape。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape  # 注释：解析维度
    if n_rep == 1:  # 注释：无需重复
        return hidden_states  # 注释：直接返回
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)  # 注释：在新维度扩展
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)  # 注释：合并 head 维度并返回
# （分隔说明：Ulysses flash attention 包装）  # 注释：替代空行，保持逐行注释
def _ulysses_flash_attention_forward(  # 注释：插入 all-to-all 的 flash attention
    query_states: torch.Tensor,  # 注释：Q 张量
    key_states: torch.Tensor,  # 注释：K 张量
    value_states: torch.Tensor,  # 注释：V 张量
    attention_mask: Optional[torch.Tensor],  # 注释：attention mask
    query_length: int,  # 注释：query 长度
    *args,  # 注释：透传参数
    position_ids: Optional[torch.Tensor] = None,  # 注释：位置编码
    **kwargs,  # 注释：透传关键字参数
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    For transformers>=4.55, the flash attention api has changed,
    we need to pass the query_length after doing ulysses all2all.
    See https://github.com/huggingface/transformers/issues/40399

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)

    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()  # 注释：获取序列并行大小

    ########## AlltoAll for Ulysses ##########  # 注释：分隔标记（原注释保留）
    # TODO: Disable sp for ViT, there's no elegent way to determine whether it's ViT or not.  # 注释：原 TODO
    # Use `position_ids` as condition since ViT doesn't pass it to flash attention.  # 注释：原注释保留
    if ulysses_sp_size > 1 and position_ids is not None:  # 注释：开启序列并行且有 position_ids
        # NOTE: repeat kv heads to be divided by sequence parallel. Instead of repeating nheads_q//nheads_k,  # 注释：原注释保留
        # we choose to repeat sp_size//nheads_k, since flash_attention supports MQA/GQA.  # 注释：原注释保留
        # For example:  # 注释：原注释保留
        # - nheads_k=4, sp=8, repeats=2  # 注释：原注释保留
        # - nheads_k=8, sp=8, repeats=1  # 注释：原注释保留
        # - nheads_k=16, sp=8, repeats=1  # 注释：原注释保留
        repeats = max(ulysses_sp_size // key_states.size(2), 1)  # 注释：计算重复次数
        key_states = repeat_kv(key_states, repeats)  # 注释：重复 K
        value_states = repeat_kv(value_states, repeats)  # 注释：重复 V

        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)  # 注释：原注释保留
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)  # 注释：all2all 交换
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)  # 注释：all2all 交换
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)  # 注释：all2all 交换

        # TODO: all_gather position_ids because `prepare_fa2_from_position_ids` needs it, we can eliminate  # 注释：原 TODO
        # this all_gather by passing cu_seq_lens_q, cu_seq_lens_k, max_length_k, max_length_q explicitly.  # 注释：原 TODO
        # https://github.com/huggingface/transformers/pull/33932  # 注释：参考链接

        # (bsz, seq_len/n) -> (bsz, seq_len)  # 注释：原注释保留
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]  # 注释：预分配列表
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())  # 注释：all_gather position_ids
        position_ids = torch.concat(position_ids_list, dim=-1)  # 注释：拼接回完整序列

    # (bsz, seq_len, n_head/n, head_dim)  # 注释：原注释保留
    query_length = query_states.size(1)  # 注释：更新 query_length（顺序并行后）
    attn_output = _flash_attention_forward(  # 注释：调用原始 flash attention
        query_states, key_states, value_states, attention_mask, query_length, *args, position_ids=position_ids, **kwargs  # 注释：参数透传
    )  # 注释：flash attention 输出

    ########## AlltoAll for Ulysses ##########  # 注释：分隔标记（原注释保留）
    if ulysses_sp_size > 1 and position_ids is not None:  # 注释：序列并行且有 position_ids
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)  # 注释：原注释保留
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)  # 注释：all2all 交换回

    return attn_output  # 注释：返回 attention 输出
# （分隔说明：VLM 输入切分 patch）  # 注释：替代空行，保持逐行注释
def patch_vlm_for_ulysses_input_slicing(model_class: type):  # 注释：给 VLM 模型打补丁以支持输入切分
    """
    Applies a monkey patch to the forward method of a given model class
    to enable Ulysses sequence parallelism input slicing.  # 注释：保留英文说明

    功能：在 forward 前对 inputs_embeds/position_ids 等进行分片。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - model_class (type)：模型类（将修改其 forward）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None（就地修改类方法）。  # 注释：返回值语义
    副作用：修改 model_class.forward。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/monkey_patch.py::patch_vlm_for_ulysses_input_slicing。  # 注释：函数位置
    - 典型调用路径：apply_monkey_patch -> patch_vlm_for_ulysses_input_slicing。  # 注释：典型调用链
    - 被谁调用：本文件 apply_monkey_patch。  # 注释：调用方说明
    - 调用了谁（项目内）：slice_input_tensor、get_ulysses_sequence_parallel_world_size。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束

    def _create_ulysses_wrapped_decoder_forward(original_forward):  # 注释：创建 wrapped forward
        def ulysses_wrapped_decoder_forward(self, *args, **kwargs):  # 注释：包裹后的 forward
            inputs_embeds = kwargs.get("inputs_embeds")  # 注释：取 inputs_embeds
            position_ids = kwargs.get("position_ids")  # 注释：取 position_ids
            visual_pos_masks = kwargs.get("visual_pos_masks")  # 注释：取视觉位置 mask
            deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds")  # 注释：取深度堆叠视觉嵌入
            call_kwargs = kwargs.copy()  # 注释：复制参数

            current_ulysses_sp_size = get_ulysses_sequence_parallel_world_size()  # 注释：获取当前序列并行 size

            slice_now = (  # 注释：是否进行切片
                inputs_embeds is not None  # 注释：有 inputs_embeds
                and current_ulysses_sp_size > 1  # 注释：并行 size > 1
                and getattr(self, "_needs_initial_slice", True)  # 注释：需要初始切片
            )  # 注释：slice_now 计算结束
            if slice_now:  # 注释：需要切片
                call_kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, padding=False)  # 注释：切分 inputs_embeds
                call_kwargs["position_ids"] = slice_input_tensor(position_ids, dim=-1, padding=False)  # 注释：切分 position_ids
                # Also slice visual_pos_masks and deepstack_visual_embeds for Qwen3 VL models  # 注释：原注释保留
                if visual_pos_masks is not None:  # 注释：存在视觉位置 mask
                    original_visual_mask = visual_pos_masks  # 注释：保存原始 mask
                    sliced_visual_mask = slice_input_tensor(visual_pos_masks, dim=1, padding=False)  # 注释：切分视觉 mask
                    call_kwargs["visual_pos_masks"] = sliced_visual_mask  # 注释：更新参数

                    if deepstack_visual_embeds is not None:  # 注释：存在视觉嵌入
                        sliced_embeds = []  # 注释：收集切分后的嵌入

                        num_visual_before = original_visual_mask.sum().item()  # 注释：原始视觉 token 总数
                        num_visual_in_shard = sliced_visual_mask.sum().item()  # 注释：本 shard 视觉 token 数

                        if num_visual_in_shard > 0 and num_visual_before > 0:  # 注释：有视觉 token
                            # Calculate which visual embeddings belong to this shard  # 注释：原注释保留
                            # We need to find the offset of visual tokens in this shard  # 注释：原注释保留
                            from verl.utils.ulysses import get_ulysses_sequence_parallel_rank  # 注释：获取 rank

                            rank = get_ulysses_sequence_parallel_rank()  # 注释：当前 rank
                            seq_len = original_visual_mask.shape[1]  # 注释：序列长度
                            local_seq_len = seq_len // current_ulysses_sp_size  # 注释：本 shard 长度
                            start_idx = rank * local_seq_len  # 注释：切片起点
                            end_idx = start_idx + local_seq_len  # 注释：切片终点

                            # Get total visual tokens before and up to the end of the shard's sequence slice  # 注释：原注释保留
                            # This correctly handles batches by summing across all samples  # 注释：原注释保留
                            visual_start = original_visual_mask[:, :start_idx].sum().item() if start_idx > 0 else 0  # 注释：视觉 token 起始偏移
                            visual_end = original_visual_mask[:, :end_idx].sum().item()  # 注释：视觉 token 结束偏移

                            # Slice each tensor in deepstack_visual_embeds  # 注释：原注释保留
                            for embed in deepstack_visual_embeds:  # 注释：遍历每个嵌入
                                sliced_embeds.append(embed[visual_start:visual_end])  # 注释：切片并追加
                        else:  # 注释：本 shard 无视觉 token
                            # No visual tokens in this shard, create empty tensors to maintain gradient flow  # 注释：原注释保留
                            for embed in deepstack_visual_embeds:  # 注释：遍历每个嵌入
                                sliced_embeds.append(embed[:0])  # 注释：追加空切片
                        call_kwargs["deepstack_visual_embeds"] = sliced_embeds  # 注释：更新参数

                self._needs_initial_slice = False  # 注释：标记已切分
            try:  # 注释：调用原始 forward
                return original_forward(self, *args, **call_kwargs)  # 注释：执行原始 forward
            finally:  # 注释：确保恢复标记
                if slice_now:  # 注释：若做过切片
                    self._needs_initial_slice = True  # 注释：重置标记

        return ulysses_wrapped_decoder_forward  # 注释：返回包装函数

    original_forward = model_class.forward  # 注释：保存原始 forward
    wrapped_forward = _create_ulysses_wrapped_decoder_forward(original_forward)  # 注释：创建包装 forward
    model_class.forward = wrapped_forward  # 注释：替换 forward
    print(f"Monkey patch {model_class.__name__}.forward for Ulysses SP input slicing.")  # 注释：打印提示
# （分隔说明：选择 fused kernel 后端）  # 注释：替代空行，保持逐行注释
def patch_forward_with_backends(  # 注释：根据后端选择 forward
    model: PreTrainedModel,  # 注释：模型实例
    use_fused_kernels: bool = False,  # 注释：是否启用 fused kernel
    fused_kernels_backend: str = None,  # 注释：后端名称
):
    """
    Choose the forward function based on the model and backend.  # 注释：保留英文说明
    Args:
        model (PreTrainedModel): The model to apply the monkey patch.
        use_fused_kernels (bool): Whether to use fused kernels.
        fused_kernels_backend (str): The backend to use for fused kernels.

    功能：根据模型类型选择 torch/triton fused kernel 的 forward 实现。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - model：目标模型。  # 注释：参数含义
    - use_fused_kernels：是否启用 fused kernel。  # 注释：参数含义
    - fused_kernels_backend：后端名称（"triton"/"torch"）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None（就地修改 model.__class__.forward）。  # 注释：返回值语义
    副作用：修改模型类的 forward 方法。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 后端不支持时抛 ValueError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - patch_forward_with_backends(model, True, "triton")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/monkey_patch.py::patch_forward_with_backends。  # 注释：函数位置
    - 典型调用路径：apply_monkey_patch -> patch_forward_with_backends。  # 注释：典型调用链
    - 被谁调用：本文件 apply_monkey_patch。  # 注释：调用方说明
    - 调用了谁（项目内）：各模型 forward_with_*_backend。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if not use_fused_kernels or fused_kernels_backend not in ["triton", "torch"]:  # 注释：条件不满足则跳过
        print(  # 注释：打印跳过原因
            f"Skipping monkey patch for {model.__class__.__name__} as use_fused_kernels is "  # 注释：提示信息1
            f"{use_fused_kernels} or fused_kernels_backend is {fused_kernels_backend}"  # 注释：提示信息2
        )  # 注释：print 结束
        return  # 注释：提前返回

    forward_with_torch_backend_function = model.__class__.forward  # 注释：默认 torch 后端 forward
    forward_with_triton_backend_function = model.__class__.forward  # 注释：默认 triton 后端 forward
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:  # 注释：Qwen2-VL
        from verl.models.transformers.qwen2_vl import forward_with_torch_backend, forward_with_triton_backend  # 注释：导入 Qwen2-VL forward

        forward_with_torch_backend_function = forward_with_torch_backend  # 注释：设置 torch 后端
        forward_with_triton_backend_function = forward_with_triton_backend  # 注释：设置 triton 后端
    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:  # 注释：Qwen3-VL
        from verl.models.transformers.qwen3_vl import forward_with_torch_backend, forward_with_triton_backend  # 注释：导入 Qwen3-VL forward

        forward_with_torch_backend_function = forward_with_torch_backend  # 注释：设置 torch 后端
        forward_with_triton_backend_function = forward_with_triton_backend  # 注释：设置 triton 后端
    elif model.config.model_type == "glm4v":  # 注释：GLM4V
        from verl.models.transformers.glm4v import forward_with_torch_backend, forward_with_triton_backend  # 注释：导入 GLM4V forward

        forward_with_torch_backend_function = forward_with_torch_backend  # 注释：设置 torch 后端
        forward_with_triton_backend_function = forward_with_triton_backend  # 注释：设置 triton 后端
    else:  # 注释：默认 dense 模型
        from verl.models.transformers.dense_common import forward_with_torch_backend, forward_with_triton_backend  # 注释：导入通用 forward

        forward_with_torch_backend_function = forward_with_torch_backend  # 注释：设置 torch 后端
        forward_with_triton_backend_function = forward_with_triton_backend  # 注释：设置 triton 后端

    if fused_kernels_backend == "triton":  # 注释：选择 triton 后端
        model.__class__.forward = forward_with_triton_backend_function  # 注释：替换 forward
        print(f"Using Triton backend for fused kernels in {model.__class__.__name__}")  # 注释：提示信息
    elif fused_kernels_backend == "torch":  # 注释：选择 torch 后端
        model.__class__.forward = forward_with_torch_backend_function  # 注释：替换 forward
        print(f"Using Torch backend for fused kernels in {model.__class__.__name__}")  # 注释：提示信息
    else:  # 注释：非法后端
        raise ValueError(f"Unsupported fused_kernels_backend: {fused_kernels_backend}. Choose 'triton' or 'torch'.")  # 注释：抛异常
# （分隔说明：apply_monkey_patch 主入口）  # 注释：替代空行，保持逐行注释
def apply_monkey_patch(  # 注释：应用 monkey patch 主入口
    model: PreTrainedModel,  # 注释：模型实例
    ulysses_sp_size: int = 1,  # 注释：序列并行 size
    use_remove_padding: bool = True,  # 注释：是否启用移除 padding
    use_fused_kernels: bool = False,  # 注释：是否启用 fused kernels
    fused_kernels_backend: str = None,  # 注释：后端名称
):
    """
    Apply monkey patch to the models for ulysses sequence parallel and fused kernel.

    In the end of this function forward function of the model is patched for fused kernel.
    If the model is not supported with fused kernel, please return after patch.

    功能：为不同模型打补丁，支持 Ulysses 序列并行与 fused kernel。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - model：目标模型。  # 注释：参数含义
    - ulysses_sp_size：序列并行大小。  # 注释：参数含义
    - use_remove_padding：是否移除 padding（flash-attn rmpad）。  # 注释：参数含义
    - use_fused_kernels：是否启用 fused kernel。  # 注释：参数含义
    - fused_kernels_backend：后端名称（triton/torch）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None（就地修改模型类与模块函数）。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 修改 transformers 模块/模型类的 forward 实现。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - heads 与 sp_size 不整除会 assert。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - apply_monkey_patch(model, ulysses_sp_size=2, use_remove_padding=True)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/monkey_patch.py::apply_monkey_patch。  # 注释：函数位置
    - 典型调用路径：fsdp_workers -> apply_monkey_patch -> patch_forward_with_backends。  # 注释：典型调用链
    - 被谁调用：verl/workers/fsdp_workers.py、verl/trainer/fsdp_sft_trainer.py、tests/models/test_transformers_ulysses.py。  # 注释：调用方示例
    - 调用了谁（项目内）：_ulysses_flash_attention_forward、patch_forward_with_backends、patch_vlm_for_ulysses_input_slicing。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers 模型类。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束

    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""  # 注释：原注释保留
    module = sys.modules[model.__module__]  # 注释：获取模型所在模块

    try:  # 注释：尝试从 config 获取头数
        num_attention_heads, num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads  # 注释：读取头数
    except AttributeError:  # 注释：部分 VLM 结构不同
        num_attention_heads, num_key_value_heads = (  # 注释：从 text_config 获取
            model.config.text_config.num_attention_heads,  # 注释：文本头数
            model.config.text_config.num_key_value_heads,  # 注释：文本 KV 头数
        )  # 注释：赋值结束

    assert num_attention_heads % ulysses_sp_size == 0, (  # 注释：attention heads 必须可整除
        f"num_attention_heads {num_attention_heads} must be divisible by ulysses_sp_size {ulysses_sp_size}"  # 注释：错误信息
    )  # 注释：assert 结束
    assert num_key_value_heads % ulysses_sp_size == 0 or ulysses_sp_size % num_key_value_heads == 0, (  # 注释：KV 头数可整除或被整除
        f"num_key_value_heads {num_key_value_heads} must be divisible by ulysses_sp_size "  # 注释：错误信息1
        f"{ulysses_sp_size}or vise versa. Upon ulysses_sp_size % num_key_value_heads == 0,"  # 注释：错误信息2
        f"kv heads are repeated to ensure correctness."  # 注释：错误信息3
    )  # 注释：assert 结束

    if is_trl_available():  # 注释：若 trl 可用
        from trl import AutoModelForCausalLMWithValueHead  # type: ignore  # 注释：导入 trl 模型

        def state_dict(self, *args, **kwargs):  # 注释：覆盖 state_dict
            return torch.nn.Module.state_dict(self, *args, **kwargs)  # 注释：调用基类 state_dict

        AutoModelForCausalLMWithValueHead.state_dict = state_dict  # 注释：替换 state_dict
        print("Monkey patch state_dict in AutoModelForCausalLMWithValueHead. ")  # 注释：打印提示

    # TODO: VLM models only, unify monkey patch to LLM models.  # 注释：原 TODO
    if model.config.model_type in ["qwen2_5_vl", "qwen2_vl"]:  # 注释：Qwen2-VL
        # Step 1: patch model to support image-text mixed data  # 注释：原注释保留
        if is_transformers_version_in_range(min_version="4.52.0"):  # 注释：新版本导入路径
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # 注释：导入 Qwen2.5-VL 模型
                Qwen2_5_VLForConditionalGeneration,  # 注释：条件生成模型
                Qwen2_5_VLModel,  # 注释：基础模型
                Qwen2_5_VLTextModel,  # 注释：文本子模型
            )  # 注释：导入结束
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (  # 注释：导入 Qwen2-VL 模型
                Qwen2VLForConditionalGeneration,  # 注释：条件生成模型
                Qwen2VLModel,  # 注释：基础模型
                Qwen2VLTextModel,  # 注释：文本子模型
            )  # 注释：导入结束
        else:  # 注释：旧版本导入路径
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration  # 注释：条件生成模型
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel as Qwen2_5_VLTextModel  # 注释：旧版文本模型别名
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration  # 注释：条件生成模型
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as Qwen2VLTextModel  # 注释：旧版文本模型别名

            Qwen2_5_VLModel = SimpleNamespace(forward=None)  # 注释：占位对象
            Qwen2VLModel = SimpleNamespace(forward=None)  # 注释：占位对象

        from verl.models.transformers.qwen2_vl import forward_with_normal_backend, qwen2_vl_base_forward  # 注释：导入 patch forward

        Qwen2_5_VLModel.forward = qwen2_vl_base_forward  # 注释：替换基础模型 forward
        Qwen2VLModel.forward = qwen2_vl_base_forward  # 注释：替换基础模型 forward
        Qwen2_5_VLForConditionalGeneration.forward = forward_with_normal_backend  # 注释：替换生成模型 forward
        Qwen2VLForConditionalGeneration.forward = forward_with_normal_backend  # 注释：替换生成模型 forward
        print(f"Monkey patch {model.__class__.__name__} model forward")  # 注释：打印提示

        # Step 2: patch attention to support ulysses parallelism  # 注释：原注释保留
        if is_transformers_version_in_range(min_version="4.54.0"):  # 注释：新版 attention 类
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention  # 注释：Qwen2.5-VL attention
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention  # 注释：Qwen2-VL attention
        elif is_transformers_version_in_range(min_version="4.53.0"):  # 注释：不支持版本
            raise RuntimeError("Transformers 4.53.* is bugged. Use transformers 4.54.0 or later.")  # 注释：抛异常
        else:  # 注释：旧版类名
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # 注释：旧版 attention 类
                Qwen2_5_VLFlashAttention2 as Qwen2_5_VLAttention,  # 注释：别名为 attention
            )  # 注释：导入结束
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2 as Qwen2VLAttention  # 注释：旧版 attention 类

        if use_remove_padding or ulysses_sp_size > 1:  # 注释：需要 patch attention
            from verl.models.transformers.qwen2_vl import qwen2_vl_attn_forward  # 注释：导入 attention forward

            Qwen2_5_VLAttention.forward = qwen2_vl_attn_forward  # 注释：替换 Qwen2.5-VL attention forward
            Qwen2VLAttention.forward = qwen2_vl_attn_forward  # 注释：替换 Qwen2-VL attention forward
            print(f"Monkey patch {model.__class__.__name__} attention layer")  # 注释：打印提示

        # Step 3: patch input for multimodal sequence parallelism  # 注释：原注释保留
        if ulysses_sp_size > 1:  # 注释：开启序列并行
            patch_vlm_for_ulysses_input_slicing(Qwen2_5_VLTextModel)  # 注释：patch Qwen2.5 文本模型
            patch_vlm_for_ulysses_input_slicing(Qwen2VLTextModel)  # 注释：patch Qwen2 文本模型

    elif model.config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:  # 注释：Qwen3-VL
        # Step 1: patch model to support image-text mixed data  # 注释：原注释保留
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # 注释：导入 Qwen3-VL 模型
            Qwen3VLForConditionalGeneration,  # 注释：生成模型
            Qwen3VLModel,  # 注释：基础模型
            Qwen3VLTextModel,  # 注释：文本模型
        )  # 注释：导入结束
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (  # 注释：导入 Qwen3-VL-MoE 模型
            Qwen3VLMoeForConditionalGeneration,  # 注释：MoE 生成模型
            Qwen3VLMoeModel,  # 注释：MoE 基础模型
            Qwen3VLMoeTextModel,  # 注释：MoE 文本模型
        )  # 注释：导入结束

        from verl.models.transformers.qwen3_vl import (  # 注释：导入 Qwen3-VL patch
            forward_with_normal_backend,  # 注释：正常后端 forward
            patch_qwen3_vl_moe_sparse_moe_block_forward,  # 注释：修复 MoE block
            qwen3_vl_base_forward,  # 注释：基础 forward
        )  # 注释：导入结束

        Qwen3VLModel.forward = qwen3_vl_base_forward  # 注释：替换基础模型 forward
        Qwen3VLMoeModel.forward = qwen3_vl_base_forward  # 注释：替换 MoE 基础模型 forward
        Qwen3VLForConditionalGeneration.forward = forward_with_normal_backend  # 注释：替换生成模型 forward
        Qwen3VLMoeForConditionalGeneration.forward = forward_with_normal_backend  # 注释：替换 MoE 生成模型 forward
        print(f"Monkey patch {model.__class__.__name__} model forward")  # 注释：打印提示

        # Step 1.5: patch Qwen3VLMoeTextSparseMoeBlock to fix transformers 4.57.3 bug  # 注释：原注释保留
        if model.config.model_type == "qwen3_vl_moe" and is_transformers_version_in_range(max_version="4.57.3"):  # 注释：触发 bug 修复
            patch_qwen3_vl_moe_sparse_moe_block_forward()  # 注释：打补丁

        # Step 2: patch input for multimodal sequence parallelism  # 注释：原注释保留
        if ulysses_sp_size > 1:  # 注释：开启序列并行
            patch_vlm_for_ulysses_input_slicing(Qwen3VLTextModel)  # 注释：patch Qwen3 文本模型
            patch_vlm_for_ulysses_input_slicing(Qwen3VLMoeTextModel)  # 注释：patch MoE 文本模型

    elif model.config.model_type == "glm4v":  # 注释：GLM4V
        # Step 1: patch model to support image-text mixed data  # 注释：原注释保留

        from transformers.models.glm4v.modeling_glm4v import (  # 注释：导入 GLM4V 模型
            Glm4vForConditionalGeneration,  # 注释：生成模型
            Glm4vModel,  # 注释：基础模型
            Glm4vTextAttention,  # 注释：文本注意力
            Glm4vTextModel,  # 注释：文本模型
        )  # 注释：导入结束

        from verl.models.transformers.glm4v import forward_with_normal_backend, glm4v_base_forward  # 注释：导入 patch forward

        Glm4vModel.forward = glm4v_base_forward  # 注释：替换基础模型 forward
        Glm4vForConditionalGeneration.forward = forward_with_normal_backend  # 注释：替换生成模型 forward
        print(f"Monkey patch {model.__class__.__name__} model forward")  # 注释：打印提示

        # Step 2: patch attention to support ulysses parallelism  # 注释：原注释保留
        if use_remove_padding or ulysses_sp_size > 1:  # 注释：需要 patch attention
            from verl.models.transformers.glm4v import glm4v_attn_forward  # 注释：导入 attention forward

            Glm4vTextAttention.forward = glm4v_attn_forward  # 注释：替换 attention forward
            print(f"Monkey patch {model.__class__.__name__} attention layer")  # 注释：打印提示

        # Step 3: patch input for multimodal sequence parallelism  # 注释：原注释保留
        if ulysses_sp_size > 1:  # 注释：开启序列并行
            patch_vlm_for_ulysses_input_slicing(Glm4vTextModel)  # 注释：patch 文本模型

    elif model.config.model_type == "kimi_vl":  # 注释：KimiVL
        if use_remove_padding or ulysses_sp_size > 1:  # 注释：需要 patch attention
            # TODO: Changes need to be made when transformers are adapted.  # 注释：原 TODO
            from verl.models.transformers.kimi_vl import _ulysses_flash_attn_forward  # 注释：导入 attention forward

            module.DeepseekV3FlashAttention2.forward = _ulysses_flash_attn_forward  # 注释：替换 attention forward
            print("Monkey patch FlashAttention2.forward in KimiVL")  # 注释：打印提示

        if ulysses_sp_size > 1:  # 注释：开启序列并行
            patch_vlm_for_ulysses_input_slicing(module.DeepseekV3ForCausalLM)  # 注释：patch 模型输入切分

        if use_fused_kernels:  # 注释：不支持 fused kernels
            print("Not support fused kernels for KimiVL")  # 注释：打印提示

        return  # 注释：KimiVL 直接返回

    if use_remove_padding or ulysses_sp_size > 1:  # 注释：需要替换 flash attention
        if hasattr(module, "_flash_attention_forward"):  # 注释：旧版 transformers 或 legacy 模型
            module._flash_attention_forward = _ulysses_flash_attention_forward  # 注释：替换 module 内函数
            print(f"Monkey patch _flash_attention_forward in {model.__module__}")  # 注释：打印提示
        else:  # 注释：新版 transformers 集成路径
            from transformers.integrations import flash_attention  # 注释：导入集成模块

            flash_attention._flash_attention_forward = _ulysses_flash_attention_forward  # 注释：替换集成模块函数
            print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")  # 注释：打印提示

    patch_forward_with_backends(model, use_fused_kernels=use_fused_kernels, fused_kernels_backend=fused_kernels_backend)  # 注释：根据后端替换 forward
