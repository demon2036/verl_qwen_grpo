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
模块用途：为 Qwen2 注意力提供支持 Ulysses 序列并行的 forward 实现。  # 注释：模块用途
输入/输出：输入 Q/K/V 与 mask 等，输出 attention 输出与 cache。  # 注释：模块输入输出概览
关键依赖：transformers flash attention、RoPE、Ulysses 并行工具。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- Qwen2Attention.forward = qwen2_flash_attn_forward  # 注释：运行时替换 forward
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：模型补丁逻辑（如 monkey_patch 中动态替换 forward）。  # 注释：上层入口举例
- 典型链路：apply_monkey_patch -> 替换 attention forward -> qwen2_*_forward。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
from typing import Callable, Optional  # 注释：类型注解
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import torch  # 注释：张量操作
from transformers.cache_utils import Cache  # 注释：KV cache 类型
from transformers.modeling_flash_attention_utils import _flash_attention_forward  # 注释：flash attention 前向
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv  # 注释：RoPE 与 KV 重复
from transformers.utils import logging  # 注释：transformers 日志
# （分隔说明：兼容性封装）  # 注释：替代空行，保持逐行注释
# Import compatibility wrapper for flash_attn_supports_top_left_mask  # 注释：原注释保留
from verl.utils.transformers_compat import flash_attn_supports_top_left_mask  # 注释：mask 支持检测
from verl.utils.ulysses import (  # 注释：Ulysses 序列并行工具
    gather_heads_scatter_seq,  # 注释：all2all（head->seq）
    gather_seq_scatter_heads,  # 注释：all2all（seq->head）
    get_ulysses_sequence_parallel_world_size,  # 注释：获取序列并行 size
    validate_ulysses_config,  # 注释：校验配置
)  # 注释：导入结束
# （分隔说明：logger）  # 注释：替代空行，保持逐行注释
logger = logging.get_logger(__name__)  # 注释：获取 logger
# （分隔说明：Qwen2 flash-attn forward）  # 注释：替代空行，保持逐行注释
def qwen2_flash_attn_forward(  # 注释：Qwen2 flash-attn 版本 forward
    self,  # 注释：attention 模块实例
    hidden_states: torch.Tensor,  # 注释：输入 hidden states
    attention_mask: Optional[torch.Tensor] = None,  # 注释：attention mask
    position_ids: Optional[torch.LongTensor] = None,  # 注释：position ids
    past_key_value: Optional[Cache] = None,  # 注释：KV cache
    output_attentions: bool = False,  # 注释：是否输出注意力权重
    use_cache: bool = False,  # 注释：是否使用 cache
    cache_position: Optional[torch.LongTensor] = None,  # 注释：cache position
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46  # 注释：外部 RoPE embeddings
):
    """
    Adapted from transformers 4.47.1 to support Ulysses sequence parallelism.

    NOTE: This function is only tested on transformers versions between 4.45.0 and 4.47.1.

    功能：Qwen2 flash-attn 前向，支持 Ulysses 序列并行 all2all。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - hidden_states：输入隐藏状态 (B,S,H)。  # 注释：参数含义
    - attention_mask：注意力 mask。  # 注释：参数含义
    - position_ids：位置索引。  # 注释：参数含义
    - past_key_value：KV cache。  # 注释：参数含义
    - output_attentions：是否返回注意力权重。  # 注释：参数含义
    - use_cache：是否使用 cache（接口保留）。  # 注释：参数含义
    - cache_position：cache position。  # 注释：参数含义
    - position_embeddings：外部 RoPE embeddings。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (attn_output, attn_weights, past_key_value)。  # 注释：返回值语义
    副作用：可能更新 KV cache。  # 注释：副作用说明
    异常/边界条件：仅在 4.45~4.47.1 测试。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - Qwen2Attention.forward = qwen2_flash_attn_forward。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/qwen2.py::qwen2_flash_attn_forward。  # 注释：函数位置
    - 典型调用路径：运行时 patch -> Qwen2Attention.forward -> qwen2_flash_attn_forward。  # 注释：典型调用链
    - 被谁调用：运行时 monkey patch（仓库内未直接引用）。  # 注释：调用方说明
    - 调用了谁（项目内）：gather_seq_scatter_heads/gather_heads_scatter_seq。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：_flash_attention_forward、apply_rotary_pos_emb、repeat_kv。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    bsz, q_len, _ = hidden_states.size()  # 注释：解析 batch/seq 维度

    query_states = self.q_proj(hidden_states)  # 注释：投影 Q
    key_states = self.k_proj(hidden_states)  # 注释：投影 K
    value_states = self.v_proj(hidden_states)  # 注释：投影 V

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # 注释：reshape 多头 Q
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # 注释：reshape 多头 K
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # 注释：reshape 多头 V

    ########## AlltoAll for Ulysses ##########  # 注释：分隔标记
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()  # 注释：获取序列并行 size

    if ulysses_sp_size > 1:  # 注释：开启序列并行
        validate_ulysses_config(self.num_heads, ulysses_sp_size)  # 注释：校验配置

        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)  # 注释：原注释保留
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)  # 注释：all2all 交换
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)  # 注释：all2all 交换
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)  # 注释：all2all 交换

    full_q_len = query_states.size(2)  # full seq length  # 注释：完整序列长度

    if position_embeddings is None:  # 注释：未提供外部 RoPE
        logger.warning_once(  # 注释：提示即将废弃 position_ids
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "  # 注释：提示文本1
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "  # 注释：提示文本2
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "  # 注释：提示文本3
            "removed and `position_embeddings` will be mandatory."  # 注释：提示文本4
        )  # 注释：warning 结束
        cos, sin = self.rotary_emb(value_states, position_ids)  # 注释：内部计算 RoPE
    else:  # 注释：使用外部 RoPE
        cos, sin = position_embeddings  # 注释：解包 cos/sin
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 注释：应用 RoPE

    if past_key_value is not None:  # 注释：有 KV cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models  # 注释：cache 参数
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)  # 注释：更新 cache

    # repeat k/v heads if n_kv_heads < n_heads  # 注释：原注释保留
    key_states = repeat_kv(key_states, self.num_key_value_groups)  # 注释：重复 K
    value_states = repeat_kv(value_states, self.num_key_value_groups)  # 注释：重复 V
    dropout_rate = 0.0 if not self.training else self.attention_dropout  # 注释：设置 dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons  # 注释：原注释保留
    # therefore the input hidden states gets silently casted in float32. Hence, we need  # 注释：原注释保留
    # cast them back in float16 just to be sure everything works as expected.  # 注释：原注释保留
    input_dtype = query_states.dtype  # 注释：记录输入 dtype
    if input_dtype == torch.float32:  # 注释：若为 float32
        if torch.is_autocast_enabled():  # 注释：自动混合精度
            target_dtype = torch.get_autocast_gpu_dtype()  # 注释：获取自动 dtype
        # Handle the case where the model is quantized  # 注释：原注释保留
        elif hasattr(self.config, "_pre_quantization_dtype"):  # 注释：量化模型
            target_dtype = self.config._pre_quantization_dtype  # 注释：使用预量化 dtype
        else:  # 注释：默认
            target_dtype = self.q_proj.weight.dtype  # 注释：使用权重 dtype

        logger.warning_once(  # 注释：提示 dtype 回退
            f"The input hidden states seems to be silently casted in float32, this might be related to "  # 注释：提示文本1
            f"the fact you have upcasted embedding or layer norm layers in float32. We will cast back the "  # 注释：提示文本2
            f"input in {target_dtype}."  # 注释：提示文本3
        )  # 注释：warning 结束

        query_states = query_states.to(target_dtype)  # 注释：转换 dtype
        key_states = key_states.to(target_dtype)  # 注释：转换 dtype
        value_states = value_states.to(target_dtype)  # 注释：转换 dtype

    # Reashape to the expected shape for Flash Attention  # 注释：原注释保留
    query_states = query_states.transpose(1, 2)  # 注释：转置
    key_states = key_states.transpose(1, 2)  # 注释：转置
    value_states = value_states.transpose(1, 2)  # 注释：转置

    if (  # 注释：判断 sliding window
        self.config.use_sliding_window  # 注释：是否启用 sliding window
        and getattr(self.config, "sliding_window", None) is not None  # 注释：是否配置
        and self.layer_idx >= self.config.max_window_layers  # 注释：层索引条件
    ):  # 注释：条件结束
        sliding_window = self.config.sliding_window  # 注释：设置 sliding window
    else:  # 注释：不使用 window
        sliding_window = None  # 注释：置空

    attn_output = _flash_attention_forward(  # 注释：调用 flash-attn
        query_states,  # 注释：Q
        key_states,  # 注释：K
        value_states,  # 注释：V
        attention_mask,  # 注释：mask
        full_q_len,  # 注释：完整序列长度
        position_ids=position_ids,  # 注释：位置索引
        dropout=dropout_rate,  # 注释：dropout
        sliding_window=sliding_window,  # 注释：sliding window
        is_causal=self.is_causal,  # 注释：因果标志
        use_top_left_mask=flash_attn_supports_top_left_mask(),  # 注释：mask 支持检测
    )  # 注释：flash-attn 输出

    # use full_q_len to reshape  # 注释：原注释保留
    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()  # 注释：恢复形状
    ########## AlltoAll for Ulysses ##########  # 注释：分隔标记
    if ulysses_sp_size > 1:  # 注释：开启序列并行
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)  # 注释：all2all 交换回
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()  # 注释：恢复形状
    attn_output = self.o_proj(attn_output)  # 注释：输出投影

    if not output_attentions:  # 注释：若不输出注意力权重
        attn_weights = None  # 注释：置空

    return attn_output, attn_weights, past_key_value  # 注释：返回结果


# （分隔说明：Qwen2 attention forward（新版 transformers））  # 注释：替代空行，保持逐行注释
def qwen2_attn_forward(  # 注释：Qwen2 attention forward（>=4.48）
    self,  # 注释：attention 模块实例
    hidden_states: torch.Tensor,  # 注释：输入 hidden states
    position_embeddings: tuple[torch.Tensor, torch.Tensor],  # 注释：外部 RoPE embeddings
    attention_mask: Optional[torch.Tensor],  # 注释：attention mask
    past_key_value: Optional[Cache] = None,  # 注释：KV cache
    cache_position: Optional[torch.LongTensor] = None,  # 注释：cache position
    **kwargs,  # 注释：其他参数
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:  # 注释：返回类型
    """
    Adapted from transformers 4.49.0 to support Ulysses sequence parallelism for transformers >= 4.48.0.

    NOTE: This function has been tested only on transformers versions between 4.48.0 and 4.50.0.

    功能：新版 transformers 注意力 forward，支持 Ulysses 序列并行。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - hidden_states：输入 hidden states。  # 注释：参数含义
    - position_embeddings：外部 RoPE cos/sin。  # 注释：参数含义
    - attention_mask：attention mask。  # 注释：参数含义
    - past_key_value：KV cache。  # 注释：参数含义
    - cache_position：cache position。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (attn_output, attn_weights, past_key_value)。  # 注释：返回值语义
    副作用：可能更新 KV cache。  # 注释：副作用说明
    异常/边界条件：仅在 4.48~4.50 测试。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - Qwen2Attention.forward = qwen2_attn_forward。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/qwen2.py::qwen2_attn_forward。  # 注释：函数位置
    - 典型调用路径：运行时 patch -> Qwen2Attention.forward -> qwen2_attn_forward。  # 注释：典型调用链
    - 被谁调用：运行时 monkey patch（仓库内未直接引用）。  # 注释：调用方说明
    - 调用了谁（项目内）：gather_seq_scatter_heads/gather_heads_scatter_seq。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：ALL_ATTENTION_FUNCTIONS、apply_rotary_pos_emb。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # 注释：导入注意力函数表

    bsz, q_len, _ = hidden_states.shape  # 注释：解析维度
    hidden_shape = (bsz, q_len, -1, self.head_dim)  # 注释：多头形状

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # 注释：投影并 reshape Q
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # 注释：投影并 reshape K
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # 注释：投影并 reshape V

    ########## AlltoAll for Ulysses ##########  # 注释：分隔标记
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()  # 注释：获取序列并行 size

    if ulysses_sp_size > 1:  # 注释：开启序列并行
        validate_ulysses_config(self.config.num_attention_heads, ulysses_sp_size)  # 注释：校验配置

        # (bsz, n_head, seq_len/n, head_dim) -> (bsz, n_head/n, seq_len, head_dim)  # 注释：原注释保留
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)  # 注释：all2all 交换
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)  # 注释：all2all 交换
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)  # 注释：all2all 交换

    full_q_len = query_states.size(2)  # 注释：完整序列长度

    cos, sin = position_embeddings  # 注释：解包 RoPE
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # 注释：应用 RoPE

    if past_key_value is not None:  # 注释：有 KV cache
        # sin and cos are specific to RoPE models; cache_position needed for the static cache  # 注释：原注释保留
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # 注释：cache 参数
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)  # 注释：更新 cache

    sliding_window = None  # 注释：默认不使用 sliding window
    if (  # 注释：判断是否启用 sliding window
        self.config.use_sliding_window  # 注释：是否开启 sliding window
        and getattr(self.config, "sliding_window", None) is not None  # 注释：是否配置
        and self.layer_idx >= self.config.max_window_layers  # 注释：层索引条件
    ):  # 注释：条件结束
        sliding_window = self.config.sliding_window  # 注释：设置 sliding window

    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward  # 注释：导入 eager attention

    attention_interface: Callable = eager_attention_forward  # 注释：默认 eager attention
    if self.config._attn_implementation != "eager":  # 注释：非 eager 实现
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):  # 注释：sdpa 不支持输出权重
            logger.warning_once(  # 注释：提示回退
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. "  # 注释：提示文本1
                "Falling back to eager attention. This warning can be removed using the argument "  # 注释：提示文本2
                '`attn_implementation="eager"` when loading the model.'  # 注释：提示文本3
            )  # 注释：warning 结束
        else:  # 注释：使用指定实现
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]  # 注释：选择 attention 函数

    attn_output, attn_weights = attention_interface(  # 注释：调用 attention 实现
        self,  # 注释：attention 模块
        query_states,  # 注释：Q
        key_states,  # 注释：K
        value_states,  # 注释：V
        attention_mask,  # 注释：mask
        dropout=0.0 if not self.training else self.attention_dropout,  # 注释：dropout
        scaling=self.scaling,  # 注释：缩放因子
        sliding_window=sliding_window,  # main diff with Llama  # 注释：滑窗配置
        **kwargs,  # 注释：透传其余参数
    )  # 注释：attention 输出

    attn_output = attn_output.reshape(bsz, full_q_len, -1, self.head_dim).contiguous()  # 注释：恢复形状
    ########## AlltoAll for Ulysses ##########  # 注释：分隔标记
    if ulysses_sp_size > 1:  # 注释：开启序列并行
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)  # 注释：原注释保留
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)  # 注释：all2all 交换回
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()  # 注释：恢复形状
    attn_output = self.o_proj(attn_output)  # 注释：输出投影
    return attn_output, attn_weights  # 注释：返回结果
