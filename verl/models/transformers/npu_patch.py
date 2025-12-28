# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team  # 注释：版权声明（Qwen 与 HF）
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：提示许可证链接
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：为 Ascend NPU 场景提供 Qwen 系列模型的高性能算子替换。  # 注释：模块用途
输入/输出：输入张量与模型对象，输出为被替换 forward/函数的模型（就地修改）。  # 注释：模块输入输出概览
关键依赖：torch_npu、transformers Qwen2/Qwen3/Qwen3-VL 等模型实现。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- import verl.models.transformers.npu_patch  # 注释：导入即应用 patch
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：Ascend NPU 训练/推理启动脚本。  # 注释：上层入口举例
- 典型链路：导入模块 -> 替换 modeling_qwen*.forward -> NPU 优化实现。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束

import torch  # 注释：张量运算
import torch.nn.functional as F  # 注释：softmax 等函数
import torch_npu  # 注释：Ascend NPU 算子
from torch import nn  # 注释：nn 模块
from transformers.activations import ACT2FN  # 注释：激活函数映射表
from transformers.models.qwen2 import modeling_qwen2  # 注释：Qwen2 模型实现
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl  # 注释：Qwen2.5-VL 模型实现
from transformers.models.qwen3 import modeling_qwen3  # 注释：Qwen3 模型实现
from transformers.models.qwen3_moe import modeling_qwen3_moe  # 注释：Qwen3 MoE 模型实现
from transformers.models.qwen3_vl import modeling_qwen3_vl  # 注释：Qwen3-VL 模型实现
from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe  # 注释：Qwen3-VL MoE 模型实现
from transformers.utils import logging  # 注释：transformers 日志

logger = logging.get_logger(__name__)  # 注释：获取 logger


def rms_norm_forward_npu(self, x):  # 注释：NPU 优化 RMSNorm
    """NPU optimized implementation for RMSNorm.  # 注释：保留英文说明

    功能：使用 torch_npu.npu_rms_norm 加速 RMSNorm。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - x (Tensor)：输入张量。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - Tensor：归一化后的输出。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：输入 dtype 与 weight 不一致会自动 cast。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - modeling_qwen2.Qwen2RMSNorm.forward = rms_norm_forward_npu。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/npu_patch.py::rms_norm_forward_npu。  # 注释：函数位置
    - 典型调用路径：Qwen* RMSNorm.forward -> rms_norm_forward_npu。  # 注释：典型调用链
    - 被谁调用：本文件末尾 patch 语句。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch_npu.npu_rms_norm。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if x.dtype != self.weight.dtype:  # 注释：dtype 不一致
        x = x.to(self.weight.dtype)  # 注释：强制转换 dtype
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]  # 注释：调用 NPU RMSNorm


def silu_forward_npu(self, hidden_state):  # 注释：NPU 优化 SiLU/MLP
    """NPU optimized implementation for SiLU in `forward` func in MLP layer.  # 注释：保留英文说明

    功能：使用 npu_swiglu 加速 MLP 前向（gate+up+down）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - hidden_state (Tensor)：输入隐藏状态。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - Tensor：MLP 输出。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - modeling_qwen2.Qwen2MLP.forward = silu_forward_npu。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/npu_patch.py::silu_forward_npu。  # 注释：函数位置
    - 典型调用路径：Qwen* MLP.forward -> silu_forward_npu。  # 注释：典型调用链
    - 被谁调用：本文件末尾 patch 语句。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch_npu.npu_swiglu。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    gate_up = torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1)  # 注释：拼接 gate/up
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))  # 注释：swiglu 后再 down_proj


def apply_rotary_pos_emb_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):  # 注释：NPU 优化 RoPE
    """NPU optimized implementation for RoPE.  # 注释：保留英文说明

    功能：使用 npu_rotary_mul 加速 RoPE。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - q/k：查询/键张量。  # 注释：参数含义
    - cos/sin：RoPE 旋转参数。  # 注释：参数含义
    - position_ids：保留接口参数（未使用）。  # 注释：参数含义
    - unsqueeze_dim：扩展维度。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (q_embed, k_embed)。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_npu。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/models/transformers/npu_patch.py::apply_rotary_pos_emb_npu。  # 注释：函数位置
    - 典型调用路径：Qwen* 注意力 forward -> apply_rotary_pos_emb_npu。  # 注释：典型调用链
    - 被谁调用：本文件末尾 patch 语句。  # 注释：调用方说明
    - 调用了谁（关键外部依赖）：torch_npu.npu_rotary_mul。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    cos = cos.unsqueeze(unsqueeze_dim)  # 注释：扩展 cos 维度
    sin = sin.unsqueeze(unsqueeze_dim)  # 注释：扩展 sin 维度
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)  # 注释：NPU RoPE 计算 q
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)  # 注释：NPU RoPE 计算 k
    return q_embed.to(q.dtype), k_embed.to(k.dtype)  # 注释：返回并转回原 dtype


class NPUGmmFunction(torch.autograd.Function):  # 注释：NPU grouped matmul 自定义 autograd
    @staticmethod  # 注释：静态方法
    def forward(ctx, x, weight, group_list, group_list_type=1):  # 注释：前向计算
        """
        Grouped Matmul(GMM) for Ascend NPU.

        Args:
            x (torch.Tensor): Input tensor, shape (tokens_num * top_k, hidden_size)
            weight (torch.Tensor): Expert weights, shape (n_experts, hidden_size, intermediate_size)
            group_list (torch.Tensor): Expert token counts, shape (n_experts,)
                - type 0: cumsum of tokens per expert
                - type 1: direct tokens per expert (default)

        功能：使用 NPU 分组矩阵乘实现 MoE 高效计算。  # 注释：函数用途
        返回：  # 注释：返回值说明标题
        - Tensor：GMM 输出。  # 注释：返回值语义
        副作用：保存 ctx 以备反向。  # 注释：副作用说明
        最小示例：  # 注释：最小示例标题
        - NPUGmmFunction.apply(x, w, tokens_per_expert)。  # 注释：示例
        """  # 注释：函数 docstring 结束
        ctx.save_for_backward(x, weight)  # 注释：保存张量用于反向
        ctx.group_list = group_list  # 注释：保存 group_list
        ctx.group_list_type = group_list_type  # 注释：保存 group_list_type

        output = torch_npu.npu_grouped_matmul(  # 注释：调用 NPU grouped matmul
            [x], [weight], bias=None, group_list=group_list, split_item=2, group_type=0, group_list_type=group_list_type  # 注释：参数设置
        )[0]  # 注释：取输出

        return output  # 注释：返回输出

    @staticmethod  # 注释：静态方法
    def backward(ctx, grad_output):  # 注释：反向传播
        x, weight = ctx.saved_tensors  # 注释：读取保存的张量
        group_list = ctx.group_list  # 注释：读取 group_list
        group_list_type = ctx.group_list_type  # 注释：读取 group_list_type

        dx = torch_npu.npu_grouped_matmul(  # 注释：计算输入梯度
            [grad_output],
            [weight.transpose(1, 2)],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=group_list_type,
        )[0]  # 注释：取 dx

        dw = torch_npu.npu_grouped_matmul(  # 注释：计算权重梯度
            [x.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=group_list_type,
        )[0]  # 注释：取 dw

        return dx, dw, None, None  # 注释：返回梯度


def qwen3_moe_sparse_moe_block_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 注释：NPU 优化 MoE block
    """NPU optimized implementation for `forward` in Qwen3MoeSparseMoeBlock.  # 注释：保留英文说明

    功能：使用 NPU grouped matmul 加速 Qwen3 MoE 前向。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - hidden_states (Tensor)：输入隐藏状态。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (final_hidden_states, router_logits)。  # 注释：返回值语义
    最小示例：  # 注释：最小示例标题
    - modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = qwen3_moe_sparse_moe_block_forward_npu。  # 注释：示例
    """  # 注释：函数 docstring 结束
    # hidden_states: (batch_size, sequence_length, hidden_size)  # 注释：输入形状说明
    hidden_dim = hidden_states.shape[-1]  # 注释：隐藏维度
    hidden_states = hidden_states.view(-1, hidden_dim)  # 注释：展平 batch*seq
    # router_logits: (batch * sequence_length, n_experts)  # 注释：router 输出形状
    router_logits = self.gate(hidden_states)  # 注释：计算路由 logits

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)  # 注释：softmax 得到权重
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)  # 注释：选择 top-k 专家
    if self.norm_topk_prob:  # only diff with mixtral sparse moe block!  # 注释：是否归一化 top-k
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # 注释：归一化
    # we cast back to the input dtype  # 注释：原注释保留
    routing_weights = routing_weights.to(hidden_states.dtype)  # 注释：转回输入 dtype

    # Loop over all available experts in the model and perform the computation on each expert  # 注释：原注释保留
    # Concat all weights  # 注释：原注释保留
    input_dtype = hidden_states.dtype  # 注释：记录 dtype
    up_weight_list = [e.up_proj.weight for e in self.experts]  # 注释：收集 up_proj 权重
    gate_weight_list = [e.gate_proj.weight for e in self.experts]  # 注释：收集 gate_proj 权重
    down_weight_list = [e.down_proj.weight for e in self.experts]  # 注释：收集 down_proj 权重
    w1 = torch.stack(up_weight_list).transpose(1, 2).to(input_dtype)  # 注释：拼接 up 权重
    w2 = torch.stack(gate_weight_list).transpose(1, 2).to(input_dtype)  # 注释：拼接 gate 权重
    w3 = torch.stack(down_weight_list).transpose(1, 2).to(input_dtype)  # 注释：拼接 down 权重

    permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))  # 注释：token 重排
    tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)  # 注释：统计每专家 token 数

    up_res = NPUGmmFunction.apply(permuted_tokens, w1, tokens_per_expert)  # 注释：上投影
    gate_res = NPUGmmFunction.apply(permuted_tokens, w2, tokens_per_expert)  # 注释：门控投影
    act_res = torch_npu.npu_swiglu(torch.cat([gate_res, up_res], dim=-1))  # 注释：swiglu 激活
    down_res = NPUGmmFunction.apply(act_res, w3, tokens_per_expert)  # 注释：下投影

    final_hidden_states = torch_npu.npu_moe_token_unpermute(down_res, row_ids_map, probs=routing_weights)  # 注释：反重排

    return final_hidden_states, router_logits  # 注释：返回输出与路由 logits


class NPUQwen3VLMoeTextExperts(nn.Module):  # 注释：NPU 优化 Qwen3-VL MoE 专家实现
    """NPU optimized implementation for Qwen3VLMoeTextExperts."""  # 注释：保留英文说明

    def __init__(self, config):  # 注释：初始化专家权重
        super().__init__()  # 注释：初始化父类
        self.num_experts = config.num_experts  # 注释：专家数量
        self.intermediate_size = config.moe_intermediate_size  # 注释：中间层维度
        self.hidden_size = config.hidden_size  # 注释：隐藏维度
        self.expert_dim = self.intermediate_size  # 注释：专家维度
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))  # 注释：gate+up 权重
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))  # 注释：down 权重
        self.act_fn = ACT2FN[config.hidden_act]  # 注释：激活函数

    def forward(  # 注释：前向
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor  # 注释：输入参数
    ) -> torch.Tensor:  # 注释：返回类型
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once.
        By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """  # 注释：函数 docstring 结束
        batch_size = hidden_states.shape[0]  # 注释：batch 大小
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)  # 注释：展平
        if self.training:  # 注释：训练路径
            permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(  # 注释：重排 token
                hidden_states, router_indices.to(torch.int32)  # 注释：输入与 indices
            )  # 注释：重排结束
            tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)  # 注释：统计 token
            intermediate_hidden_states = NPUGmmFunction.apply(  # 注释：GMM 前向
                permuted_hidden_states, self.gate_up_proj, tokens_per_expert  # 注释：输入/权重/分组
            )  # 注释：GMM 输出
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)  # 注释：swiglu 激活
            output = NPUGmmFunction.apply(intermediate_activations, self.down_proj, tokens_per_expert)  # 注释：下投影
            next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)  # 注释：反重排
            next_states = next_states.view(batch_size, -1, self.hidden_size)  # 注释：恢复形状
        else:  # 注释：推理路径
            hidden_states = hidden_states.repeat(self.num_experts, 1)  # 注释：重复输入
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)  # 注释：重排形状
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)  # 注释：批量矩阵乘
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors  # 注释：拆分 gate/up
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)  # 注释：激活并下投影
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)  # 注释：重排
            next_states = (  # 注释：加权汇聚
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]  # 注释：加权
            )  # 注释：加权结束
            next_states = next_states.sum(dim=0)  # 注释：沿专家维求和
        return next_states  # 注释：返回输出


class NPUQwen3VLMoeTextSparseMoeBlock(nn.Module):  # 注释：NPU 优化 Qwen3-VL MoE block
    """NPU optimized implementation for Qwen3VLMoeTextSparseMoeBlock."""  # 注释：保留英文说明

    def __init__(self, config):  # 注释：初始化 MoE block
        super().__init__()  # 注释：初始化父类
        self.hidden_size = config.hidden_size  # 注释：隐藏维度
        self.num_experts = config.num_experts  # 注释：专家数量
        self.top_k = config.num_experts_per_tok  # 注释：每 token 专家数
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)  # 注释：门控网络
        self.experts = NPUQwen3VLMoeTextExperts(config)  # 注释：专家实现

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # 注释：前向
        batch_size = hidden_states.shape[0]  # 注释：batch 大小
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # 注释：展平
        router_logits = self.gate(hidden_states)  # 注释：路由 logits
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)  # 注释：softmax 权重
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # 注释：top-k 选择
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # 注释：归一化权重
        routing_weights = routing_weights.to(router_logits.dtype)  # 注释：转 dtype
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)  # 注释：恢复形状
        if not self.training:  # 注释：推理时需要稀疏权重
            routing_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)  # 注释：构造稀疏权重
        routed_out = self.experts(hidden_states, routing_weights, router_indices)  # 注释：专家前向
        return routed_out  # 注释：返回输出


# Patches for Qwen2 Model  # 注释：Qwen2 patch
modeling_qwen2.Qwen2RMSNorm.forward = rms_norm_forward_npu  # 注释：替换 RMSNorm
modeling_qwen2.Qwen2MLP.forward = silu_forward_npu  # 注释：替换 MLP
modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb_npu  # 注释：替换 RoPE

# Patches for Qwen2.5-VL Model  # 注释：Qwen2.5-VL patch
modeling_qwen2_5_vl.Qwen2RMSNorm.forward = rms_norm_forward_npu  # 注释：替换 RMSNorm
modeling_qwen2_5_vl.Qwen2_5_VLMLP.forward = silu_forward_npu  # 注释：替换 MLP

# Patches for Qwen3 Model  # 注释：Qwen3 patch
modeling_qwen3.Qwen3RMSNorm.forward = rms_norm_forward_npu  # 注释：替换 RMSNorm
modeling_qwen3.Qwen3MLP.forward = silu_forward_npu  # 注释：替换 MLP
modeling_qwen3.apply_rotary_pos_emb = apply_rotary_pos_emb_npu  # 注释：替换 RoPE

# Patches for Qwen3 MoE Model  # 注释：Qwen3 MoE patch
modeling_qwen3_moe.Qwen3MoeRMSNorm.forward = rms_norm_forward_npu  # 注释：替换 RMSNorm
modeling_qwen3_moe.Qwen3MoeSparseMoeBlock.forward = qwen3_moe_sparse_moe_block_forward_npu  # 注释：替换 MoE block forward
modeling_qwen3_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_npu  # 注释：替换 RoPE

# Patches for Qwen3 VL Model  # 注释：Qwen3-VL patch
modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = rms_norm_forward_npu  # 注释：替换 RMSNorm
modeling_qwen3_vl.Qwen3VLTextMLP.forward = silu_forward_npu  # 注释：替换 MLP

# Patches for Qwen3-VL MoE Model  # 注释：Qwen3-VL MoE patch
modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = NPUQwen3VLMoeTextSparseMoeBlock  # 注释：替换 MoE block 类
modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm.forward = rms_norm_forward_npu  # 注释：替换 RMSNorm
modeling_qwen3_vl_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_npu  # 注释：替换 RoPE
