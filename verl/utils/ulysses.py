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

"""（模块说明：Ulysses 序列并行通信与张量重排工具，支持 all-to-all 实现序列维度与头维度的切换）

模块用途：
    本模块提供 Ulysses 序列并行（Sequence Parallelism, SP）的核心通信工具。
    Ulysses SP 是一种在注意力机制中对序列维度进行并行切分的技术，通过 all-to-all 通信在"序列切分"与"头切分"之间转换。
    主要功能包括：
    1. 序列并行进程组的管理（设置/获取全局 SP 组）
    2. all-to-all 通信：序列维度聚合 + 头维度切分（或反向）
    3. 输入张量的 padding 与切分（为 SP 做准备）
    4. 输出张量的聚合与去 padding（SP 后的还原）
    5. 自定义 autograd Function 支持前向/反向的通信

输入/输出：
    - 输入：
      * PyTorch 张量（通常形状为 [batch, seq, heads, dim]）
      * 序列维度索引（seq_dim）、头维度索引（head_dim）
      * 分布式进程组（ProcessGroup）
      * padding 相关参数（unpadded_dim_size、padding_size 等）
    - 输出：
      * 重排后的张量（序列/头维度切换）
      * 聚合后的全序列张量（可选去 padding）
      * 切分后的子序列张量（每个 rank 持有一部分）

关键依赖：
    - torch：张量计算与 autograd 机制
    - torch.distributed：分布式通信（all_to_all、all_gather、ProcessGroup）
    - typing：类型标注（Optional、Any）

典型用法：
    >>> from verl.utils.ulysses import gather_seq_scatter_heads, gather_outputs_and_unpad
    >>> # 场景1：注意力前向（序列聚合 + 头切分）
    >>> x = torch.randn(1, 4, 8, 64)  # [batch, seq/sp_world, heads, dim]
    >>> x = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)  # -> [batch, seq, heads/sp_world, dim]
    >>> # 场景2：聚合输出并去 padding
    >>> out = gather_outputs_and_unpad(x, gather_dim=1, unpad_dim=1, padding_size=2)

调用路径概览：
    入口脚本（如 verl/trainer/fsdp_sft_trainer.py 或 verl/trainer/main_ppo.py）
    -> verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager.__enter__()
    -> set_ulysses_sequence_parallel_group(sp_group)
    -> 模型前向传播
    -> verl/workers/engine/fsdp/transformer_impl.py 或模型自定义 forward
    -> 本模块函数（gather_seq_scatter_heads、gather_outputs_and_unpad 等）
    -> torch.distributed.all_to_all / all_gather（底层通信）

所在位置：
    - 路径：verl/utils/ulysses.py
    - 模块：verl.utils.ulysses

被谁调用：
    - verl/workers/sharding_manager/fsdp_ulysses.py（设置 SP 组）
    - verl/workers/engine/fsdp/transformer_impl.py（模型前向中调用通信工具）
    - verl/workers/actor/dp_actor.py（padding 与切分输入）
    - verl/trainer/fsdp_sft_trainer.py（初始化 sharding manager）
    - recipe/prime/prime_dp_rm.py（聚合输出）
    - tests/models/test_transformers_ulysses.py（单元测试）

调用了谁（项目内）：
    - 无（本模块为底层通信工具，不依赖其他项目内模块）

调用了谁（外部依赖）：
    - torch.distributed.all_to_all（all-to-all 通信）
    - torch.distributed.all_gather_into_tensor（all-gather 通信）
    - torch.distributed.get_world_size/get_rank（查询进程组信息）
    - torch.autograd.Function（自定义反向传播）
    - torch.cat / torch.zeros / torch.nn.functional.pad（张量操作）

注意事项：
    1. 序列长度必须能被 sp_world_size 整除，否则需要 padding
    2. 头数（num_heads）必须能被 sp_world_size 整除（见 validate_ulysses_config）
    3. all-to-all 通信会引入同步点，影响性能
    4. padding/unpadding 操作需要在前向与反向中保持一致
    5. 全局 SP 组通过模块级变量 _ULYSSES_SEQUENCE_PARALLEL_GROUP 管理，需在训练前设置

记忆提示：
    - Ulysses = 序列并行，通过 all-to-all 在"序列切分"与"头切分"之间转换
    - 前向：gather_seq_scatter_heads（序列聚合 + 头切分）
    - 反向：gather_heads_scatter_seq（头聚合 + 序列切分）
    - 输入准备：ulysses_pad_and_slice_inputs（padding + 切分）
    - 输出还原：gather_outputs_and_unpad（聚合 + 去 padding）
"""  # 注释：模块级 docstring 结束，下面是依赖导入

# ===== 标准库导入 =====
from typing import Any, Optional  # 注释：类型标注工具，Any 表示任意类型，Optional 表示可选参数

# ===== 第三方依赖导入 =====
import torch  # 注释：张量计算与 autograd 核心库
import torch.distributed as dist  # 注释：分布式通信接口（all_to_all、all_gather、ProcessGroup 等）
from torch import Tensor  # 注释：张量类型别名，用于类型标注
from torch.distributed import ProcessGroup  # 注释：进程组类型，用于指定通信范围

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None  # 注释：模块级全局变量，缓存 Ulysses 序列并行进程组（初始为 None）


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):  # 注释：函数定义，设置全局 Ulysses SP 进程组
    """（函数说明：设置 Ulysses 序列并行进程组，供后续通信使用）

    本函数用于在训练初始化阶段设置全局的 Ulysses 序列并行进程组。
    设置后，本模块的其他函数（如 gather_seq_scatter_heads）会使用该进程组进行通信。

    参数：
        group (ProcessGroup): 需要设置为全局的进程组对象
            - 类型：torch.distributed.ProcessGroup
            - 来源：通常由 torch.distributed.new_group() 创建
            - 作用：定义哪些进程参与序列并行通信
            - 取值范围：任意有效的 ProcessGroup 对象，或 None（清空进程组）

    返回：
        无（仅副作用，修改模块级全局变量）

    副作用：
        - 修改模块级全局变量 _ULYSSES_SEQUENCE_PARALLEL_GROUP
        - 影响后续所有调用 get_ulysses_sequence_parallel_group() 的函数

    异常/边界条件：
        - 允许传入 None 以清空进程组（SP 退化为单进程）
        - 多次调用会覆盖之前的设置

    最小示例（手算验证）：
        >>> import torch.distributed as dist
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group
        >>> # 初始状态
        >>> assert get_ulysses_sequence_parallel_group() is None
        >>> # 创建并设置进程组（假设 world_size=4，取 rank 0-1 作为 SP 组）
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 验证：后续调用可获取该进程组
        >>> assert get_ulysses_sequence_parallel_group() is sp_group

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：set_ulysses_sequence_parallel_group(group: ProcessGroup)

    典型调用路径：
        verl/trainer/fsdp_sft_trainer.py（或其他训练入口）
        -> verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager.__enter__()
        -> set_ulysses_sequence_parallel_group(sp_group)

    被谁调用：
        - verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager（上下文管理器）

    调用了谁（项目内）：
        - 无（仅修改全局变量）

    调用了谁（外部依赖）：
        - 无

    记忆提示：
        - 通常在训练初始化阶段调用一次
        - 设置后，所有 Ulysses 通信函数会自动使用该进程组
    """  # 注释：函数 docstring 结束
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP  # 注释：声明使用模块级全局变量（允许修改）
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group  # 注释：将传入的进程组赋值给全局变量，后续调用 get_ulysses_sequence_parallel_group() 会返回此值


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:  # 注释：函数定义，获取全局 Ulysses SP 进程组
    """（函数说明：获取当前全局的 Ulysses 序列并行进程组）

    本函数返回通过 set_ulysses_sequence_parallel_group() 设置的全局进程组。
    如果未设置，返回 None。

    参数：
        无

    返回：
        Optional[ProcessGroup]: 已设置的进程组对象，或 None（未设置时）
            - 类型：torch.distributed.ProcessGroup 或 None
            - 语义：定义哪些进程参与 Ulysses 序列并行通信
            - None 表示未启用 SP 或 SP 已清空

    副作用：
        无（纯读取全局变量）

    异常/边界条件：
        - 未调用 set_ulysses_sequence_parallel_group() 时返回 None
        - 多次调用返回相同的进程组对象（除非中间被 set 修改）

    最小示例（手算验证）：
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_group
        >>> # 场景1：未设置时返回 None
        >>> assert get_ulysses_sequence_parallel_group() is None
        >>> # 场景2：设置后返回该进程组
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> assert get_ulysses_sequence_parallel_group() is sp_group

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：get_ulysses_sequence_parallel_group() -> Optional[ProcessGroup]

    典型调用路径：
        本模块内的通信函数（如 gather_seq_scatter_heads、all_to_all_tensor）
        -> get_ulysses_sequence_parallel_group()
        -> 返回进程组或 None

    被谁调用：
        - 本模块内多处（gather_seq_scatter_heads、get_ulysses_sequence_parallel_world_size 等）
        - verl/workers/sharding_manager/fsdp_ulysses.py（间接使用）

    调用了谁（项目内）：
        - 无（仅读取全局变量）

    调用了谁（外部依赖）：
        - 无

    记忆提示：
        - 通常作为其他函数的默认参数（group=None 时自动调用此函数）
    """  # 注释：函数 docstring 结束
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP  # 注释：声明读取模块级全局变量（不修改）
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP  # 注释：返回当前全局 SP 进程组（可能为 None）


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:  # 注释：函数定义，获取 SP 进程组的 world size
    """（函数说明：获取 Ulysses 序列并行进程组的 world size）

    本函数返回指定进程组（或全局 SP 组）的进程数量。
    如果未启用 SP（group 为 None 且未设置全局组），返回 1（退化为单进程）。

    参数：
        group (ProcessGroup, optional): 指定进程组，默认为 None
            - None：使用全局 SP 组（通过 get_ulysses_sequence_parallel_group() 获取）
            - ProcessGroup：使用指定的进程组
            - 默认值：None

    返回：
        int: 进程组大小（world size）
            - 取值范围：>= 1
            - 语义：参与 SP 通信的进程数量
            - 1 表示未启用 SP 或 SP 组为空

    副作用：
        无（纯查询）

    异常/边界条件：
        - group 为 None 且未设置全局组时返回 1（退化）
        - 进程组无效时 torch.distributed.get_world_size 可能抛异常

    最小示例（手算验证）：
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size
        >>> # 场景1：未设置全局组时返回 1
        >>> assert get_ulysses_sequence_parallel_world_size() == 1
        >>> # 场景2：设置全局组后返回组大小
        >>> sp_group = dist.new_group([0, 1])  # 假设创建大小为 2 的组
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> assert get_ulysses_sequence_parallel_world_size() == 2

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int

    典型调用路径：
        本模块内的通信函数（如 gather_seq_scatter_heads、slice_input_tensor）
        -> get_ulysses_sequence_parallel_world_size(group)
        -> 返回 world size 用于计算切分长度

    被谁调用：
        - 本模块内多处（gather_seq_scatter_heads、all_to_all_tensor、ulysses_pad 等）
        - verl/workers/engine/fsdp/transformer_impl.py（间接使用）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取全局组）

    调用了谁（外部依赖）：
        - torch.distributed.get_world_size(group)（查询进程组大小）

    记忆提示：
        - world size = 参与 SP 的进程数量
        - 用于计算序列/头维度的切分份数
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择进程组：None 时使用全局组，否则使用传入组
    return dist.get_world_size(group) if group else 1  # 注释：有效组时返回其 world size，无组时退化返回 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:  # 注释：函数定义，获取当前进程在 SP 组中的 rank
    """（函数说明：获取当前进程在 Ulysses 序列并行组中的 rank）

    本函数返回当前进程在指定进程组（或全局 SP 组）中的 rank。
    如果未启用 SP（group 为 None 且未设置全局组），返回 0（退化为单进程）。

    参数：
        group (ProcessGroup, optional): 指定进程组，默认为 None
            - None：使用全局 SP 组（通过 get_ulysses_sequence_parallel_group() 获取）
            - ProcessGroup：使用指定的进程组
            - 默认值：None

    返回：
        int: 本进程在进程组中的 rank
            - 取值范围：0 到 world_size - 1
            - 语义：标识当前进程在 SP 组中的位置
            - 0 表示未启用 SP 或 SP 组为空

    副作用：
        无（纯查询）

    异常/边界条件：
        - group 为 None 且未设置全局组时返回 0（退化）
        - 进程组无效时 torch.distributed.get_rank 可能抛异常

    最小示例（手算验证）：
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_rank
        >>> # 场景1：未设置全局组时返回 0
        >>> assert get_ulysses_sequence_parallel_rank() == 0
        >>> # 场景2：设置全局组后返回当前 rank
        >>> sp_group = dist.new_group([0, 1])  # 假设当前进程是 rank 0
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> assert get_ulysses_sequence_parallel_rank() == 0  # 假设当前进程在组中的 rank 为 0

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int

    典型调用路径：
        本模块内的切分函数（如 slice_input_tensor）
        -> get_ulysses_sequence_parallel_rank(group)
        -> 返回 rank 用于计算当前进程应持有的切片索引

    被谁调用：
        - 本模块内多处（slice_input_tensor）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取全局组）

    调用了谁（外部依赖）：
        - torch.distributed.get_rank(group)（查询当前进程在组中的 rank）

    记忆提示：
        - rank = 当前进程在 SP 组中的索引（从 0 开始）
        - 用于确定当前进程应持有哪个切片
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择进程组：None 时使用全局组，否则使用传入组
    return dist.get_rank(group) if group else 0  # 注释：有效组时返回当前 rank，无组时退化返回 0


def gather_seq_scatter_heads(  # 注释：函数定义，序列维度聚合 + head 维度切分（Ulysses SP 核心通信）
    x: Tensor,  # 注释：输入张量，序列维度已被 SP 切分
    seq_dim: int,  # 注释：序列维度的索引
    head_dim: int,  # 注释：head 维度的索引
    unpadded_dim_size: int = 0,  # 注释：未 padding 的序列长度（用于反向去 padding）
    group: ProcessGroup = None,  # 注释：指定进程组，None 时使用全局 SP 组
) -> Tensor:  # 注释：返回序列聚合、head 切分后的张量
    """（函数说明：使用 all-to-all 进行"序列维度聚合 + head 维度切分"，Ulysses SP 的核心操作）

    本函数是 Ulysses 序列并行的核心通信原语之一。
    前向阶段：将 SP 切分的序列维度（seq/sp_world）聚合为全序列（seq），同时将 head 维度（h）切分为（h/sp_world）。
    反向阶段：通过 SeqAllToAll.backward 自动执行反向通信（head 聚合 + 序列切分）。

    变换示意（sp_world=2 为例）：
        前向：[batch, seq/2, h, dim] -> all-to-all -> [batch, seq, h/2, dim]
        反向：梯度 [batch, seq, h/2, dim] -> all-to-all -> [batch, seq/2, h, dim]

    参数：
        x (Tensor): 输入张量，序列维度被 SP 切分
            - 形状：[batch, seq/sp_world, heads, dim]（示例）
            - 设备：需在 GPU/NPU 上
            - 要求：序列维度长度需能被 sp_world 整除（否则需 padding）
        seq_dim (int): 序列维度的索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定哪个维度是序列维度
            - 常见值：1（对于形状 [batch, seq, ...]）
        head_dim (int): head 维度的索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定哪个维度是 head 维度
            - 常见值：2（对于形状 [batch, seq, heads, ...]）
        unpadded_dim_size (int): 未 padding 的序列长度
            - 取值范围：>= 0
            - 语义：如果输入序列做了 padding，此参数指定原始长度
            - 0 表示未做 padding 或不需要去 padding
            - 非 0 且不能整除 sp_world 时会执行去 padding
        group (ProcessGroup, optional): 指定进程组
            - None：使用全局 SP 组
            - ProcessGroup：使用指定组
            - 默认值：None

    返回：
        Tensor: 序列聚合、head 切分后的张量
            - 形状：[batch, seq, heads/sp_world, dim]（示例）
            - 设备：与输入相同
            - 若 unpadded_dim_size 非 0 且不能整除 sp_world，会去除 padding

    副作用：
        - 在分布式组内执行 all-to-all 通信（阻塞操作）
        - 可能修改张量形状（去 padding）

    异常/边界条件：
        - group 为 None 且未设置全局组时直接返回输入（退化）
        - 若 unpadded_dim_size 非 0 且不能整除 sp_world，则执行去 padding
        - head 维度长度需能被 sp_world 整除（否则 all-to-all 可能失败）

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, gather_seq_scatter_heads
        >>> # 假设 sp_world=2，创建进程组（示例）
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入张量：[batch=1, seq/sp_world=4, heads=8, dim=64]
        >>> x = torch.randn(1, 4, 8, 64)
        >>> # 执行 all-to-all：序列聚合 + head 切分
        >>> y = gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)
        >>> # 预期输出形状：[1, 8, 4, 64]（序列从 4 变为 8，head 从 8 变为 4）
        >>> assert y.shape == (1, 8, 4, 64)

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：gather_seq_scatter_heads(x, seq_dim, head_dim, unpadded_dim_size, group)

    典型调用路径：
        模型前向（如 Attention 模块）
        -> verl/workers/engine/fsdp/transformer_impl.py 或自定义模型
        -> gather_seq_scatter_heads(x, seq_dim=1, head_dim=2)
        -> SeqAllToAll.apply(...)
        -> all_to_all_tensor(...)
        -> torch.distributed.all_to_all(...)

    被谁调用：
        - verl/workers/engine/fsdp/transformer_impl.py（注意力模块前向）
        - 用户自定义模型（实现 Ulysses SP 的注意力层）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取 SP 组）
        - get_ulysses_sequence_parallel_world_size()（获取 world size）
        - SeqAllToAll.apply()（执行 all-to-all 通信）
        - _unpad_tensor()（去 padding）

    调用了谁（外部依赖）：
        - torch.distributed.all_to_all（通过 all_to_all_tensor 间接调用）

    记忆提示：
        - gather_seq = 序列维度聚合（从切分状态恢复为全序列）
        - scatter_heads = head 维度切分（从全 head 切分为子 head）
        - all-to-all = 双向通信，A 维度聚合 + B 维度切分
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择进程组：None 时使用全局 SP 组
    if not group:  # 注释：未启用 SP 时直接返回输入（退化为单进程）
        return x  # 注释：无需通信，直接返回原张量
    sp_world = get_ulysses_sequence_parallel_world_size(group)  # 注释：获取 SP world size，用于后续计算
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)  # 注释：执行 all-to-all 通信（head 维度 scatter，seq 维度 gather）
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:  # 注释：需要去 padding 的情况（原始长度不能整除 sp_world）
        padding_size = x.size(seq_dim) - unpadded_dim_size  # 注释：计算 padding 长度（当前长度 - 原始长度）
        x = _unpad_tensor(x, seq_dim, padding_size)  # 注释：去除末尾 padding，恢复原始长度
    return x  # 注释：返回处理后的张量


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:  # 注释：函数定义，head 维度聚合 + 序列维度切分（gather_seq_scatter_heads 的逆操作）
    """（函数说明：使用 all-to-all 进行"head 维度聚合 + 序列维度切分"，gather_seq_scatter_heads 的反向操作）

    本函数是 Ulysses 序列并行的另一个核心通信原语。
    前向阶段：将 head 维度（h/sp_world）聚合为全 head（h），同时将序列维度（seq）切分为（seq/sp_world）。
    通常用于注意力计算后，将输出从"全序列 + 切分 head"状态恢复为"切分序列 + 全 head"状态。

    变换示意（sp_world=2 为例）：
        前向：[batch, seq, h/2, dim] -> all-to-all -> [batch, seq/2, h, dim]

    参数：
        x (Tensor): 输入张量，head 维度已切分
            - 形状：[batch, seq, heads/sp_world, dim]（示例）
            - 设备：需在 GPU/NPU 上
            - 要求：序列维度长度需能被 sp_world 整除（否则先 padding）
        head_dim (int): head 维度的索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定哪个维度是 head 维度
            - 常见值：2
        seq_dim (int): 序列维度的索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定哪个维度是序列维度
            - 常见值：1
        group (ProcessGroup, optional): 指定进程组
            - None：使用全局 SP 组
            - ProcessGroup：使用指定组
            - 默认值：None

    返回：
        Tensor: head 聚合、序列切分后的张量
            - 形状：[batch, seq/sp_world, heads, dim]（示例）
            - 设备：与输入相同
            - 若序列长度不能整除 sp_world，会先 padding

    副作用：
        - 在分布式组内执行 all-to-all 通信（阻塞操作）
        - 可能修改张量形状（padding）

    异常/边界条件：
        - group 为 None 且未设置全局组时直接返回输入（退化）
        - 若序列维度长度不能整除 sp_world，会先 padding

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, gather_heads_scatter_seq
        >>> # 假设 sp_world=2
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入张量：[batch=1, seq=8, heads/sp_world=4, dim=64]
        >>> x = torch.randn(1, 8, 4, 64)
        >>> # 执行 all-to-all：head 聚合 + 序列切分
        >>> y = gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
        >>> # 预期输出形状：[1, 4, 8, 64]（序列从 8 变为 4，head 从 4 变为 8）
        >>> assert y.shape == (1, 4, 8, 64)

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：gather_heads_scatter_seq(x, head_dim, seq_dim, group)

    典型调用路径：
        模型前向（如 Attention 模块输出后）
        -> gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
        -> SeqAllToAll.apply(...)
        -> all_to_all_tensor(...)
        -> torch.distributed.all_to_all(...)

    被谁调用：
        - verl/workers/engine/fsdp/transformer_impl.py（注意力模块输出后）
        - 用户自定义模型（实现 Ulysses SP 的注意力层）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取 SP 组）
        - get_ulysses_sequence_parallel_world_size()（获取 world size）
        - _pad_tensor()（padding）
        - SeqAllToAll.apply()（执行 all-to-all 通信）

    调用了谁（外部依赖）：
        - torch.distributed.all_to_all（通过 all_to_all_tensor 间接调用）

    记忆提示：
        - gather_heads = head 维度聚合（从切分状态恢复为全 head）
        - scatter_seq = 序列维度切分（从全序列切分为子序列）
        - 通常与 gather_seq_scatter_heads 配对使用
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择进程组：None 时使用全局 SP 组
    if not group:  # 注释：未启用 SP 时直接返回输入（退化）
        return x  # 注释：无需通信，直接返回原张量
    dim_size = x.size(seq_dim)  # 注释：获取当前序列维度长度
    sp_world = get_ulysses_sequence_parallel_world_size(group)  # 注释：获取 SP world size
    if dim_size % sp_world != 0:  # 注释：序列长度不能整除 sp_world，需要 padding
        padding_size = sp_world - (dim_size % sp_world)  # 注释：计算需要补齐的长度
        x = _pad_tensor(x, seq_dim, padding_size)  # 注释：在序列维度末尾补 0
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)  # 注释：执行 all-to-all 通信（seq 维度 scatter，head 维度 gather）


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:  # 注释：函数定义，在指定维度上对张量进行零填充（内部工具）
    """（函数说明：在指定维度上对张量进行零填充，用于对齐 SP 通信要求）

    本函数是内部工具函数，用于在指定维度末尾补零，使维度长度能被 sp_world 整除。
    通常在 gather_heads_scatter_seq 或 slice_input_tensor 前调用。

    参数：
        x (Tensor): 原始张量
            - 形状：任意
            - 设备：任意（CPU/GPU/NPU）
        dim (int): 需要 padding 的维度索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定在哪个维度末尾补零
        padding_size (int): 需要补齐的长度
            - 取值范围：>= 0
            - 语义：在 dim 维度末尾补多少个零元素
            - 0 表示不补（仍会拼接一个空张量）

    返回：
        Tensor: 补齐后的张量
            - 形状：x.shape[dim] 维度长度增加 padding_size
            - 设备：与输入相同
            - 数据类型：与输入相同

    副作用：
        无（纯函数，创建新张量）

    异常/边界条件：
        - padding_size 为 0 时仍会拼接一个空张量（无实际影响）
        - 负 padding_size 会导致创建形状异常

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import _pad_tensor
        >>> # 输入：[2, 3]，在 dim=1 补 1 个元素
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = _pad_tensor(x, dim=1, padding_size=1)
        >>> # 预期输出：[2, 4]，末尾补 0
        >>> assert y.shape == (2, 4)
        >>> assert torch.allclose(y, torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]]))

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：_pad_tensor(x, dim, padding_size)

    典型调用路径：
        gather_heads_scatter_seq / slice_input_tensor
        -> _pad_tensor(x, dim, padding_size)
        -> torch.zeros(...) + torch.cat(...)

    被谁调用：
        - 仅在本文件内调用（gather_heads_scatter_seq、slice_input_tensor、ulysses_pad）

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - torch.zeros（创建零张量）
        - torch.cat（拼接张量）

    记忆提示：
        - 在指定维度末尾补零
        - 用于对齐 SP 通信要求（维度长度需能被 sp_world 整除）
    """  # 注释：函数 docstring 结束
    shape = list(x.shape)  # 注释：复制输入张量的形状列表（避免修改原形状）
    shape[dim] = padding_size  # 注释：将指定维度替换为 padding 大小
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)  # 注释：创建零张量，形状、数据类型、设备与输入一致
    return torch.cat([x, pad], dim=dim)  # 注释：在指定维度上拼接原张量与零张量（末尾补零）


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:  # 注释：函数定义，在指定维度上去除末尾 padding（内部工具）
    """（函数说明：在指定维度上去除末尾 padding，恢复原始长度）

    本函数是内部工具函数，用于在指定维度末尾去除 padding（通过切片）。
    通常在 gather_seq_scatter_heads 或 gather_outputs_and_unpad 后调用，恢复原始序列长度。

    参数：
        x (Tensor): 带 padding 的张量
            - 形状：任意
            - 设备：任意（CPU/GPU/NPU）
        dim (int): 需要去 padding 的维度索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定在哪个维度末尾去除 padding
        padding_size (int): 需要去除的长度
            - 取值范围：>= 0
            - 语义：在 dim 维度末尾去除多少个元素
            - 0 表示不去除（slice 等价于原张量）

    返回：
        Tensor: 去 padding 后的张量视图
            - 形状：x.shape[dim] 维度长度减少 padding_size
            - 设备：与输入相同
            - 内存：与输入共享（视图）

    副作用：
        无（返回视图，不修改原张量）

    异常/边界条件：
        - padding_size 为 0 时 slice 等价于原张量
        - padding_size >= x.size(dim) 会导致负切片（可能返回空张量）

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import _unpad_tensor
        >>> # 输入：[2, 4]，去除 dim=1 的末尾 1 个元素
        >>> x = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])
        >>> y = _unpad_tensor(x, dim=1, padding_size=1)
        >>> # 预期输出：[2, 3]，去除末尾 0
        >>> assert y.shape == (2, 3)
        >>> assert torch.allclose(y, torch.tensor([[1, 2, 3], [4, 5, 6]]))

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：_unpad_tensor(x, dim, padding_size)

    典型调用路径：
        gather_seq_scatter_heads / gather_outputs_and_unpad
        -> _unpad_tensor(x, dim, padding_size)
        -> Python slicing

    被谁调用：
        - 仅在本文件内调用（gather_seq_scatter_heads、gather_outputs_and_unpad）

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - Python slicing（通过 __getitem__）

    记忆提示：
        - 在指定维度末尾去除 padding
        - 用于恢复原始序列长度
        - 返回视图，不创建新张量
    """  # 注释：函数 docstring 结束
    slc = [slice(None)] * len(x.shape)  # 注释：构造全维切片列表（初始为 [:, :, :, ...]）
    slc[dim] = slice(0, -padding_size)  # 注释：在目标维度设置切片为 [0:-padding_size]（去除末尾 padding）
    return x[tuple(slc)]  # 注释：返回切片后的视图（与原张量共享内存）


def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:  # 注释：函数定义，沿指定维度将张量切分为 SP 子块，并取当前 rank 的分片
    """（函数说明：沿指定维度将张量切分为 SP 子块，并取当前 rank 的分片）

    本函数用于 Ulysses SP 的输入准备阶段，将全张量（如全序列）沿指定维度切分为 sp_world 份，
    并返回当前 rank 对应的分片。
    如果维度长度不能整除 sp_world，可选择先 padding 再切分。

    参数：
        x (Tensor): 输入张量
            - 形状：任意
            - 设备：任意（CPU/GPU/NPU）
        dim (int): 切分维度索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：指定在哪个维度上切分
            - 常见值：1（对于序列维度）
        padding (bool): 是否在切分前补齐长度
            - True：若维度长度不能整除 sp_world，先 padding 再切分
            - False：直接切分（可能导致各 rank 切片长度不一致）
            - 默认值：True
        group (ProcessGroup, optional): 指定进程组
            - None：使用全局 SP 组
            - ProcessGroup：使用指定组
            - 默认值：None

    返回：
        Tensor: 当前 rank 的切片（contiguous）
            - 形状：x.shape[dim] 维度长度变为原长度 / sp_world（向上取整）
            - 设备：与输入相同
            - 内存：连续内存（调用 .contiguous()）

    副作用：
        无（纯函数，创建新张量或视图）

    异常/边界条件：
        - 维度长度不能整除 sp_world 且 padding=False 时，各 rank 切片长度可能不一致
        - group 为 None 且未设置全局组时，直接返回输入（退化）

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, slice_input_tensor
        >>> # 假设 sp_world=2，当前 rank=0
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入：[1, 5]，dim=1，padding=True
        >>> x = torch.tensor([[1, 2, 3, 4, 5]])
        >>> y = slice_input_tensor(x, dim=1, padding=True)
        >>> # 预期：先 padding 到 6，再切分为 2 份，rank 0 得到前 3 个元素
        >>> assert y.shape == (1, 3)
        >>> assert torch.allclose(y, torch.tensor([[1, 2, 3]]))

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：slice_input_tensor(x, dim, padding, group)

    典型调用路径：
        ulysses_pad_and_slice_inputs
        -> slice_input_tensor(x, dim=1, padding=False)
        -> _pad_tensor (可选) + Python slicing

    被谁调用：
        - 本模块内（ulysses_pad_and_slice_inputs）
        - verl/workers/actor/dp_actor.py（间接通过 ulysses_pad_and_slice_inputs）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取 SP 组）
        - get_ulysses_sequence_parallel_rank()（获取当前 rank）
        - _pad_tensor()（可选 padding）

    调用了谁（外部依赖）：
        - torch.distributed.get_world_size（查询 world size）
        - Python slicing（切片）

    记忆提示：
        - 将全张量切分为 sp_world 份，取当前 rank 的分片
        - 用于 SP 输入准备
        - 可选 padding 以对齐切分
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择 SP 组
    sp_world_size = dist.get_world_size(group)  # 注释：获取 world size
    sp_rank = get_ulysses_sequence_parallel_rank()  # 注释：获取当前 rank
    dim_size = x.size(dim)  # 注释：获取切分维度长度
    # pad before slice  # 注释：在切分前 padding（如果需要）
    if padding and dim_size % sp_world_size:  # 注释：需要 padding 的情况（长度不能整除且 padding=True）
        padding_size = sp_world_size - (dim_size % sp_world_size)  # 注释：计算补齐长度
        x = _pad_tensor(x, dim, padding_size)  # 注释：执行 padding
    # slice the input tensor  # 注释：切分输入张量
    parts = x.size(dim) // sp_world_size  # 注释：每个 rank 的片段长度（整除）
    slc = [slice(None)] * len(x.shape)  # 注释：构造全维切片列表
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)  # 注释：选择当前 rank 的区间 [rank*parts, (rank+1)*parts)
    return x[tuple(slc)].contiguous()  # 注释：返回连续内存的切片


def all_to_all_tensor(  # 注释：函数定义，执行 all-to-all 通信，并在指定维度上 scatter/gather
    local_input: Tensor,  # 注释：当前 rank 输入张量
    scatter_dim: int,  # 注释：切分维度（scatter 维度）
    gather_dim: int,  # 注释：拼接维度（gather 维度）
    group: Optional[dist.ProcessGroup] = None,  # 注释：进程组，None 时使用全局 SP 组
    async_op: bool = False,  # 注释：是否异步通信
):  # 注释：返回拼接后的张量（同步）或 wait 函数（异步）
    """（函数说明：执行 all-to-all 通信，并在指定维度上 scatter/gather）

    本函数是 Ulysses SP 的底层通信原语，封装了 torch.distributed.all_to_all。
    工作流程：
    1. 将 local_input 沿 scatter_dim 切分为 sp_world 份
    2. 将第 i 份发送给 rank i，从 rank i 接收一份数据
    3. 将接收到的 sp_world 份数据沿 gather_dim 拼接

    参数：
        local_input (Tensor): 当前 rank 输入张量
            - 形状：任意
            - 设备：需在 GPU/NPU 上
            - scatter_dim 维度长度需能被 sp_world 整除
        scatter_dim (int): 切分维度索引
            - 取值范围：0 到 local_input.ndim - 1
            - 语义：在哪个维度上切分并分发
        gather_dim (int): 拼接维度索引
            - 取值范围：0 到 local_input.ndim - 1
            - 语义：在哪个维度上拼接接收到的数据
        group (ProcessGroup, optional): 进程组
            - None：使用全局 SP 组
            - ProcessGroup：使用指定组
            - 默认值：None
        async_op (bool): 是否异步通信
            - True：返回 wait 函数，需要手动调用 wait() 等待完成
            - False：阻塞等待通信完成并返回结果
            - 默认值：False

    返回：
        Tensor（同步模式）或 Callable（异步模式）:
            - 同步模式：返回拼接后的张量
            - 异步模式：返回 wait() 函数，调用后返回拼接张量
            - 形状：scatter_dim 维度长度不变，gather_dim 维度长度变为原长度 * sp_world

    副作用：
        - 启动分布式 all-to-all 通信
        - 异步模式下，通信在后台进行

    异常/边界条件：
        - group 为 None 时需要已设置全局 SP 组
        - scatter_dim 维度长度需能被 sp_world 整除

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, all_to_all_tensor
        >>> # 假设 sp_world=2，当前 rank=0
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入：[2, 4]，scatter_dim=1，gather_dim=0
        >>> x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> y = all_to_all_tensor(x, scatter_dim=1, gather_dim=0)
        >>> # 预期：scatter_dim=1 切分为 2 份，gather_dim=0 拼接
        >>> # 具体结果取决于各 rank 的输入

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    典型调用路径：
        SeqAllToAll.forward
        -> all_to_all_tensor(...)
        -> torch.distributed.all_to_all(...)

    被谁调用：
        - 本文件内（SeqAllToAll.forward、SeqAllToAll.backward）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取 SP 组）

    调用了谁（外部依赖）：
        - torch.distributed.all_to_all（底层通信）
        - torch.tensor_split（切分张量）
        - torch.cat（拼接张量）

    记忆提示：
        - all-to-all = 双向通信，scatter + gather
        - scatter_dim 切分，gather_dim 拼接
        - 支持异步模式（通过 async_op=True）
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择 SP 组
    seq_world_size = dist.get_world_size(group)  # 注释：获取 world size
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]  # 注释：按 scatter_dim 切分为 sp_world 份（每份连续内存）
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]  # 注释：预分配输出列表（每个元素形状与 input_list[0] 相同）
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)  # 注释：启动 all-to-all 通信（input_list[i] 发送给 rank i，从 rank i 接收到 output_list[i]）
    if async_op:  # 注释：异步模式返回 wait 函数

        def wait():  # 注释：定义异步等待函数
            """等待通信完成并拼接输出。（注释：异步等待函数说明）"""  # 注释：wait 函数 docstring
            comm.wait()  # 注释：阻塞等待通信结束
            return torch.cat(output_list, dim=gather_dim).contiguous()  # 注释：拼接输出并返回连续内存张量

        return wait  # 注释：返回等待函数（调用者需手动调用 wait()）
    return torch.cat(output_list, dim=gather_dim).contiguous()  # 注释：同步模式直接返回拼接结果


def all_gather_tensor(local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):  # 注释：函数定义，在进程组内对张量进行 all_gather，并输出拼接后的张量
    """（函数说明：在进程组内对张量进行 all_gather，沿 dim0 拼接）

    本函数封装了 torch.distributed.all_gather_into_tensor，用于 Ulysses SP 的 gather 操作。
    工作流程：
    1. 在进程组内聚合所有 rank 的 local_tensor
    2. 沿 dim0 拼接为一个大张量（dim0 长度变为原长度 * sp_world）

    参数：
        local_tensor (Tensor): 本地张量
            - 形状：任意
            - 设备：需在 GPU/NPU 上
            - 要求：各 rank 的 local_tensor 形状一致
        group (ProcessGroup, optional): 进程组
            - None：使用全局 SP 组
            - ProcessGroup：使用指定组
            - 默认值：None
        async_op (bool): 是否异步通信
            - True：启动异步通信（需要手动同步）
            - False：阻塞等待通信完成
            - 默认值：False

    返回：
        Tensor: 拼接后的张量
            - 形状：dim0 长度变为原长度 * sp_world，其他维度不变
            - 设备：与输入相同

    副作用：
        - 执行分布式 all_gather 通信

    异常/边界条件：
        - group 需已初始化
        - 各 rank 的 local_tensor 形状需一致

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, all_gather_tensor
        >>> # 假设 sp_world=2
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入：[2, 3]
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = all_gather_tensor(x)
        >>> # 预期输出：[4, 3]（dim0 翻倍）

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：all_gather_tensor(local_tensor, group, async_op)

    典型调用路径：
        Gather.forward
        -> all_gather_tensor(...)
        -> torch.distributed.all_gather_into_tensor(...)

    被谁调用：
        - 本文件内（Gather.forward）

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取 SP 组）

    调用了谁（外部依赖）：
        - torch.distributed.all_gather_into_tensor（底层通信）

    记忆提示：
        - all_gather = 聚合所有 rank 的张量
        - 沿 dim0 拼接
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择 SP 组
    sp_world_size = dist.get_world_size(group=group)  # 注释：获取 world size
    output_shape = list(local_tensor.shape)  # 注释：拷贝形状列表
    output_shape[0] = output_shape[0] * sp_world_size  # 注释：扩展 dim0 长度为原长度 * sp_world
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)  # 注释：预分配输出张量
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)  # 注释：执行 all_gather，将各 rank 的 local_tensor 拼接到 output
    return output  # 注释：返回聚合后的张量


class SeqAllToAll(torch.autograd.Function):  # 注释：类定义，自定义 autograd Function，包装 all-to-all 通信的前向/反向
    """（类说明：自定义 autograd Function，实现 all-to-all 通信的前向与反向传播）

    本类封装了 all-to-all 通信，使其支持 PyTorch autograd 机制。
    前向传播：执行 all-to-all 通信（scatter_dim -> gather_dim）。
    反向传播：对梯度执行反向 all-to-all（gather_dim -> scatter_dim），恢复原布局。

    说明：
        - 前向：scatter_dim -> gather_dim 的 all-to-all
        - 反向：gather_dim -> scatter_dim 的 all-to-all（还原梯度布局）

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 类：SeqAllToAll(torch.autograd.Function)

    典型调用路径：
        gather_seq_scatter_heads / gather_heads_scatter_seq
        -> SeqAllToAll.apply(group, x, scatter_dim, gather_dim)
        -> SeqAllToAll.forward(...) / SeqAllToAll.backward(...)

    被谁调用：
        - 本模块内通信工具（gather_seq_scatter_heads、gather_heads_scatter_seq）

    调用了谁（项目内）：
        - all_to_all_tensor（前向与反向通信）

    调用了谁（外部依赖）：
        - torch.autograd.Function（基类）
        - torch.cat（异步模式拼接梯度）

    记忆提示：
        - 前向 = all-to-all(scatter_dim -> gather_dim)
        - 反向 = all-to-all(gather_dim -> scatter_dim)
        - 支持异步模式
    """  # 注释：类 docstring 结束

    @staticmethod  # 注释：静态方法装饰器
    def forward(  # 注释：前向传播方法定义
        ctx: Any,  # 注释：上下文对象，用于保存前向信息供反向使用
        group: dist.ProcessGroup,  # 注释：进程组
        local_input: Tensor,  # 注释：本地输入张量
        scatter_dim: int,  # 注释：scatter 维度
        gather_dim: int,  # 注释：gather 维度
        async_op: bool = False,  # 注释：是否异步通信
    ) -> Tensor:  # 注释：返回通信后的张量
        """（方法说明：前向传播，执行 all-to-all 通信）

        本方法在前向传播时被调用，执行 all-to-all 通信。
        保存必要的信息（group、scatter_dim、gather_dim、async_op）到 ctx，供反向传播使用。

        参数：
            ctx (Any): autograd 上下文对象
            group (ProcessGroup): 进程组
            local_input (Tensor): 本地输入张量
            scatter_dim (int): scatter 维度
            gather_dim (int): gather 维度
            async_op (bool): 是否异步通信

        返回：
            Tensor: 通信后的张量

        副作用：
            - 在进程组内执行 all-to-all 通信
            - 保存信息到 ctx

        最小示例：
            见 all_to_all_tensor

        调用路径依赖：

        所在位置：
            - 路径：verl/utils/ulysses.py
            - 方法：SeqAllToAll.forward(ctx, group, local_input, scatter_dim, gather_dim, async_op)

        被谁调用：
            - SeqAllToAll.apply（autograd 自动调用）

        调用了谁（项目内）：
            - all_to_all_tensor（执行通信）

        调用了谁（外部依赖）：
            - torch.distributed.all_to_all（通过 all_to_all_tensor）
        """  # 注释：方法 docstring 结束
        ctx.group = group  # 注释：保存进程组到 ctx，供反向使用
        ctx.scatter_dim = scatter_dim  # 注释：保存 scatter 维度到 ctx
        ctx.gather_dim = gather_dim  # 注释：保存 gather 维度到 ctx
        ctx.async_op = async_op  # 注释：保存是否异步到 ctx
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)  # 注释：执行 all-to-all 通信并返回结果

    @staticmethod  # 注释：静态方法装饰器
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:  # 注释：反向传播方法定义
        """（方法说明：反向传播，对梯度执行反向 all-to-all）

        本方法在反向传播时被调用，对梯度执行反向 all-to-all 通信，恢复原布局。

        参数：
            ctx (Any): autograd 上下文对象（包含前向保存的信息）
            *grad_output (Tensor): 梯度输出（异步模式下可能有多个）

        返回：
            tuple: 梯度元组，对应 forward 的参数位置
                - None（group 无梯度）
                - Tensor（local_input 的梯度）
                - None（scatter_dim 无梯度）
                - None（gather_dim 无梯度）
                - None（async_op 无梯度）
                - None（占位）

        副作用：
            - 在进程组内执行 all-to-all 通信

        调用路径依赖：

        所在位置：
            - 路径：verl/utils/ulysses.py
            - 方法：SeqAllToAll.backward(ctx, *grad_output)

        被谁调用：
            - autograd 反向图（自动调用）

        调用了谁（项目内）：
            - all_to_all_tensor（执行反向通信）

        调用了谁（外部依赖）：
            - torch.cat（异步模式拼接梯度）
        """  # 注释：方法 docstring 结束
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]  # 注释：异步模式需要拼接多个梯度片段
        return (
            None,  # 注释：group 无梯度
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),  # 注释：执行反向 all-to-all（gather_dim -> scatter_dim）
            None,  # 注释：scatter_dim 无梯度
            None,  # 注释：gather_dim 无梯度
            None,  # 注释：async_op 无梯度
            None,  # 注释：占位保持返回长度一致
        )


class Gather(torch.autograd.Function):  # 注释：类定义，自定义 autograd Function，实现 all_gather 的前向与反向
    """（类说明：自定义 autograd Function，实现 all_gather 通信的前向与反向传播）

    本类封装了 all_gather 通信，使其支持 PyTorch autograd 机制。
    前向传播：执行 all_gather 并沿 gather_dim 拼接。
    反向传播：从拼接梯度中切回本地分片，可选梯度缩放。

    说明：
        - 前向：all_gather 后按 gather_dim 拼接
        - 反向：从拼接梯度中切出本地部分，可选缩放

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 类：Gather(torch.autograd.Function)

    典型调用路径：
        gather_outputs_and_unpad
        -> Gather.apply(group, x, gather_dim, grad_scaler)
        -> Gather.forward(...) / Gather.backward(...)

    被谁调用：
        - 本模块内（gather_outputs_and_unpad）

    调用了谁（项目内）：
        - all_gather_tensor（前向通信）

    调用了谁（外部依赖）：
        - torch.autograd.Function（基类）
        - torch.split（反向切分梯度）

    记忆提示：
        - 前向 = all_gather + 拼接
        - 反向 = 切分 + 可选缩放
    """  # 注释：类 docstring 结束

    @staticmethod  # 注释：静态方法装饰器
    def forward(  # 注释：前向传播方法定义
        ctx: Any,  # 注释：上下文对象
        group: dist.ProcessGroup,  # 注释：进程组
        local_tensor: Tensor,  # 注释：本地张量
        gather_dim: int,  # 注释：拼接维度
        grad_scaler: bool = True,  # 注释：是否进行梯度缩放
        async_op=False,  # 注释：是否异步通信
    ) -> Tensor:  # 注释：返回拼接后的张量
        """（方法说明：前向传播，all_gather 并在 gather_dim 上拼接）

        本方法在前向传播时被调用，执行 all_gather 并拼接。

        参数：
            ctx (Any): autograd 上下文对象
            group (ProcessGroup): 进程组
            local_tensor (Tensor): 本地张量
            gather_dim (int): 拼接维度
            grad_scaler (bool): 是否进行梯度缩放
            async_op (bool): 是否异步通信

        返回：
            Tensor: 拼接后的张量

        副作用：
            - 执行 all_gather 通信
            - 保存信息到 ctx

        调用路径依赖：

        所在位置：
            - 路径：verl/utils/ulysses.py
            - 方法：Gather.forward(ctx, group, local_tensor, gather_dim, grad_scaler, async_op)

        被谁调用：
            - Gather.apply（autograd 自动调用）

        调用了谁（项目内）：
            - all_gather_tensor（执行 all_gather）

        调用了谁（外部依赖）：
            - torch.distributed.get_world_size/get_rank（查询进程信息）
        """  # 注释：方法 docstring 结束
        ctx.group = group  # 注释：保存进程组到 ctx
        ctx.gather_dim = gather_dim  # 注释：保存 gather 维度到 ctx
        ctx.grad_scaler = grad_scaler  # 注释：保存是否梯度缩放到 ctx
        ctx.async_op = async_op  # 注释：保存是否异步到 ctx

        sp_world_size = dist.get_world_size(group=group)  # 注释：获取 world size
        ctx.sp_world_size = sp_world_size  # 注释：缓存 world size 到 ctx

        sp_rank = dist.get_rank(group=group)  # 注释：获取当前 rank
        ctx.sp_rank = sp_rank  # 注释：缓存 rank 到 ctx

        local_shape = list(local_tensor.size())  # 注释：保存局部形状
        split_size = local_shape[0]  # store original size  # 注释：按 dim0 分割的块大小
        part_size = local_shape[gather_dim]  # store original size  # 注释：保存 gather_dim 原始长度
        ctx.part_size = part_size  # 注释：缓存用于反向切分

        output = all_gather_tensor(local_tensor, group, async_op)  # 注释：执行 all_gather
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)  # 注释：按 gather_dim 拼接

    @staticmethod  # 注释：静态方法装饰器
    def backward(ctx: Any, grad_output: Tensor) -> Any:  # 注释：反向传播方法定义
        """（方法说明：反向传播，从拼接梯度中切回本地分片）

        本方法在反向传播时被调用，从拼接梯度中切出本地分片，可选梯度缩放。

        参数：
            ctx (Any): autograd 上下文对象
            grad_output (Tensor): 拼接后的梯度

        返回：
            tuple: 梯度元组，对应 forward 的参数位置

        副作用：
            无（纯函数）

        调用路径依赖：

        所在位置：
            - 路径：verl/utils/ulysses.py
            - 方法：Gather.backward(ctx, grad_output)

        被谁调用：
            - autograd 反向图（自动调用）

        调用了谁（项目内）：
            - 无

        调用了谁（外部依赖）：
            - torch.split（切分梯度）
        """  # 注释：方法 docstring 结束
        if ctx.grad_scaler:  # 注释：需要梯度缩放时乘以 world size
            grad_output = grad_output * ctx.sp_world_size  # 注释：缩放梯度
        return (
            None,  # 注释：group 无梯度
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),  # 注释：切回本地分片
            None,  # 注释：gather_dim 无梯度
            None,  # 注释：grad_scaler 无梯度
            None,  # 注释：async_op 无梯度
            None,  # 注释：占位
        )


def gather_outpus_and_unpad(*args, **kwargs):  # 注释：函数定义，兼容旧拼写的占位函数，直接抛错引导使用正确接口
    """（函数说明：兼容旧拼写的占位函数，直接抛错引导使用正确接口）

    本函数是为了处理历史代码中可能存在的拼写错误（outpus -> outputs）。
    调用时会直接抛出 RuntimeError，提示使用正确的函数名 gather_outputs_and_unpad。

    参数：
        *args, **kwargs: 任意参数（兼容旧调用签名）

    返回：
        无（总是抛异常）

    副作用：
        抛出 RuntimeError

    异常/边界条件：
        总是抛错

    最小示例：
        >>> from verl.utils.ulysses import gather_outpus_and_unpad
        >>> gather_outpus_and_unpad(x)  # 抛出 RuntimeError

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：gather_outpus_and_unpad(*args, **kwargs)

    典型调用路径：
        旧代码误用时触发

    被谁调用：
        - 无固定调用方（防错接口）

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - RuntimeError

    记忆提示：
        - 拼写错误兼容函数
        - 引导使用 gather_outputs_and_unpad
    """  # 注释：函数 docstring 结束
    raise RuntimeError(  # 注释：直接抛错提醒改用正确函数名
        "please use verl.utils.ulysses.gather_outputs_and_unpad instead of verl.utils.ulysses.gather_outpus_and_unpad"
    )


def gather_outputs_and_unpad(  # 注释：函数定义，在进程组内 gather 输出，并按需去除 padding
    x: Tensor,  # 注释：本地输出张量
    gather_dim: int,  # 注释：聚合维度
    unpad_dim: int = None,  # 注释：去 padding 的维度，None 表示不去 padding
    padding_size: int = 0,  # 注释：需要去除的 padding 长度
    grad_scaler: bool = True,  # 注释：是否进行梯度缩放
    group: Optional[dist.ProcessGroup] = None,  # 注释：进程组，None 时使用全局 SP 组
):  # 注释：返回聚合后的张量（可选去 padding）
    """（函数说明：在进程组内 gather 输出，并按需去除 padding）

    本函数用于 Ulysses SP 的输出还原阶段，将各 rank 的输出张量聚合为全输出，
    并可选地去除 padding 以恢复原始长度。

    参数：
        x (Tensor): 本地输出张量
            - 形状：任意
            - 设备：需在 GPU/NPU 上
        gather_dim (int): 聚合维度索引
            - 取值范围：0 到 x.ndim - 1
            - 语义：在哪个维度上聚合
            - 常见值：1（对于序列维度）
        unpad_dim (int, optional): 去 padding 的维度索引
            - None：不去 padding
            - int：在该维度去除 padding
            - 默认值：None
        padding_size (int): 需要去除的 padding 长度
            - 取值范围：>= 0
            - 语义：在 unpad_dim 维度末尾去除多少个元素
            - 0 表示不去除
            - 默认值：0
        grad_scaler (bool): 是否进行梯度缩放
            - True：反向传播时梯度乘以 sp_world_size
            - False：梯度不缩放
            - 默认值：True
        group (ProcessGroup, optional): 进程组
            - None：使用全局 SP 组
            - ProcessGroup：使用指定组
            - 默认值：None

    返回：
        Tensor: 聚合后的张量（可选去 padding）
            - 形状：gather_dim 维度长度变为原长度 * sp_world
            - 若 unpad_dim 非 None 且 padding_size > 0，会去除 padding

    副作用：
        - 执行分布式 all_gather（通过 Gather）

    异常/边界条件：
        - group 为 None 且未设置全局组时直接返回输入
        - padding_size 需为 int

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, gather_outputs_and_unpad
        >>> # 假设 sp_world=2
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入：[2, 3]，gather_dim=0，padding_size=1
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = gather_outputs_and_unpad(x, gather_dim=0, unpad_dim=0, padding_size=1)
        >>> # 预期：先 gather 到 [4, 3]，再去除 dim=0 的末尾 1 个元素 -> [3, 3]

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：gather_outputs_and_unpad(x, gather_dim, unpad_dim, padding_size, grad_scaler, group)

    典型调用路径：
        模型前向输出
        -> verl/workers/actor/dp_actor.py 或模型实现
        -> gather_outputs_and_unpad(x, gather_dim=1, unpad_dim=1, padding_size=pad)
        -> Gather.apply(...) + _unpad_tensor(...)

    被谁调用：
        - verl/workers/actor/dp_actor.py
        - verl/trainer/fsdp_sft_trainer.py
        - verl/workers/engine/fsdp/transformer_impl.py
        - recipe/prime/prime_dp_rm.py

    调用了谁（项目内）：
        - get_ulysses_sequence_parallel_group()（获取 SP 组）
        - Gather.apply()（执行 gather 并拼接）
        - _unpad_tensor()（去 padding）

    调用了谁（外部依赖）：
        - torch.distributed（通过 Gather）

    记忆提示：
        - gather = 聚合各 rank 输出
        - unpad = 去除 padding
        - 用于 SP 输出还原
    """  # 注释：函数 docstring 结束
    group = get_ulysses_sequence_parallel_group() if group is None else group  # 注释：选择 SP 组
    if group is None:  # 注释：未启用 SP 时直接返回输入
        return x  # 注释：无需通信，直接返回原张量
    x = Gather.apply(group, x, gather_dim, grad_scaler)  # 注释：执行 gather 并拼接
    if unpad_dim is not None:  # 注释：需要去 padding 的情况
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"  # 注释：参数校验
        if padding_size == 0:  # 注释：无需去 padding
            return x  # 注释：直接返回
        x = _unpad_tensor(x, unpad_dim, padding_size)  # 注释：去除 padding
    return x  # 注释：返回结果


def ulysses_pad(input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor] = None, sp_size: int = 1):  # 注释：函数定义，对输入 token/position ids 进行 padding，使序列长度可被 sp_size 整除
    """（函数说明：对输入 token/position ids 进行 padding，使序列长度可被 sp_size 整除）

    本函数用于 Ulysses SP 的输入准备阶段，对去除 padding 的输入序列进行补齐，
    使序列长度能被 sp_size 整除，便于后续切分。

    参数：
        input_ids_rmpad (Tensor): 去 padding 的输入 token ids
            - 形状：[batch, seqlen]
            - 设备：任意（CPU/GPU/NPU）
            - 语义：输入 token 序列
        position_ids_rmpad (Tensor, optional): 位置编码 ids
            - 形状：[batch, seqlen] 或 [1, seqlen]
            - 设备：与 input_ids_rmpad 相同
            - 语义：位置编码序列
            - 默认值：None
        sp_size (int): 序列并行大小
            - 取值范围：>= 1
            - 语义：SP world size
            - 1 表示未启用 SP（直接返回）
            - 默认值：1

    返回：
        tuple: (input_ids_rmpad, position_ids_rmpad, pad_size)
            - input_ids_rmpad (Tensor): padding 后的 input ids
            - position_ids_rmpad (Tensor): padding 后的 position ids（若输入非 None）
            - pad_size (int): padding 长度（0 表示未 padding）

    副作用：
        无（纯函数，创建新张量）

    异常/边界条件：
        - position_ids_rmpad 若存在，需与 input_ids_rmpad 对齐
        - sp_size <= 1 时不做 padding

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import ulysses_pad
        >>> # 输入：seqlen=5，sp_size=2
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 4]])
        >>> input_ids_pad, position_ids_pad, pad_size = ulysses_pad(input_ids, position_ids, sp_size=2)
        >>> # 预期：pad_size=1（5 -> 6），seqlen 变为 6
        >>> assert input_ids_pad.shape == (1, 6)
        >>> assert pad_size == 1

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：ulysses_pad(input_ids_rmpad, position_ids_rmpad, sp_size)

    典型调用路径：
        ulysses_pad_and_slice_inputs
        -> ulysses_pad(...)
        -> torch.nn.functional.pad + torch.cat

    被谁调用：
        - 本模块内（ulysses_pad_and_slice_inputs）
        - verl/workers/actor/dp_actor.py（间接）

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - torch.nn.functional.pad（padding token ids）
        - torch.arange（生成 position ids）
        - torch.cat（拼接 position ids）

    记忆提示：
        - 对序列进行 padding 以对齐 sp_size
        - 同时处理 input_ids 和 position_ids
    """  # 注释：函数 docstring 结束
    if position_ids_rmpad is not None:  # 注释：若有 position ids，需要对齐校验
        assert position_ids_rmpad.size(-2) == 1  # 注释：期望 batch 维度为 1 或固定格式
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)  # 注释：序列长度一致
    if sp_size <= 1:  # 注释：sp_size 为 1 时无需 padding
        return input_ids_rmpad, position_ids_rmpad, 0  # 注释：直接返回原数据与 pad_size=0
    _, total_seq_len = input_ids_rmpad.shape  # 注释：获取序列长度
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size  # 注释：计算补齐长度（取模两次避免已整除时补齐 sp_size）
    if pad_size > 0:  # 注释：需要 padding 的情况
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=0)  # 注释：在序列末尾补 0（padding token）
        if position_ids_rmpad is not None:  # 注释：同步 padding position ids
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)  # 注释：构造 padding position ids（从 0 递增）
            if position_ids_rmpad.dim() == 3:  # 注释：多 batch/多模态时对齐维度
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(position_ids_rmpad.size(0), 1, 1)  # 注释：扩展 batch 维度
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)  # 注释：拼接 position ids
    return input_ids_rmpad, position_ids_rmpad, pad_size  # 注释：返回 padding 结果与 padding 长度


def ulysses_pad_and_slice_inputs(  # 注释：函数定义，对 input_ids/position_ids 进行 padding 后按 SP 切分
    input_ids_rmpad: torch.Tensor,  # 注释：去 padding 的 input ids
    position_ids_rmpad: Optional[torch.Tensor] = None,  # 注释：去 padding 的 position ids
    sp_size: int = 1,  # 注释：序列并行大小
    skip_position_ids_rmpad: bool = False,  # 注释：是否跳过 position_ids 的切分
):  # 注释：返回 padding 且切分后的 input_ids、position_ids、padding 长度
    """（函数说明：对 input_ids/position_ids 进行 padding 后按 SP 切分）

    本函数是 Ulysses SP 输入准备的完整流程，包含：
    1. padding 序列（ulysses_pad）
    2. 按 SP 切分（slice_input_tensor）

    参数：
        input_ids_rmpad (Tensor): 去 padding 的输入 token ids
            - 形状：[batch, seqlen]
            - 设备：任意（CPU/GPU/NPU）
        position_ids_rmpad (Tensor, optional): 去 padding 的位置编码 ids
            - 形状：[batch, seqlen] 或 [1, seqlen]
            - 设备：与 input_ids_rmpad 相同
            - 默认值：None
        sp_size (int): 序列并行大小
            - 取值范围：>= 1
            - 1 表示未启用 SP（仅返回 padding 结果）
            - 默认值：1
        skip_position_ids_rmpad (bool): 是否跳过 position_ids 的切分
            - True：position_ids 不切分（保留全序列）
            - False：position_ids 也切分
            - 默认值：False

    返回：
        tuple: (input_ids_rmpad, position_ids_rmpad, pad_size)
            - input_ids_rmpad (Tensor): padding 且切分后的 input ids
            - position_ids_rmpad (Tensor): padding 且切分后的 position ids（若未跳过）
            - pad_size (int): padding 长度

    副作用：
        无（纯函数）

    异常/边界条件：
        - sp_size <= 1 时仅返回 padding 结果，不切分

    最小示例（手算验证）：
        >>> import torch
        >>> from verl.utils.ulysses import set_ulysses_sequence_parallel_group, ulysses_pad_and_slice_inputs
        >>> # 假设 sp_world=2，当前 rank=0
        >>> sp_group = dist.new_group([0, 1])
        >>> set_ulysses_sequence_parallel_group(sp_group)
        >>> # 输入：seqlen=5，sp_size=2
        >>> input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        >>> position_ids = torch.tensor([[0, 1, 2, 3, 4]])
        >>> input_ids_sliced, position_ids_sliced, pad_size = ulysses_pad_and_slice_inputs(
        ...     input_ids, position_ids, sp_size=2
        ... )
        >>> # 预期：padding 到 6，切成 2 份，rank 0 得到前 3 个元素
        >>> assert input_ids_sliced.shape == (1, 3)
        >>> assert pad_size == 1

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size, skip_position_ids_rmpad)

    典型调用路径：
        模型前向输入准备
        -> verl/workers/actor/dp_actor.py
        -> ulysses_pad_and_slice_inputs(...)
        -> ulysses_pad(...) + slice_input_tensor(...)

    被谁调用：
        - verl/workers/actor/dp_actor.py
        - verl/workers/engine/fsdp/transformer_impl.py
        - recipe/flowrl/flowrl_actor.py

    调用了谁（项目内）：
        - ulysses_pad（padding 序列）
        - slice_input_tensor（切分序列）

    调用了谁（外部依赖）：
        - 无

    记忆提示：
        - padding + slice = Ulysses SP 输入准备完整流程
        - 可选跳过 position_ids 切分
    """  # 注释：函数 docstring 结束
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(input_ids_rmpad, position_ids_rmpad, sp_size)  # 注释：先 padding 序列
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)  # 注释：再按 SP 切分 input_ids（dim=1 为序列维度，padding=False 因为已 padding）
    if position_ids_rmpad is not None and not skip_position_ids_rmpad:  # 注释：按需切分 position_ids
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)  # 注释：切分 position_ids
    return input_ids_rmpad, position_ids_rmpad, pad_size  # 注释：返回切分结果与 pad_size


def validate_ulysses_config(num_heads, ulysses_sequence_size):  # 注释：函数定义，校验 Ulysses 序列并行配置是否与注意力头数匹配
    """（函数说明：校验 Ulysses 序列并行配置是否与注意力头数匹配）

    本函数用于在初始化 Ulysses SP 时校验配置合法性。
    核心约束：当启用 Ulysses SP 时（ulysses_sequence_size > 1），
    注意力头数（num_heads）必须能被 ulysses_sequence_size 整除。

    参数：
        num_heads (int): 注意力头数
            - 取值范围：> 0
            - 语义：模型的注意力头数量
        ulysses_sequence_size (int): 序列并行大小
            - 取值范围：>= 1
            - 语义：SP world size
            - 1 表示未启用 SP（不校验）

    返回：
        无（仅校验）

    副作用：
        - 断言失败会抛异常

    异常/边界条件：
        - ulysses_sequence_size > 1 时要求 num_heads 可整除
        - ulysses_sequence_size <= 1 时不校验

    最小示例（手算验证）：
        >>> from verl.utils.ulysses import validate_ulysses_config
        >>> # 场景1：通过（16 % 4 == 0）
        >>> validate_ulysses_config(num_heads=16, ulysses_sequence_size=4)
        >>> # 场景2：失败（16 % 3 != 0）
        >>> validate_ulysses_config(num_heads=16, ulysses_sequence_size=3)  # AssertionError

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/ulysses.py
        - 函数：validate_ulysses_config(num_heads, ulysses_sequence_size)

    典型调用路径：
        模型配置加载
        -> 初始化 Ulysses SP
        -> validate_ulysses_config(...)

    被谁调用：
        - 未在仓库内直接检索到显式调用（可能由配置/初始化逻辑调用）

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - 断言机制

    记忆提示：
        - num_heads 必须能被 ulysses_sequence_size 整除
        - 用于配置合法性校验
    """  # 注释：函数 docstring 结束
    if ulysses_sequence_size > 1:  # 注释：仅在启用序列并行时校验
        assert num_heads % ulysses_sequence_size == 0, (  # 注释：要求整除
            f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"
        )
