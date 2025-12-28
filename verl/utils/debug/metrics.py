# Copyright 2025 Individual Contributor: TomQunChaoA
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

"""（模块说明：Rollout 与 Actor 概率分布一致性调试指标计算工具）

模块用途：
    本模块提供调试诊断工具，用于检测 RLHF/PPO 训练过程中 Rollout 阶段和 Actor 训练阶段的概率分布差异。
    主要功能包括：
    1. 计算 Rollout 生成时的 log_probs 与 Actor 前向传播的 log_probs 之间的差异
    2. 计算 token 序列的逐位差异统计（用于检测数值不稳定性）
    3. 计算 Pearson 相关系数（评估两个概率分布的线性相关性）
    4. 提供详细的统计指标（max、mean、std）用于调试

    应用场景：
    - 调试 Rollout 与 Actor 之间的数值不一致问题
    - 检测浮点精度误差或数值溢出
    - 验证 Rollout 缓存是否与 Actor 重新计算一致
    - 评估模型训练的稳定性

输入/输出：
    - 输入：
      * DataProto 对象：包含 rollout_log_probs、old_log_probs（actor）、response_mask 等
    - 输出：
      * 字典：包含以下调试指标
        - "training/rollout_probs_diff_valid": 输入有效性标志（1=有效，0=无效）
        - "training/rollout_probs_diff_max": log_probs 差异的最大值
        - "training/rollout_probs_diff_mean": log_probs 差异的均值
        - "training/rollout_probs_diff_std": log_probs 差异的标准差
        - "training/rollout_actor_probs_pearson_corr": Pearson 相关系数

关键依赖：
    - torch：用于张量运算和统计计算
    - verl.protocol.DataProto：VERL 项目的统一数据协议类
    - logging：用于日志输出（警告和调试信息）

典型用法：
    >>> from verl.utils.debug.metrics import calculate_debug_metrics
    >>> from verl.protocol import DataProto
    >>> # 假设 data 是一个 DataProto 对象，包含 rollout_log_probs 和 old_log_probs
    >>> metrics = calculate_debug_metrics(data)
    >>> print(metrics)
    {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": 0.0023,
        "training/rollout_probs_diff_mean": 0.0001,
        "training/rollout_probs_diff_std": 0.0005,
        "training/rollout_actor_probs_pearson_corr": 0.9998
    }

调用路径概览：
    入口脚本（如 verl/trainer/main_ppo.py）
    -> verl/trainer/ppo/ray_trainer.py::PPOTrainer.fit()
    -> （可选）启用调试模式，计算 rollout vs actor 指标
    -> verl/utils/debug/metrics.py::calculate_debug_metrics(data)
    -> 返回调试指标字典，记录到日志或 WandB/MLflow

所在位置：
    - 路径：verl/utils/debug/metrics.py
    - 模块：verl.utils.debug.metrics

被谁调用：
    - verl/trainer/ppo/ray_trainer.py（PPO/GRPO 训练循环中，用于调试）
    - verl/workers/actor/dp_actor.py（Actor worker 中，用于调试）
    - 用户自定义调试脚本

调用了谁（项目内）：
    - verl.protocol.DataProto（数据协议类，提供 batch 字典）

调用了谁（外部依赖）：
    - torch.masked_select：根据 mask 提取有效元素
    - torch.corrcoef：计算 Pearson 相关系数
    - torch.abs、torch.max、torch.mean、torch.std：统计计算
    - logging.getLogger、logging.debug、logging.warning：日志输出

注意事项：
    1. rollout_log_probs 和 old_log_probs 必须形状一致
    2. response_mask 用于过滤 prompt 部分，只计算 response 部分的差异
    3. Pearson 相关系数接近 1 表示两个分布高度线性相关（理想情况）
    4. 如果差异较大（max > 0.1），可能存在数值不稳定或 bug
    5. 本模块仅用于调试，不影响训练流程

参考文献：
    - Pearson 相关系数定义：https://arxiv.org/pdf/2506.13585
    - RLHF 训练中的数值稳定性问题：https://arxiv.org/abs/2203.02155

最小手算示例（Pearson 相关系数）：
    假设有两个概率分布（已转换为 exp(log_probs)）：
    - actor_probs = [0.1, 0.2, 0.7]
    - rollout_probs = [0.11, 0.19, 0.71]

    步骤 1：计算均值
    - mean_actor = (0.1 + 0.2 + 0.7) / 3 = 0.333
    - mean_rollout = (0.11 + 0.19 + 0.71) / 3 = 0.337

    步骤 2：计算方差
    - var_actor = [(0.1-0.333)^2 + (0.2-0.333)^2 + (0.7-0.333)^2] / 3 = 0.0756
    - var_rollout = [(0.11-0.337)^2 + (0.19-0.337)^2 + (0.71-0.337)^2] / 3 = 0.0756

    步骤 3：计算协方差
    - cov = [(0.1-0.333)*(0.11-0.337) + (0.2-0.333)*(0.19-0.337) + (0.7-0.333)*(0.71-0.337)] / 3
          = [(-0.233)*(-0.227) + (-0.133)*(-0.147) + (0.367)*(0.373)] / 3
          = [0.0529 + 0.0195 + 0.1369] / 3 = 0.0698

    步骤 4：计算 Pearson 相关系数
    - pearson = cov / (sqrt(var_actor) * sqrt(var_rollout))
              = 0.0698 / (sqrt(0.0756) * sqrt(0.0756))
              = 0.0698 / 0.0756 = 0.923

    解释：相关系数 0.923 接近 1，说明两个分布高度相关，数值差异较小。

记忆提示：
    - 模块名：metrics（调试指标）
    - 核心函数：calculate_debug_metrics
    - 口诀：rollout vs actor，log_probs 差多少，Pearson 相关看一看
"""  # 注释：模块级 docstring 结束，下面是依赖导入

# 注释：导入 Python 标准库 logging，用于记录调试信息和警告
import logging  # 注释：logging 提供分级日志功能（DEBUG、INFO、WARNING、ERROR、CRITICAL）

# 注释：导入 PyTorch 库，用于张量运算和统计计算
import torch  # 注释：torch 是深度学习框架，提供张量（Tensor）数据结构和各种数学运算

# 注释：导入 VERL 项目的数据协议类，用于统一数据格式
from verl.protocol import DataProto  # 注释：DataProto 是 VERL 项目中用于封装训练数据的统一协议类

# 注释：创建当前模块的 logger 对象，用于输出调试信息
logger = logging.getLogger(__file__)  # 注释：logger 的名称是当前文件的路径，便于追踪日志来源


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # 注释：函数定义，计算两个 token 序列的逐位差异统计
    """（函数说明：计算两个 token 序列的逐位差异统计）

    本函数用于检测两个 token 序列（如 rollout 生成的 token 与 actor 重新生成的 token）之间的差异。
    主要用于调试数值不稳定性问题（如浮点误差、随机性等）。

    参数：
        tensor1 (torch.Tensor): 第一个 token 序列，形状为 [batch_size, seq_len]
            - 每个元素是一个 token ID（整数）
        tensor2 (torch.Tensor): 第二个 token 序列，形状为 [batch_size, seq_len]
            - 应与 tensor1 形状相同
        mask (torch.Tensor): 有效位置掩码，形状为 [batch_size, seq_len]
            - 1 表示有效位置，0 表示无效位置（如 padding）

    返回：
        torch.Tensor: 每个样本的差异 token 数量，形状为 [batch_size]
            - 例如：[2, 0, 5] 表示第 1 个样本有 2 个 token 不同，第 2 个样本完全相同，第 3 个样本有 5 个 token 不同

    副作用：
        - 如果输入形状不匹配，打印警告信息

    异常/边界条件：
        - 如果 tensor1 或 tensor2 为空（numel() == 0），返回全 0 张量
        - 如果形状不匹配，打印警告并返回全 1 张量（表示所有位置都不同）
        - 自动将 tensor2 和 mask 移动到与 tensor1 相同的设备（CPU 或 GPU）

    最小示例（手算验证）：
        >>> tensor1 = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0]])  # 2 个样本，长度 4
        >>> tensor2 = torch.tensor([[1, 2, 4, 0], [4, 5, 6, 0]])  # 第 1 个样本的第 3 个 token 不同
        >>> mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])     # 最后一个位置是 padding，无效
        >>> diff_counts = calculate_token_list_diff(tensor1, tensor2, mask)
        >>> # 预期输出：[1, 0]（第 1 个样本有 1 个差异，第 2 个样本无差异）
        >>> # 手算过程：
        >>> # - 样本 1：位置 0, 1, 2 有效，其中位置 2 不同（3 vs 4），差异数 = 1
        >>> # - 样本 2：位置 0, 1, 2 有效，全部相同，差异数 = 0
        >>> assert diff_counts.tolist() == [1, 0]

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/debug/metrics.py
        - 函数：calculate_token_list_diff(tensor1, tensor2, mask)

    典型调用路径：
        （本函数目前未被直接调用，但可用于调试 token 序列差异）

    被谁调用：
        - 调试脚本（用户自定义）

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - torch.numel()：获取张量元素总数
        - torch.Tensor.to()：将张量移动到指定设备
        - torch.sum()：沿指定维度求和

    记忆提示：
        - 函数名：calculate_token_list_diff（计算 token 列表差异）
        - 核心逻辑：逐位比较 -> 过滤有效位置 -> 统计差异数量
        - 返回值：每个样本的差异 token 数量
    """  # 注释：函数 docstring 结束，下面进入函数体

    # verify inputs
    # 注释：步骤 1：验证输入有效性（检查是否为空张量）
    if tensor1.numel() == 0 or tensor2.numel() == 0:  # 注释：numel() 返回张量元素总数，0 表示空张量
        # 注释：如果任一张量为空，返回形状为 [batch_size] 的全 0 张量
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)  # 注释：dtype=torch.long 表示整数类型（int64）

    # 注释：步骤 2：验证形状一致性（三个张量必须形状相同）
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:  # 注释：检查所有张量形状是否相同
        # 注释：打印警告信息（包含三个张量的形状）
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"  # 注释：f-string 格式化，= 后缀会同时打印变量名和值
        )
        # 注释：返回形状为 [batch_size, seq_len] 的全 1 张量（表示所有位置都不同）
        return torch.ones_like(tensor1)  # 注释：torch.ones_like 创建与 tensor1 形状和设备相同的全 1 张量

    # transfer to same device
    # 注释：步骤 3：确保所有张量在同一设备上（CPU 或 GPU）
    if tensor2.device != tensor1.device:  # 注释：检查 tensor2 是否与 tensor1 在同一设备
        tensor2 = tensor2.to(tensor1.device)  # 注释：将 tensor2 移动到 tensor1 的设备（可能触发 GPU <-> CPU 拷贝）
    if mask.device != tensor1.device:  # 注释：检查 mask 是否与 tensor1 在同一设备
        mask = mask.to(tensor1.device)  # 注释：将 mask 移动到 tensor1 的设备

    # calculate diff
    # 注释：步骤 4：计算逐位差异（生成 bool 掩码，True 表示不同）
    diff_mask = tensor1 != tensor2  # 注释：逐元素比较，形状为 [batch_size, seq_len]，True 表示对应位置的 token 不同

    # 注释：步骤 5：过滤有效位置（只统计 mask=1 的位置）
    valid_diff_mask = diff_mask & (mask == 1)  # 注释：逻辑与运算，只保留有效位置的差异（忽略 padding 等无效位置）

    # 注释：步骤 6：统计每个样本的差异 token 数量（沿 seq_len 维度求和）
    diff_counts = valid_diff_mask.sum(dim=1)  # 注释：sum(dim=1) 沿第 1 维（seq_len）求和，返回形状为 [batch_size]

    # 注释：返回差异统计结果
    return diff_counts  # 注释：返回每个样本的差异 token 数量


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # 注释：函数定义，计算两个张量的 Pearson 相关系数
    """（函数说明：计算两个张量的 Pearson 相关系数）

    Pearson 相关系数是一个介于 -1 和 1 之间的值，用于衡量两个变量之间的线性相关程度：
    - r = 1：完全正相关（两个变量完全线性相关，且斜率为正）
    - r = 0：无线性相关（两个变量之间没有线性关系）
    - r = -1：完全负相关（两个变量完全线性相关，且斜率为负）

    在 RLHF 训练中，rollout_probs 和 actor_probs 应该高度相关（r 接近 1），
    如果相关系数较低（如 r < 0.95），说明数值不稳定或存在 bug。

    参数：
        tensor1 (torch.Tensor): 第一个张量，形状为 [batch_size, seq_len]
        tensor2 (torch.Tensor): 第二个张量，形状为 [batch_size, seq_len]
        mask (torch.Tensor): 有效位置掩码，形状为 [batch_size, seq_len]
            - True 或 1 表示有效位置，False 或 0 表示无效位置

    返回：
        float: Pearson 相关系数，范围 [-1, 1]
            - 如果形状不匹配，返回 0

    副作用：
        - 无

    异常/边界条件：
        - 如果形状不匹配，返回 0
        - 如果有效元素数量 < 2，torch.corrcoef 会抛出异常

    最小示例（手算验证）：
        >>> # 假设两个概率分布非常接近
        >>> tensor1 = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        >>> tensor2 = torch.tensor([[0.11, 0.19, 0.71], [0.29, 0.41, 0.31]])
        >>> mask = torch.tensor([[True, True, True], [True, True, True]])
        >>> r = pearson_correlation_coefficient(tensor1, tensor2, mask)
        >>> # 预期输出：接近 1（两个分布高度相关）
        >>> print(r)  # 约 0.9998

    参考文献：
        - https://arxiv.org/pdf/2506.13585
        - Pearson 相关系数定义：https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/debug/metrics.py
        - 函数：pearson_correlation_coefficient(tensor1, tensor2, mask)

    典型调用路径：
        calculate_debug_metrics(data)
        -> pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)  # 当前函数
        -> 返回 Pearson 相关系数

    被谁调用：
        - calculate_debug_metrics()

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - torch.masked_select：根据 mask 提取有效元素
        - torch.corrcoef：计算相关系数矩阵
        - torch.stack：将两个张量堆叠为 [2, num_valid_elements] 形状

    记忆提示：
        - 函数名：pearson_correlation_coefficient（Pearson 相关系数）
        - 核心逻辑：masked_select -> stack -> corrcoef -> 返回 [0][1]
        - 返回值：接近 1 表示高度相关，接近 0 表示无关
    """  # 注释：函数 docstring 结束，下面进入函数体

    # implemention of https://arxiv.org/pdf/2506.13585
    # 注释：实现参考论文：https://arxiv.org/pdf/2506.13585

    # 注释：步骤 1：验证形状一致性
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:  # 注释：检查所有张量形状是否相同
        return 0  # 注释：形状不匹配，返回 0（无相关性）

    # 注释：步骤 2：根据 mask 提取有效元素（过滤 padding 等无效位置）
    mt1 = torch.masked_select(tensor1, mask)  # 注释：masked_select 返回 1D 张量，只包含 mask=True 位置的元素
    mt2 = torch.masked_select(tensor2, mask)  # 注释：mt1 和 mt2 的长度相同（有效元素数量）

    # 注释：步骤 3：计算 Pearson 相关系数矩阵
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))  # 注释：stack 将两个 1D 张量堆叠为 [2, num_valid_elements]
    # 注释：corrcoef 返回 [2, 2] 相关系数矩阵：
    # 注释：[[corr(mt1, mt1), corr(mt1, mt2)],
    # 注释： [corr(mt2, mt1), corr(mt2, mt2)]]
    # 注释：对角线元素为 1（自身相关），非对角线元素为互相关系数

    # 注释：步骤 4：提取 mt1 与 mt2 的相关系数（result[0][1] 或 result[1][0]，两者相同）
    return result[0][1].detach().item()  # 注释：detach() 从计算图中分离，item() 转换为 Python 标量（float）


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # 注释：函数定义，计算两个 log_probs 的绝对差异
    """（函数说明：计算两个 log_probs 的绝对差异，并过滤有效位置）

    本函数用于计算两个 log_probs（如 rollout_log_probs 和 actor_log_probs）之间的差异，
    并根据 mask 提取有效位置的差异值。

    参数：
        log_probs1 (torch.Tensor): 第一个 log_probs，形状为 [batch_size, seq_len]
        log_probs2 (torch.Tensor): 第二个 log_probs，形状为 [batch_size, seq_len]
        mask (torch.Tensor): 有效位置掩码，形状为 [batch_size, seq_len]
            - True 或 1 表示有效位置，False 或 0 表示无效位置

    返回：
        torch.Tensor: 有效位置的差异值，形状为 [num_valid_elements]
            - 例如：如果批次中有 100 个有效 token，返回形状为 [100] 的张量

    副作用：
        - 无

    异常/边界条件：
        - 如果 mask 全为 False，返回空张量

    最小示例（手算验证）：
        >>> log_probs1 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> log_probs2 = torch.tensor([[0.11, 0.19, 0.31], [0.39, 0.51, 0.59]])
        >>> mask = torch.tensor([[True, True, False], [True, True, True]])  # 第 1 个样本的第 3 个位置无效
        >>> diff = calculate_log_prob_diff(log_probs1, log_probs2, mask)
        >>> # 预期输出：[0.01, 0.01, 0.01, 0.01, 0.01]（5 个有效位置的差异）
        >>> # 手算过程：
        >>> # - 位置 [0, 0]: |0.1 - 0.11| = 0.01
        >>> # - 位置 [0, 1]: |0.2 - 0.19| = 0.01
        >>> # - 位置 [0, 2]: 跳过（mask=False）
        >>> # - 位置 [1, 0]: |0.4 - 0.39| = 0.01
        >>> # - 位置 [1, 1]: |0.5 - 0.51| = 0.01
        >>> # - 位置 [1, 2]: |0.6 - 0.59| = 0.01
        >>> assert diff.tolist() == [0.01, 0.01, 0.01, 0.01, 0.01]

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/debug/metrics.py
        - 函数：calculate_log_prob_diff(log_probs1, log_probs2, mask)

    典型调用路径：
        calculate_debug_metrics(data)
        -> calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)  # 当前函数
        -> 返回差异张量

    被谁调用：
        - calculate_debug_metrics()

    调用了谁（项目内）：
        - 无

    调用了谁（外部依赖）：
        - torch.abs：计算绝对值
        - torch.masked_select：根据 mask 提取有效元素

    记忆提示：
        - 函数名：calculate_log_prob_diff（计算 log_probs 差异）
        - 核心逻辑：abs(diff) -> masked_select
        - 返回值：1D 张量（有效位置的差异值）
    """  # 注释：函数 docstring 结束，下面进入函数体

    # 注释：步骤 1：计算逐元素绝对差异
    full_diff = torch.abs(log_probs1 - log_probs2)  # 注释：abs 计算绝对值，形状为 [batch_size, seq_len]

    # 注释：步骤 2：根据 mask 提取有效位置的差异值
    return torch.masked_select(full_diff, mask)  # 注释：返回 1D 张量，只包含 mask=True 位置的差异值


def calculate_debug_metrics(data: DataProto) -> dict:  # 注释：函数定义，计算 rollout vs actor 的调试指标
    """（函数说明：计算 Rollout 与 Actor 概率分布的调试指标）

    本函数是模块的主要入口，用于计算 Rollout 阶段生成的 log_probs 与 Actor 前向传播的 log_probs 之间的差异。
    这些指标用于调试 RLHF/PPO 训练过程中的数值稳定性问题。

    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
                注释：Rollout 阶段生成 response 时记录的 log_probs
            old_log_probs(actor log probs): log_probs record when actor forward tokens
                注释：Actor 阶段重新前向传播时计算的 log_probs（也称为 old_log_probs）
            loss_mask or attention_mask: to mask unrelated token
                注释：用于过滤无效 token（如 padding）的掩码，优先使用 response_mask
            responses: the response tokens, for calculating size
                注释：生成的 response token 序列，用于确定 response 长度

    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
                注释：输入有效性标志（1=有效，0=无效，用于日志系统过滤无效数据）
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
                注释：log_probs 差异的最大值（检测最大偏差）
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
                注释：log_probs 差异的均值（评估平均偏差）
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
                注释：log_probs 差异的标准差（评估分布离散程度）
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
                注释：Pearson 相关系数（评估线性相关性，接近 1 为理想）

    副作用：
        - 可能输出 debug 或 warning 日志

    异常/边界条件：
        - 如果 data.batch 中缺少必要字段，会抛出 KeyError

    最小示例（手算验证）：
        >>> from verl.protocol import DataProto
        >>> import torch
        >>> # 构造测试数据
        >>> batch = {
        ...     "rollout_log_probs": torch.tensor([[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6]]),
        ...     "old_log_probs": torch.tensor([[-0.11, -0.19, -0.31], [-0.39, -0.51, -0.59]]),
        ...     "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        ...     "responses": torch.tensor([[10, 20, 0], [30, 40, 50]]),
        ... }
        >>> data = DataProto(batch=batch)
        >>> metrics = calculate_debug_metrics(data)
        >>> print(metrics)
        {
            "training/rollout_probs_diff_valid": 1,
            "training/rollout_probs_diff_max": 0.01,
            "training/rollout_probs_diff_mean": 0.01,
            "training/rollout_probs_diff_std": 0.0,
            "training/rollout_actor_probs_pearson_corr": 1.0
        }

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/debug/metrics.py
        - 函数：calculate_debug_metrics(data: DataProto) -> dict

    典型调用路径：
        verl/trainer/main_ppo.py (入口脚本)
        -> verl/trainer/ppo/ray_trainer.py::PPOTrainer.fit()
        -> （可选）启用调试模式
        -> verl/utils/debug/metrics.py::calculate_debug_metrics(data)  # 当前函数
        -> 返回调试指标字典

    被谁调用：
        - verl/trainer/ppo/ray_trainer.py（PPO/GRPO 训练循环）
        - 用户自定义调试脚本

    调用了谁（项目内）：
        - pearson_correlation_coefficient()
        - calculate_log_prob_diff()

    调用了谁（外部依赖）：
        - torch.exp：计算指数（log_probs -> probs）
        - torch.max、torch.mean、torch.std：统计计算
        - logging.debug、logging.warning：日志输出

    记忆提示：
        - 函数名：calculate_debug_metrics（计算调试指标）
        - 核心逻辑：提取 log_probs -> 计算差异 -> 统计指标
        - 返回值：包含 5 个调试指标的字典
    """  # 注释：函数 docstring 结束，下面进入函数体

    # 注释：步骤 1：从 DataProto 对象中提取必要的数据字段
    rollout_old_log_probs = data.batch["rollout_log_probs"]  # 注释：Rollout 阶段记录的 log_probs，形状 [batch_size, seq_len]
    actor_old_log_probs = data.batch["old_log_probs"]  # 注释：Actor 阶段计算的 log_probs（也称为 old_log_probs），形状 [batch_size, seq_len]

    # 注释：步骤 2：确定使用哪个 mask（优先级：response_mask > attention_mask > 全 1 mask）
    if "response_mask" in data.batch:  # 注释：检查是否存在 response_mask（专门用于标记 response 部分）
        logger.debug("response mask found, use it to mask log probs")  # 注释：debug 日志，只在启用 DEBUG 级别时输出
        log_prob_mask = data.batch["response_mask"]  # 注释：使用 response_mask（推荐，只关注 response 部分）
    elif "attention_mask" in data.batch:  # 注释：如果没有 response_mask，尝试使用 attention_mask
        log_prob_mask = data.batch["attention_mask"]  # 注释：使用 attention_mask（会包含 prompt 部分，不够精确）
    else:  # 注释：如果两者都不存在，打印警告并使用全 1 mask（所有位置都有效）
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")  # 注释：warning 日志，提示缺少 mask 信息
        log_prob_mask = torch.ones_like(rollout_old_log_probs)  # 注释：创建全 1 mask（形状与 log_probs 相同）

    # 注释：步骤 3：提取 responses 字段，用于确定 response 长度
    responses = data.batch["responses"]  # 注释：response token 序列，形状 [batch_size, response_len]
    response_length = responses.size(1)  # 注释：response 的序列长度（第 1 维）

    # 注释：步骤 4：提取 response 部分的 mask（只关注 response，忽略 prompt）
    response_mask = log_prob_mask[:, -response_length:]  # 注释：切片取最后 response_length 列，形状 [batch_size, response_len]

    # calculate pearson corrcoef
    # 注释：步骤 5：将 log_probs 转换为 probs（exp(log_probs)）
    actor_probs = torch.exp(actor_old_log_probs)  # 注释：exp 将 log_probs 转换为概率，形状 [batch_size, seq_len]
    rollout_probs = torch.exp(rollout_old_log_probs)  # 注释：同上

    # 注释：步骤 6：将 mask 转换为 bool 类型（torch.corrcoef 需要 bool mask）
    response_mask_bool = response_mask.bool()  # 注释：将 0/1 转换为 False/True

    # 注释：步骤 7：计算 Pearson 相关系数（评估两个概率分布的线性相关性）
    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)  # 注释：调用前面定义的函数

    # 注释：步骤 8：计算 log_probs 差异（绝对值）
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)  # 注释：调用前面定义的函数，返回 1D 张量

    # 注释：步骤 9：返回调试指标字典
    return {
        "training/rollout_probs_diff_valid": 1,  # 注释：输入有效性标志（始终为 1，表示数据有效）
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),  # 注释：差异的最大值（标量）
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),  # 注释：差异的均值（标量）
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),  # 注释：差异的标准差（标量）
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,  # 注释：Pearson 相关系数（标量）
    }  # 注释：返回字典，键名符合 WandB/MLflow 日志格式（training/ 前缀）
