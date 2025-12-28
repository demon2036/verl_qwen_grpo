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

"""（模块说明：训练过程中的 FLOPS 统计与 MFU 计算工具）

模块用途：
    本模块提供 FLOPS（Floating Point Operations Per Second，每秒浮点运算次数）统计功能，
    用于计算训练过程中的模型浮点运算吞吐量（MFU - Model FLOPs Utilization）。
    主要功能包括：
    1. 获取不同 GPU/NPU 设备的理论峰值 FLOPS
    2. 根据模型架构（Qwen2/LLaMA/DeepSeek V3/MoE 等）估算实际 FLOPS
    3. 计算 MFU = 实际 FLOPS / 理论峰值 FLOPS，评估硬件利用率

    支持的模型架构：
    - Dense 模型：Qwen2、LLaMA、Gemma3、Apertus、MiniCPM 等
    - MoE 模型：Qwen2-MoE、Qwen3-MoE、DeepSeek V3
    - VL 模型：Qwen2-VL、Qwen2.5-VL、Qwen3-VL、GLM4V

    支持的硬件：
    - NVIDIA: GB200/B200/H100/H200/H800/A100/A800/L40S/L40/A40/L20/H20/RTX 系列
    - AMD: MI300X
    - Ascend NPU: 910B/Ascend910
    - CPU（理论值）

输入/输出：
    - 输入：
      * PretrainedConfig：模型配置对象（包含 hidden_size、num_layers 等）
      * batch_seqlens：批次中每个样本的有效 token 数列表，例如 [128, 256, 512]
      * delta_time：处理该批次的时间（秒）
    - 输出：
      * estimated_flops：根据 token 数和时间估算的实际 FLOPS（单位：TFLOPS，即 10^12 FLOPS/s）
      * promised_flops：当前设备的理论峰值 FLOPS（单位：TFLOPS）
      * mfu：MFU = estimated_flops / promised_flops（范围：0~1，越接近 1 表示硬件利用率越高）

关键依赖：
    - torch：用于设备检测（torch.cpu、torch.cuda.get_device_name 等）
    - transformers.PretrainedConfig：模型配置类，包含架构参数
    - verl.utils.device.get_torch_device：获取当前 PyTorch 设备对象

典型用法：
    >>> from transformers import AutoConfig
    >>> from verl.utils.flops_counter import FlopsCounter
    >>> # 加载 Qwen2-7B 的配置
    >>> config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
    >>> flops_counter = FlopsCounter(config)
    >>> # 假设一个批次包含 3 个样本，长度分别为 128、256、512
    >>> batch_seqlens = [128, 256, 512]
    >>> delta_time = 0.5  # 处理耗时 0.5 秒
    >>> estimated_flops, promised_flops = flops_counter.estimate_flops(batch_seqlens, delta_time)
    >>> mfu = estimated_flops / promised_flops
    >>> print(f"实际 FLOPS: {estimated_flops:.2f} TFLOPS")
    >>> print(f"理论峰值: {promised_flops:.2f} TFLOPS")
    >>> print(f"MFU: {mfu:.2%}")

调用路径概览：
    入口脚本（如 verl/trainer/main_ppo.py 或 verl/trainer/fsdp_sft_trainer.py）
    -> 训练循环中初始化 FlopsCounter(config)
    -> 每个训练步骤结束后调用 flops_counter.estimate_flops(batch_seqlens, delta_time)
    -> 计算 MFU 并记录到日志或 WandB/MLflow
    -> （可选）在 verl/utils/profiler/performance.py 中聚合多步 MFU 指标

所在位置：
    - 路径：verl/utils/flops_counter.py
    - 模块：verl.utils.flops_counter

被谁调用：
    - verl/trainer/fsdp_sft_trainer.py（SFT 训练中计算 MFU）
    - verl/trainer/ppo/ray_trainer.py（PPO/GRPO 训练中计算 MFU）
    - verl/workers/fsdp_workers.py（FSDP worker 中可能统计 MFU）
    - verl/utils/profiler/performance.py（性能分析工具）

调用了谁（项目内）：
    - verl.utils.device.get_torch_device()（获取当前设备对象）

调用了谁（外部依赖）：
    - torch.cpu、torch.cuda.get_device_name()（设备检测）
    - transformers.PretrainedConfig（模型配置）

注意事项：
    1. FLOPS 估算基于理论计算公式，假设前向+反向传播共 6 倍参数量的 FLOPS
    2. 不同模型架构的 FLOPS 计算公式差异较大（如 MoE 需要考虑 topk 和专家数）
    3. 实际 FLOPS 受以下因素影响：
       - 激活函数类型（SwiGLU vs GeGLU vs XIELU）
       - 注意力类型（全注意力 vs 滑动窗口 vs MLA）
       - 批次内序列长度分布（长序列会增加注意力 FLOPS）
    4. MFU 通常在 0.3~0.6 之间（30%~60%），超过 0.7 说明优化得很好
    5. 如果模型架构不在 ESTIMATE_FUNC 中，MFU 会始终为 0

最小手算示例（Qwen2-0.5B 单层 FLOPS）：
    假设 Qwen2-0.5B 配置：
    - hidden_size = 896
    - num_layers = 24
    - num_heads = 14
    - intermediate_size = 4864
    - vocab_size = 151936
    - 批次：1 个样本，长度 128 token
    - 时间：0.1 秒

    步骤 1：计算单层参数量（不含注意力）
    - MLP: hidden * intermediate * 3（SwiGLU 有 gate、up、down）
           = 896 * 4864 * 3 = 13,070,848
    - Attention: hidden * (Q + K + V + O) = 896 * (896 + 896 + 896 + 896) = 3,211,264
    - 单层参数 N_layer = 13,070,848 + 3,211,264 = 16,282,112

    步骤 2：计算所有层参数量（不含注意力 QKV 乘法）
    - 24 层 + Embedding + LM Head
    - N_dense = 16,282,112 * 24 + 151936 * 896 * 2 = 663,558,528

    步骤 3：计算非注意力 FLOPS（前向+反向=6倍）
    - dense_flops = 6 * N_dense * tokens = 6 * 663,558,528 * 128 = 509,707,156,480（约 510 GFLOPS）

    步骤 4：计算注意力 QKV FLOPS（前向+反向=12倍）
    - seqlen^2 = 128 * 128 = 16,384
    - head_dim = 896 / 14 = 64
    - attn_flops = 12 * 16,384 * 64 * 14 * 24 = 424,673,280（约 0.42 GFLOPS）

    步骤 5：总 FLOPS
    - total = 509,707,156,480 + 424,673,280 = 510,131,829,760（约 510 GFLOPS）
    - 时间 0.1 秒，FLOPS/s = 510 / 0.1 = 5100 GFLOPS = 5.1 TFLOPS

    步骤 6：计算 MFU（假设 H100 理论峰值 989 TFLOPS）
    - MFU = 5.1 / 989 ≈ 0.005（0.5%，非常低，因为批次太小）

    记忆提示：
    - FLOPS = 6 * params * tokens（密集层）+ 12 * seqlen^2 * head_dim * num_heads * num_layers（注意力）
    - MFU = 实际 FLOPS / 理论峰值
    - SwiGLU 有 3 个矩阵（gate、up、down），GeGLU 也是 3 个，但 XIELU 只有 2 个（up、down）
"""  # 注释：模块级 docstring 结束，下面是依赖导入

# 注释：导入 PyTorch 库，用于设备检测（torch.cpu、torch.cuda.get_device_name 等）
import torch  # 注释：torch 提供设备抽象层，可以获取当前设备类型和名称

# 注释：导入 Hugging Face transformers 的模型配置基类，所有模型配置都继承自它
from transformers import PretrainedConfig  # 注释：PretrainedConfig 包含模型架构参数（hidden_size、num_layers 等）

# 注释：导入 VERL 项目的设备工具函数，用于获取当前 PyTorch 设备对象
from verl.utils.device import get_torch_device  # 注释：get_torch_device() 返回 torch.device 对象，封装了 GPU/CPU/NPU 检测逻辑

# 注释：定义设备理论峰值 FLOPS 字典（单位：FLOPS/s，即每秒浮点运算次数）
# 注释：键为设备名称（包含在 torch.cuda.get_device_name() 返回值中的子串），值为理论峰值 FLOPS
_DEVICE_FLOPS = {  # 注释：字典存储不同 GPU/NPU 的理论性能，用于 MFU 计算的分母
    # 注释：CPU 理论峰值（单核约 448 GFLOPS，多核需乘以核心数，此处为示例值）
    "CPU": 448e9,  # 注释：448 GFLOPS = 4.48 * 10^11 FLOPS/s
    # 注释：NVIDIA 最新一代 GPU（GB200/B200 系列）
    "GB200": 2.5e15,  # 注释：2.5 PFLOPS（FP16/BF16 峰值，约 2500 TFLOPS）
    "B200": 2.25e15,  # 注释：2.25 PFLOPS
    # 注释：AMD MI300X（数据中心 GPU）
    "MI300X": 1336e12,  # 注释：1336 TFLOPS（FP16 峰值）
    # 注释：NVIDIA H 系列（Hopper 架构，适用于 AI 训练）
    "H100": 989e12,  # 注释：989 TFLOPS（FP16 Tensor Core 峰值）
    "H800": 989e12,  # 注释：H800 是 H100 的中国出口版本，性能相同
    "H200": 989e12,  # 注释：H200 是 H100 的高内存版本（141GB HBM3e vs 80GB HBM3）
    # 注释：NVIDIA A 系列（Ampere 架构）
    "A100": 312e12,  # 注释：312 TFLOPS（FP16 Tensor Core 峰值）
    "A800": 312e12,  # 注释：A800 是 A100 的中国出口版本
    # 注释：NVIDIA L 系列（推理优化）
    "L40S": 362.05e12,  # 注释：362 TFLOPS（FP16 峰值）
    "L40": 181.05e12,  # 注释：181 TFLOPS
    "L20": 119.5e12,  # 注释：119.5 TFLOPS
    # 注释：NVIDIA A40（数据中心 GPU）
    "A40": 149.7e12,  # 注释：149.7 TFLOPS
    # 注释：NVIDIA H20（中国市场专供版本）
    "H20": 148e12,  # 注释：148 TFLOPS
    # 注释：华为昇腾 NPU（Ascend 910 系列）
    "910B": 354e12,  # 注释：354 TFLOPS（FP16 峰值）
    "Ascend910": 354e12,  # 注释：Ascend910 是昇腾 910 的全称
    # 注释：NVIDIA 消费级 GPU（桌面/工作站）
    "RTX 3070 Ti": 21.75e12,  # 注释：21.75 TFLOPS（FP16 峰值）
}  # 注释：字典结束，后续可以按需添加更多设备


def get_device_flops(unit="T", device_name=None):  # 注释：函数定义，获取当前设备的理论峰值 FLOPS
    """（函数说明：获取当前设备的理论峰值 FLOPS，支持单位转换）

    本函数根据设备名称查询理论峰值 FLOPS，并转换为指定单位。
    如果设备名称未在 _DEVICE_FLOPS 字典中，返回 float('inf')（无限大），避免 MFU 计算出错。

    参数：
        unit (str): 返回值的单位，支持以下选项：
            - "B" - Billion (10^9，十亿)
            - "K" - Thousand (10^3，千)
            - "M" - Million (10^6，百万)
            - "G" - Giga (10^9，十亿，与 "B" 相同)
            - "T" - Tera (10^12，万亿，默认值)
            - "P" - Peta (10^15，千万亿)
        device_name (str | None): 设备名称（可选，仅用于测试）
            - 如果为 None，自动检测当前设备名称
            - 如果提供，直接使用该名称查询（便于单元测试）

    返回：
        float: 当前设备的理论峰值 FLOPS，单位为 unit 指定的单位
            - 如果设备名称在 _DEVICE_FLOPS 中找到，返回对应的 FLOPS 值
            - 如果找不到，返回 float('inf')（无限大），避免除零错误

    副作用：
        - 无

    异常/边界条件：
        - 如果 unit 不在支持列表中，单位转换可能不正确（但不会抛出异常）
        - 如果设备名称包含多个匹配项（如 "H100" 和 "RTX 3070 Ti" 都包含在名称中），
          返回字典中第一个匹配的值（按逆序排序后的第一个）

    最小示例（手算验证）：
        >>> # 场景 1：查询 H100 的 FLOPS（默认单位 TFLOPS）
        >>> flops = get_device_flops(unit="T", device_name="NVIDIA H100 80GB")
        >>> # 预期输出：989 TFLOPS（因为 "H100" 在 device_name 中，_DEVICE_FLOPS["H100"] = 989e12）
        >>> # 单位转换：989e12 / 1e12 = 989.0
        >>> assert flops == 989.0

        >>> # 场景 2：查询 A100 的 FLOPS（单位为 GFLOPS）
        >>> flops = get_device_flops(unit="G", device_name="NVIDIA A100-SXM4-40GB")
        >>> # 预期输出：312000.0 GFLOPS（因为 "A100" 在 device_name 中，_DEVICE_FLOPS["A100"] = 312e12）
        >>> # 单位转换：312e12 / 1e9 = 312000.0
        >>> assert flops == 312000.0

        >>> # 场景 3：查询未知 GPU（返回 inf）
        >>> flops = get_device_flops(unit="T", device_name="Unknown GPU XYZ")
        >>> # 预期输出：inf（因为 "Unknown GPU XYZ" 不包含任何 _DEVICE_FLOPS 的键）
        >>> assert flops == float('inf')

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/flops_counter.py
        - 函数：get_device_flops(unit="T", device_name=None)

    典型调用路径：
        FlopsCounter.__init__(config)
        -> FlopsCounter.estimate_flops(batch_seqlens, delta_time)
        -> get_device_flops(unit="T")  # 当前函数
        -> 返回理论峰值 FLOPS

    被谁调用：
        - FlopsCounter.estimate_flops()（内部调用）
        - 测试代码（直接调用以验证设备 FLOPS 查询）

    调用了谁（项目内）：
        - verl.utils.device.get_torch_device()（获取当前设备对象）

    调用了谁（外部依赖）：
        - torch.cpu（检测是否为 CPU 设备）
        - torch.device.get_device_name()（获取 GPU 设备名称）

    记忆提示：
        - 函数名：get_device_flops（获取设备 FLOPS）
        - 核心逻辑：根据设备名称子串匹配 _DEVICE_FLOPS 字典
        - 未知设备返回 inf，避免除零错误
        - 单位转换：从 FLOPS/s 转换为 TFLOPS/GFLOPS 等
    """  # 注释：函数 docstring 结束，下面进入函数体

    # 注释：定义内部辅助函数，用于单位转换（将 FLOPS/s 转换为指定单位）
    def unit_convert(number, level):  # 注释：number 为原始 FLOPS 值，level 为目标单位（"T"/"G"/"M" 等）
        """（内部函数说明：将 FLOPS 数值从基础单位转换为指定单位）

        参数：
            number (float): 原始 FLOPS 值（单位：FLOPS/s，即 1）
            level (str): 目标单位（"B"/"K"/"M"/"G"/"T"/"P"）

        返回：
            float: 转换后的 FLOPS 值

        工作原理：
            - 从基础单位（1 FLOPS/s）开始，每次除以 1000，向上移动一个单位级别
            - 例如：1e12 FLOPS/s -> 1000 GFLOPS -> 1 TFLOPS
        """  # 注释：内部函数 docstring 结束

        # 注释：定义单位列表（从小到大：Billion -> Thousand -> Million -> Giga -> Tera -> Peta）
        units = ["B", "K", "M", "G", "T", "P"]  # 注释：每个单位相差 1000 倍（10^3）

        # 注释：特殊处理：如果数值 <= 0，直接返回原值（避免除以 1000 导致的浮点误差）
        if number <= 0:  # 注释：0 或负数不需要单位转换
            return number  # 注释：直接返回 0 或负数

        # 注释：初始化指针，指向单位列表的起始位置（"B"）
        ptr = 0  # 注释：ptr 表示当前单位在 units 列表中的索引

        # 注释：循环除以 1000，直到找到目标单位或遍历完所有单位
        while ptr < len(units) and units[ptr] != level:  # 注释：条件 1：未超出列表范围；条件 2：当前单位不是目标单位
            number /= 1000  # 注释：每次除以 1000，向上移动一个单位级别（例如：GFLOPS -> TFLOPS）
            ptr += 1  # 注释：指针向后移动一位

        # 注释：返回转换后的数值
        return number  # 注释：此时 number 已经被转换为目标单位

    # 注释：步骤 1：获取设备名称（如果未提供，自动检测）
    # pass device_name is for testing purpose only
    if device_name is None:  # 注释：如果调用者未提供设备名称，需要自动检测
        # 注释：获取当前 PyTorch 设备对象（torch.device）
        device = get_torch_device()  # 注释：get_torch_device() 返回 torch.device，可能是 torch.cpu 或 torch.cuda:0 等

        # 注释：判断是否为 CPU 设备
        if device == torch.cpu:  # 注释：torch.cpu 是一个特殊的设备对象，表示 CPU
            device_name = "CPU"  # 注释：设置设备名称为 "CPU"，后续会匹配 _DEVICE_FLOPS["CPU"]

        else:  # 注释：如果不是 CPU，说明是 GPU 或 NPU
            # 注释：获取 GPU/NPU 设备名称（例如 "NVIDIA H100 80GB" 或 "Ascend910"）
            device_name = get_torch_device().get_device_name()  # 注释：get_device_name() 是 torch.device 的方法，返回设备全名

    # 注释：步骤 2：查询设备理论峰值 FLOPS（默认为 inf，表示未知设备）
    flops = float("inf")  # INF flops for unkown gpu type  # 注释：未知 GPU 类型返回无限大，避免 MFU 计算时除零错误

    # 注释：遍历 _DEVICE_FLOPS 字典，查找设备名称中包含的子串
    # 注释：逆序排序是为了优先匹配更长的名称（例如 "GB200" 比 "B200" 更长，应优先匹配 "GB200"）
    for key, value in sorted(_DEVICE_FLOPS.items(), reverse=True):  # 注释：sorted(..., reverse=True) 按键名逆序排序
        # 注释：检查设备名称是否包含当前键（子串匹配，忽略大小写）
        if key in device_name:  # 注释：例如 "NVIDIA H100 80GB" 包含 "H100"
            flops = value  # 注释：找到匹配项，设置 FLOPS 值
            break  # 注释：找到第一个匹配项后立即退出循环（逆序排序确保优先匹配长名称）

    # 注释：步骤 3：单位转换（将 FLOPS/s 转换为目标单位）
    flops_unit = unit_convert(flops, unit)  # 注释：调用内部函数 unit_convert，将 FLOPS 转换为 TFLOPS/GFLOPS 等

    # 注释：返回转换后的 FLOPS 值
    return flops_unit  # 注释：例如：989e12 FLOPS/s 转换为 989.0 TFLOPS


def _estimate_qwen2_flops(config, tokens_sum, batch_seqlens, delta_time):  # 注释：函数定义，估算 Qwen2/LLaMA 等 Dense 模型的 FLOPS
    """（函数说明：估算 Qwen2/LLaMA 等 Dense 模型的 FLOPS）

    本函数根据 Qwen2/LLaMA 的模型架构计算训练过程中的浮点运算量。
    适用于所有使用 SwiGLU MLP + 标准多头注意力的 Dense 模型。

    FLOPS 计算公式：
        1. 非注意力部分（MLP + Attention Linear）：
           FLOPS = 6 * N_dense * tokens_sum
           其中：
           - N_dense = 各层参数量之和（MLP + Attention Linear + Embedding + LM Head）
           - 6 倍来自前向传播（2 倍）+ 反向传播（4 倍）
           - tokens_sum = 批次中所有样本的 token 总数

        2. 注意力 QKV 乘法部分：
           FLOPS = 12 * seqlen_square_sum * head_dim * num_heads * num_layers
           其中：
           - seqlen_square_sum = sum([seqlen^2 for seqlen in batch_seqlens])
           - 12 倍来自前向传播（2 倍，Q@K 和 Attn@V）+ 反向传播（10 倍）
           - head_dim = hidden_size / num_heads

    参数：
        config (PretrainedConfig): 模型配置对象，必须包含以下字段：
            - hidden_size (int): 隐藏层维度（例如 Qwen2-7B 为 3584）
            - vocab_size (int): 词表大小（例如 151936）
            - num_hidden_layers (int): Transformer 层数（例如 28）
            - num_attention_heads (int): 注意力头数（例如 28）
            - num_key_value_heads (int): KV 头数（GQA 时小于 num_attention_heads）
            - intermediate_size (int): MLP 中间层维度（例如 18944）
            - head_dim (int, 可选): 每个注意力头的维度（默认为 hidden_size / num_attention_heads）
        tokens_sum (int): 批次中所有样本的 token 总数（例如 [128, 256, 512] -> 896）
        batch_seqlens (list[int]): 批次中每个样本的序列长度列表（例如 [128, 256, 512]）
        delta_time (float): 处理该批次的时间（秒）

    返回：
        float: 估算的 FLOPS（单位：TFLOPS，即 10^12 FLOPS/s）

    最小示例（手算验证）：
        假设 Qwen2-0.5B 配置：
        - hidden_size = 896
        - num_layers = 24
        - num_heads = 14
        - intermediate_size = 4864
        - vocab_size = 151936
        - 批次：1 个样本，长度 128 token
        - 时间：0.1 秒

        步骤 1：计算 MLP 参数量
        mlp_N = 896 * 4864 * 3 = 13,070,848（SwiGLU 有 gate、up、down 3 个矩阵）

        步骤 2：计算 Attention Linear 参数量
        head_dim = 896 / 14 = 64
        q_size = 14 * 64 = 896
        k_size = 14 * 64 = 896
        v_size = 14 * 64 = 896
        attn_linear_N = 896 * (896 + 896 + 896 + 896) = 3,211,264

        步骤 3：计算 Embedding + LM Head 参数量
        emd_and_lm_head_N = 151936 * 896 * 2 = 272,253,952

        步骤 4：计算总参数量
        dense_N = (13,070,848 + 3,211,264) * 24 + 272,253,952 = 663,558,528

        步骤 5：计算非注意力 FLOPS
        dense_N_flops = 6 * 663,558,528 * 128 = 509,707,156,480（约 510 GFLOPS）

        步骤 6：计算注意力 FLOPS
        seqlen_square_sum = 128 * 128 = 16,384
        attn_qkv_flops = 12 * 16,384 * 64 * 14 * 24 = 424,673,280（约 0.42 GFLOPS）

        步骤 7：总 FLOPS
        flops_all_token = 509,707,156,480 + 424,673,280 = 510,131,829,760（约 510 GFLOPS）
        flops_achieved = 510,131,829,760 / 0.1 / 1e12 = 5.1 TFLOPS
    """  # 注释：函数 docstring 结束

    # 注释：从配置中提取模型架构参数
    hidden_size = config.hidden_size  # 注释：隐藏层维度（例如 Qwen2-7B 为 3584）
    vocab_size = config.vocab_size  # 注释：词表大小（例如 151936）
    num_hidden_layers = config.num_hidden_layers  # 注释：Transformer 层数（例如 28）
    num_key_value_heads = config.num_key_value_heads  # 注释：KV 头数（GQA 时小于 num_attention_heads）
    num_attention_heads = config.num_attention_heads  # 注释：注意力头数（例如 28）
    intermediate_size = config.intermediate_size  # 注释：MLP 中间层维度（例如 18944）

    # 注释：计算每个注意力头的维度（如果配置中没有 head_dim，则自动计算）
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # 注释：head_dim = hidden_size / num_heads
    # 注释：计算 Q、K、V、O 的总参数量（用于后续 Attention Linear 参数统计）
    q_size = num_attention_heads * head_dim  # 注释：Q 的维度（通常等于 hidden_size）
    k_size = num_key_value_heads * head_dim  # 注释：K 的维度（GQA 时小于 Q）
    v_size = num_key_value_heads * head_dim  # 注释：V 的维度（GQA 时小于 Q）

    # non-attn per layer parm
    # 注释：计算单层 MLP 参数量（Qwen2/LLaMA 使用 SwiGLU，包含 gate、up、down 3 个线性层）
    # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
    mlp_N = hidden_size * intermediate_size * 3  # 注释：MLP 参数 = hidden * intermediate * 3（gate、up、down）

    # 注释：计算单层 Attention Linear 参数量（Q、K、V、O 四个投影矩阵）
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)  # 注释：最后一项是 O 投影矩阵

    # 注释：计算 Embedding + LM Head 参数量（输入嵌入层 + 输出语言模型头）
    emd_and_lm_head_N = vocab_size * hidden_size * 2  # 注释：2 倍是因为有 Embedding 和 LM Head 两个矩阵

    # non-attn all_layer parm
    # 注释：计算所有层的总参数量（不包括注意力 QKV 乘法）
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N  # 注释：总参数 = 单层参数 * 层数 + Embedding/LM Head

    # non-attn all_layer & all_token fwd & bwd flops
    # 注释：计算非注意力部分的 FLOPS（前向 + 反向 = 6 倍参数量）
    dense_N_flops = 6 * dense_N * tokens_sum  # 注释：6 倍来自前向（2 倍）+ 反向（4 倍）

    # attn all_layer & all_token fwd & bwd flops
    # 注释：计算注意力 QKV 乘法的 FLOPS（需要考虑序列长度的平方）
    seqlen_square_sum = 0  # 注释：累积所有样本的 seqlen^2
    for seqlen in batch_seqlens:  # 注释：遍历批次中每个样本的序列长度
        seqlen_square_sum += seqlen * seqlen  # 注释：注意力复杂度是 O(seqlen^2)
    # 注释：注意力 FLOPS = 12 * seqlen^2 * head_dim * num_heads * num_layers
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers  # 注释：12 倍来自前向（2 倍，Q@K 和 Attn@V）+ 反向（10 倍）

    # all_layer & all_token fwd & bwd flops
    # 注释：总 FLOPS = 非注意力 FLOPS + 注意力 FLOPS
    flops_all_token = dense_N_flops + attn_qkv_flops  # 注释：总浮点运算次数
    # 注释：FLOPS/s = 总 FLOPS / 时间，除以 1e12 转换为 TFLOPS
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12  # 注释：单位：TFLOPS（10^12 FLOPS/s）
    return flops_achieved  # 注释：返回估算的 FLOPS


def _estimate_deepseek_v3_flops(config, tokens_sum, batch_seqlens, delta_time):  # 注释：函数定义，估算 DeepSeek V3 MoE 模型的 FLOPS
    """（函数说明：估算 DeepSeek V3 MoE 模型的 FLOPS）

    DeepSeek V3 是一个大规模 MoE 模型，使用了以下关键技术：
    1. MLA（Multi-Latent Attention）：压缩 KV Cache，减少内存占用
    2. MoE（Mixture of Experts）：每个 token 只激活 topk 个专家
    3. Shared Experts：所有 token 都会经过的共享专家层
    4. First K Dense Replace：前几层使用 Dense MLP 而非 MoE

    FLOPS 计算需要考虑：
    - 路由器（gating）的参数量：hidden_size * n_routed_experts
    - 激活的专家参数量：hidden_size * moe_intermediate_size * topk * 3（SwiGLU）
    - 共享专家参数量：hidden_size * moe_intermediate_size * n_shared_experts * 3
    - MLA 注意力的参数量（包含 q_lora、kv_lora 等压缩层）
    - Dense 层参数量（前 K 层）

    参数：
        config (PretrainedConfig): DeepSeek V3 配置对象，必须包含：
            - hidden_size (int): 隐藏层维度
            - vocab_size (int): 词表大小
            - moe_intermediate_size (int): MoE 专家中间层维度
            - num_hidden_layers (int): 总层数
            - first_k_dense_replace (int): 前 K 层使用 Dense MLP
            - num_attention_heads (int): 注意力头数
            - n_routed_experts (int): 可路由的专家总数
            - num_experts_per_tok (int): 每个 token 激活的专家数（topk）
            - n_shared_experts (int): 共享专家数
            - q_lora_rank (int | None): Q 的 LoRA 秩（用于压缩 Q 投影）
            - kv_lora_rank (int): KV 的 LoRA 秩（用于压缩 KV 投影）
            - qk_nope_head_dim (int): QK 不使用 RoPE 的维度
            - qk_rope_head_dim (int): QK 使用 RoPE 的维度
            - v_head_dim (int): V 的维度
            - intermediate_size (int): Dense MLP 的中间层维度（前 K 层使用）
        tokens_sum (int): 批次中所有样本的 token 总数
        batch_seqlens (list[int]): 批次中每个样本的序列长度列表
        delta_time (float): 处理该批次的时间（秒）

    返回：
        float: 估算的 FLOPS（单位：TFLOPS）
    """  # 注释：函数 docstring 结束

    # 注释：从配置中提取模型架构参数
    hidden_size = config.hidden_size  # 注释：隐藏层维度
    vocab_size = config.vocab_size  # 注释：词表大小
    moe_intermediate_size = config.moe_intermediate_size  # 注释：MoE 专家中间层维度
    num_hidden_layers = config.num_hidden_layers  # 注释：总层数
    first_k_dense_replace = config.first_k_dense_replace  # 注释：前 K 层使用 Dense MLP
    num_query_heads = config.num_attention_heads  # 注释：注意力头数
    moe_num_expert = config.n_routed_experts  # 注释：可路由的专家总数

    moe_topk = config.num_experts_per_tok  # 注释：每个 token 激活的专家数（topk）
    share_expert_num = config.n_shared_experts  # 注释：共享专家数

    # non-attn per layer parm
    # 注释：计算单层 MoE 路由器参数量（gate 层）
    moe_gata_N = hidden_size * moe_num_expert  # 注释：路由器参数 = hidden * n_experts
    # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
    # 注释：计算单层 MoE 专家参数量（激活的 topk 个专家 + 共享专家）
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3  # 注释：3 倍来自 SwiGLU（gate、up、down）

    # MLA attn
    # 注释：计算单层 MLA 注意力参数量（包含 q_lora、kv_lora 等压缩层）
    attn_linear_N = 0  # 注释：初始化注意力参数量
    # 注释：计算 Q 头维度（RoPE 维度 + 非 RoPE 维度）
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 注释：Q 的总维度
    # 注释：判断是否使用 Q LoRA 压缩
    if config.q_lora_rank is None:  # 注释：不使用 LoRA，直接投影
        attn_linear_N += hidden_size * num_query_heads * q_head_dim  # 注释：Q 投影矩阵参数量
    else:  # 注释：使用 LoRA 压缩
        attn_linear_N += hidden_size * config.q_lora_rank  # 注释：Q 的低秩投影（down projection）
        attn_linear_N += num_query_heads * q_head_dim * config.q_lora_rank  # 注释：Q 的高秩投影（up projection）

    # 注释：计算 KV 的 LoRA 压缩参数量
    attn_linear_N += hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)  # 注释：KV 的低秩投影 + RoPE 投影
    attn_linear_N += num_query_heads * (q_head_dim - config.qk_rope_head_dim + config.v_head_dim) * config.kv_lora_rank  # 注释：KV 的高秩投影
    attn_linear_N += num_query_heads * config.v_head_dim * hidden_size  # 注释：V 的输出投影

    # 注释：计算 Embedding + LM Head 参数量
    emd_and_lm_head_N = vocab_size * hidden_size * 2  # 注释：2 倍是因为有 Embedding 和 LM Head

    # non-attn all_layer parm
    # 注释：计算所有层的总参数量（MoE 层 + Dense 层 + Embedding/LM Head）
    moe_N = (
        (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)  # 注释：MoE 层参数
        + (hidden_size * config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace  # 注释：Dense 层参数（前 K 层）
        + emd_and_lm_head_N  # 注释：Embedding + LM Head 参数
    )

    # non-attn all_layer & all_token fwd & bwd flops
    # 注释：计算非注意力部分的 FLOPS（前向 + 反向 = 6 倍参数量）
    dense_N_flops = 6 * moe_N * tokens_sum  # 注释：6 倍来自前向（2 倍）+ 反向（4 倍）

    # attn all_layer & all_token fwd & bwd flops
    # 注释：计算注意力 QKV 乘法的 FLOPS（需要考虑序列长度的平方）
    seqlen_square_sum = 0  # 注释：累积所有样本的 seqlen^2 * num_layers
    for seqlen in batch_seqlens:  # 注释：遍历批次中每个样本的序列长度
        seqlen_square_sum += seqlen * seqlen * num_hidden_layers  # 注释：注意力复杂度是 O(seqlen^2 * num_layers)

    # 注释：注意力 FLOPS = 12 * seqlen^2 * q_head_dim * num_heads
    attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads  # 注释：12 倍来自前向（2 倍）+ 反向（10 倍）

    # all_layer & all_token fwd & bwk flops
    # 注释：总 FLOPS = 非注意力 FLOPS + 注意力 FLOPS
    flops_all_token = dense_N_flops + attn_qkv_flops  # 注释：总浮点运算次数
    # 注释：FLOPS/s = 总 FLOPS / 时间，除以 1e12 转换为 TFLOPS
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12  # 注释：单位：TFLOPS

    return flops_achieved  # 注释：返回估算的 FLOPS


def _estimate_qwen2_moe_flops(config, tokens_sum, batch_seqlens, delta_time):  # 注释：函数定义，估算 Qwen2-MoE 模型的 FLOPS
    """（函数说明：估算 Qwen2-MoE 模型的 FLOPS）

    Qwen2-MoE 使用标准的 MoE 架构（与 DeepSeek V3 不同，不使用 MLA 和 Shared Experts）。
    每个 token 只激活 topk 个专家，路由器根据 token 特征选择专家。

    FLOPS 计算需要考虑：
    - 路由器（gating）的参数量：hidden_size * num_experts
    - 激活的专家参数量：hidden_size * moe_intermediate_size * topk * 3（SwiGLU）
    - 标准多头注意力参数量（Q、K、V、O）

    参数：
        config (PretrainedConfig): Qwen2-MoE 配置对象
        tokens_sum (int): 批次中所有样本的 token 总数
        batch_seqlens (list[int]): 批次中每个样本的序列长度列表
        delta_time (float): 处理该批次的时间（秒）

    返回：
        float: 估算的 FLOPS（单位：TFLOPS）
    """  # 注释：函数 docstring 结束

    # 注释：从配置中提取模型架构参数
    hidden_size = config.hidden_size  # 注释：隐藏层维度
    vocab_size = config.vocab_size  # 注释：词表大小
    num_hidden_layers = config.num_hidden_layers  # 注释：Transformer 层数
    num_key_value_heads = config.num_key_value_heads  # 注释：KV 头数（GQA）
    num_attention_heads = config.num_attention_heads  # 注释：注意力头数
    moe_intermediate_size = config.moe_intermediate_size  # 注释：MoE 专家中间层维度
    moe_topk = config.num_experts_per_tok  # 注释：每个 token 激活的专家数（topk）
    num_experts = config.num_experts  # 注释：专家总数

    # 注释：计算每个注意力头的维度
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # 注释：head_dim = hidden_size / num_heads
    # 注释：计算 Q、K、V 的总参数量
    q_size = num_attention_heads * head_dim  # 注释：Q 的维度
    k_size = num_key_value_heads * head_dim  # 注释：K 的维度
    v_size = num_key_value_heads * head_dim  # 注释：V 的维度

    # non-attn per layer parm
    # gate + moe export
    # 注释：计算单层 MoE 参数量（路由器 + 激活的 topk 个专家）
    moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts  # 注释：3 倍来自 SwiGLU（gate、up、down）
    # 注释：计算单层 Attention Linear 参数量
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)  # 注释：Q、K、V、O 四个投影矩阵
    # 注释：计算 Embedding + LM Head 参数量
    emd_and_lm_head_N = vocab_size * hidden_size * 2  # 注释：2 倍是因为有 Embedding 和 LM Head

    # non-attn all_layer parm
    # 注释：计算所有层的总参数量
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N  # 注释：总参数 = 单层参数 * 层数 + Embedding/LM Head

    # non-attn all_layer & all_token fwd & bwd flops
    # 注释：计算非注意力部分的 FLOPS
    dense_N_flops = 6 * dense_N * tokens_sum  # 注释：6 倍来自前向（2 倍）+ 反向（4 倍）

    # attn all_layer & all_token fwd & bwd flops
    # 注释：计算注意力 QKV 乘法的 FLOPS
    seqlen_square_sum = 0  # 注释：累积所有样本的 seqlen^2
    for seqlen in batch_seqlens:  # 注释：遍历批次中每个样本的序列长度
        seqlen_square_sum += seqlen * seqlen  # 注释：注意力复杂度是 O(seqlen^2)
    # 注释：注意力 FLOPS = 12 * seqlen^2 * head_dim * num_heads * num_layers
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers  # 注释：12 倍来自前向（2 倍）+ 反向（10 倍）

    # all_layer & all_token fwd & bwd flops
    # 注释：总 FLOPS = 非注意力 FLOPS + 注意力 FLOPS
    flops_all_token = dense_N_flops + attn_qkv_flops  # 注释：总浮点运算次数
    # 注释：FLOPS/s = 总 FLOPS / 时间，除以 1e12 转换为 TFLOPS
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12  # 注释：单位：TFLOPS
    return flops_achieved  # 注释：返回估算的 FLOPS


def _estimate_gemma3_flops(config, tokens_sum, batch_seqlens, delta_time):  # 注释：函数定义，估算 Gemma3 模型的 FLOPS
    """（函数说明：估算 Gemma3 模型的 FLOPS）

    Gemma3 使用了以下关键技术：
    1. GeGLU 激活函数（gelu_pytorch_tanh）：与 SwiGLU 类似，有 3 个线性层
    2. 滑动窗口注意力（Sliding Window Attention）：每隔几层使用一次全注意力，其余层使用滑动窗口
       - 全注意力：每个 token 可以看到所有历史 token，复杂度 O(seqlen^2)
       - 滑动窗口：每个 token 只能看到窗口内的 token，复杂度 O(seqlen * window_size)

    FLOPS 计算需要考虑：
    - GeGLU MLP 参数量：hidden_size * intermediate_size * 3
    - 滑动窗口对注意力 FLOPS 的影响

    参数：
        config (PretrainedConfig): Gemma3 配置对象，必须包含：
            - layer_types (list[str] | None): 每层的注意力类型（"full_attention" 或 "sliding_attention"）
            - sliding_window (int): 滑动窗口大小（默认 1024）
            - sliding_window_pattern (int): 滑动窗口模式（默认 6，表示每 6 层使用一次全注意力）
        tokens_sum (int): 批次中所有样本的 token 总数
        batch_seqlens (list[int]): 批次中每个样本的序列长度列表
        delta_time (float): 处理该批次的时间（秒）

    返回：
        float: 估算的 FLOPS（单位：TFLOPS）
    """  # 注释：函数 docstring 结束

    # 注释：从配置中提取模型架构参数
    hidden_size = config.hidden_size  # 注释：隐藏层维度
    vocab_size = config.vocab_size  # 注释：词表大小
    num_hidden_layers = config.num_hidden_layers  # 注释：Transformer 层数
    num_key_value_heads = config.num_key_value_heads  # 注释：KV 头数（GQA）
    num_attention_heads = config.num_attention_heads  # 注释：注意力头数
    intermediate_size = config.intermediate_size  # 注释：MLP 中间层维度

    # 注释：计算每个注意力头的维度
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # 注释：head_dim = hidden_size / num_heads
    # 注释：计算 Q、K、V 的总参数量
    q_size = num_attention_heads * head_dim  # 注释：Q 的维度
    k_size = num_key_value_heads * head_dim  # 注释：K 的维度
    v_size = num_key_value_heads * head_dim  # 注释：V 的维度

    # non-attn per layer parm
    # Gemma3 uses GeGLU (gelu_pytorch_tanh), having 3 matrices in MLP (inherited from Gemma2MLP)
    # 注释：计算单层 MLP 参数量（GeGLU 有 3 个线性层）
    mlp_N = hidden_size * intermediate_size * 3  # 注释：GeGLU 与 SwiGLU 类似，都有 gate、up、down
    # 注释：计算单层 Attention Linear 参数量
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)  # 注释：Q、K、V、O 四个投影矩阵
    # 注释：计算 Embedding + LM Head 参数量
    emd_and_lm_head_N = vocab_size * hidden_size * 2  # 注释：2 倍是因为有 Embedding 和 LM Head

    # non-attn all_layer parm
    # 注释：计算所有层的总参数量
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N  # 注释：总参数 = 单层参数 * 层数 + Embedding/LM Head

    # non-attn all_layer & all_token fwd & bwd flops
    # 注释：计算非注意力部分的 FLOPS
    dense_N_flops = 6 * dense_N * tokens_sum  # 注释：6 倍来自前向（2 倍）+ 反向（4 倍）

    # attn all_layer & all_token fwd & bwd flops
    # Gemma3 alternates between full and sliding window attention based on layer_types
    # 注释：计算注意力 QKV 乘法的 FLOPS（需要考虑滑动窗口）
    seqlen_square_sum = 0  # 注释：累积所有样本的有效注意力范围

    # 注释：获取滑动窗口配置
    layer_types = getattr(config, "layer_types", None)  # 注释：每层的注意力类型（"full_attention" 或 "sliding_attention"）
    sliding_window = getattr(config, "sliding_window", 1024)  # default 1024  # 注释：滑动窗口大小（默认 1024）
    # default pattern: every 6th layer is full
    sliding_window_pattern = getattr(config, "sliding_window_pattern", 6)  # 注释：每 6 层使用一次全注意力（默认值）

    # If layer_types is not provided, generate it based on sliding_window_pattern
    # 注释：如果配置中没有 layer_types，根据 sliding_window_pattern 自动生成
    if layer_types is None and sliding_window is not None and sliding_window_pattern is not None:  # 注释：需要同时满足两个条件
        # 注释：生成 layer_types 列表（每 sliding_window_pattern 层使用一次全注意力）
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"  # 注释：(i+1) % pattern == 0 时为全注意力
            for i in range(num_hidden_layers)  # 注释：遍历所有层
        ]

    # 注释：根据 layer_types 计算每层的注意力 FLOPS
    if layer_types:  # 注释：如果有 layer_types 配置
        # Calculate attention flops per layer based on attention type
        # 注释：遍历每一层，根据注意力类型计算 FLOPS
        for layer_idx in range(num_hidden_layers):  # 注释：遍历所有层
            is_sliding = False  # 注释：默认为全注意力
            # 注释：检查当前层是否使用滑动窗口
            if layer_types and layer_idx < len(layer_types):  # 注释：确保索引不越界
                is_sliding = layer_types[layer_idx] == "sliding_attention"  # 注释：判断是否为滑动注意力

            # 注释：遍历批次中每个样本，计算注意力范围
            for seqlen in batch_seqlens:  # 注释：遍历批次中每个样本的序列长度
                if is_sliding and sliding_window:  # 注释：如果使用滑动窗口
                    # Sliding window limits each token to attend to at most window_size tokens
                    # 注释：滑动窗口限制每个 token 只能看到窗口内的 token
                    effective_seqlen = min(seqlen, sliding_window)  # 注释：有效序列长度 = min(实际长度, 窗口大小)
                    seqlen_square_sum += seqlen * effective_seqlen  # 注释：复杂度 O(seqlen * window_size)
                else:  # 注释：使用全注意力
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen  # 注释：复杂度 O(seqlen^2)
    else:  # 注释：如果没有 layer_types 配置，假设所有层都使用全注意力
        # If no layer_types config, assume all layers use full attention
        for seqlen in batch_seqlens:  # 注释：遍历批次中每个样本的序列长度
            seqlen_square_sum += seqlen * seqlen  # 注释：复杂度 O(seqlen^2)
        seqlen_square_sum *= num_hidden_layers  # 注释：乘以层数

    # 注释：注意力 FLOPS = 12 * seqlen_square_sum * head_dim * num_heads
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads  # 注释：12 倍来自前向（2 倍）+ 反向（10 倍）

    # all_layer & all_token fwd & bwd flops
    # 注释：总 FLOPS = 非注意力 FLOPS + 注意力 FLOPS
    flops_all_token = dense_N_flops + attn_qkv_flops  # 注释：总浮点运算次数
    # 注释：FLOPS/s = 总 FLOPS / 时间，除以 1e12 转换为 TFLOPS
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12  # 注释：单位：TFLOPS
    return flops_achieved  # 注释：返回估算的 FLOPS


def _estimate_apertus_flops(config, tokens_sum, batch_seqlens, delta_time):  # 注释：函数定义，估算 Apertus 模型的 FLOPS
    """（函数说明：估算 Apertus 模型的 FLOPS）

    Apertus 使用了以下关键技术：
    1. XIELU 激活函数：只有 2 个线性层（up_proj、down_proj），没有 gate_proj
    2. QK Norm：对 Q 和 K 进行归一化，增加了参数量（q_norm 和 k_norm）

    FLOPS 计算需要考虑：
    - XIELU MLP 参数量：hidden_size * intermediate_size * 2（只有 up 和 down）
    - QK Norm 参数量：hidden_size + num_kv_heads * head_dim（q_norm + k_norm）

    参数：
        config (PretrainedConfig): Apertus 配置对象
        tokens_sum (int): 批次中所有样本的 token 总数
        batch_seqlens (list[int]): 批次中每个样本的序列长度列表
        delta_time (float): 处理该批次的时间（秒）

    返回：
        float: 估算的 FLOPS（单位：TFLOPS）
    """  # 注释：函数 docstring 结束

    # 注释：从配置中提取模型架构参数
    hidden_size = config.hidden_size  # 注释：隐藏层维度
    vocab_size = config.vocab_size  # 注释：词表大小
    num_hidden_layers = config.num_hidden_layers  # 注释：Transformer 层数
    num_key_value_heads = config.num_key_value_heads  # 注释：KV 头数（GQA）
    num_attention_heads = config.num_attention_heads  # 注释：注意力头数
    intermediate_size = config.intermediate_size  # 注释：MLP 中间层维度

    # 注释：计算每个注意力头的维度
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)  # 注释：head_dim = hidden_size / num_heads
    # 注释：计算 Q、K、V 的总参数量
    q_size = num_attention_heads * head_dim  # 注释：Q 的维度
    k_size = num_key_value_heads * head_dim  # 注释：K 的维度
    v_size = num_key_value_heads * head_dim  # 注释：V 的维度

    # Apertus MLP with XIELU activation uses only 2 linear layers (up_proj, down_proj)
    # No gate_proj for XIELU, unlike SwiGLU which has 3 layers
    # 注释：计算单层 MLP 参数量（XIELU 只有 2 个线性层）
    mlp_N = hidden_size * intermediate_size * 2  # 注释：XIELU 只有 up_proj 和 down_proj，没有 gate_proj
    # 注释：计算单层 Attention Linear 参数量
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)  # 注释：Q、K、V、O 四个投影矩阵

    # ApertusConfig has qk_norm defaulting to True.
    # This adds params for q_norm (on H) and k_norm (on num_kv_heads * head_dim)
    # 注释：计算 QK Norm 参数量（q_norm + k_norm）
    qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim  # q_norm + k_norm  # 注释：q_norm 作用在 hidden_size 上，k_norm 作用在 k_size 上

    # 注释：计算 Embedding + LM Head 参数量
    emd_and_lm_head_N = vocab_size * hidden_size * 2  # 注释：2 倍是因为有 Embedding 和 LM Head

    # non-attn all_layer params
    # 注释：计算所有层的总参数量（包含 QK Norm）
    dense_N = (mlp_N + attn_linear_N + qk_norm_params_per_layer) * num_hidden_layers + emd_and_lm_head_N  # 注释：总参数 = 单层参数 * 层数 + Embedding/LM Head

    # non-attn all_layer & all_token fwd & bwd flops
    # 注释：计算非注意力部分的 FLOPS
    dense_N_flops = 6 * dense_N * tokens_sum  # 注释：6 倍来自前向（2 倍）+ 反向（4 倍）

    # attn all_layer & all_token fwd & bwd flops
    # 注释：计算注意力 QKV 乘法的 FLOPS
    seqlen_square_sum = 0  # 注释：累积所有样本的 seqlen^2
    for seqlen in batch_seqlens:  # 注释：遍历批次中每个样本的序列长度
        seqlen_square_sum += seqlen * seqlen  # 注释：注意力复杂度是 O(seqlen^2)
    # 注释：注意力 FLOPS = 12 * seqlen^2 * head_dim * num_heads * num_layers
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers  # 注释：12 倍来自前向（2 倍）+ 反向（10 倍）

    # all_layer & all_token fwd & bwd flops
    # 注释：总 FLOPS = 非注意力 FLOPS + 注意力 FLOPS
    flops_all_token = dense_N_flops + attn_qkv_flops  # 注释：总浮点运算次数
    # 注释：FLOPS/s = 总 FLOPS / 时间，除以 1e12 转换为 TFLOPS
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12  # 注释：单位：TFLOPS
    return flops_achieved  # 注释：返回估算的 FLOPS


def _estimate_unknown_flops(config, tokens_sum, batch_seqlens, delta_time):  # 注释：函数定义，未知模型架构的占位函数
    """（函数说明：未知模型架构的占位函数）

    当模型架构不在 ESTIMATE_FUNC 字典中时，调用此函数返回 0。
    避免 MFU 计算时出错。

    参数：
        config: 模型配置对象（未使用）
        tokens_sum: token 总数（未使用）
        batch_seqlens: 序列长度列表（未使用）
        delta_time: 时间（未使用）

    返回：
        int: 始终返回 0
    """  # 注释：函数 docstring 结束
    return 0  # 注释：返回 0，表示无法估算 FLOPS


# 注释：定义模型架构到估算函数的映射字典（根据 model_type 选择对应的 FLOPS 估算函数）
ESTIMATE_FUNC = {  # 注释：字典键为 config.model_type，值为对应的估算函数
    # 注释：Qwen2 系列（使用 _estimate_qwen2_flops）
    "qwen2": _estimate_qwen2_flops,  # 注释：Qwen2 Dense 模型
    "llama": _estimate_qwen2_flops,  # 注释：LLaMA 与 Qwen2 架构相同（SwiGLU MLP + 标准注意力）
    "qwen2_moe": _estimate_qwen2_moe_flops,  # 注释：Qwen2-MoE 模型
    "qwen2_vl": _estimate_qwen2_flops,  # 注释：Qwen2-VL（视觉语言模型，文本部分与 Qwen2 相同）
    "qwen2_5_vl": _estimate_qwen2_flops,  # 注释：Qwen2.5-VL
    # 注释：Qwen3 系列
    "qwen3": _estimate_qwen2_flops,  # 注释：Qwen3 Dense 模型（架构与 Qwen2 兼容）
    "qwen3_moe": _estimate_qwen2_moe_flops,  # 注释：Qwen3-MoE 模型
    "qwen3_vl": _estimate_qwen2_flops,  # 注释：Qwen3-VL
    "qwen3_vl_moe": _estimate_qwen2_moe_flops,  # 注释：Qwen3-VL-MoE
    # 注释：DeepSeek V3 系列（使用 _estimate_deepseek_v3_flops）
    "deepseek_v3": _estimate_deepseek_v3_flops,  # 注释：DeepSeek V3 MoE 模型（使用 MLA 注意力）
    # 注释：其他模型
    "minicpmv": _estimate_qwen2_flops,  # 注释：MiniCPM-V（视觉语言模型）
    "minicpmo": _estimate_qwen2_flops,  # 注释：MiniCPM-O（视觉语言模型）
    "mistral": _estimate_qwen2_flops,  # 注释：Mistral（架构与 LLaMA 相似）
    "gemma3_text": _estimate_gemma3_flops,  # 注释：Gemma3（使用 GeGLU 和滑动窗口注意力）
    "seed_oss": _estimate_qwen2_flops,  # 注释：SEED-OSS（开源多模态模型）
    "apertus": _estimate_apertus_flops,  # 注释：Apertus（使用 XIELU 激活函数和 QK Norm）
    "glm4v": _estimate_qwen2_flops,  # 注释：GLM4V（视觉语言模型）
}  # 注释：字典结束，后续可以按需添加更多模型架构


class FlopsCounter:  # 注释：类定义，FLOPS 统计器
    """（类说明：训练过程中的 FLOPS 统计器）

    本类用于统计训练过程中的浮点运算吞吐量（FLOPS），并计算 MFU（Model FLOPs Utilization）。
    通过 FlopsCounter，可以评估硬件利用率，优化训练性能。

    工作原理：
    1. 初始化时，根据模型配置（config）选择对应的 FLOPS 估算函数
    2. 每个训练步骤结束后，调用 estimate_flops() 方法
    3. 根据批次 token 数和处理时间，计算实际 FLOPS
    4. 查询当前设备的理论峰值 FLOPS
    5. 计算 MFU = 实际 FLOPS / 理论峰值 FLOPS

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    参数：
        config (PretrainedConfig): 模型配置对象，必须包含 model_type 字段

    副作用：
        - 如果 model_type 不在 ESTIMATE_FUNC 中，打印警告信息

    异常/边界条件：
        - 如果 config.model_type 不在 ESTIMATE_FUNC 中，MFU 会始终为 0

    最小示例：
        >>> from transformers import AutoConfig
        >>> config = AutoConfig.from_pretrained("Qwen/Qwen2-7B-Instruct")
        >>> flops_counter = FlopsCounter(config)
        >>> batch_seqlens = [128, 256, 512]
        >>> delta_time = 0.5
        >>> estimated_flops, promised_flops = flops_counter.estimate_flops(batch_seqlens, delta_time)
        >>> mfu = estimated_flops / promised_flops
        >>> print(f"MFU: {mfu:.2%}")

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/flops_counter.py
        - 类：FlopsCounter

    典型调用路径：
        verl/trainer/fsdp_sft_trainer.py (SFT 训练入口)
        -> 初始化 FlopsCounter(config)
        -> 训练循环中调用 flops_counter.estimate_flops(batch_seqlens, delta_time)
        -> 计算 MFU 并记录到日志

    被谁调用：
        - verl/trainer/fsdp_sft_trainer.py（SFT 训练）
        - verl/trainer/ppo/ray_trainer.py（PPO/GRPO 训练）
        - verl/workers/fsdp_workers.py（FSDP worker）

    调用了谁（项目内）：
        - get_device_flops()（获取设备理论峰值 FLOPS）
        - ESTIMATE_FUNC 中的估算函数（如 _estimate_qwen2_flops）

    调用了谁（外部依赖）：
        - transformers.PretrainedConfig（模型配置）

    记忆提示：
        - 类名：FlopsCounter（FLOPS 统计器）
        - 核心方法：estimate_flops()
        - MFU = 实际 FLOPS / 理论峰值 FLOPS
    """  # 注释：类 docstring 结束

    def __init__(self, config: PretrainedConfig):  # 注释：构造函数，初始化 FlopsCounter
        """（构造函数说明：初始化 FlopsCounter 对象）

        参数：
            config (PretrainedConfig): 模型配置对象，必须包含 model_type 字段

        副作用：
            - 如果 model_type 不在 ESTIMATE_FUNC 中，打印警告信息
            - 设置 self.config 为 text_config（如果存在）或原始 config
        """  # 注释：构造函数 docstring 结束

        # 注释：定义支持的模型架构列表（从 ESTIMATE_FUNC 字典中提取键）
        VALID_CONFIG_TYPE = ESTIMATE_FUNC.keys()  # 注释：所有支持的 model_type
        # 注释：检查当前模型架构是否在支持列表中
        if config.model_type not in VALID_CONFIG_TYPE:  # 注释：如果 model_type 不在字典中
            # 注释：打印警告信息，提示 MFU 会始终为 0
            print(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will always be "
                f"zero."  # 注释：警告信息包含支持的模型架构列表和当前模型架构
            )

        # 注释：提取文本配置（对于 VL 模型，config 包含 text_config 和 vision_config，只需要 text_config）
        self.config = getattr(config, "text_config", config)  # 注释：如果有 text_config 属性，使用它；否则使用原始 config

    # TODO: actually we can make this a static method
    def estimate_flops(self, batch_seqlens, delta_time):  # 注释：方法定义，估算 FLOPS
        """（方法说明：估算当前批次的 FLOPS 和 MFU）

        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
                例如：[128, 256, 512] 表示批次中有 3 个样本，长度分别为 128、256、512
            delta_time (float): The time taken to process the batch, in seconds.
                例如：0.5 表示处理该批次耗时 0.5 秒

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
                单位：TFLOPS（10^12 FLOPS/s）
            promised_flops (float): The expected FLOPS of the current device.
                单位：TFLOPS（10^12 FLOPS/s）

        最小示例：
            >>> flops_counter = FlopsCounter(config)
            >>> batch_seqlens = [128, 256, 512]
            >>> delta_time = 0.5
            >>> estimated_flops, promised_flops = flops_counter.estimate_flops(batch_seqlens, delta_time)
            >>> mfu = estimated_flops / promised_flops
            >>> print(f"实际 FLOPS: {estimated_flops:.2f} TFLOPS")
            >>> print(f"理论峰值: {promised_flops:.2f} TFLOPS")
            >>> print(f"MFU: {mfu:.2%}")
        """  # 注释：方法 docstring 结束

        # 注释：计算批次中所有样本的 token 总数
        tokens_sum = sum(batch_seqlens)  # 注释：例如 [128, 256, 512] -> 896

        # 注释：根据 model_type 选择对应的 FLOPS 估算函数
        func = ESTIMATE_FUNC.get(self.config.model_type, _estimate_unknown_flops)  # 注释：如果 model_type 不在字典中，使用 _estimate_unknown_flops（返回 0）

        # 注释：调用估算函数，计算实际 FLOPS
        estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time)  # 注释：估算函数根据模型架构计算 FLOPS

        # 注释：获取当前设备的理论峰值 FLOPS
        promised_flops = get_device_flops()  # 注释：默认单位为 TFLOPS

        # 注释：返回实际 FLOPS 和理论峰值 FLOPS
        return estimated_flops, promised_flops  # 注释：调用者可以计算 MFU = estimated_flops / promised_flops
