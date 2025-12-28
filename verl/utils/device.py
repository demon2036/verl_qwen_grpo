# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE
"""
模块用途：统一获取设备类型（CPU/CUDA/NPU）、设备 ID、通信后端与内存配置。（注释：模块职责）
输入/输出：
  - 输入：torch 运行时、环境变量与配置对象。（注释：输入来源）
  - 输出：设备名称、设备 ID、NCCL/HCCL 后端字符串等。（注释：输出内容）
关键依赖：torch、logging。（注释：依赖说明）
典型用法（最小示例）：
  - `device = get_device_name()`；`device_id = get_device_id()`。（注释：最常见调用）
调用路径概览：
  - 训练/worker 初始化 -> `verl.utils.device` -> 设备/后端选择。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import logging  # 注释：日志记录

# ===== 第三方依赖导入 =====
import torch  # 注释：设备与张量库

# ===== 日志器 =====
logger = logging.getLogger(__name__)  # 注释：模块级 logger


def is_torch_npu_available() -> bool:
    """
    检查 torch 是否支持 NPU 并可用。（注释：函数用途）

    参数：无。（注释：无输入）
    返回：
      - bool：True 表示 NPU 可用。（注释：返回说明）
    副作用：无。（注释：纯查询）
    异常/边界条件：torch 未安装 NPU 子模块时返回 False。（注释：边界条件）
    最小示例：
      - 输入：无。（注释：示例输入）
      - 输出：True/False。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::is_torch_npu_available`。（注释：定位）
      - 典型调用路径：`get_device_name` -> `is_torch_npu_available`。（注释：典型路径）
      - 被谁调用：`get_device_name`、`auto_set_ascend_device_name`。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.npu.is_available`。（注释：外部依赖）
    """
    try:  # 注释：捕获 torch.npu 不存在的情况
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)):  # 注释：检查接口可用性
            return torch.npu.is_available()  # 注释：返回 NPU 可用性
        return False  # 注释：无 npu 子模块时不可用
    except ImportError:
        return False  # 注释：torch.npu 导入失败视为不可用


is_cuda_available = torch.cuda.is_available()  # 注释：CUDA 可用性全局缓存
is_npu_available = is_torch_npu_available()  # 注释：NPU 可用性全局缓存


def get_visible_devices_keyword() -> str:
    """
    获取可见设备的环境变量关键字。（注释：函数用途）

    返回：`CUDA_VISIBLE_DEVICES` 或 `ASCEND_RT_VISIBLE_DEVICES`。（注释：返回说明）
    副作用：无。（注释：纯函数）
    最小示例：
      - 输入：CUDA 可用。（注释：示例输入）
      - 输出：`CUDA_VISIBLE_DEVICES`。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::get_visible_devices_keyword`。（注释：定位）
      - 典型调用路径：设备配置 -> `get_visible_devices_keyword`。（注释：典型路径）
      - 被谁调用：未在仓库内直接检索到显式调用（可由脚本使用）。（注释：调用方）
      - 调用了谁（项目内）：`is_cuda_available`。（注释：内部依赖）
      - 调用了谁（外部依赖）：无。（注释：外部依赖）
    """
    return "CUDA_VISIBLE_DEVICES" if is_cuda_available else "ASCEND_RT_VISIBLE_DEVICES"  # 注释：CUDA/NPU 对应环境变量


def get_device_name() -> str:
    """
    获取当前设备名称（cpu/cuda/npu）。（注释：函数用途）

    返回：设备字符串。（注释：返回说明）
    副作用：无。（注释：纯函数）
    最小示例：
      - 输入：CUDA 可用。（注释：示例输入）
      - 输出："cuda"。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::get_device_name`。（注释：定位）
      - 典型调用路径：训练初始化 -> `get_device_name`。（注释：典型路径）
      - 被谁调用：`verl/utils/fsdp_utils.py`、`verl/trainer/fsdp_sft_trainer.py` 等广泛调用。（注释：调用方）
      - 调用了谁（项目内）：`is_cuda_available`、`is_npu_available`。（注释：内部依赖）
      - 调用了谁（外部依赖）：无。（注释：外部依赖）
    """
    if is_cuda_available:  # 注释：优先 CUDA
        device = "cuda"
    elif is_npu_available:  # 注释：次选 NPU
        device = "npu"
    else:  # 注释：最后回退 CPU
        device = "cpu"
    return device  # 注释：返回设备名称


def get_torch_device() -> any:
    """
    根据设备名称返回 torch 设备命名空间。（注释：函数用途）

    返回：
      - module：如 torch.cuda / torch.npu / torch.cpu。（注释：返回说明）
    副作用：可能记录 warning。（注释：副作用）
    异常/边界条件：若找不到设备命名空间，回退到 torch.cuda。（注释：边界条件）
    最小示例：
      - 输入：CUDA 可用。（注释：示例输入）
      - 输出：torch.cuda。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::get_torch_device`。（注释：定位）
      - 典型调用路径：`get_device_id` -> `get_torch_device`。（注释：典型路径）
      - 被谁调用：`verl/utils/fsdp_utils.py`、`verl/utils/distributed.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`get_device_name`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`getattr(torch, ...)`。（注释：外部依赖）
    """
    device_name = get_device_name()  # 注释：获取设备名称字符串
    try:
        return getattr(torch, device_name)  # 注释：获取对应 torch 命名空间
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")  # 注释：记录警告
        return torch.cuda  # 注释：回退到 torch.cuda


def get_device_id() -> int:
    """
    获取当前设备 ID。（注释：函数用途）

    返回：设备索引。（注释：返回说明）
    副作用：无。（注释：纯查询）
    最小示例：
      - 输出：0。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::get_device_id`。（注释：定位）
      - 典型调用路径：`verl/utils/fsdp_utils.py` -> `get_device_id`。（注释：典型路径）
      - 被谁调用：`verl/utils/fsdp_utils.py`、`verl/utils/ulysses.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.cuda.current_device` / `torch.npu.current_device`。（注释：外部依赖）
    """
    return get_torch_device().current_device()  # 注释：通过 torch 设备命名空间获取当前设备 ID


def get_nccl_backend() -> str:
    """
    获取分布式通信后端名称（nccl/hccl）。（注释：函数用途）

    返回：后端字符串。（注释：返回说明）
    副作用：无。（注释：纯函数）
    最小示例：
      - 输入：NPU 可用。（注释：示例输入）
      - 输出："hccl"。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::get_nccl_backend`。（注释：定位）
      - 典型调用路径：`initialize_global_process_group` -> `get_nccl_backend`。（注释：典型路径）
      - 被谁调用：`verl/utils/distributed.py`、`verl/trainer/fsdp_sft_trainer.py`。（注释：调用方）
      - 调用了谁（项目内）：`is_npu_available`。（注释：内部依赖）
      - 调用了谁（外部依赖）：无。（注释：外部依赖）
    """
    if is_npu_available:  # 注释：NPU 使用 HCCL
        return "hccl"
    else:
        # default to nccl
        return "nccl"  # 注释：默认使用 NCCL


def set_expandable_segments(enable: bool) -> None:
    """
    开关 CUDA 的 expandable segments 分配策略。（注释：函数用途）

    参数：
      - enable (bool)：是否启用该特性。（注释：输入说明）
    返回：无。（注释：仅副作用）
    副作用：修改 CUDA allocator 设置。（注释：副作用说明）
    异常/边界条件：仅在 CUDA 可用时生效。（注释：边界条件）
    最小示例：
      - 输入：enable=True。（注释：示例输入）
      - 输出：CUDA allocator 切换策略。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::set_expandable_segments`。（注释：定位）
      - 典型调用路径：训练初始化 -> `set_expandable_segments`。（注释：典型路径）
      - 被谁调用：未在仓库内直接检索到显式调用（可供外部脚本使用）。（注释：调用方）
      - 调用了谁（项目内）：`is_cuda_available`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.cuda.memory._set_allocator_settings`。（注释：外部依赖）
    """
    if is_cuda_available:  # 注释：仅在 CUDA 下生效
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")  # 注释：设置 allocator 参数


def auto_set_ascend_device_name(config):
    """
    当检测到 Ascend NPU 时，自动修正配置中的设备名称。（注释：函数用途）

    参数：
      - config：配置对象（通常为 Hydra/OmegaConf）。（注释：输入说明）
    返回：无。（注释：仅修改 config）
    副作用：可能修改 `config.trainer.device`。（注释：副作用说明）
    异常/边界条件：config/字段为空时不处理。（注释：边界条件）
    最小示例：
      - 输入：config.trainer.device="cuda" 且 NPU 可用。（注释：示例输入）
      - 输出：device 被改为 "npu"。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::auto_set_ascend_device_name`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` 初始化配置 -> `auto_set_ascend_device_name`。（注释：典型路径）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`。（注释：调用方）
      - 调用了谁（项目内）：`is_torch_npu_available`。（注释：内部依赖）
      - 调用了谁（外部依赖）：logging。（注释：外部依赖）
    """
    if config and config.trainer and config.trainer.device:  # 注释：确保配置与字段存在
        if is_torch_npu_available():  # 注释：检测 NPU 是否可用
            if config.trainer.device != "npu":  # 注释：若未设置为 npu，则提示并修正
                logger.warning(
                    f"Detect setting config.trainer.device to {config.trainer.device} for Ascend NPU, maybe"
                    f"from default value in config file, automatically set to `npu` instead."
                )

            config.trainer.device = "npu"  # 注释：强制设置为 npu


def get_device_capability(device_id: int = 0) -> tuple[int, int]:
    """
    获取 CUDA 设备计算能力（major, minor）。（注释：函数用途）

    参数：
      - device_id (int)：设备索引，默认 0。（注释：输入说明）
    返回：
      - (major, minor)：CUDA 计算能力；非 CUDA 环境返回 (None, None)。（注释：返回说明）
    副作用：无。（注释：纯查询）
    异常/边界条件：仅 CUDA 可用时查询。（注释：边界条件）
    最小示例：
      - 输入：device_id=0。（注释：示例输入）
      - 输出：(8, 0) 等。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/device.py::get_device_capability`。（注释：定位）
      - 典型调用路径：设备特性检测 -> `get_device_capability`。（注释：典型路径）
      - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
      - 调用了谁（项目内）：`is_cuda_available`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.cuda.get_device_capability`。（注释：外部依赖）
    """
    major, minor = None, None  # 注释：默认返回值
    if is_cuda_available:  # 注释：仅 CUDA 可用时查询
        major, minor = torch.cuda.get_device_capability(device_id)  # 注释：获取计算能力

    return major, minor  # 注释：返回结果
