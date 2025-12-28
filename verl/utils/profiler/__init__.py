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
模块用途：统一导出性能分析/标记工具，按环境选择 NVTX 或 MSTX 实现。（注释：模块职责）
输入/输出：
  - 输入：是否支持 NVTX/NPU 的环境探测结果。（注释：输入来源）
  - 输出：统一的标记函数与 Profiler 类集合。（注释：输出内容）
关键依赖：`verl.utils.import_utils`、`verl.utils.device`、`verl.utils.profiler.*` 子模块。（注释：依赖说明）
典型用法（最小示例）：
  - `from verl.utils.profiler import mark_start_range, mark_end_range`。（注释：最小使用方式）
调用路径概览：
  - 训练/推理入口 -> `verl.utils.profiler` -> 具体后端实现（NVTX/MSTX/纯 Python）。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 项目内依赖导入 =====
from ..device import is_npu_available  # 注释：检测是否为 NPU 环境
from ..import_utils import is_nvtx_available  # 注释：检测 NVTX 是否可用
from .performance import GPUMemoryLogger, log_gpu_memory_usage, simple_timer  # 注释：通用性能工具
from .profile import DistProfiler, DistProfilerExtension, ProfilerConfig  # 注释：分布式 profiler 类

# Select marker implementations by availability, but keep DistProfiler as our dispatcher
if is_nvtx_available():  # 注释：优先使用 NVTX 标记
    from .nvtx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer  # 注释：NVTX 实现
elif is_npu_available:  # 注释：NPU 环境使用 MSTX
    from .mstx_profile import mark_annotate, mark_end_range, mark_start_range, marked_timer  # 注释：MSTX 实现
else:  # 注释：无 NVTX/NPU 时使用纯 Python 实现
    from .performance import marked_timer  # 注释：回退的计时工具
    from .profile import mark_annotate, mark_end_range, mark_start_range  # 注释：回退标记函数

__all__ = [  # 注释：显式导出 API 列表
    "GPUMemoryLogger",
    "log_gpu_memory_usage",
    "mark_start_range",
    "mark_end_range",
    "mark_annotate",
    "DistProfiler",
    "DistProfilerExtension",
    "ProfilerConfig",
    "simple_timer",
    "marked_timer",
]
