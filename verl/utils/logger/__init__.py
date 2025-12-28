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
模块用途：统一导出日志辅助工具（按 rank 打印、装饰器 logger 等）。（注释：模块职责）
输入/输出：
  - 输入：日志消息、rank、logger 实例。（注释：输入说明）
  - 输出：控制台或日志系统输出。（注释：输出说明）
关键依赖：`verl.utils.logger.aggregate_logger`。（注释：依赖说明）
典型用法（最小示例）：
  - `from verl.utils.logger import log_with_rank`。（注释：最常见用法）
调用路径概览：
  - 训练/worker -> `verl.utils.logger` -> `aggregate_logger`。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 项目内依赖导入 =====
from .aggregate_logger import (  # 注释：导出聚合日志工具
    DecoratorLoggerBase,
    LocalLogger,
    log_with_rank,
    print_rank_0,
    print_with_rank,
    print_with_rank_and_timer,
)

__all__ = [  # 注释：显式导出列表
    "LocalLogger",
    "DecoratorLoggerBase",
    "print_rank_0",
    "print_with_rank",
    "print_with_rank_and_timer",
    "log_with_rank",
]
