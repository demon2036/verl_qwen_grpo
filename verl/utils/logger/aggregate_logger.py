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
模块用途：提供本地/装饰器日志工具与按 rank 输出的辅助函数。（注释：模块职责）
输入/输出：
  - 输入：日志字典、step、rank、logger 实例。（注释：输入说明）
  - 输出：控制台或 logging 输出。（注释：输出说明）
关键依赖：logging、torch.distributed（可选）。（注释：依赖说明）
典型用法（最小示例）：
  - `log_with_rank("msg", rank=0, logger=logger)`。（注释：最常见用法）
调用路径概览：
  - 训练/worker -> `verl.utils.logger` -> 本模块函数。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import datetime  # 注释：时间戳
import logging  # 注释：日志
import numbers  # 注释：数值类型判断
import pprint  # 注释：格式化输出

# ===== 第三方依赖导入 =====
import torch  # 注释：分布式 rank 检测


def concat_dict_to_str(dict: dict, step):
    """
    将日志字典拼接为字符串。（注释：函数用途）

    参数：
      - dict (dict)：日志字典。（注释：输入说明）
      - step：当前 step。（注释：输入说明）
    返回：
      - str：拼接后的字符串。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/logger/aggregate_logger.py::concat_dict_to_str`。（注释：定位）
      - 被谁调用：`LocalLogger.log`。（注释：调用方）
      - 调用了谁（外部依赖）：`pprint.pformat`。（注释：外部依赖）
    """
    output = [f"step:{step}"]  # 注释：起始包含 step
    for k, v in dict.items():  # 注释：遍历日志项
        if isinstance(v, numbers.Number):  # 注释：仅记录数值
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)  # 注释：拼接为字符串
    return output_str  # 注释：返回结果


class LocalLogger:
    """
    本地控制台 logger。（注释：类用途）

    参数：
      - print_to_console (bool)：是否输出到控制台。（注释：参数说明）
    调用路径依赖：
      - 所在位置：`verl/utils/logger/aggregate_logger.py::LocalLogger`。（注释：定位）
      - 被谁调用：`verl/utils/tracking.py`、`verl/utils/logger/__init__.py`。（注释：调用方）
      - 调用了谁（项目内）：`concat_dict_to_str`。（注释：内部依赖）
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console  # 注释：是否打印到控制台

    def flush(self):
        """占位 flush 接口。（注释：保持与 logger 接口一致）"""
        pass  # 注释：无实际操作

    def log(self, data, step):
        """记录日志到控制台。（注释：方法用途）"""
        if self.print_to_console:  # 注释：仅在开启时打印
            print(concat_dict_to_str(data, step=step), flush=True)  # 注释：拼接并输出


class DecoratorLoggerBase:
    """
    装饰器日志基类。（注释：类用途）

    参数：
      - role (str)：日志角色/前缀。（注释：参数说明）
      - logger (logging.Logger)：外部 logger 实例。（注释：参数说明）
      - level (int)：日志等级。（注释：参数说明）
      - rank (int)：当前 rank。（注释：参数说明）
      - log_only_rank_0 (bool)：是否仅 rank0 输出。（注释：参数说明）
    调用路径依赖：
      - 所在位置：`verl/utils/logger/aggregate_logger.py::DecoratorLoggerBase`。（注释：定位）
      - 被谁调用：`verl/utils/profiler/performance.py` 等。（注释：调用方）
      - 调用了谁（外部依赖）：logging。（注释：外部依赖）
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role  # 注释：日志前缀
        self.logger = logger  # 注释：外部 logger
        self.level = level  # 注释：日志等级
        self.rank = rank  # 注释：rank
        self.log_only_rank_0 = log_only_rank_0  # 注释：仅 rank0 输出
        self.logging_function = self.log_by_logging  # 注释：默认使用 logging
        if logger is None:  # 注释：无 logger 时回退到 print
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        """使用 print 输出日志。（注释：方法用途）"""
        if not self.log_only_rank_0 or self.rank == 0:  # 注释：按 rank 过滤
            print(f"{self.role} {log_str}", flush=True)  # 注释：输出

    def log_by_logging(self, log_str):
        """使用 logging.Logger 输出日志。（注释：方法用途）"""
        if self.logger is None:  # 注释：logger 未初始化
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:  # 注释：按 rank 过滤
            self.logger.log(self.level, f"{self.role} {log_str}")  # 注释：输出日志


def print_rank_0(message):
    """
    仅在 rank0 打印消息（若分布式已初始化）。（注释：函数用途）

    调用路径依赖：
      - 所在位置：`verl/utils/logger/aggregate_logger.py::print_rank_0`。（注释：定位）
      - 被谁调用：`verl/utils/logger/__init__.py` 对外导出。（注释：调用方）
      - 调用了谁（外部依赖）：`torch.distributed.get_rank`。（注释：外部依赖）
    """
    if torch.distributed.is_initialized():  # 注释：分布式已初始化
        if torch.distributed.get_rank() == 0:  # 注释：仅 rank0 输出
            print(message, flush=True)
    else:
        print(message, flush=True)  # 注释：非分布式直接输出


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """
    打印带 rank 前缀的消息。（注释：函数用途）

    参数：
      - message (str)：日志内容。（注释：输入说明）
      - rank (int)：当前 rank。（注释：输入说明）
      - log_only_rank_0 (bool)：是否仅 rank0 输出。（注释：输入说明）
    返回：无。（注释：仅输出）
    调用路径依赖：
      - 所在位置：`verl/utils/logger/aggregate_logger.py::print_with_rank`。（注释：定位）
      - 被谁调用：`verl/utils/logger/__init__.py` 对外导出。（注释：调用方）
      - 调用了谁（外部依赖）：`print`。（注释：外部依赖）
    """
    if not log_only_rank_0 or rank == 0:  # 注释：按 rank 过滤
        print(f"[Rank {rank}] {message}", flush=True)  # 注释：输出消息


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """
    打印带时间戳与 rank 前缀的消息。（注释：函数用途）

    参数同 `print_with_rank`。（注释：输入说明）
    返回：无。（注释：仅输出）
    """
    now = datetime.datetime.now()  # 注释：获取当前时间
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"  # 注释：拼接前缀
    if not log_only_rank_0 or rank == 0:  # 注释：按 rank 过滤
        print(message, flush=True)  # 注释：输出消息


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """
    使用 logging.Logger 输出带 rank 前缀的日志。（注释：函数用途）

    参数：
      - message (str)：日志内容。（注释：输入说明）
      - rank (int)：当前 rank。（注释：输入说明）
      - logger (logging.Logger)：logger 实例。（注释：输入说明）
      - level (int)：日志等级。（注释：输入说明）
      - log_only_rank_0 (bool)：是否仅 rank0 输出。（注释：输入说明）
    返回：无。（注释：仅输出）
    调用路径依赖：
      - 所在位置：`verl/utils/logger/aggregate_logger.py::log_with_rank`。（注释：定位）
      - 被谁调用：`verl/utils/checkpoint/*`、`verl/trainer/fsdp_sft_trainer.py` 等。（注释：调用方）
      - 调用了谁（外部依赖）：`logger.log`。（注释：外部依赖）
    """
    if not log_only_rank_0 or rank == 0:  # 注释：按 rank 过滤
        logger.log(level, f"[Rank {rank}] {message}")  # 注释：输出日志
