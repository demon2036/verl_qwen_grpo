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
模块用途：reward_loop.reward_manager 子包入口，集中导出各 RewardManager 实现与注册表。（注释：模块功能概述）
输入：无（仅导入导出符号）。（注释：输入形态说明）
输出：RewardManager 类与注册/查询函数。（注释：输出形态说明）
关键依赖：registry.py、naive.py、dapo.py、limited.py。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls
调用路径概览：（注释：全局调用关系）
  - reward_loop.py::_init_reward_fn -> get_reward_manager_cls -> 选择具体 RewardManager
"""

from .registry import get_reward_manager_cls, register  # noqa: I001  # 注释：导出注册表 API
from .dapo import DAPORewardManager  # 注释：导出 DAPO RewardManager
from .naive import NaiveRewardManager  # 注释：导出 Naive RewardManager
from .limited import RateLimitedRewardManager  # 注释：导出限速 RewardManager

__all__ = [  # 注释：显式导出列表
    "DAPORewardManager",
    "NaiveRewardManager",
    "RateLimitedRewardManager",
    "register",
    "get_reward_manager_cls",
]
