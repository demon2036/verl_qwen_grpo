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
模块用途：reward_loop 子包入口，导出 RewardLoopManager/Worker 与 RewardModelManager。（注释：模块功能概述）
输入：无（仅导入导出符号）。（注释：输入形态说明）
输出：对外暴露的类对象。（注释：输出形态说明）
关键依赖：reward_loop.py、reward_model.py。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - from verl.experimental.reward_loop import RewardLoopManager
调用路径概览：（注释：全局调用关系）
  - verl/experimental/reward_loop/reward_loop.py 在上层被 ray_trainer 或 reward.py 调用
"""

from .reward_loop import RewardLoopManager, RewardLoopWorker  # 注释：导出 RewardLoop 管理器与 Worker
from .reward_model import RewardModelManager  # 注释：导出 RewardModel 管理器

__all__ = ["RewardModelManager", "RewardLoopWorker", "RewardLoopManager"]  # 注释：显式导出列表
