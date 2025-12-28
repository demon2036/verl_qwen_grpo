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
模块用途：
  - 暴露 Ray 单控制器（single_controller.ray）层的关键类与工厂函数。
  - 统一对外导出 RayWorkerGroup/资源池等对象，供训练入口与 worker 构建调用。

输入：
  - 无直接输入（仅用于导入与符号导出）。

输出：
  - 无直接输出（导出 __all__ 中的类/函数）。

关键依赖：
  - `verl/single_controller/ray/base.py` 中的 Ray 资源池与 worker group 定义。

典型用法（最小示例）：
  - `from verl.single_controller.ray import RayWorkerGroup`  # 获取 Ray 版 WorkerGroup。

调用路径概览：
  - `verl/trainer/ppo/ray_trainer.py`
    -> `verl/single_controller/ray/__init__.py`
    -> `RayWorkerGroup` / `RayResourcePool` 等。
"""

from .base import (  # 导入 Ray 版本的 WorkerGroup 与辅助工厂
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    create_colocated_worker_cls,
    create_colocated_worker_cls_fused,
)

__all__ = [  # 对外导出符号列表，限制 `from ... import *` 的暴露范围
    "RayClassWithInitArgs",
    "RayResourcePool",
    "RayWorkerGroup",
    "create_colocated_worker_cls",
    "create_colocated_worker_cls_fused",
]
