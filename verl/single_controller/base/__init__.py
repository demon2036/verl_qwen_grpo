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
  - 暴露 single_controller.base 层的基础 Worker/WorkerGroup/ResourcePool 抽象。
  - 供 Ray/其他运行时后端继承复用，形成统一的 Worker 调度接口。

输入：
  - 无直接输入（模块仅用于符号导出）。

输出：
  - 无直接输出（导出 __all__ 中的类）。

关键依赖：
  - `verl/single_controller/base/worker.py`
  - `verl/single_controller/base/worker_group.py`

典型用法（最小示例）：
  - `from verl.single_controller.base import WorkerGroup`
  - `from verl.single_controller.base import Worker`

调用路径概览：
  - `verl/trainer/ppo/ray_trainer.py`
    -> `verl/single_controller/base/__init__.py`
    -> `WorkerGroup` / `Worker`。
"""

from .worker import Worker  # 基础 Worker 抽象
from .worker_group import ClassWithInitArgs, ResourcePool, WorkerGroup  # WorkerGroup 与资源池

__all__ = ["Worker", "WorkerGroup", "ClassWithInitArgs", "ResourcePool"]  # 对外导出的公共符号
