# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
模块用途：RewardLoop 专用 RewardManager 注册表，实现名称到类的映射。（注释：模块功能概述）
输入：RewardManagerBase 子类与名称字符串。（注释：输入形态说明）
输出：通过名称查询到的 RewardManager 类。（注释：输出形态说明）
关键依赖：RewardManagerBase。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - @register("naive")
    class NaiveRewardManager(RewardManagerBase): ...
  - cls = get_reward_manager_cls("naive")
调用路径概览：（注释：全局调用关系）
  - reward_loop.py::_init_reward_fn -> get_reward_manager_cls
"""

from typing import Callable  # 注释：装饰器类型提示

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase  # 注释：reward_loop 基类

__all__ = ["register", "get_reward_manager_cls"]  # 注释：对外导出 API

REWARD_LOOP_MANAGER_REGISTRY: dict[str, type[RewardManagerBase]] = {}  # 注释：全局注册表


def register(name: str) -> Callable[[type[RewardManagerBase]], type[RewardManagerBase]]:  # 注释：注册装饰器
    """
    功能：将 RewardManager 类注册到全局注册表。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - name (str): 注册名称（唯一键）。（注释：注册键）
    返回：（注释：返回值说明）
      - decorator：接收类并完成注册的装饰器。（注释：返回装饰器）
    副作用：（注释：副作用说明）
      - 修改全局字典 REWARD_LOOP_MANAGER_REGISTRY。（注释：全局状态变更）
    异常/边界条件：（注释：异常与边界）
      - 若同名已注册且类不同，抛出 ValueError。（注释：重复注册保护）
    最小示例：（注释：最小可理解示例）
      - 输入：@register("naive") class NaiveRewardManager(...)
      - 输出：REWARD_LOOP_MANAGER_REGISTRY["naive"] = NaiveRewardManager
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/reward_manager/registry.py::register`
      - 典型调用路径：`reward_manager/naive.py` -> `@register("naive")`
      - 被谁调用：`reward_manager/naive.py`、`reward_manager/dapo.py`、`reward_manager/limited.py`
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：无
    """

    def decorator(cls: type[RewardManagerBase]) -> type[RewardManagerBase]:  # 注释：真正执行注册的内部函数
        """
        功能：检查冲突并写入注册表。（注释：内部函数说明）
        参数：（注释：函数参数说明）
          - cls (type[RewardManagerBase]): 待注册类。（注释：类对象）
        返回：（注释：返回值说明）
          - cls：原类对象。（注释：保持装饰器语义）
        副作用：（注释：副作用说明）
          - 写入全局注册表。（注释：全局状态变更）
        异常/边界条件：（注释：异常与边界）
          - 同名不同类时抛 ValueError。（注释：重复注册保护）
        最小示例：（注释：最小可理解示例）
          - 输入：decorator(NaiveRewardManager)
          - 输出：NaiveRewardManager
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/registry.py::register.decorator`
          - 典型调用路径：`register` -> `decorator`
          - 被谁调用：`register`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：无
        """
        if name in REWARD_LOOP_MANAGER_REGISTRY and REWARD_LOOP_MANAGER_REGISTRY[name] != cls:  # 注释：避免同名冲突
            raise ValueError(  # 注释：同名不同类时拒绝注册
                f"reward manager {name} has already been registered: {REWARD_LOOP_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_LOOP_MANAGER_REGISTRY[name] = cls  # 注释：写入注册表
        return cls  # 注释：返回类本身

    return decorator  # 注释：返回装饰器


def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:  # 注释：按名称查询 RewardManager 类
    """
    功能：根据名称从注册表获取 RewardManager 类。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - name (str): RewardManager 注册名称。（注释：查询键）
    返回：（注释：返回值说明）
      - type[RewardManagerBase]：对应类。（注释：返回类型）
    副作用：（注释：副作用说明）
      - 无（只读查询）。（注释：无副作用）
    异常/边界条件：（注释：异常与边界）
      - name 不存在时抛 ValueError。（注释：未知名称）
    最小示例：（注释：最小可理解示例）
      - 输入：get_reward_manager_cls("naive")
      - 输出：NaiveRewardManager
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/reward_manager/registry.py::get_reward_manager_cls`
      - 典型调用路径：`reward_loop.py::_init_reward_fn` -> `get_reward_manager_cls`
      - 被谁调用：`reward_loop.py`
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：无
    """
    if name not in REWARD_LOOP_MANAGER_REGISTRY:  # 注释：名称不存在则报错
        raise ValueError(f"Unknown reward manager: {name}")  # 注释：提示未知名称
    return REWARD_LOOP_MANAGER_REGISTRY[name]  # 注释：返回注册的类
