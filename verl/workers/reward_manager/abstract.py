# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
模块用途：定义 RewardManager 的抽象接口与通用辅助方法，用于统一奖励计算的调用协议。（注释：模块功能概述）
输入：DataProto batch、tokenizer、compute_score 等。（注释：输入形态说明）
输出：奖励张量或包含 reward_tensor 的字典。（注释：输出形态说明）
关键依赖：torch（张量）、verl.protocol.DataProto（数据容器）。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - class NaiveRewardManager(AbstractRewardManager): ...
  - rm = NaiveRewardManager(tokenizer, num_examine=1, compute_score=default_compute_score)
  - reward = rm(data)
调用路径概览：（注释：全局调用关系）
  - verl/trainer/ppo/reward.py::compute_reward -> RewardManager.__call__ -> _extract_reward_from_rm_scores
"""

from abc import ABC, abstractmethod  # 注释：抽象基类与抽象方法装饰器
from typing import Any, Callable  # 注释：类型提示

import torch  # 注释：奖励张量类型

from verl.protocol import DataProto  # 注释：训练数据容器

RawRewardFn = Callable[..., Any]  # 注释：评分函数类型别名（输入/输出不固定）


class AbstractRewardManager(ABC):  # 注释：RewardManager 抽象基类
    """
    功能：统一奖励管理器接口，子类需实现 __init__ 与 __call__。（注释：类的职责）
    典型子类：NaiveRewardManager、DAPORewardManager 等。（注释：子类示例）
    """

    @abstractmethod  # 注释：声明抽象方法（子类必须实现）
    def __init__(  # 注释：抽象初始化接口
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        """
        功能：初始化 RewardManager。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - tokenizer (Any): 用于解码 token 的 tokenizer。（注释：文本还原）
          - num_examine (int): 调试打印的样本数。（注释：调试控制）
          - compute_score (RawRewardFn|None): 实际评分函数。（注释：评分逻辑）
          - reward_fn_key (str): 数据源字段名。（注释：数据源定位）
          - **kwargs: 扩展参数。（注释：可扩展）
        返回：（注释：返回值说明）
          - None（初始化过程）。
        副作用：（注释：副作用说明）
          - 由子类决定（可能保存状态、打印日志等）。
        异常/边界条件：（注释：异常与边界）
          - 由子类实现决定（如参数校验失败）。
        最小示例：（注释：最小可理解示例）
          - 输入：tokenizer=..., num_examine=1, compute_score=default_compute_score
          - 输出：完成对象初始化
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/reward_manager/abstract.py::AbstractRewardManager.__init__`
          - 典型调用路径：`verl/trainer/ppo/reward.py::load_reward_manager` -> 子类 __init__
          - 被谁调用：各 RewardManager 子类构造过程（如 NaiveRewardManager）
          - 调用了谁（项目内）：由子类决定
          - 调用了谁（外部依赖）：由子类决定
        """
        pass  # 注释：抽象方法，不提供实现

    @abstractmethod  # 注释：声明抽象方法（子类必须实现）
    def __call__(  # 注释：抽象调用接口
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """
        功能：根据 DataProto 计算奖励。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 包含 prompts/responses/metadata 的 batch。（注释：输入批次）
          - return_dict (bool): 是否返回包含 reward_extra_info 的字典。（注释：返回控制）
        返回：（注释：返回值说明）
          - torch.Tensor 或 dict[str, Any]。（注释：返回类型）
        副作用：（注释：副作用说明）
          - 由子类实现决定（可能打印或记录日志）。
        异常/边界条件：（注释：异常与边界）
          - 由子类实现决定（如缺失字段）。
        最小示例：（注释：最小可理解示例）
          - 输入：data=batch, return_dict=True
          - 输出：{"reward_tensor": tensor(...), "reward_extra_info": {...}}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/reward_manager/abstract.py::AbstractRewardManager.__call__`
          - 典型调用路径：`verl/trainer/ppo/reward.py::compute_reward` -> reward_fn(data)
          - 被谁调用：`verl/trainer/ppo/reward.py::compute_reward`
          - 调用了谁（项目内）：由子类决定（通常会调用 _extract_reward_from_rm_scores）
          - 调用了谁（外部依赖）：由子类决定
        """
        pass  # 注释：抽象方法，不提供实现

    def _extract_reward_from_rm_scores(  # 注释：从 data.batch 的 rm_scores 中提取奖励
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        """
        功能：如果 DataProto 已包含 rm_scores（由 reward loop 预计算），直接提取并返回。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 当前 batch。（注释：输入批次）
          - return_dict (bool): 是否返回包含 reward_extra_info 的字典。（注释：返回控制）
        返回：（注释：返回值说明）
          - 存在 rm_scores 时：torch.Tensor 或 dict；不存在则返回 None。（注释：返回分支）
        副作用：（注释：副作用说明）
          - 无（纯读取）。
        异常/边界条件：（注释：异常与边界）
          - data.batch 缺失 rm_scores 时返回 None。（注释：缺失字段处理）
          - reward_extra_keys 缺失时默认空列表。（注释：边界默认值）
        最小示例：（注释：最小可理解示例）
          - 输入：data.batch["rm_scores"]=tensor([1.0])，return_dict=True
          - 输出：{"reward_tensor": tensor([1.0]), "reward_extra_info": {}}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/reward_manager/abstract.py::_extract_reward_from_rm_scores`
          - 典型调用路径：`verl/workers/reward_manager/naive.py::__call__` -> `_extract_reward_from_rm_scores`
          - 被谁调用：`verl/workers/reward_manager/naive.py`、`verl/workers/reward_manager/dapo.py`、
            `verl/workers/reward_manager/batch.py`、`verl/workers/reward_manager/prime.py`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：无
        """
        if "rm_scores" not in data.batch.keys():  # 注释：若无 rm_scores 则直接返回 None
            return None  # 注释：表示无法直接提取

        if return_dict:  # 注释：需要返回字典格式
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])  # 注释：读取额外信息字段列表
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}  # 注释：组装附加信息
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}  # 注释：返回字典
        else:  # 注释：仅返回张量格式
            return data.batch["rm_scores"]  # 注释：直接返回 rm_scores
