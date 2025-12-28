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
模块用途：定义 RewardLoop 版本 RewardManager 的抽象基类与共享初始化逻辑。（注释：模块功能概述）
输入：DictConfig、AutoTokenizer、DataProto。（注释：输入形态说明）
输出：子类需实现的奖励计算接口。（注释：输出形态说明）
关键依赖：omegaconf、transformers、verl.utils.ray_utils.get_event_loop。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - class NaiveRewardManager(RewardManagerBase): ...
调用路径概览：（注释：全局调用关系）
  - reward_loop.py::_init_reward_fn -> RewardManagerBase 子类实例化
"""

import logging  # 注释：日志记录
import os  # 注释：读取环境变量
from abc import ABC, abstractmethod  # 注释：抽象基类与抽象方法

from omegaconf import DictConfig  # 注释：配置类型
from transformers import AutoTokenizer  # 注释：分词器类型

from verl import DataProto  # 注释：数据容器
from verl.utils.ray_utils import get_event_loop  # 注释：获取事件循环

logger = logging.getLogger(__file__)  # 注释：模块级日志器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：读取环境变量设置日志级别


class RewardManagerBase(ABC):  # 注释：RewardLoop 抽象基类
    _class_initialized = False  # 注释：类级初始化标记（全局共享）

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):  # 注释：基类初始化
        """
        功能：保存配置与 tokenizer，并初始化共享类状态与事件循环。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config (DictConfig): RewardLoop 配置。（注释：配置对象）
          - tokenizer (AutoTokenizer): 用于编码/解码的 tokenizer。（注释：分词器）
        返回：（注释：返回值说明）
          - None。（注释：初始化无显式返回）
        副作用：（注释：副作用说明）
          - 初始化事件循环引用。（注释：事件循环副作用）
          - 触发类级初始化 init_class。（注释：类级状态）
        异常/边界条件：（注释：异常与边界）
          - get_event_loop 失败可能抛出异常。（注释：事件循环异常）
        最小示例：（注释：最小可理解示例）
          - 输入：RewardManagerBase 子类(config, tokenizer)
          - 输出：对象初始化完成
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/base.py::RewardManagerBase.__init__`
          - 典型调用路径：`reward_loop.py::_init_reward_fn` -> 子类 __init__
          - 被谁调用：`NaiveRewardManager`、`DAPORewardManager`、`RateLimitedRewardManager`
          - 调用了谁（项目内）：`get_event_loop`、`init_class`
          - 调用了谁（外部依赖）：无
        """
        self.config = config  # 注释：保存配置
        self.tokenizer = tokenizer  # 注释：保存 tokenizer
        self.loop = get_event_loop()  # 注释：获取事件循环
        self.init_class(config, tokenizer)  # 注释：执行类级初始化

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):  # 注释：类级共享初始化
        """
        功能：初始化所有实例共享的类级状态。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config (DictConfig): RewardLoop 配置。（注释：配置对象）
          - tokenizer (AutoTokenizer): 分词器。（注释：分词器）
        返回：（注释：返回值说明）
          - None。（注释：无显式返回）
        副作用：（注释：副作用说明）
          - 修改类变量 _class_initialized。（注释：类级状态变更）
        异常/边界条件：（注释：异常与边界）
          - 已初始化时直接返回。（注释：幂等逻辑）
        最小示例：（注释：最小可理解示例）
          - 输入：RewardManagerBase.init_class(config, tokenizer)
          - 输出：_class_initialized=True
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/base.py::RewardManagerBase.init_class`
          - 典型调用路径：`RewardManagerBase.__init__` -> `init_class`
          - 被谁调用：各 RewardManager 子类构造时
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：无
        """
        if cls._class_initialized:  # 注释：已初始化则直接返回
            return  # 注释：避免重复初始化
        cls._class_initialized = True  # 注释：标记已初始化

    @abstractmethod
    async def run_single(self, data: DataProto):  # 注释：异步处理单条样本
        """
        功能：子类实现单条样本的 reward 计算逻辑。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 单条样本的 DataProto（len==1）。（注释：输入样本）
        返回：（注释：返回值说明）
          - dict：包含 reward_score / reward_extra_info 等字段。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 可能调用外部服务或记录日志。（注释：外部副作用）
        异常/边界条件：（注释：异常与边界）
          - 子类决定（如缺字段/网络错误）。（注释：异常由子类处理）
        最小示例：（注释：最小可理解示例）
          - 输入：data=DataProto(len=1)
          - 输出：{"reward_score": 1.0, "reward_extra_info": {...}}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/base.py::RewardManagerBase.run_single`
          - 典型调用路径：`RewardLoopWorker.compute_score` -> `reward_loop.run_single`
          - 被谁调用：`RewardLoopWorker`、`RateLimitedRewardManager.__call__`（批量包装）
          - 调用了谁（项目内）：由子类实现
          - 调用了谁（外部依赖）：由子类实现
        """
        raise NotImplementedError  # 注释：抽象方法占位
