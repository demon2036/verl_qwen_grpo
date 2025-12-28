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
模块用途：提供 PPO/GRPO 训练中的角色枚举、worker 类型别名，以及是否启用 critic/ref policy/reward model 的判断逻辑。（注释：模块级用途概述）
输入/输出：（注释：模块级 I/O 概览）
  - 输入：角色映射字典、Hydra 配置（DictConfig）。（注释：说明输入来源）
  - 输出：布尔判断结果、角色字符串。（注释：说明输出形态）
关键依赖：（注释：列出关键依赖）
  - omegaconf.DictConfig（配置对象）。（注释：关键依赖）
  - verl.trainer.ppo.core_algos.AdvantageEstimator（优势估计枚举）。（注释：关键依赖）
典型用法（最小示例）：（注释：最小可运行片段）
  >>> from verl.trainer.ppo.utils import Role, need_critic  # 导入工具（示例）
  >>> Role.from_string("actor_rollout")  # 解析角色（示例）
  >>> need_critic(cfg)  # 判断是否启用 critic（示例）
调用路径概览：（注释：说明入口链路）
  - examples/grpo_trainer/run_qwen2-7b.sh -> verl.trainer.main_ppo.TaskRunner.run -> need_critic/need_reference_policy。（注释：调用路径）
"""  # 注释：模块 docstring 结束

# 标准库导入（注释：用于发出警告）
import warnings
# 标准库导入（注释：Enum 用于角色枚举）
from enum import Enum

# 第三方依赖（注释：Hydra 配置类型）
from omegaconf import DictConfig

# 项目内依赖（注释：Worker 基类）
from verl.single_controller.base import Worker
# 项目内依赖（注释：优势估计枚举）
from verl.trainer.ppo.core_algos import AdvantageEstimator

# Worker 类型别名（注释：用于类型标注）
WorkerType = type[Worker]


class Role(Enum):
    """
    训练中各类 Worker 的角色枚举，用于统一角色命名与资源映射。（注释：类用途概述）

    参数：（注释：Enum 不显式接收参数）
      - 无。（注释：枚举项在类体内定义）
    返回：（注释：Enum 实例可转为字符串）
      - `Role` 枚举成员。（注释：输出类型）
    副作用：（注释：无副作用）
      - 无。（注释：说明无状态修改）
    异常/边界条件：（注释：异常说明）
      - `from_string` 传入未知名称会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> Role.ActorRollout  # 直接使用枚举值（示例）
      >>> Role.from_string("ref")  # 转换字符串（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/utils.py`（注释：文件路径）
      - 类：`Role`（注释：类名）
      典型调用路径
      ------------
      - `verl/trainer/main_ppo.py::TaskRunner` -> `Role.*`（注释：典型调用）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`、`verl/trainer/main_ppo.py` 等角色映射处。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - 无。（注释：枚举本身不调用项目函数）
      调用了谁（关键外部依赖）
      ----------------------
      - Python `Enum`。（注释：外部依赖）
    """

    Actor = 0  # 注释：Actor 训练/更新角色
    Rollout = 1  # 注释：Rollout 生成角色
    ActorRollout = 2  # 注释：Actor+Rollout 融合角色
    Critic = 3  # 注释：价值函数/critic 角色
    RefPolicy = 4  # 注释：参考策略（KL 约束用）
    RewardModel = 5  # 注释：奖励模型角色
    ActorRolloutRef = 6  # 注释：Actor+Rollout+Ref 融合角色
    Env = 7  # 注释：环境/外部交互角色

    def __str__(self):
        """返回角色的字符串表示（用于日志/配置匹配）。（注释：方法用途）"""
        return self._get_role_string()  # 注释：统一走内部映射方法

    def _get_role_string(self):
        """将枚举值映射为配置/日志使用的字符串。（注释：内部工具方法）"""
        role_mapping = {
            Role.Actor: "actor",  # 注释：Actor -> "actor"
            Role.Rollout: "rollout",  # 注释：Rollout -> "rollout"
            Role.ActorRollout: "actor_rollout",  # 注释：ActorRollout -> "actor_rollout"
            Role.Critic: "critic",  # 注释：Critic -> "critic"
            Role.RefPolicy: "ref",  # 注释：RefPolicy -> "ref"
            Role.RewardModel: "rm",  # 注释：RewardModel -> "rm"
            Role.ActorRolloutRef: "actor_rollout_ref",  # 注释：ActorRolloutRef -> "actor_rollout_ref"
        }
        return role_mapping.get(self, self.name.lower())  # 注释：未显式映射则回退到枚举名

    @classmethod
    def from_string(cls, name: str):
        """
        将字符串解析为 Role 枚举。（注释：类方法用途）

        参数：（注释：参数说明）
          - name (str): 角色字符串，如 "actor"/"ref"。（注释：输入类型与语义）
        返回：（注释：返回值说明）
          - Role: 对应枚举成员。（注释：输出类型）
        副作用：（注释：无副作用）
          - 无。（注释：仅做映射）
        异常/边界条件：（注释：异常说明）
          - 未知字符串会抛 ValueError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> Role.from_string("actor_rollout") == Role.ActorRollout  # True（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/ppo/utils.py`（注释：文件路径）
          - 方法：`Role.from_string(name)`（注释：方法名）
          典型调用路径
          ------------
          - `verl/trainer/ppo/ray_trainer.py` 等读取字符串角色时调用。（注释：典型调用）
          被谁调用
          --------
          - 角色字符串解析处（如 worker 映射/配置解析逻辑）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - 无。（注释：不依赖其他项目函数）
          调用了谁（关键外部依赖）
          ----------------------
          - Python `Enum` / `str.lower`。（注释：外部依赖）
        """
        string_mapping = {
            "actor": cls.Actor,  # 注释：actor 映射
            "rollout": cls.Rollout,  # 注释：rollout 映射
            "actor_rollout": cls.ActorRollout,  # 注释：actor_rollout 映射
            "critic": cls.Critic,  # 注释：critic 映射
            "ref": cls.RefPolicy,  # 注释：ref 映射
            "rm": cls.RewardModel,  # 注释：rm 映射
            "actor_rollout_ref": cls.ActorRolloutRef,  # 注释：actor_rollout_ref 映射
        }
        role = string_mapping.get(name.lower())  # 注释：大小写不敏感映射
        if role is None:
            raise ValueError(f"No Role found for string: {name}")  # 注释：未知字符串报错
        return role  # 注释：返回枚举成员


def need_reference_policy(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """
    判断是否需要参考策略（RefPolicy）或 ActorRolloutRef 融合角色。（注释：函数用途）

    参数：（注释：参数说明）
      - role_worker_mapping (dict[Role, WorkerType]): 角色到 Worker 的映射。（注释：输入含义）
    返回：（注释：返回值说明）
      - bool: True 表示需要参考策略。（注释：输出含义）
    副作用：（注释：无副作用）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - mapping 为空时返回 False。（注释：边界）
    最小示例：（注释：最小示例）
      >>> need_reference_policy({Role.RefPolicy: object})  # True（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/utils.py`（注释：文件路径）
      - 函数：`need_reference_policy(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/main_ppo.py::TaskRunner.run` -> `validate_config` 前判断。（注释：典型调用）
      被谁调用
      --------
      - `verl/trainer/main_ppo.py` / `verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - 无。（注释：无项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - Python `in` 操作符。（注释：外部依赖）
    """
    return Role.RefPolicy in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping  # 注释：判断逻辑


def need_reward_model(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """
    判断是否需要 RewardModel Worker。（注释：函数用途）

    参数：（注释：参数说明）
      - role_worker_mapping (dict[Role, WorkerType]): 角色到 Worker 的映射。（注释：输入含义）
    返回：（注释：返回值说明）
      - bool: True 表示需要奖励模型。（注释：输出含义）
    副作用：（注释：无副作用）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - mapping 为空时返回 False。（注释：边界）
    最小示例：（注释：最小示例）
      >>> need_reward_model({Role.RewardModel: object})  # True（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/utils.py`（注释：文件路径）
      - 函数：`need_reward_model(...)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.__init__`。（注释：典型调用）
      被谁调用
      --------
      - `verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - 无。（注释：无项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - Python `in` 操作符。（注释：外部依赖）
    """
    return Role.RewardModel in role_worker_mapping  # 注释：判断 RewardModel 是否存在


def need_critic(config: DictConfig) -> bool:
    """
    判断是否需要 critic（价值网络）训练。（注释：函数用途）

    参数：（注释：参数说明）
      - config (DictConfig): 训练配置，包含 critic 与算法设置。（注释：输入含义）
    返回：（注释：返回值说明）
      - bool: True 表示启用 critic。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 可能发出 warnings.warn 提示（adv_estimator 非 GAE 且未显式配置 critic）。（注释：副作用）
    异常/边界条件：（注释：异常说明）
      - config.critic.enable 为 None 且 adv_estimator 非 GAE 时，返回 False。（注释：边界）
    最小示例：（注释：最小示例）
      >>> cfg.critic.enable = None; cfg.algorithm.adv_estimator = AdvantageEstimator.GAE  # 示例输入
      >>> need_critic(cfg)  # True（示例输出）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/utils.py`（注释：文件路径）
      - 函数：`need_critic(config)`（注释：函数名）
      典型调用路径
      ------------
      - `verl/trainer/main_ppo.py::TaskRunner.run` -> `validate_config`。（注释：典型调用）
      被谁调用
      --------
      - `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `AdvantageEstimator` 枚举。（注释：项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `warnings.warn`。（注释：外部依赖）
    """
    if config.critic.enable is not None:
        return bool(config.critic.enable)  # 注释：显式配置时直接返回
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True  # 注释：GAE 需要 critic
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )  # 注释：非 GAE 且未显式启用时给出提示
        return False  # 注释：默认不启用 critic
