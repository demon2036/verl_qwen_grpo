# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
模块用途：基于 Ray 的 PPO/GRPO 训练调度器，负责数据流、worker 调度、训练循环与指标汇总。（注释：模块级用途概述）
输入/输出：（注释：模块级 I/O 概览）
  - 输入：Ray 资源配置、DataProto 批数据、Trainer 配置等。（注释：输入来源）
  - 输出：训练日志、checkpoint、验证指标等。（注释：输出形态）
关键依赖：（注释：列出关键依赖）
  - ray / torch / omegaconf / numpy。（注释：外部依赖）
  - verl.trainer.ppo.core_algos（优势估计与损失）。（注释：项目内依赖）
典型用法（最小示例）：（注释：最小调用片段）
  >>> trainer = RayPPOTrainer(config, tokenizer, role_worker_mapping, resource_pool_manager)  # 创建 trainer（示例）
  >>> trainer.init_workers(); trainer.fit()  # 启动训练（示例）
调用路径概览：（注释：入口链路）
  - `examples/grpo_trainer/run_qwen2-7b.sh` -> `verl.trainer.main_ppo.TaskRunner.run` -> `RayPPOTrainer.fit`。（注释：调用链）
备注：（注释：说明设计）
  - 支持不同后端（FSDP/Megatron/vLLM）与多种优势估计器。（注释：功能概览）
"""  # 注释：模块 docstring 结束

# 标准库导入（注释：系统与工具）
import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

# 第三方依赖（注释：数值/分布式/张量/数据加载）
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

# 项目内依赖（注释：核心数据结构与工具）
from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    资源池管理器：根据配置创建 RayResourcePool，并提供查询接口。（注释：类用途概述）

    属性：（注释：属性说明）
      - resource_pool_spec (dict[str, list[int]]): 资源池 -> 每节点 GPU 数列表。（注释：输入配置）
      - mapping (dict[Role, str]): Role -> 资源池名称映射。（注释：角色资源分配）
      - resource_pool_dict (dict[str, RayResourcePool]): 已创建的资源池对象。（注释：运行期状态）
    最小示例：（注释：最小示例）
      >>> rpm = ResourcePoolManager({"global_pool":[8]}, mapping)  # 构造（示例）
      >>> rpm.create_resource_pool()  # 创建资源池（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/ray_trainer.py`（注释：文件路径）
      - 类：`ResourcePoolManager`（注释：类名）
      典型调用路径
      ------------
      - `TaskRunner.init_resource_pool_mgr` -> `ResourcePoolManager.create_resource_pool`。（注释：入口链路）
      被谁调用
      --------
      - `TaskRunner.init_resource_pool_mgr` / `RayPPOTrainer.init_workers`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `RayResourcePool`（Ray 资源池封装）。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `ray._private.state.available_resources_per_node`。（注释：外部依赖）
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """
        创建 RayResourcePool 并检查资源可用性。（注释：方法用途）

        返回：（注释：返回值说明）
          - None。（注释：仅创建并写入 resource_pool_dict）
        副作用：（注释：副作用说明）
          - 修改 self.resource_pool_dict。（注释：内部状态更新）
        异常/边界条件：（注释：异常说明）
          - 资源不足时 _check_resource_available 会抛 ValueError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> rpm.create_resource_pool()  # 创建资源池（示例）
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.init_workers` -> `create_resource_pool`。（注释：典型调用）
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # 逐个资源池创建（注释：遍历配置）
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()  # 注释：检查资源是否足够

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """
        根据角色获取资源池。（注释：方法用途）

        参数：（注释：参数说明）
          - role (Role): 角色枚举。（注释：输入含义）
        返回：（注释：返回值说明）
          - RayResourcePool: 对应资源池对象。（注释：输出含义）
        """
        return self.resource_pool_dict[self.mapping[role]]  # 注释：根据 mapping 查找资源池

    def get_n_gpus(self) -> int:
        """
        统计配置中总 GPU 数。（注释：方法用途）

        返回：（注释：返回值说明）
          - int: 总 GPU 数量。（注释：输出含义）
        """
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """
        检查 Ray 集群资源是否满足需求。（注释：方法用途）

        异常/边界条件：（注释：异常说明）
          - 资源不足时抛 ValueError。（注释：边界）
        """
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """
    在 token 级奖励中加入 KL 惩罚项。（注释：函数用途）

    参数：（注释：参数说明）
      - data (DataProto): 包含 old_log_probs/ref_log_prob/token_level_scores 的批数据。（注释：输入含义）
      - kl_ctrl (AdaptiveKLController): 自适应 KL 系数控制器。（注释：输入含义）
      - kl_penalty (str): KL 惩罚类型（默认 "kl"）。（注释：输入含义）
    返回：（注释：返回值说明）
      - data (DataProto): 更新了 token_level_rewards。（注释：输出含义）
      - metrics (dict): KL 惩罚相关指标。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 修改 data.batch["token_level_rewards"]。（注释：原地修改）
      - 更新 kl_ctrl 内部系数。（注释：状态更新）
    异常/边界条件：（注释：异常说明）
      - 缺少 log_probs 字段会触发 KeyError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> data, metrics = apply_kl_penalty(data, kl_ctrl, kl_penalty="kl")  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/ray_trainer.py`（注释：文件路径）
      - 函数：`apply_kl_penalty(...)`（注释：函数名）
      典型调用路径
      ------------
      - `RayPPOTrainer.fit` -> `apply_kl_penalty`。（注释：训练循环）
      被谁调用
      --------
      - `RayPPOTrainer.fit`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `core_algos.kl_penalty` / `masked_mean`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.mean`。（注释：外部依赖）
    """
    # 取出必要字段（注释：mask 与 scores）
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # 计算 KL（注释：old vs ref）
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask  # 注释：mask 掉 padding
    beta = kl_ctrl.value  # 注释：当前 KL 系数

    # 奖励减去 KL 惩罚（注释：token 级）
    token_level_rewards = token_level_scores - beta * kld

    # 统计当前 KL（注释：序列平均后再 batch 平均）
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    # 更新 KL 控制器（注释：自适应系数）
    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards  # 注释：写回奖励

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}  # 注释：指标

    return data, metrics  # 注释：返回更新后的数据与指标


def compute_response_mask(data: DataProto):
    """
    从 attention_mask 中切出 response 部分的 mask。（注释：函数用途）

    参数：（注释：参数说明）
      - data (DataProto): 包含 responses 与 attention_mask 的批数据。（注释：输入含义）
    返回：（注释：返回值说明）
      - torch.Tensor: response 区域的 mask，形状 (B, response_len)。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 无。（注释：纯函数）
    异常/边界条件：（注释：异常说明）
      - 缺失字段会抛 KeyError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> resp_mask = compute_response_mask(batch)  # 返回 response mask（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/ray_trainer.py`（注释：文件路径）
      - 函数：`compute_response_mask(data)`（注释：函数名）
      典型调用路径
      ------------
      - `RayPPOTrainer.fit` -> `compute_response_mask`。（注释：训练循环）
      被谁调用
      --------
      - `RayPPOTrainer.fit`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - 无。（注释：无项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.Tensor.size`。（注释：外部依赖）
    """
    responses = data.batch["responses"]  # 注释：response token 序列
    response_length = responses.size(1)  # 注释：response 长度
    attention_mask = data.batch["attention_mask"]  # 注释：全序列 mask
    return attention_mask[:, -response_length:]  # 注释：切出 response 部分


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """
    计算优势（GAE/GRPO/REINFORCE 等）并写回 batch。（注释：函数用途）

    参数：（注释：参数说明）
      - data (DataProto): 含奖励/values/response_mask 的批数据。（注释：输入含义）
      - adv_estimator (AdvantageEstimator): 优势估计器类型。（注释：输入含义）
      - gamma (float): 折扣因子。（注释：输入含义）
      - lam (float): GAE λ 参数。（注释：输入含义）
      - num_repeat (int): 采样重复次数（GRPO 等用到）。（注释：输入含义）
      - norm_adv_by_std_in_grpo (bool): GRPO 是否按标准差归一化。（注释：输入含义）
      - config (AlgoConfig|None): 算法配置。（注释：输入含义）
    返回：（注释：返回值说明）
      - DataProto: 更新了 advantages/returns 的 data。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 修改 data.batch 中的 advantages/returns。（注释：原地修改）
    异常/边界条件：（注释：异常说明）
      - 未注册的优势估计器会抛 ValueError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> data = compute_advantage(data, AdvantageEstimator.GRPO, num_repeat=5)  # 示例
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/ppo/ray_trainer.py`（注释：文件路径）
      - 函数：`compute_advantage(...)`（注释：函数名）
      典型调用路径
      ------------
      - `RayPPOTrainer.fit` -> `compute_advantage`。（注释：训练循环）
      被谁调用
      --------
      - `RayPPOTrainer.fit`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `core_algos.compute_gae_advantage_return` / `core_algos.compute_grpo_outcome_advantage` / `core_algos.get_adv_estimator_fn`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch` 张量运算。（注释：外部依赖）
    """
    # 兼容旧 trainer：若未计算 response_mask 则补齐（注释：兼容性）
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # 按优势估计器分支（注释：GAE/GRPO/其他）
    if adv_estimator == AdvantageEstimator.GAE:
        # GAE：依赖 values（注释：通用优势估计）
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages  # 注释：写回优势
        data.batch["returns"] = returns  # 注释：写回回报
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )  # 注释：可选 PF-PPO 重加权
    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO：只依赖 token_level_rewards（注释：基于 outcome）
        grpo_calculation_mask = data.batch["response_mask"]  # 注释：默认 mask

        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages  # 注释：写回优势
        data.batch["returns"] = returns  # 注释：写回回报
    else:
        # 其他估计器（注释：通过注册表获取）
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]  # 注释：按 UID 聚合
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]  # 注释：REMAX 基线

        # 计算优势（注释：调用注册函数）
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages  # 注释：写回优势
        data.batch["returns"] = returns  # 注释：写回回报
    return data  # 注释：返回更新后的 data


class RayPPOTrainer:
    """
    基于 Ray 的 PPO/GRPO 分布式训练器。（注释：类用途概述）

    功能概述：（注释：说明核心职责）
      - 创建/管理 Actor、Critic、Ref、Reward 等 worker group。（注释：调度角色）
      - 执行 rollout、奖励计算、优势估计与参数更新。（注释：训练流程）
      - 记录指标、保存/加载 checkpoint。（注释：训练管理）
    关键依赖：（注释：列出依赖）
      - `RayWorkerGroup` / `RayResourcePool`（分布式调度）。（注释：依赖）
      - `core_algos`（优势/损失计算）。（注释：依赖）
    最小示例：（注释：最小示例）
      >>> trainer = RayPPOTrainer(config, tokenizer, role_worker_mapping, resource_pool_manager)  # 创建
      >>> trainer.init_workers(); trainer.fit()  # 启动训练
    调用路径依赖：（注释：调用关系说明）
      - `TaskRunner.run` -> `RayPPOTrainer.fit`。（注释：入口链路）
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        初始化 RayPPOTrainer（运行在 driver 进程）。（注释：方法用途）

        参数：（注释：参数说明）
          - config (DictConfig): 训练配置。（注释：输入含义）
          - tokenizer: tokenizer 实例。（注释：输入含义）
          - role_worker_mapping (dict[Role, WorkerType]): 角色 -> worker 类映射。（注释：输入含义）
          - resource_pool_manager (ResourcePoolManager): 资源池管理器。（注释：输入含义）
          - ray_worker_group_cls (type): RayWorkerGroup 类。（注释：输入含义）
          - processor: 可选 processor（多模态用）。（注释：输入含义）
          - reward_fn: 训练奖励函数。（注释：输入含义）
          - val_reward_fn: 验证奖励函数。（注释：输入含义）
          - train_dataset/val_dataset: 可选数据集。（注释：输入含义）
          - collate_fn: collate 函数。（注释：输入含义）
          - train_sampler: 训练采样器。（注释：输入含义）
          - device_name: 设备名（cuda/npu 等）。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：构造函数无返回）
        副作用：（注释：副作用说明）
          - 初始化内部状态，并创建 DataLoader。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - hybrid_engine=False 会触发断言。（注释：边界）
        最小示例：（注释：最小示例）
          >>> RayPPOTrainer(config, tokenizer, role_map, pool_mgr)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `TaskRunner.run` -> `RayPPOTrainer.__init__`。（注释：典型调用）
        """

        # 保存关键对象（注释：tokenizer/processor/config/reward）
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # 仅支持 hybrid engine（注释：当前实现限制）
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )  # 注释：必须存在 actor_rollout 相关角色

        # 保存角色映射与资源池管理器（注释：后续初始化用）
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)  # 注释：是否启用 ref
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)  # 注释：是否启用 RM
        self.use_reward_loop = self.config.reward_model.use_reward_loop  # 注释：reward loop 开关

        # 训练开关与设备（注释：critic/worker/device）
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )  # 注释：验证样本日志

        # ref_in_actor：ref policy 复用 actor（注释：LoRA 情况）
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # KL-in-reward 控制器（注释：如启用 KL 奖励）
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # 新旧 worker 实现开关（注释：用于后续分支）
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # 创建 DataLoader（注释：训练/验证数据流）
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        构建训练/验证 DataLoader，并计算总训练步数。（注释：方法用途）

        参数：（注释：参数说明）
          - train_dataset / val_dataset: 可选外部传入的数据集。（注释：输入含义）
          - collate_fn: 可选 collate 函数。（注释：输入含义）
          - train_sampler: 可选训练采样器。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：内部初始化 dataloader）
        副作用：（注释：副作用说明）
          - 设置 self.train_dataloader / self.val_dataloader / self.total_training_steps。（注释：状态修改）
        异常/边界条件：（注释：异常说明）
          - 训练/验证 dataloader 为空会触发断言。（注释：边界）
        最小示例：（注释：最小示例）
          >>> self._create_dataloader(None, None, None, None)  # 由配置创建（示例）
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.__init__` -> `_create_dataloader`。（注释：构造过程）
        """
        # TODO: batch_size 需可被 dp size 整除（注释：未来校验）
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler  # 注释：复用入口函数

        # 构建训练集（注释：若未显式传入）
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        # 构建验证集（注释：若未显式传入）
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset  # 注释：保存数据集

        # 训练采样器（注释：若未传入则创建）
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        # collate_fn（注释：若未传入则使用默认）
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]  # 注释：DataLoader worker 数

        # 训练 DataLoader（注释：使用 StatefulDataLoader 支持恢复）
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # 验证 batch_size（注释：优先使用配置）
        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        # 验证 DataLoader（注释：可配置 shuffle）
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"  # 注释：训练集非空
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"  # 注释：验证集非空

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )  # 注释：打印 dataloader 大小

        # 计算总训练步数（注释：len(dataloader)*epochs）
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps  # 注释：优先使用显式配置

        self.total_training_steps = total_training_steps  # 注释：保存总步数
        print(f"Total training steps: {self.total_training_steps}")  # 注释：打印总步数

        # 尝试写回 total_training_steps 到配置（注释：供优化器使用）
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """
        将生成结果写入 JSONL 文件（rollout/validation 用）。（注释：方法用途）

        参数：（注释：参数说明）
          - inputs/outputs/gts/scores: 生成样本与分数列表。（注释：输入含义）
          - reward_extra_infos_dict (dict): 额外字段（可选）。（注释：输入含义）
          - dump_path (str): 输出目录。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：仅写文件）
        副作用：（注释：副作用说明）
          - 在 dump_path 下创建 jsonl 文件。（注释：文件写入）
        异常/边界条件：（注释：异常说明）
          - 目录不可写会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> self._dump_generations(["q"], ["a"], ["gt"], [1.0], {}, "./dump")  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer._log_rollout_data` / `_validate` -> `_dump_generations`。（注释：内部调用）
        """
        os.makedirs(dump_path, exist_ok=True)  # 注释：确保输出目录存在
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")  # 注释：文件名含 global_steps

        n = len(inputs)  # 注释：样本数量
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }  # 注释：基础字段

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v  # 注释：追加额外字段（长度匹配才写）

        lines = []  # 注释：每行一个 JSON
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}  # 注释：构造单条记录
            lines.append(json.dumps(entry, ensure_ascii=False))  # 注释：序列化为 JSON

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")  # 注释：写入 JSONL

        print(f"Dumped generations to {filename}")  # 注释：日志提示

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """
        将 rollout 生成样本写入磁盘。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): rollout 输出 batch。（注释：输入含义）
          - reward_extra_infos_dict (dict): 额外奖励信息。（注释：输入含义）
          - timing_raw (dict): 计时信息（用于打点）。（注释：输入含义）
          - rollout_data_dir (str): 输出目录。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：仅写文件）
        副作用：（注释：副作用说明）
          - 写入 JSONL 文件。（注释：文件输出）
        最小示例：（注释：最小示例）
          >>> self._log_rollout_data(batch, extra_infos, timing, "./rollout")  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_log_rollout_data` -> `_dump_generations`。（注释：训练流程）
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)  # 注释：解码 prompt
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)  # 注释：解码输出
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()  # 注释：序列分数
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]  # 注释：GT

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """
        按配置将验证样本写入日志系统（如 wandb/swanlab）。（注释：方法用途）

        参数：（注释：参数说明）
          - inputs/outputs/scores: 验证样本与分数列表。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：仅记录日志）
        副作用：（注释：副作用说明）
          - 向日志后端写入表格数据。（注释：外部可见）
        最小示例：（注释：最小示例）
          >>> self._maybe_log_val_generations(["q"], ["a"], [1.0])  # 示例
        """

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return  # 注释：不记录直接返回

        import numpy as np

        # 构造 (input, output, score) 并排序（注释：保证确定性）
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # 固定随机种子（注释：确保可复现）
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # 取前 N 个样本（注释：控制日志数量）
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        计算或提取 reward（支持 reward_loop 直接读取 rm_scores）。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 含奖励或需计算奖励的 batch。（注释：输入含义）
          - reward_fn: 奖励函数（若 rm_scores 不存在时使用）。（注释：输入含义）
          - return_dict (bool): 是否返回 dict（用于验证记录额外信息）。（注释：输入含义）
          - sum_reward (bool): 是否沿最后维度求和（REMAX 基线用）。（注释：输入含义）
        返回：（注释：返回值说明）
          - dict 或 Tensor 或 (Tensor, extra_info)：视 return_dict/sum_reward 而定。（注释：输出说明）
        副作用：（注释：副作用说明）
          - 无（不修改 batch 内容）。  # 注释：保持纯读
        异常/边界条件：（注释：异常说明）
          - reward_fn 缺失且 rm_scores 不存在会抛 ValueError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> reward_tensor, extra = self._compute_or_extract_reward(batch, reward_fn=fn)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` / `_validate` -> `_compute_or_extract_reward`。（注释：调用链）
        """
        # 若 rm_scores 已存在则直接提取（注释：无需重复计算）
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # 否则调用 reward_fn 计算（注释：外部奖励函数）
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """
        构造用于生成的 batch：去除无关字段，保留奖励相关信息。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 原始 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto: 用于 generate_sequences 的 batch。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 会对 batch 执行 pop（返回的新 batch）。（注释：数据被拆分）
        最小示例：（注释：最小示例）
          >>> gen_batch = self._get_gen_batch(batch)  # 示例
        """
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # 生成阶段不需要的张量字段（注释：减少传输）
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # agent loop 需要保留 reward model keys（注释：异步 rollout）
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch  # 注释：返回生成 batch

    def _validate(self):
        """
        执行验证：生成输出、计算奖励并汇总验证指标。（注释：方法用途）

        返回：（注释：返回值说明）
          - dict: 验证指标字典（可能为空）。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 可能写入验证样本日志/文件。（注释：外部输出）
        异常/边界条件：（注释：异常说明）
          - 若使用模型奖励且配置不支持验证，会提前返回空字典。（注释：边界）
        最小示例：（注释：最小示例）
          >>> metrics = self._validate()  # 返回验证指标（示例）
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_validate`。（注释：训练循环）
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """
        初始化 Ray worker group（actor/critic/ref/reward 等）。（注释：方法用途）

        返回：（注释：返回值说明）
          - None。（注释：仅初始化内部状态）
        副作用：（注释：副作用说明）
          - 创建资源池与 worker group；可能启动远程进程。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 资源不足会在 ResourcePoolManager 中报错。（注释：边界）
        最小示例：（注释：最小示例）
          >>> trainer.init_workers()  # 初始化 worker（示例）
        调用路径依赖：（注释：调用关系说明）
          - `TaskRunner.run` -> `RayPPOTrainer.init_workers`。（注释：调用链）
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            can_reward_loop_parallelize = self.config.actor_rollout_ref.rollout.mode == "async" and (
                not self.use_rm or self.config.reward_model.enable_resource_pool
            )
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            # Support custom AgentLoopManager via config
            manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            if manager_class_fqn:
                AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
            else:
                from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        """
        保存 actor/critic 模型与 dataloader 状态。（注释：方法用途）

        副作用：（注释：副作用说明）
          - 在本地/远端目录写入 checkpoint 文件。（注释：文件输出）
        异常/边界条件：（注释：异常说明）
          - 路径不可写会抛异常。（注释：边界）
        """
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        """
        从本地（或未来 HDFS）加载 checkpoint。（注释：方法用途）

        返回：（注释：返回值说明）
          - int: 恢复后的 global_steps（0 表示从头训练）。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 更新模型权重与 dataloader 状态。（注释：状态恢复）
        异常/边界条件：（注释：异常说明）
          - resume_mode 配置不合法或路径不存在可能报错。（注释：边界）
        """
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """
        按配置启动各 worker 的 profiling。（注释：方法用途）

        参数：（注释：参数说明）
          - do_profile (bool): 是否启用 profiling。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 触发 worker 的 start_profile。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> self._start_profiling(True)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` 中按步触发。（注释：调用链）
        """
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """
        按配置停止各 worker 的 profiling。（注释：方法用途）

        参数：（注释：参数说明）
          - do_profile (bool): 是否停用 profiling。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 触发 worker 的 stop_profile。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> self._stop_profiling(True)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` 中按步触发。（注释：调用链）
        """
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """
        重新排序 batch，使各 DP rank 的 token 负载更均衡。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 当前 batch。（注释：输入含义）
          - metrics (dict): 指标字典（用于记录均衡统计）。（注释：输入含义）
          - logging_prefix (str): 统计前缀。（注释：输入含义）
          - keep_minibatch (bool): 是否保持 mini-batch 内部顺序。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：batch 原地重排）
        副作用：（注释：副作用说明）
          - 修改 batch 内部顺序并更新 metrics。（注释：原地修改）
        最小示例：（注释：最小示例）
          >>> self._balance_batch(batch, metrics)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` 在 balance_batch 配置开启时调用。（注释：调用链）
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                workload_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (workload_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        """
        计算 critic 的 value 预测。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 含 input/attention_mask 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto: 包含 values 的输出。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 远程调用 critic worker。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> values = self._compute_values(batch)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_compute_values`。（注释：调用链）
        """
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """
        计算参考策略的 log_prob。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 含 prompt/response 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto: 包含 ref_log_prob。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 远程调用 ref_policy worker。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> ref_log = self._compute_ref_log_prob(batch)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_compute_ref_log_prob`。（注释：调用链）
        """
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        """
        计算旧策略 log_prob（作为 PPO 参考分布）。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 含输入/响应的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - (DataProto, mfu): 包含 old_log_probs 与 MFU 指标。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 远程调用 actor_rollout worker。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> old_log, mfu = self._compute_old_log_prob(batch)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_compute_old_log_prob`。（注释：调用链）
        """
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        """
        更新 Actor 参数（PPO policy 更新）。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 含优势/returns 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto: 包含 actor 更新指标。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 触发 actor worker 反向更新。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> metrics = self._update_actor(batch)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_update_actor`。（注释：调用链）
        """
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        """
        更新 Critic 参数（价值函数回归）。（注释：方法用途）

        参数：（注释：参数说明）
          - batch (DataProto): 含 returns/values 的 batch。（注释：输入含义）
        返回：（注释：返回值说明）
          - DataProto: 包含 critic 更新指标。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 触发 critic worker 反向更新。（注释：外部副作用）
        最小示例：（注释：最小示例）
          >>> metrics = self._update_critic(batch)  # 示例
        调用路径依赖：（注释：调用关系说明）
          - `RayPPOTrainer.fit` -> `_update_critic`。（注释：调用链）
        """
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        PPO/GRPO 主训练循环。（注释：方法用途）

        核心流程：（注释：简要概述）
          - rollout 生成 -> 奖励计算 -> 优势估计 -> actor/critic 更新 -> 指标记录。（注释：流程概览）
        返回：（注释：返回值说明）
          - None。（注释：训练到结束即返回）
        副作用：（注释：副作用说明）
          - 触发远程 worker 计算与更新；写日志/保存 checkpoint。（注释：外部副作用）
        异常/边界条件：（注释：异常说明）
          - 关键配置缺失会导致运行时异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> trainer.fit()  # 启动训练（示例）
        调用路径依赖：（注释：调用关系说明）
          - `TaskRunner.run` -> `RayPPOTrainer.fit`。（注释：调用链）
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
