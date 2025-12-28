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
模块用途：PPO/GRPO 训练入口（Hydra + Ray），负责初始化 Ray、构建数据集与 Trainer 并启动训练。（注释：模块级用途概述）
输入/输出：（注释：模块级 I/O 概览）
  - 输入：Hydra 配置（yaml + CLI override）、数据文件路径、模型权重路径。（注释：输入来源）
  - 输出：训练过程日志、checkpoint、评测指标。（注释：输出形态）
关键依赖：（注释：列出关键依赖）
  - hydra / omegaconf（配置管理）、ray（分布式调度）、torch（数据集）。（注释：关键依赖）
  - verl.trainer.ppo.ray_trainer.RayPPOTrainer（训练主循环）。（注释：项目内依赖）
典型用法（最小示例）：（注释：最小可运行命令）
  python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=...  # 参见 examples/grpo_trainer/run_qwen2-7b.sh（注释：示例）
调用路径概览：（注释：入口调用链）
  - `examples/grpo_trainer/run_qwen2-7b.sh` -> `verl.trainer.main_ppo:main` -> `run_ppo` -> `RayPPOTrainer.fit`。（注释：调用链）
备注：（注释：说明模块设计选择）
  - 该入口未与 ray_trainer 合并，因为 ray_trainer 也被其他入口复用。（注释：设计说明）
"""  # 注释：模块 docstring 结束

# 标准库导入（注释：用于路径与环境信息）
import os
# 标准库导入（注释：用于打印主机名）
import socket

# 第三方依赖（注释：Hydra 配置入口）
import hydra
# 第三方依赖（注释：Ray 分布式）
import ray
# 第三方依赖（注释：配置合并）
from omegaconf import OmegaConf

# 项目内依赖（注释：课程采样器基类）
from verl.experimental.dataset.sampler import AbstractSampler
# 项目内依赖（注释：Ray runtime_env 生成）
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
# 项目内依赖（注释：PPO/GRPO Trainer）
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# 项目内依赖（注释：奖励函数管理器）
from verl.trainer.ppo.reward import load_reward_manager
# 项目内依赖（注释：是否需要 critic/ref 的判断）
from verl.trainer.ppo.utils import need_critic, need_reference_policy
# 项目内依赖（注释：配置校验）
from verl.utils.config import validate_config
# 项目内依赖（注释：Ascend 设备自动设置与 CUDA 检查）
from verl.utils.device import auto_set_ascend_device_name, is_cuda_available
# 项目内依赖（注释：动态加载外部类）
from verl.utils.import_utils import load_extern_object


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """
    Hydra 入口函数：解析配置并启动 PPO/GRPO 训练。（注释：函数用途）

    参数：（注释：参数说明）
      - config (DictConfig): Hydra 合成后的配置对象。（注释：输入含义）
    返回：（注释：返回值说明）
      - None。（注释：入口函数不返回结果）
    副作用：（注释：副作用说明）
      - 可能修改 config.trainer.device（Ascend 自动设置）。  # 注释：说明对配置的影响
      - 启动 Ray 并触发训练流程（见 run_ppo）。  # 注释：启动训练副作用
    异常/边界条件：（注释：异常说明）
      - 配置缺失关键字段会在下游报错。  # 注释：边界条件
    最小示例：（注释：最小示例）
      >>> python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo  # CLI 入口示例（注释）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
      - 函数：`main(config)`（注释：函数名）
      典型调用路径
      ------------
      - `examples/grpo_trainer/run_qwen2-7b.sh` -> `main` -> `run_ppo`。（注释：入口链路）
      被谁调用
      --------
      - Python 模块入口（`python -m verl.trainer.main_ppo`）。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `auto_set_ascend_device_name` / `run_ppo`。（注释：内部调用）
      调用了谁（关键外部依赖）
      ----------------------
      - `hydra.main`（配置注入）。（注释：外部依赖）
    """
    # 自动检测 Ascend NPU 并设置设备名（注释：Ascend 平台适配）
    auto_set_ascend_device_name(config)

    # 进入核心训练流程（注释：调用 run_ppo）
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """
    初始化 Ray 集群并启动分布式 PPO/GRPO 训练。（注释：函数用途）

    参数：（注释：参数说明）
      - config (DictConfig): 训练配置（含 Ray 初始化、模型路径、超参等）。（注释：输入含义）
      - task_runner_class: 可选，替换默认 TaskRunner（用于 recipe/扩展）。（注释：输入含义）
    返回：（注释：返回值说明）
      - None。（注释：无返回）
    副作用：（注释：副作用说明）
      - 若 Ray 未初始化，会调用 ray.init 启动集群。（注释：副作用）
      - 启动远程 TaskRunner 并执行训练。（注释：副作用）
    异常/边界条件：（注释：异常说明）
      - NVTX 不可用且配置 nsys profiling 会断言失败。（注释：边界）
    最小示例：（注释：最小示例）
      >>> run_ppo(cfg)  # 直接启动训练（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
      - 函数：`run_ppo(config, task_runner_class=None)`（注释：函数名）
      典型调用路径
      ------------
      - `main` -> `run_ppo` -> `TaskRunner.run` -> `RayPPOTrainer.fit`。（注释：入口链路）
      被谁调用
      --------
      - `main`（本文件）。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `get_ppo_ray_runtime_env` / `TaskRunner.run`。（注释：项目内依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `ray.init` / `ray.remote` / `ray.get`。（注释：外部依赖）
    """
    # 若 Ray 尚未初始化，则进行初始化（注释：Ray 集群启动入口）
    if not ray.is_initialized():
        # 1) 读取默认 runtime_env（注释：默认环境变量集合）
        default_runtime_env = get_ppo_ray_runtime_env()
        # 2) 从配置中获取 ray.init 参数（注释：支持用户覆盖）
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # 3) 根据 transfer_queue 配置追加 env_vars（注释：启用传输队列）
        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})  # 注释：已有 env_vars
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"  # 注释：开启 transfer queue
            runtime_env_kwargs["env_vars"] = runtime_env_vars  # 注释：写回配置

        # 4) 合并 runtime_env（注释：默认值 + 用户覆盖）
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")  # 注释：打印 Ray 初始化参数
        ray.init(**OmegaConf.to_container(ray_init_kwargs))  # 注释：启动 Ray

    # 若未指定 TaskRunner 类，则使用默认远程类（注释：默认路径）
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # 注释：避免 main_task 在 head 上调度

    # 根据 profiler 配置决定是否附加 nsight options（注释：GPU profiling 分支）
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available  # 注释：按需导入 NVTX 检测

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )  # 注释：nsight 运行参数
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()  # 注释：创建 runner
    else:
        runner = task_runner_class.remote()  # 注释：普通方式创建 runner
    ray.get(runner.run.remote(config))  # 注释：阻塞等待训练完成

    # 可选：导出 Ray timeline 文件用于性能分析（注释：性能追踪）
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)  # 注释：写出 timeline JSON


class TaskRunner:
    """
    Ray 远程任务执行器：封装训练主流程，作为 Ray Actor 运行。（注释：类用途概述）

    属性：（注释：属性说明）
      - role_worker_mapping (dict): Role -> Ray Worker 类映射。（注释：用于创建 worker）
      - mapping (dict): Role -> 资源池 ID 映射。（注释：资源调度）
    输入/输出：（注释：类级输入输出）
      - 输入：通过 `run(config)` 接收完整训练配置。（注释：输入来源）
      - 输出：训练副产物（日志/ckpt）；`run` 无返回。（注释：输出形态）
    关键依赖：（注释：关键依赖）
      - Ray Actor、RayWorkerGroup、RayPPOTrainer。（注释：项目内依赖）
    最小示例：（注释：最小示例）
      >>> runner = ray.remote(TaskRunner).remote()  # 创建远程 actor（示例）
      >>> ray.get(runner.run.remote(cfg))  # 运行训练（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
      - 类：`TaskRunner`（注释：类名）
      典型调用路径
      ------------
      - `run_ppo` -> `TaskRunner.run` -> `RayPPOTrainer.fit`。（注释：调用链）
      被谁调用
      --------
      - `run_ppo`（本文件）。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `RayPPOTrainer` / `validate_config` / `load_reward_manager` 等。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `ray.remote` / `ray.get`。（注释：外部依赖）
    """

    def __init__(self):
        """初始化 TaskRunner 内部状态。（注释：构造函数用途）"""
        self.role_worker_mapping = {}  # 注释：Role -> Worker 类映射
        self.mapping = {}  # 注释：Role -> 资源池 ID 映射

    def add_actor_rollout_worker(self, config):
        """
        根据配置选择 Actor/Rollout Worker，并注册到 role_worker_mapping。（注释：函数用途）

        参数：（注释：参数说明）
          - config (DictConfig): 训练配置，包含 actor/rollout/strategy 等字段。（注释：输入含义）
        返回：（注释：返回值说明）
          - tuple: (actor_rollout_cls, ray_worker_group_cls)。（注释：返回类以供后续创建）
        副作用：（注释：副作用说明）
          - 更新 self.role_worker_mapping 与 self.mapping。（注释：写入映射）
        异常/边界条件：（注释：异常说明）
          - 未支持的 strategy 会抛 NotImplementedError。（注释：边界）
          - rollout.mode == "sync" 会抛 ValueError（已弃用）。（注释：边界）
        最小示例：（注释：最小示例）
          >>> actor_cls, wg_cls = self.add_actor_rollout_worker(cfg)  # 返回 worker 类（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
          - 方法：`TaskRunner.add_actor_rollout_worker`（注释：方法名）
          典型调用路径
          ------------
          - `TaskRunner.run` -> `add_actor_rollout_worker`。（注释：调用链）
          被谁调用
          --------
          - `TaskRunner.run`（本文件）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - `verl.workers.fsdp_workers` / `verl.workers.megatron_workers` / `verl.workers.engine_workers`。（注释：内部依赖）
          调用了谁（关键外部依赖）
          ----------------------
          - `ray.remote`。（注释：外部依赖）
        """
        # 引入 Ray worker group 与 Role 枚举（注释：后续用于映射）
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        # 读取是否启用旧版 worker 实现（注释：控制新旧引擎分支）
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # 使用新模型引擎实现（注释：Actor/Ref/ Rollout 融合）
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker  # 注释：新引擎 worker

            actor_rollout_cls = ActorRolloutRefWorker  # 注释：新引擎类
            ray_worker_group_cls = RayWorkerGroup  # 注释：Ray worker group 类
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role = Role.ActorRolloutRef  # 注释：需要 ref 时使用融合角色
            else:
                role = Role.ActorRollout  # 注释：仅 actor+rollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)  # 注释：注册远程 worker
            self.mapping[role] = "global_pool"  # 注释：绑定到全局资源池
            return actor_rollout_cls, ray_worker_group_cls  # 注释：返回类用于后续创建

        # 旧实现不支持 sync rollout（注释：强制 async）
        if config.actor_rollout_ref.rollout.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` to use the native server rollout."
            )

        # 按 actor strategy 选择 worker 类（注释：fsdp/fsdp2 分支）
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )  # 注释：async 与 sync 对应不同 worker
            ray_worker_group_cls = RayWorkerGroup  # 注释：Ray worker group 类

        # Megatron 分支（注释：使用 Megatron worker）
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )  # 注释：async 与 sync 对应不同 worker
            ray_worker_group_cls = RayWorkerGroup  # 注释：Ray worker group 类

        else:
            raise NotImplementedError  # 注释：未知策略不支持

        # 旧实现下注册 ActorRollout 角色（注释：角色映射）
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)  # 注释：注册远程 worker
        self.mapping[Role.ActorRollout] = "global_pool"  # 注释：绑定全局资源池
        return actor_rollout_cls, ray_worker_group_cls  # 注释：返回 worker 类与 group 类

    def add_critic_worker(self, config):
        """
        根据配置添加 Critic Worker 到角色映射。（注释：函数用途）

        参数：（注释：参数说明）
          - config (DictConfig): 训练配置，包含 critic.strategy 等字段。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：仅更新内部状态）
        副作用：（注释：副作用说明）
          - 更新 self.role_worker_mapping 与 self.mapping。（注释：写入映射）
        异常/边界条件：（注释：异常说明）
          - 未支持的 strategy 或 use_legacy_worker_impl 值会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> self.add_critic_worker(cfg)  # 注册 critic worker（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
          - 方法：`TaskRunner.add_critic_worker`（注释：方法名）
          典型调用路径
          ------------
          - `TaskRunner.run` -> `add_critic_worker`。（注释：调用链）
          被谁调用
          --------
          - `TaskRunner.run`（本文件）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - `verl.workers.fsdp_workers` / `verl.workers.megatron_workers` / `verl.workers.engine_workers`。（注释：内部依赖）
          调用了谁（关键外部依赖）
          ----------------------
          - `ray.remote`。（注释：外部依赖）
        """
        # 读取新旧 worker 实现开关（注释：控制不同路径）
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        # 选择 critic worker（注释：按策略分支）
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker  # 注释：旧版 FSDP critic
            elif use_legacy_worker_impl == "disable":
                # we don't need to specialize critic worker. Just use TrainingWorker
                from verl.workers.engine_workers import TrainingWorker  # 注释：新引擎统一 worker

                CriticWorker = TrainingWorker  # 注释：别名赋值
                print("Using new worker implementation")  # 注释：提示日志
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")  # 注释：非法值

        elif config.critic.strategy == "megatron":
            # TODO: switch this to TrainingWorker as well
            from verl.workers.megatron_workers import CriticWorker  # 注释：Megatron critic

        else:
            raise NotImplementedError  # 注释：未知策略不支持

        # 注册 Critic 角色（注释：写入映射）
        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)  # 注释：注册远程 worker
        self.mapping[Role.Critic] = "global_pool"  # 注释：绑定全局资源池

    def init_resource_pool_mgr(self, config):
        """
        初始化资源池管理器（ResourcePoolManager）。（注释：函数用途）

        参数：（注释：参数说明）
          - config (DictConfig): 训练配置（包含 trainer/reward_model 的资源设置）。（注释：输入含义）
        返回：（注释：返回值说明）
          - ResourcePoolManager: 资源池管理器实例。（注释：输出含义）
        副作用：（注释：副作用说明）
          - 无（仅创建对象）。（注释：无状态修改）
        异常/边界条件：（注释：异常说明）
          - reward_model 资源池参数 <=0 会抛 ValueError。（注释：边界）
        最小示例：（注释：最小示例）
          >>> rpm = self.init_resource_pool_mgr(cfg)  # 构建资源池管理器（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
          - 方法：`TaskRunner.init_resource_pool_mgr`（注释：方法名）
          典型调用路径
          ------------
          - `TaskRunner.run` -> `init_resource_pool_mgr`。（注释：调用链）
          被谁调用
          --------
          - `TaskRunner.run`（本文件）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - `verl/trainer/ppo/ray_trainer.py::ResourcePoolManager`。（注释：内部依赖）
          调用了谁（关键外部依赖）
          ----------------------
          - 无。（注释：外部依赖）
        """
        # 全局资源池 ID（注释：默认池）
        global_pool_id = "global_pool"
        # 构造资源池规格：每个节点的 GPU 数列表（注释：形如 [gpus]*nnodes）
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # TODO: 可扩展为动态注册角色（注释：未来扩展点）
        if config.reward_model.enable_resource_pool:
            # 校验 reward_pool 配置（注释：避免非法值）
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            # 构建 reward_pool（注释：奖励模型专用资源池）
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        # 创建资源池管理器（注释：在 Ray 侧用于资源分配）
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager  # 注释：返回管理器实例

    def add_reward_model_worker(self, config):
        """
        按配置添加 Reward Model Worker。（注释：函数用途）

        参数：（注释：参数说明）
          - config (DictConfig): 奖励模型相关配置。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：仅更新内部映射）
        副作用：（注释：副作用说明）
          - 更新 self.role_worker_mapping 与 self.mapping。（注释：写入映射）
        异常/边界条件：（注释：异常说明）
          - 未支持的 strategy 或 use_legacy_worker_impl 会抛异常。（注释：边界）
        最小示例：（注释：最小示例）
          >>> self.add_reward_model_worker(cfg)  # 注册奖励模型 worker（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
          - 方法：`TaskRunner.add_reward_model_worker`（注释：方法名）
          典型调用路径
          ------------
          - `TaskRunner.run` -> `add_reward_model_worker`。（注释：调用链）
          被谁调用
          --------
          - `TaskRunner.run`（本文件）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - `verl.workers.fsdp_workers` / `verl.workers.megatron_workers`。（注释：内部依赖）
          调用了谁（关键外部依赖）
          ----------------------
          - `ray.remote`。（注释：外部依赖）
        """
        from verl.trainer.ppo.ray_trainer import Role  # 注释：导入角色枚举

        # 仅在启用 reward_model 时创建 worker（注释：功能开关）
        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")  # 注释：新旧实现开关
            if use_legacy_worker_impl in ["auto", "enable", "disable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker  # 注释：FSDP 奖励模型
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker  # 注释：Megatron 奖励模型
                else:
                    raise NotImplementedError  # 注释：未知策略
            # elif use_legacy_worker_impl == "disable":
            #     from verl.workers.engine_workers import RewardModelWorker
            #
            #     print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")  # 注释：非法值

            # 注册 RewardModel 角色（注释：写入映射）
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"  # 注释：使用专用资源池
            else:
                self.mapping[Role.RewardModel] = "global_pool"  # 注释：使用全局资源池

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """
        根据配置添加参考策略（RefPolicy）worker。（注释：函数用途）

        参数：（注释：参数说明）
          - config (DictConfig): 训练配置，含 KL 相关开关。（注释：输入含义）
          - ref_policy_cls: 参考策略 worker 类（通常与 actor_rollout 类一致）。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：仅更新内部映射）
        副作用：（注释：副作用说明）
          - 更新 self.role_worker_mapping 与 self.mapping。（注释：写入映射）
        异常/边界条件：（注释：异常说明）
          - 新引擎（use_legacy_worker_impl=disable）直接返回，不单独创建 RefPolicy。（注释：边界）
        最小示例：（注释：最小示例）
          >>> self.add_ref_policy_worker(cfg, actor_cls)  # 启用 KL 时注册 ref（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
          - 方法：`TaskRunner.add_ref_policy_worker`（注释：方法名）
          典型调用路径
          ------------
          - `TaskRunner.run` -> `add_ref_policy_worker`。（注释：调用链）
          被谁调用
          --------
          - `TaskRunner.run`（本文件）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - `Role.RefPolicy` / `ray.remote`。（注释：内部依赖）
          调用了谁（关键外部依赖）
          ----------------------
          - Ray 远程 actor API。（注释：外部依赖）
        """
        from verl.trainer.ppo.ray_trainer import Role  # 注释：导入角色枚举

        # 新引擎中 ref policy 已融合到 ActorRolloutRefWorker（注释：无需单独创建）
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return  # 注释：直接返回

        # 仅在使用 KL reward 或 KL loss 时创建 RefPolicy（注释：条件开关）
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)  # 注释：注册 ref worker
            self.mapping[Role.RefPolicy] = "global_pool"  # 注释：绑定全局资源池

    def run(self, config):
        """
        执行主训练流程：构建 worker、数据集、奖励函数并启动 RayPPOTrainer。（注释：函数用途）

        参数：（注释：参数说明）
          - config (DictConfig): 完整训练配置。（注释：输入含义）
        返回：（注释：返回值说明）
          - None。（注释：训练过程无返回）
        副作用：（注释：副作用说明）
          - 会创建 Ray worker、读取模型/数据并开始训练。（注释：副作用）
        异常/边界条件：（注释：异常说明）
          - 配置不合法将触发 validate_config 报错。（注释：边界）
        最小示例：（注释：最小示例）
          >>> ray.get(ray.remote(TaskRunner).remote().run.remote(cfg))  # 远程运行（示例）
        调用路径依赖：（注释：调用关系说明）
          所在位置
          --------
          - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
          - 方法：`TaskRunner.run`（注释：方法名）
          典型调用路径
          ------------
          - `run_ppo` -> `TaskRunner.run` -> `RayPPOTrainer.fit`。（注释：调用链）
          被谁调用
          --------
          - `run_ppo`（本文件）。（注释：调用方）
          调用了谁（项目内）
          ----------------
          - `add_actor_rollout_worker` / `add_critic_worker` / `validate_config` / `RayPPOTrainer`。（注释：内部依赖）
          调用了谁（关键外部依赖）
          ----------------------
          - `ray` / `OmegaConf` / `torch`。（注释：外部依赖）
        """
        # 打印配置（resolve=True 会展开插值变量）（注释：便于调试）
        from pprint import pprint

        from omegaconf import OmegaConf  # 注释：用于解析与打印配置

        from verl.utils.fs import copy_to_local  # 注释：将远端权重拷贝到本地

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")  # 注释：打印宿主信息
        pprint(OmegaConf.to_container(config, resolve=True))  # 注释：打印解析后的配置
        OmegaConf.resolve(config)  # 注释：就地解析配置插值

        # 创建 actor/rollout worker（注释：决定训练执行角色）
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        # 创建 critic worker（注释：如启用 critic）
        self.add_critic_worker(config)

        # 创建奖励模型 worker（注释：支持规则/模型/混合奖励）
        self.add_reward_model_worker(config)

        # 如需 KL 约束，创建参考策略 worker（注释：ref policy）
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # 校验配置与角色依赖（注释：提前发现不一致）
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # 下载模型权重到本地（注释：支持 HDFS/远端路径）
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # 初始化 tokenizer / processor（注释：文本与多模态处理）
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)  # 注释：是否信任远程代码
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)  # 注释：加载 tokenizer
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # 注释：加载处理器

        # 加载奖励函数（注释：训练与验证各自一套）
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # 初始化资源池管理器（注释：Ray 资源调度）
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn  # 注释：默认 collate_fn

        # 构建训练/验证数据集（注释：RL 数据读取与处理）
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)  # 注释：构建采样器

        # 初始化 PPO/GRPO Trainer（注释：训练主流程对象）
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # 初始化分布式 worker（注释：启动 Ray worker 进程）
        trainer.init_workers()

        # 进入训练循环（注释：开始 PPO/GRPO）
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """
    根据配置创建 RL 训练/验证数据集。（注释：函数用途）

    参数：（注释：参数说明）
      - data_paths (list[str]): 数据文件路径列表。（注释：输入含义）
      - data_config (DictConfig): 数据相关配置。（注释：输入含义）
      - tokenizer: tokenizer 实例。（注释：输入含义）
      - processor: processor 实例（多模态可用，允许 None）。（注释：输入含义）
      - is_train (bool): 是否训练集（决定是否启用 datagen）。（注释：输入含义）
      - max_samples (int): 最大样本数，-1 表示不限制。（注释：输入含义）
    返回：（注释：返回值说明）
      - dataset (torch.utils.data.Dataset): 数据集实例。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 可能打印所选 dataset 类名。（注释：日志输出）
    异常/边界条件：（注释：异常说明）
      - custom_cls 若不继承 Dataset 会抛 TypeError。（注释：边界）
    最小示例：（注释：最小示例）
      >>> ds = create_rl_dataset(["train.parquet"], cfg.data, tokenizer, None)  # 返回 Dataset（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
      - 函数：`create_rl_dataset(...)`（注释：函数名）
      典型调用路径
      ------------
      - `TaskRunner.run` -> `create_rl_dataset`。（注释：入口链路）
      被谁调用
      --------
      - `TaskRunner.run` / `RayPPOTrainer._create_dataloader`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `load_extern_object` / `verl.utils.dataset.rl_dataset.RLHFDataset`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torch.utils.data.Dataset`。（注释：外部依赖）
    """
    # 依赖 Dataset 基类用于校验（注释：确保自定义数据集规范）
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset  # 注释：默认 RLHF 数据集

    # 优先使用自定义数据集类（注释：支持用户自定义）
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)  # 注释：动态加载
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )  # 注释：类型校验
    # 训练阶段可使用动态生成数据集（注释：datagen 分支）
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset  # 注释：动态生成数据集

        dataset_cls = DynamicGenDataset  # 注释：选择动态生成类
        print("Using DynamicGenDataset for data generation.")  # 注释：日志提示
    else:
        dataset_cls = RLHFDataset  # 注释：默认数据集
    print(f"Using dataset class: {dataset_cls.__name__}")  # 注释：打印最终选择

    # 实例化数据集（注释：传入路径与处理器）
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset  # 注释：返回数据集


def create_rl_sampler(data_config, dataset):
    """
    根据配置创建数据采样器（支持 curriculum / random / sequential）。（注释：函数用途）

    参数：（注释：参数说明）
      - data_config (DictConfig): 数据配置（含 sampler/shuffle 等字段）。（注释：输入含义）
      - dataset (Dataset): 数据集实例。（注释：输入含义）
    返回：（注释：返回值说明）
      - sampler (Sampler): 采样器实例。（注释：输出含义）
    副作用：（注释：副作用说明）
      - 若设置随机种子，会影响 torch 的随机状态。（注释：副作用）
    异常/边界条件：（注释：异常说明）
      - curriculum sampler 需 num_workers=0，否则断言失败。（注释：边界）
    最小示例：（注释：最小示例）
      >>> sampler = create_rl_sampler(cfg.data, dataset)  # 返回 Sampler（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/main_ppo.py`（注释：文件路径）
      - 函数：`create_rl_sampler(...)`（注释：函数名）
      典型调用路径
      ------------
      - `TaskRunner.run` -> `create_rl_sampler`。（注释：入口链路）
      被谁调用
      --------
      - `TaskRunner.run` / `RayPPOTrainer._create_dataloader`。（注释：调用方）
      调用了谁（项目内）
      ----------------
      - `load_extern_object` / `AbstractSampler`。（注释：内部依赖）
      调用了谁（关键外部依赖）
      ----------------------
      - `torchdata.stateful_dataloader.sampler.RandomSampler` / `torch.utils.data.SequentialSampler`。（注释：外部依赖）
    """
    import torch  # 注释：用于随机数生成器
    from torch.utils.data import SequentialSampler  # 注释：顺序采样器

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler  # 注释：可恢复的随机采样器

    # 1) curriculum sampler 分支（注释：支持动态课程学习）
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )  # 注释：动态加载 sampler 类
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )  # 注释：实例化 sampler
        assert isinstance(sampler, AbstractSampler)  # 注释：类型校验
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )  # 注释：避免 DataLoader 缓存影响课程排序

    # 2) shuffle=True -> RandomSampler（注释：随机采样，利于训练）
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()  # 注释：随机数生成器
        seed = data_config.get("seed")  # 注释：可选随机种子
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)  # 注释：设置随机种子
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)  # 注释：随机采样器
    else:
        # 3) shuffle=False -> 顺序采样（注释：保持原始顺序）
        sampler = SequentialSampler(data_source=dataset)  # 注释：顺序采样器

    return sampler  # 注释：返回采样器


if __name__ == "__main__":  # 注释：脚本直接执行入口
    main()  # 注释：调用 Hydra 入口
