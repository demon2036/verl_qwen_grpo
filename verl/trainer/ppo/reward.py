# Copyright 2025 Individual Contributor: Thibaut Barroyer
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
模块用途：PPO/GRPO 训练阶段的奖励管理入口，负责加载 Reward Manager、包装自定义奖励函数，并提供同步/异步 reward 计算接口。（注释：说明模块核心职责）
输入：来自 rollout 的 DataProto、训练配置 DictConfig、tokenizer、以及 reward manager 相关配置。（注释：说明模块输入形态）
输出：奖励张量（与 responses 对齐）与可选的奖励附加信息字典。（注释：说明模块输出形态）
关键依赖：ray（远程执行）、torch（张量）、multiprocessing（并发控制）、omegaconf DictConfig、verl.utils.reward_score（规则评分）、transfer_queue（可选）（注释：列出关键外部依赖）
典型用法：（注释：给出最小调用示例）
  - reward_fn = load_reward_manager(config, tokenizer, num_examine=1)
  - reward_tensor, extra = compute_reward(batch, reward_fn)
调用路径概览：（注释：说明模块在全局训练链路中的位置）
  - examples/grpo_trainer/run_qwen2-7b.sh -> verl/trainer/main_ppo.py -> verl/trainer/ppo/ray_trainer.py
    -> compute_reward(...) -> reward_fn(...) -> verl/utils/reward_score/default_compute_score -> gsm8k.compute_score
  - 可选：verl/experimental/reward_loop/reward_loop.py -> get_custom_reward_fn(...)
"""
from __future__ import annotations  # 注释：允许前置引用类型注解，减少运行时导入依赖

import inspect  # 注释：用于判断自定义奖励函数是否为协程函数
import multiprocessing  # 注释：用于创建跨进程同步原语（Semaphore）以限制 sandbox 并发
import warnings  # 注释：用于发出弃用警告
from functools import partial  # 注释：用于将额外 kwargs 绑定到奖励函数
from typing import TYPE_CHECKING, Any, Optional, cast  # 注释：类型提示与条件导入辅助

import ray  # 注释：Ray 远程执行框架，用于异步计算 reward
import torch  # 注释：用于创建/返回奖励张量

from verl.utils.reward_score import default_compute_score  # 注释：默认规则评分函数（按数据源路由）
from verl.utils.transferqueue_utils import tqbridge  # 注释：transfer queue 装饰器（可选能力）

if TYPE_CHECKING:  # 注释：仅在类型检查时导入，避免运行时强依赖
    from omegaconf import DictConfig  # 注释：训练配置类型

    from verl import DataProto  # 注释：rollout 数据容器类型
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase  # 注释：reward loop 管理器基类
    from verl.trainer.config.config import ModuleConfig, RewardManagerConfig  # 注释：奖励管理器配置类型
    from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn  # 注释：奖励管理器抽象与回调类型
else:  # 注释：运行时尝试导入可选 reward loop 基类
    try:  # 注释：reward_loop 可能不存在，需捕获 ImportError
        from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase  # 注释：reward loop 基类（可选）
    except ImportError:  # 注释：可选依赖缺失时回退
        RewardManagerBase = None  # type: ignore[assignment,misc]  # 注释：用 None 占位，后续分支判断


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):  # 注释：同步包装器，合并额外 kwargs 后调用
    """
    功能：将 extra_kwargs 合并进调用时 kwargs，并以 extra_kwargs 为准。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - raw_fn (Callable): 原始评分函数。（注释：被包装的目标函数）
      - extra_kwargs (dict): 需要强制覆盖的额外关键字参数。（注释：额外参数来源）
      - *args: 位置参数透传。（注释：保持与 raw_fn 兼容）
      - **kwargs: 原调用的关键字参数。（注释：与 extra_kwargs 合并）
    返回：（注释：返回值说明）
      - raw_fn 的返回值（原样返回）。
    副作用：（注释：副作用说明）
      - 无（仅调用 raw_fn）。
    异常/边界条件：（注释：异常与边界）
      - raw_fn 抛出的异常会原样向上抛出。（注释：异常不拦截）
      - kwargs 与 extra_kwargs 冲突时，以 extra_kwargs 覆盖。（注释：合并策略）
    最小示例：（注释：最小可理解示例）
      - 输入：raw_fn=lambda a, b=1: a + b；extra_kwargs={"b": 2}；args=(3,)
      - 输出：5
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/trainer/ppo/reward.py::_call_with_kwargs`
      - 典型调用路径：`get_custom_reward_fn` -> `partial` -> `_call_with_kwargs`
      - 被谁调用：仅本文件 `get_custom_reward_fn` 通过 `partial` 间接调用
      - 调用了谁（项目内）：无（仅调用传入 raw_fn）
      - 调用了谁（外部依赖）：无（仅调用传入 raw_fn）
    """
    merged_kwargs = {**kwargs, **extra_kwargs}  # 注释：合并调用时 kwargs，extra_kwargs 覆盖同名键
    return raw_fn(*args, **merged_kwargs)  # 注释：调用原始函数并返回结果


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):  # 注释：异步包装器，合并 kwargs 后 await 调用
    """
    功能：异步版本的 kwargs 合并调用器。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - raw_fn (Callable): 原始异步评分函数。（注释：需要 await 的协程函数）
      - extra_kwargs (dict): 需要强制覆盖的额外关键字参数。（注释：额外参数来源）
      - *args: 位置参数透传。（注释：保持与 raw_fn 兼容）
      - **kwargs: 原调用的关键字参数。（注释：与 extra_kwargs 合并）
    返回：（注释：返回值说明）
      - raw_fn 的 await 结果（原样返回）。
    副作用：（注释：副作用说明）
      - 无（仅 await raw_fn）。
    异常/边界条件：（注释：异常与边界）
      - raw_fn 抛出的异常会原样向上抛出。（注释：异常不拦截）
      - kwargs 与 extra_kwargs 冲突时，以 extra_kwargs 覆盖。（注释：合并策略）
    最小示例：（注释：最小可理解示例）
      - 输入：raw_fn=async lambda a, b=1: a + b；extra_kwargs={"b": 2}；args=(3,)
      - 输出：5
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/trainer/ppo/reward.py::_call_with_kwargs_async`
      - 典型调用路径：`get_custom_reward_fn` -> `partial` -> `_call_with_kwargs_async`
      - 被谁调用：仅本文件 `get_custom_reward_fn` 通过 `partial` 间接调用
      - 调用了谁（项目内）：无（仅调用传入 raw_fn）
      - 调用了谁（外部依赖）：无（仅调用传入 raw_fn）
    """
    merged_kwargs = {**kwargs, **extra_kwargs}  # 注释：合并调用时 kwargs，extra_kwargs 覆盖同名键
    return await raw_fn(*args, **merged_kwargs)  # 注释：await 原始协程函数并返回结果


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:  # 注释：从外部文件加载自定义奖励函数
    """
    功能：从配置指定的外部文件加载奖励函数，并合并配置中的 reward_kwargs。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - config (DictConfig): PPO/GRPO 训练配置，需包含 custom_reward_function 子段。（注释：输入配置来源）
    返回：（注释：返回值说明）
      - RawRewardFn 或 None：存在配置则返回包装后的函数，否则返回 None。（注释：返回类型说明）
    副作用：（注释：副作用说明）
      - 动态导入外部模块（可能触发文件 I/O）。（注释：动态加载副作用）
    异常/边界条件：（注释：异常与边界）
      - FileNotFoundError/RuntimeError/AttributeError 可能由 load_extern_object 抛出。（注释：外部加载失败）
      - 当 custom_reward_function.path 未配置时直接返回 None。（注释：空配置边界）
    最小示例：（注释：最小可理解示例）
      - 输入：config.custom_reward_function={"path": "/tmp/my_reward.py", "name": "score", "reward_kwargs": {"k": 1}}
      - 输出：可调用函数，调用时会自动附带 k=1
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/trainer/ppo/reward.py::get_custom_reward_fn`
      - 典型调用路径：`verl/trainer/main_ppo.py` -> `load_reward_manager` -> `get_custom_reward_fn`
      - 被谁调用：`verl/trainer/ppo/reward.py::load_reward_manager`、`verl/experimental/reward_loop/reward_loop.py`、
        `verl/trainer/main_eval.py`、`recipe/*/main_*.py`（如 recipe/r1/main_eval.py）
      - 调用了谁（项目内）：`verl/utils/import_utils.load_extern_object`、`_call_with_kwargs`/`_call_with_kwargs_async`
      - 调用了谁（外部依赖）：`inspect.iscoroutinefunction`
    """
    reward_fn_config = config.get("custom_reward_function") or {}  # 注释：读取自定义奖励配置，若无则用空 dict
    module_path = reward_fn_config.get("path")  # 注释：外部模块路径（文件路径）
    if not module_path:  # 注释：未配置路径则直接返回 None
        return None  # 注释：表示不启用自定义奖励函数

    fn_name = reward_fn_config.get("name")  # 注释：读取函数名
    assert fn_name is not None  # 注释：函数名必须存在，否则属于配置错误

    from verl.utils.import_utils import load_extern_object  # 注释：延迟导入以减少启动依赖

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)  # 注释：动态加载外部函数对象

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))  # 注释：读取需要注入的额外 kwargs
    if not inspect.iscoroutinefunction(raw_fn):  # 注释：同步函数走同步包装器
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)  # 注释：绑定 kwargs 并返回可调用对象
    else:  # 注释：异步函数走异步包装器
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)  # 注释：绑定 kwargs 并返回可调用对象


def load_reward_manager(  # 注释：根据配置加载并实例化 RewardManager
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:  # 注释：返回具体 RewardManager 实例
    """
    功能：根据配置选择 RewardManager 类，构造 compute_score，并实例化返回。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - config (DictConfig): PPO/GRPO 训练配置，含 reward_manager/reward_model 子段。（注释：配置来源）
      - tokenizer (Any): 用于解码的 tokenizer。（注释：用于 reward 计算时还原文本）
      - num_examine (int): 需要打印调试样本的数量。（注释：调试控制）
      - **reward_kwargs: 额外透传给 RewardManager 的关键字参数。（注释：可扩展参数）
    返回：（注释：返回值说明）
      - AbstractRewardManager 子类实例。（注释：具体 reward manager）
    副作用：（注释：副作用说明）
      - 可能启动 multiprocessing.Manager 用于并发控制。（注释：进程/资源副作用）
      - 动态导入 RewardManager 类或外部函数。（注释：动态导入副作用）
    异常/边界条件：（注释：异常与边界）
      - 配置缺失 module.path 时会触发断言。（注释：配置校验）
      - unknown reward manager 名称会由注册表抛出 ValueError。（注释：注册表错误）
    最小示例：（注释：最小可理解示例）
      - 输入：reward_manager.source="register"，reward_manager.name="naive"
      - 输出：NaiveRewardManager 实例（compute_score 使用 default_compute_score）
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/trainer/ppo/reward.py::load_reward_manager`
      - 典型调用路径：`examples/grpo_trainer/run_qwen2-7b.sh` -> `verl/trainer/main_ppo.py` -> `load_reward_manager`
      - 被谁调用：`verl/trainer/main_ppo.py`、`recipe/*/main_*.py`（如 recipe/dapo/main_dapo.py）、
        `recipe/fully_async_policy/fully_async_trainer.py`、`tests/experimental/agent_loop/test_basic_agent_loop.py`
      - 调用了谁（项目内）：`get_custom_reward_fn`、`verl/workers/reward_manager.get_reward_manager_cls`、
        `verl/utils.import_utils.load_extern_object`、`verl/utils.reward_score.default_compute_score`
      - 调用了谁（外部依赖）：`multiprocessing.Manager`
    """

    # 先尝试加载用户自定义奖励函数  # 注释：优先使用配置指定的自定义 reward 逻辑
    # 自定义奖励函数可通过 custom_reward_function 配置注册  # 注释：说明配置来源
    compute_score = get_custom_reward_fn(config)  # 注释：尝试从外部加载 custom reward fn
    final_compute_score = compute_score  # 注释：默认使用 custom reward fn（若存在）

    reward_manager_cfg: RewardManagerConfig = config.reward_manager  # 注释：读取 reward_manager 配置段
    reward_manager_cls: type[AbstractRewardManager]  # 注释：类型提示：要实例化的 RewardManager 类
    if reward_manager_cfg.source == "register":  # 注释：从注册表获取 RewardManager 类
        from verl.workers.reward_manager import get_reward_manager_cls  # 注释：注册表查询函数

        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)  # 注释：按名称取出类
    elif reward_manager_cfg.source == "importlib":  # 注释：从外部模块动态加载 RewardManager 类
        from verl.utils.import_utils import load_extern_object  # 注释：延迟导入动态加载工具

        module_cfg: ModuleConfig | None = reward_manager_cfg.module  # 注释：读取 module 配置
        assert module_cfg is not None and module_cfg.path is not None, (  # 注释：必须提供 module.path
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        reward_manager_cls_name = reward_manager_cfg.name  # 注释：类名字符串
        reward_manager_cls = cast(  # 注释：类型转换，告诉类型检查器返回值类型
            "type[AbstractRewardManager]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cls_name),
        )

    if compute_score is None:  # 注释：若未配置自定义奖励函数，则构造默认评分函数
        sandbox_config = config.reward_model.get("sandbox_fusion")  # 注释：读取 sandbox 配置（用于代码题）
        sandbox_url = sandbox_config.get("url") if sandbox_config else None  # 注释：sandbox 服务 URL
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024  # 注释：内存限制
        if sandbox_url:  # 注释：启用 sandbox fusion 时加并发控制
            sandbox_manager = multiprocessing.Manager()  # 注释：创建进程级 Manager
            # 创建信号量限制并发请求  # 注释：避免 sandbox 被过载
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))  # 注释：并发上限
            final_compute_score = partial(  # 注释：绑定 sandbox 参数到默认评分函数
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:  # 注释：无 sandbox 配置则直接使用默认评分函数
            final_compute_score = default_compute_score  # 注释：默认评分逻辑

    # 按 reward manager 类型实例化并返回  # 注释：构造完成后直接返回
    # RewardManagerBase 子类（如 RateLimitedRewardLoopManager）不接受 num_examine  # 注释：reward loop 版本签名不同
    # AbstractRewardManager 子类（如 NaiveRewardManager）需要 num_examine  # 注释：传统 reward manager 参数需求
    if RewardManagerBase is not None and issubclass(reward_manager_cls, RewardManagerBase):  # 注释：reward loop 管理器分支
        # RewardManagerBase 管理器使用不同的参数签名  # 注释：reward loop 分支说明
        return reward_manager_cls(  # 注释：构造 reward loop 管理器
            config=config,
            tokenizer=tokenizer,
            compute_score=final_compute_score,
            **reward_kwargs,
        )
    else:  # 注释：传统 AbstractRewardManager 分支
        # 传统 AbstractRewardManager 构造方式  # 注释：非 reward loop 分支说明
        return reward_manager_cls(  # 注释：构造传统 reward manager
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=final_compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )


@tqbridge(put_data=False)  # 注释：transfer queue 装饰器（可能包装 DataProto 传输逻辑）
def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:  # 注释：同步计算 reward
    """
    功能：对一个 batch 的 DataProto 计算奖励，并返回 reward 张量与附加信息。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - data (DataProto): 包含 prompts/responses/metadata 的 batch。（注释：输入批次）
      - reward_fn (AbstractRewardManager): 用于计算奖励的 reward manager。（注释：评分器）
    返回：（注释：返回值说明）
      - (reward_tensor, reward_extra_info_dict)：奖励张量与额外信息字典。（注释：返回结构）
    副作用：（注释：副作用说明）
      - 当 reward_fn(return_dict=True) 失败时，会打印错误并回退到 reward_fn(data)。（注释：日志副作用）
    异常/边界条件：（注释：异常与边界）
      - reward_fn 内部异常会在第二次调用中继续抛出或返回异常。（注释：异常传播）
      - return_dict=True 不被支持时触发 except 分支。（注释：兼容旧接口）
    最小示例：（注释：最小可理解示例）
      - 输入：reward_fn 返回 {"reward_tensor": tensor([[0,1]]), "reward_extra_info": {"score":[1]}}
      - 输出：reward_tensor=tensor([[0,1]]), reward_extra_info_dict={"score":[1]}
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/trainer/ppo/reward.py::compute_reward`
      - 典型调用路径：`verl/trainer/ppo/ray_trainer.py` -> `compute_reward`
      - 被谁调用：`verl/trainer/ppo/ray_trainer.py`、`recipe/*/ray_trainer.py`（如 recipe/dapo/dapo_ray_trainer.py）、
        `recipe/fully_async_policy/ray_trainer.py`、`tests/experimental/agent_loop/test_basic_agent_loop.py`
      - 调用了谁（项目内）：`AbstractRewardManager.__call__`
      - 调用了谁（外部依赖）：无（仅调用 reward_fn）
    """
    try:  # 注释：优先尝试使用 return_dict=True 的新接口
        reward_result = reward_fn(data, return_dict=True)  # 注释：期望返回字典形式
        reward_tensor = reward_result["reward_tensor"]  # 注释：提取 reward 张量
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})  # 注释：提取额外信息（若有）
    except Exception as e:  # 注释：兼容旧接口或 reward_fn 未实现 return_dict
        print(f"Error in reward_fn: {e}")  # 注释：打印错误，提示回退策略
        reward_tensor = reward_fn(data)  # 注释：回退到旧接口（仅返回张量）
        reward_extra_infos_dict = {}  # 注释：旧接口无额外信息

    return reward_tensor, reward_extra_infos_dict  # 注释：返回奖励张量与附加信息


@ray.remote(num_cpus=1)  # 注释：将函数注册为 Ray 远程任务，分配 1 CPU
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):  # 注释：异步 reward 计算入口
    """
    功能：在 Ray worker 中加载 reward manager（如需要）并计算 reward。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - data (DataProto): batch 数据。（注释：输入批次）
      - config (DictConfig|None): 当 reward_fn=None 时用于构建 reward manager。（注释：配置）
      - tokenizer (Any|None): 当 reward_fn=None 时用于构建 reward manager。（注释：分词器）
      - reward_fn (AbstractRewardManager|None): 直接传入的 reward manager。（注释：评分器）
    返回：（注释：返回值说明）
      - 与 compute_reward 相同：(reward_tensor, reward_extra_info_dict)。（注释：返回结构）
    副作用：（注释：副作用说明）
      - 若 reward_fn 为空则会发出弃用警告并动态加载 reward manager。（注释：警告与动态加载）
    异常/边界条件：（注释：异常与边界）
      - reward_fn 与 config/tokenizer 同时为空会触发断言。（注释：参数约束）
      - reward_fn 内部异常可能向上抛出。（注释：异常传播）
    最小示例：（注释：最小可理解示例）
      - 输入：compute_reward_async.remote(batch, reward_fn=my_reward_fn)
      - 输出：Ray ObjectRef，解析后得到 (reward_tensor, reward_extra_info_dict)
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/trainer/ppo/reward.py::compute_reward_async`
      - 典型调用路径：`verl/trainer/ppo/ray_trainer.py` -> `compute_reward_async.remote(...)`
      - 被谁调用：`verl/trainer/ppo/ray_trainer.py`、`recipe/one_step_off_policy/ray_trainer.py`、`recipe/sppo/sppo_ray_trainer.py`
      - 调用了谁（项目内）：`load_reward_manager`、`compute_reward`
      - 调用了谁（外部依赖）：`warnings.warn`
    """
    if reward_fn is None:  # 注释：未提供 reward_fn 时按旧逻辑构建
        assert config is not None and tokenizer is not None, (  # 注释：确保必要配置存在
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)  # 注释：弃用提示
        reward_fn = load_reward_manager(  # 注释：动态加载 reward manager
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)  # 注释：复用同步逻辑计算 reward
