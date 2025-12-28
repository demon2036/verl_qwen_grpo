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
模块用途：提供 FSDP/FSDP2 相关的包装策略、参数迁移、分片加载与 LoRA 收集工具。（注释：模块职责）
输入/输出：
  - 输入：PyTorch 模型、FSDP 包装模块、分布式进程组与 checkpoint 路径。（注释：输入说明）
  - 输出：wrap policy、状态字典、CPU/GPU 迁移后的模型/优化器等。（注释：输出说明）
关键依赖：torch.distributed.fsdp、torch.distributed.checkpoint、transformers。（注释：依赖说明）
典型用法（最小示例）：
  - `auto_wrap_policy = get_fsdp_wrap_policy(model, config)`。（注释：wrap 策略）
  - `offload_fsdp_model_to_cpu(model)`。（注释：模型卸载）
调用路径概览：
  - `verl/trainer/fsdp_sft_trainer.py`/`verl/workers/fsdp_workers.py` -> 本模块函数。（注释：典型入口）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import functools  # 注释：函数式工具
import itertools  # 注释：迭代器工具
import json  # 注释：加载 safetensors 索引
import math  # 注释：向上取整等计算
import os  # 注释：路径操作
from abc import ABC  # 注释：抽象基类
from collections import OrderedDict  # 注释：有序字典
from contextlib import contextmanager, nullcontext  # 注释：上下文管理器

# ===== 第三方依赖导入 =====
import torch  # 注释：PyTorch
import torch.distributed as dist  # 注释：分布式通信
import torch.nn as nn  # 注释：神经网络模块
from packaging import version  # 注释：版本比较
from torch.distributed import DeviceMesh  # 注释：设备网格
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 注释：FSDP 封装
from torch.distributed.fsdp._runtime_utils import _lazy_init  # 注释：FSDP 延迟初始化
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy  # 注释：wrap 策略
from transformers.trainer_pt_utils import get_module_class_from_name  # 注释：根据名称获取模块类

# ===== 项目内依赖导入 =====
from verl.utils.device import get_device_id, get_device_name, get_torch_device  # 注释：设备工具
from verl.utils.model import check_exclude_modules, check_target_modules  # 注释：LoRA 目标模块筛选

if version.parse(torch.__version__) >= version.parse("2.6"):  # 注释：torch>=2.6 使用新 FSDP2 API
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard  # 注释：FSDP2 组件
    from torch.distributed.tensor import Shard  # 注释：DTensor 分片描述

    fully_shard_module = torch.distributed.fsdp._fully_shard._fully_shard  # 注释：内部 fully_shard 模块
elif version.parse(torch.__version__) >= version.parse("2.4"):  # 注释：torch>=2.4 但 <2.6 的可组合 API
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard  # 注释：可组合 FSDP2

    fully_shard_module = torch.distributed._composable.fsdp.fully_shard  # 注释：内部 fully_shard 模块
else:  # 注释：低版本不支持 FSDP2
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy, fully_shard_module = None, None, None, None, None


def init_fn(x: torch.nn.Module):
    """
    FSDP 初始化辅助函数：非 rank0 使用 empty 参数创建以节省显存。（注释：函数用途）

    参数：
      - x (torch.nn.Module)：待初始化的模块。（注释：输入说明）
    返回：
      - torch.nn.Module：可能被移动到 empty 参数的模块。（注释：返回说明）
    副作用：非 rank0 时释放显存缓存。（注释：副作用）
    异常/边界条件：依赖已初始化的 torch.distributed。（注释：边界条件）
    最小示例：
      - 输入：rank!=0 时 module 参数被 to_empty。（注释：示例输入）
      - 输出：节省显存的 module。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::init_fn`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `init_fn`。（注释：典型路径）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`recipe/*` 等导入使用。（注释：调用方）
      - 调用了谁（项目内）：`get_device_id`、`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.get_rank`、`Module.to_empty`。（注释：外部依赖）
    """
    if torch.distributed.get_rank() != 0:  # 注释：非 rank0 进行 empty 初始化
        x = x.to_empty(device=get_device_id(), recurse=False)  # 注释：仅在本层创建空参数
        get_torch_device().empty_cache()  # 注释：清理缓存
    return x  # 注释：返回模块


def get_init_weight_context_manager(use_meta_tensor=True, mesh: DeviceMesh = None):
    """
    获取权重初始化的上下文管理器（meta 或 CPU）。（注释：函数用途）

    参数：
      - use_meta_tensor (bool)：是否使用 meta 初始化。（注释：输入说明）
      - mesh (DeviceMesh, optional)：若提供，按 mesh coordinate 决定 rank。（注释：输入说明）
    返回：
      - context manager：用于模型权重初始化。（注释：返回说明）
    副作用：无。（注释：纯选择逻辑）
    异常/边界条件：依赖 accelerate.init_empty_weights。（注释：边界条件）
    最小示例：
      - 输入：use_meta_tensor=True 且 rank!=0。（注释：示例输入）
      - 输出：init_empty_weights。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::get_init_weight_context_manager`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `get_init_weight_context_manager`。（注释：典型路径）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`recipe/*`。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`accelerate.init_empty_weights`。（注释：外部依赖）
    """
    from accelerate import init_empty_weights  # 注释：延迟导入 accelerate

    cpu_init_weights = lambda: torch.device("cpu")  # 注释：CPU 初始化上下文
    if use_meta_tensor:  # 注释：选择 meta 初始化
        if mesh is None:  # 注释：无 mesh 时按全局 rank 判断
            init_context = init_empty_weights if torch.distributed.get_rank() != 0 else cpu_init_weights
        else:  # 注释：有 mesh 时按 mesh 坐标判断
            init_context = init_empty_weights if mesh.get_coordinate()[-1] != 0 else cpu_init_weights
    else:
        init_context = cpu_init_weights  # 注释：不使用 meta，固定 CPU
    return init_context  # 注释：返回上下文管理器


# Copyright 2020-present the HuggingFace Inc. team.
# Adapted from https://github.com/huggingface/transformers/src/transformers/trainer.py
def get_fsdp_wrap_policy(module, config=None, is_lora=False):
    """
    生成 FSDP 自动包裹策略（wrap policy）。（注释：函数用途）

    参数：
      - module：待包裹的模型。（注释：输入说明）
      - config：wrap policy 配置。（注释：输入说明）
      - is_lora (bool)：是否启用 LoRA 的 lambda 策略。（注释：输入说明）
    返回：
      - callable|None：wrap policy 函数，或 None 表示不启用。（注释：返回说明）
    副作用：无。（注释：纯配置计算）
    异常/边界条件：
      - 找不到指定层类名时抛异常。（注释：边界条件）
    最小示例：
      - 输入：config.min_num_params=1e6。（注释：示例输入）
      - 输出：size_based_auto_wrap_policy。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::get_fsdp_wrap_policy`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `get_fsdp_wrap_policy`。（注释：典型路径）
      - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py`、`recipe/*`。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.fsdp.wrap`、`transformers.trainer_pt_utils`。（注释：外部依赖）
    """
    if config is None:  # 注释：默认空配置
        config = {}

    # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this
    # once we have make all config in verl from OmegaConf to data class.
    def _get_attr(attr_name, default_value=None):
        """兼容 OmegaConf/dict/dataclass 的属性读取。（注释：内部函数用途）"""
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)  # 注释：dict/OmegaConf 读取
        else:
            return config.__getattribute__(attr_name)  # 注释：dataclass 读取

    if _get_attr("disable", False):  # 注释：配置禁用则返回 None
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)  # 注释：默认不切分层
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )  # 注释：读取配置的包裹层类名
    min_num_params = _get_attr("min_num_params", 0)  # 注释：按参数量包裹的阈值
    auto_wrap_policy = None  # 注释：最终 policy

    policies = []  # 注释：候选策略列表

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy  # 注释：FSDP wrap 工具

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:  # 注释：LoRA 时添加 lambda policy

        def lambda_policy_fn(module):
            """LoRA 包裹策略：仅包裹叶子参数且可训练的权重。（注释：内部函数用途）"""
            return bool(
                len(list(module.named_children())) == 0
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)  # 注释：构造 lambda policy
        policies.append(lambda_policy)  # 注释：加入策略列表

    if min_num_params > 0:  # 注释：优先使用按参数量的策略
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:  # 注释：否则按 transformer 层包裹
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:  # 注释：合并多个策略
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy  # 注释：返回 wrap policy


@torch.no_grad()
def offload_fsdp_model_to_cpu(model: FSDP, empty_cache: bool = True):
    """
    将 FSDP1 模型参数卸载到 CPU（FSDP2 走专用路径）。（注释：函数用途）

    参数：
      - model (FSDP)：FSDP 包装模型。（注释：输入说明）
      - empty_cache (bool)：是否清空显存缓存。（注释：输入说明）
    返回：无。（注释：仅副作用）
    副作用：模型参数移动到 CPU；可能清理显存。（注释：副作用）
    异常/边界条件：仅支持 root FSDP 模型。（注释：边界条件）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::offload_fsdp_model_to_cpu`。（注释：定位）
      - 典型调用路径：`recipe/sppo/sppo_worker.py` -> `offload_fsdp_model_to_cpu`。（注释：典型路径）
      - 被谁调用：`recipe/sppo/sppo_worker.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`fsdp_version`、`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.fsdp._lazy_init`。（注释：外部依赖）
    """
    if fsdp_version(model) == 2:  # 注释：FSDP2 走专用接口
        offload_fsdp2_model_to_cpu(model, empty_cache)
        return

    assert isinstance(model, FSDP)  # 注释：确保为 FSDP1
    # lazy init FSDP model
    _lazy_init(model, model)  # 注释：确保句柄初始化
    assert model._is_root, "Only support root model offloading to CPU"  # 注释：仅支持 root
    for handle in model._all_handles:  # 注释：遍历所有 handle
        if handle._offload_params:  # 注释：已配置 offload 则跳过
            continue
        flat_param = handle.flat_param  # 注释：当前扁平参数
        assert (
            flat_param.data.data_ptr() == flat_param._local_shard.data_ptr()
            and id(flat_param.data) != id(flat_param._local_shard)
            and flat_param.data.size() == flat_param._local_shard.size()
        )  # 注释：确认扁平参数与本地分片一致
        handle.flat_param_to(torch.device("cpu"), non_blocking=True)  # 注释：迁移到 CPU
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data  # 注释：更新本地分片引用
        assert id(flat_param._local_shard) != id(flat_param.data)  # 注释：保证引用不同
    if empty_cache:  # 注释：按需清理缓存
        get_torch_device().empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True):
    """
    将 FSDP2 模型卸载到 CPU。（注释：函数用途）

    参数：model/empty_cache。（注释：输入说明）
    返回：无。（注释：仅副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::offload_fsdp2_model_to_cpu`。（注释：定位）
      - 被谁调用：`offload_fsdp_model_to_cpu`。（注释：调用方）
      - 调用了谁（项目内）：`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.nn.Module.cpu`。（注释：外部依赖）
    """
    model.cpu()  # 注释：直接移动到 CPU
    if empty_cache:  # 注释：按需清理缓存
        get_torch_device().empty_cache()


@torch.no_grad()
def load_fsdp_model_to_gpu(model: FSDP):
    """
    将 FSDP1 模型加载回 GPU（FSDP2 走专用路径）。（注释：函数用途）

    参数：model（FSDP）。（注释：输入说明）
    返回：无。（注释：仅副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::load_fsdp_model_to_gpu`。（注释：定位）
      - 被谁调用：`verl/utils/fsdp_utils.py` 内部或外部工具。（注释：调用方）
      - 调用了谁（项目内）：`fsdp_version`、`get_device_id`、`get_device_name`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.fsdp._lazy_init`。（注释：外部依赖）
    """
    if fsdp_version(model) == 2:  # 注释：FSDP2 走专用接口
        load_fsdp2_model_to_gpu(model)
        return

    assert isinstance(model, FSDP)  # 注释：确保为 FSDP1
    # lazy init FSDP model
    _lazy_init(model, model)  # 注释：确保句柄初始化
    assert model._is_root, "Only support root model loading to GPU"  # 注释：仅支持 root
    device_id = get_device_id()  # 注释：获取设备 ID
    for handle in model._all_handles:  # 注释：遍历 handle
        if handle._offload_params:  # 注释：已配置 offload 则跳过
            continue
        flat_param = handle.flat_param  # 注释：扁平参数
        handle.flat_param_to(torch.device(f"{get_device_name()}:{device_id}"), non_blocking=True)  # 注释：迁移到 GPU
        # the following still keeps id(._local_shard) != id(.data)
        flat_param._local_shard = flat_param.data  # 注释：更新本地分片引用


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    """
    将 FSDP2 模型加载到 GPU。（注释：函数用途）

    参数：model。（注释：输入说明）
    返回：无。（注释：仅副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::load_fsdp2_model_to_gpu`。（注释：定位）
      - 被谁调用：`load_fsdp_model_to_gpu`。（注释：调用方）
      - 调用了谁（项目内）：`get_device_id`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.nn.Module.to`。（注释：外部依赖）
    """
    device = get_device_id()  # 注释：获取设备 ID
    model.to(device)  # 注释：移动到设备


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    """
    将优化器状态张量迁移到 CPU。（注释：函数用途）

    参数：
      - optimizer：优化器实例。（注释：输入说明）
    返回：无。（注释：仅副作用）
    副作用：优化器 state 中的张量被移动到 CPU。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::offload_fsdp_optimizer`。（注释：定位）
      - 被谁调用：`recipe/sppo/sppo_worker.py` 等。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.Tensor.to`。（注释：外部依赖）
    """
    if not optimizer.state:  # 注释：无状态则直接返回
        return
    for param_group in optimizer.param_groups:  # 注释：遍历参数组
        for param in param_group["params"]:  # 注释：遍历参数
            state = optimizer.state[param]  # 注释：获取 state
            for key, value in state.items():  # 注释：遍历 state 条目
                if isinstance(value, torch.Tensor):  # 注释：仅处理张量
                    state[key] = value.to("cpu", non_blocking=True)  # 注释：迁移到 CPU


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    """
    将优化器状态张量迁移回指定设备。（注释：函数用途）

    参数：
      - optimizer：优化器实例。（注释：输入说明）
      - device_id：目标设备 ID。（注释：输入说明）
    返回：无。（注释：仅副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::load_fsdp_optimizer`。（注释：定位）
      - 被谁调用：`recipe/sppo/sppo_worker.py` 等。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.Tensor.to`。（注释：外部依赖）
    """
    if not optimizer.state:  # 注释：无状态则直接返回
        return
    for param_group in optimizer.param_groups:  # 注释：遍历参数组
        for param in param_group["params"]:  # 注释：遍历参数
            state = optimizer.state[param]  # 注释：获取 state
            for key, value in state.items():  # 注释：遍历 state 条目
                if isinstance(value, torch.Tensor):  # 注释：仅处理张量
                    state[key] = value.to(device_id, non_blocking=True)  # 注释：迁移到设备


@contextmanager
def meta_device_init():
    """
    使用 meta device 创建模型参数（节省内存）。（注释：函数用途）

    说明：buffer 仍在默认设备初始化，避免 meta 无法表达实际值。（注释：行为说明）
    返回：上下文管理器。（注释：返回说明）
    副作用：临时替换 `nn.Module.register_parameter`。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::meta_device_init`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用（可供外部使用）。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.device("meta")`、`nn.Module.register_parameter`。（注释：外部依赖）
    """
    device = torch.device("meta")  # 注释：meta 设备
    old_register_parameter = nn.Module.register_parameter  # 注释：保存原始方法
    registered = set()  # 注释：记录已替换参数，避免共享参数重复处理

    def register_empty_parameter(module, name, param):
        """注册参数并将其转换为 meta 参数。（注释：内部函数用途）"""
        old_register_parameter(module, name, param)  # 注释：先调用原始注册逻辑
        # we will skip register shared parameters as it
        # is already registered previously
        if param is not None and param not in registered:  # 注释：避免处理共享参数
            param_cls = type(module._parameters[name])  # 注释：参数类
            kwargs = module._parameters[name].__dict__  # 注释：参数属性
            kwargs["requires_grad"] = param.requires_grad  # 注释：保持 requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)  # 注释：替换为 meta 参数
            registered.add(module._parameters[name])  # 注释：记录已处理

    try:
        nn.Module.register_parameter = register_empty_parameter  # 注释：替换注册方法
        yield  # 注释：进入上下文
    finally:
        registered.clear()  # 注释：清理记录
        nn.Module.register_parameter = old_register_parameter  # 注释：恢复原始方法


def parallel_load_safetensors(filepath):
    """
    并行加载 HuggingFace safetensors checkpoint。（注释：函数用途）

    说明：每个 rank 只加载自己负责的分片文件，其它参数记录为拥有者 rank。（注释：行为说明）
    返回：dict，key 为参数名，value 为 Tensor 或拥有该参数的 rank。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::parallel_load_safetensors`。（注释：定位）
      - 被谁调用：`parallel_init_module_fn` 生成的初始化流程。（注释：调用方）
      - 调用了谁（项目内）：`get_device_id`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`safetensors.torch.load_file`。（注释：外部依赖）
    """
    from safetensors.torch import load_file  # 注释：延迟导入 safetensors

    safetensors2param = {}  # 注释：文件 -> 参数名列表

    index_file = os.path.join(filepath, "model.safetensors.index.json")  # 注释：索引文件
    if os.path.exists(index_file):  # 注释：分片索引存在
        index = json.load(open(index_file, "rb"))  # 注释：读取索引
        for param_name, filename in index["weight_map"].items():
            safetensors2param.setdefault(filename, []).append(param_name)  # 注释：建立映射
    else:
        # in this case, the model is small and we can load it all at once
        param_file = os.path.join(filepath, "model.safetensors")
        assert os.path.exists(param_file), f"Cannot find {param_file}"
        states = load_file(param_file)  # 注释：加载完整文件
        for param_name in states:
            safetensors2param.setdefault("model.safetensors", []).append(param_name)
        del states  # 注释：释放内存

    total_files = len(safetensors2param)  # 注释：总分片数
    ckpt_chunks = sorted(safetensors2param.keys())  # 注释：分片文件名排序
    world_size = dist.get_world_size()  # 注释：world size
    size = int(math.ceil(total_files / world_size))  # 注释：每 rank 处理文件数
    ckpt_chunks = [ckpt_chunks[rank * size : rank * size + size] for rank in range(world_size)]  # 注释：按 rank 切分

    shard_states = {}  # 注释：本 rank 加载的状态字典
    device = get_device_id()  # 注释：当前设备 ID
    for rank, files in enumerate(ckpt_chunks):  # 注释：遍历每个 rank 的文件列表
        if rank == dist.get_rank():  # 注释：本 rank 负责加载
            for file in files:
                file = os.path.join(filepath, file)
                states = load_file(file, device=device)  # 注释：加载到设备
                # print(f"rank {rank} loading {file}...")
                shard_states.update(states)  # 注释：合并到状态字典
        else:  # 注释：其它 rank 的参数仅记录其拥有者
            for file in files:
                for param_name in safetensors2param[file]:
                    shard_states[param_name] = rank
    return shard_states  # 注释：返回状态映射


def parallel_init_module_fn(module: torch.nn.Module, shard_states: dict[str, torch.nn.Parameter]):
    """
    生成用于初始化子模块参数的函数（基于分片 checkpoint）。（注释：函数用途）

    参数：
      - module：全局模型。（注释：输入说明）
      - shard_states：分片参数字典（参数名 -> Tensor 或 rank）。（注释：输入说明）
    返回：
      - init_fn：用于递归初始化子模块的函数。（注释：返回说明）
    副作用：会修改模块参数/缓冲区对象。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::parallel_init_module_fn`。（注释：定位）
      - 被谁调用：通常与 `parallel_load_safetensors` 配合使用。（注释：调用方）
      - 调用了谁（项目内）：`get_device_id`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.broadcast`。（注释：外部依赖）
    """

    state2fqn = {}  # 注释：state 对象 -> FQN 列表
    for name, state in itertools.chain(
        module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)
    ):
        state2fqn.setdefault(state, []).append(name)  # 注释：记录共享参数/缓冲区
    # remove standalone parameters and buffers
    shared = {s for s, names in state2fqn.items() if len(names) > 1}  # 注释：共享参数集合
    materialized_states = {}  # 注释：已实例化的共享参数

    @torch.no_grad()
    def create_and_sync_state(param_name, state, is_param):
        """创建参数/缓冲区并在进程间同步。（注释：内部函数用途）"""
        assert param_name in shard_states, f"{param_name} not loaded"  # 注释：必须在 shard_states 中
        device = get_device_id()  # 注释：当前设备
        if is_param:
            param = torch.nn.Parameter(torch.empty_like(state.data, device=device), requires_grad=state.requires_grad)
        else:  # buffer
            param = torch.empty_like(state.data, device=device)
        loaded = shard_states[param_name]  # 注释：加载到的 shard 或 rank
        if isinstance(loaded, torch.nn.Parameter | torch.Tensor):
            # NOTE: loaded.dtype can be different with param.dtype
            param.data.copy_(loaded.data)  # 注释：复制本地数据
            dist.broadcast(param.data, src=dist.get_rank())  # 注释：广播到其它 rank
        else:
            assert isinstance(loaded, int)  # the rank that holds the state
            dist.broadcast(param.data, src=loaded)  # 注释：从拥有者 rank 广播
        shard_states.pop(param_name)  # 注释：移除已处理项
        del loaded  # 注释：释放引用
        return param  # 注释：返回参数/缓冲区

    def init_fn(sub_mod: torch.nn.Module, recurse: bool = True):
        """递归初始化子模块的参数/缓冲区。（注释：内部函数用途）"""
        param_and_buffers = tuple(sub_mod.named_parameters(recurse=False)) + tuple(sub_mod.named_buffers(recurse=False))
        # param_and_buffers = sorted(sub_mod.named_parameters(recurse=False), key=lambda x: x[0])
        for name, state in param_and_buffers:  # 注释：遍历本层参数/缓冲区
            if not state.is_meta:  # 注释：仅处理 meta 参数
                continue
            is_param = name in sub_mod._parameters  # 注释：判断是参数还是缓冲区
            fqn = state2fqn[state].pop(0)  # 注释：取出 FQN
            # non-persistent buffers will not be saved in state dict, we can safely skip it
            if (not is_param) and fqn not in shard_states:
                if state.is_meta:
                    raise RuntimeError(
                        f"find a non-persistent buffer ({fqn}) initiated with device meta. Such buffer is not saved "
                        f"in checkpoint and user should guarantee to init in CPU / GPU device."
                    )
                continue
            # for shared parameter, we get it from the first time it is created
            if state in shared:
                if state not in materialized_states:
                    materialized_states[state] = create_and_sync_state(fqn, state, is_param)
                else:
                    if fqn in shard_states:
                        shard_states.pop(fqn)
                materialize_state = materialized_states[state]
            # for not shared parameter, we create it directly
            else:
                materialize_state = create_and_sync_state(fqn, state, is_param)
            if is_param:
                sub_mod._parameters[name] = materialize_state  # 注释：替换参数
            else:
                sub_mod._buffers[name] = materialize_state  # 注释：替换缓冲区
        if recurse:  # 注释：递归处理子模块
            for module in sub_mod.children():
                init_fn(module, recurse=True)

        # for debug
        # if len(shard_states) == 0: print("clear")
        return sub_mod  # 注释：返回已初始化的子模块

    return init_fn  # 注释：返回初始化函数


def fsdp_version(model):
    """
    判断模型的 FSDP 版本类型。（注释：函数用途）

    返回：1 表示 FSDP1，2 表示 FSDP2，0 表示非 FSDP。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::fsdp_version`。（注释：定位）
      - 被谁调用：`get_fsdp_full_state_dict`、`FSDPCheckpointManager` 等。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`isinstance`。（注释：外部依赖）
    """
    if isinstance(model, FSDP):  # 注释：FSDP1
        return 1
    elif isinstance(model, FSDPModule):  # 注释：FSDP2
        return 2
    else:  # 注释：非 FSDP
        return 0


def get_fsdp_state_ctx(model, state_type, state_cfg, optim_cfg):
    """
    获取 FSDP 状态字典上下文（FSDP1）或空上下文（FSDP2/非 FSDP）。（注释：函数用途）

    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::get_fsdp_state_ctx`。（注释：定位）
      - 被谁调用：`get_fsdp_full_state_dict`、`FSDPCheckpointManager`。（注释：调用方）
      - 调用了谁（项目内）：`fsdp_version`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`FSDP.state_dict_type`。（注释：外部依赖）
    """
    if fsdp_version(model) == 1:  # 注释：FSDP1 使用 state_dict_type
        return FSDP.state_dict_type(model, state_type, state_cfg, optim_cfg)
    else:  # 注释：FSDP2/非 FSDP 无需上下文
        return nullcontext()


def get_fsdp_full_state_dict(model: torch.nn.Module, offload_to_cpu: bool = True, rank0_only: bool = True):
    """
    获取 FSDP 模型的完整 state_dict。（注释：函数用途）

    参数：
      - offload_to_cpu：是否把 state_dict 放到 CPU。（注释：输入说明）
      - rank0_only：是否仅 rank0 获取。（注释：输入说明）
    返回：dict。（注释：返回说明）
    异常：未知 FSDP 版本抛 NotImplementedError。（注释：异常说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::get_fsdp_full_state_dict`。（注释：定位）
      - 被谁调用：`FSDPCheckpointManager.save_checkpoint`、`fsdp2_load_full_state_dict`。（注释：调用方）
      - 调用了谁（项目内）：`fsdp_version`、`get_fsdp_state_ctx`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.checkpoint`。（注释：外部依赖）
    """
    if fsdp_version(model) == 1:  # 注释：FSDP1 分支
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        state_dict_config = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=rank0_only)
        with get_fsdp_state_ctx(
            model, state_type=StateDictType.FULL_STATE_DICT, state_cfg=state_dict_config, optim_cfg=None
        ):
            state_dict = model.state_dict()
        return state_dict
    elif fsdp_version(model) == 2:  # 注释：FSDP2 分支
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        state_dict_config = StateDictOptions(
            full_state_dict=True, cpu_offload=offload_to_cpu, broadcast_from_rank0=not rank0_only
        )
        state_dict = get_model_state_dict(model, options=state_dict_config)
        return state_dict
    else:
        raise NotImplementedError(f"Unknown FSDP version {fsdp_version}")


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, device_mesh=None, cpu_offload=None):
    """
    将完整 state_dict 加载到 FSDP2 分片模型中。（注释：函数用途）

    参数：
      - model：FSDP2 模型。（注释：输入说明）
      - full_state：完整 state_dict（通常仅 rank0 有）。（注释：输入说明）
    返回：无。（注释：仅副作用）
    副作用：广播参数并修改模型。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict`。（注释：定位）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`（加载全量权重时）。（注释：调用方）
      - 调用了谁（项目内）：`get_device_id`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.checkpoint.set_model_state_dict`。（注释：外部依赖）
    """

    if version.parse(torch.__version__) >= version.parse("2.7.0"):
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    else:
        # official torch 2.6.0 set_model_state_dict API leads to OOM
        # use torch 2.7.0 copy from verl/third_party/torch/distributed/checkpoint
        from verl.third_party.torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    # To broadcast, it needs to be instantiated in the GPU.
    if dist.get_rank() == 0:  # 注释：rank0 将模型放到 GPU
        model = model.to(device=get_device_id(), non_blocking=True)
    else:  # 注释：其它 rank 用 empty 参数占位
        model = model.to_empty(device=get_device_id())

    cpu_offload = cpu_offload is not None  # 注释：是否 CPU offload
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)  # 注释：设置模型参数

    # rotary_emb is not in state_dict, so we need to broadcast it manually
    for name, buf in model.named_buffers():  # 注释：广播缓冲区
        dist.broadcast(buf, src=0)

    if cpu_offload:  # 注释：如需 offload，则移回 CPU
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(get_device_id())


@contextmanager
def maybe_patch_fsdp_module(model):
    """
    临时替换 FSDPModule 的基类以兼容 ABC 检测。（注释：函数用途）

    参数：
      - model：待检查模型。（注释：输入说明）
    返回：上下文管理器。（注释：返回说明）
    副作用：可能临时修改 `fully_shard_module.FSDPModule`。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::maybe_patch_fsdp_module`。（注释：定位）
      - 被谁调用：`apply_fsdp2`。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.fsdp._fully_shard`。（注释：外部依赖）
    """
    if fully_shard_module is None:  # 注释：不支持 FSDP2 时直接透传
        yield
        return

    orig_fsdp_module = fully_shard_module.FSDPModule  # 注释：保存原始 FSDPModule

    class FSDPModuleABC(ABC, orig_fsdp_module):
        """用于兼容 ABC 检测的临时类。（注释：内部类用途）"""
        pass

    try:
        if isinstance(model, ABC):  # 注释：模型为 ABC 时替换类
            fully_shard_module.FSDPModule = FSDPModuleABC
        yield  # 注释：进入上下文
    finally:
        fully_shard_module.FSDPModule = orig_fsdp_module  # 注释：恢复原始类


def apply_fsdp2(model, fsdp_kwargs, config):
    """
    对模型应用 FSDP2 的 fully_shard 包裹。（注释：函数用途）

    参数：
      - model：待包裹的模型（通常 AutoModelForCausalLM）。（注释：输入说明）
      - fsdp_kwargs：fully_shard 的参数。（注释：输入说明）
      - config：wrap_policy 配置。（注释：输入说明）
    返回：无。（注释：仅副作用）
    副作用：就地包裹模型子模块与根模块。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::apply_fsdp2`。（注释：定位）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`tests/utils/test_activation_offload.py`。（注释：调用方）
      - 调用了谁（项目内）：`maybe_patch_fsdp_module`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`fully_shard`。（注释：外部依赖）
    """
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"

    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)  # 注释：默认包裹层列表
    fsdp_transformer_layer_cls_to_wrap = config.get("wrap_policy", {}).get(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )  # 注释：读取配置

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):  # 注释：单字符串转列表
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None  # 注释：校验

    modules = []  # 注释：待包裹模块列表
    for name, module in model.named_modules():  # 注释：遍历子模块
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)  # 注释：收集包裹候选

    for idx, module in enumerate(modules):  # 注释：逐个包裹子模块
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f"wrap module {module.__class__.__name__}")
        with maybe_patch_fsdp_module(module):  # 注释：兼容 ABC 检测
            fully_shard(module, **fsdp_kwargs)  # 注释：应用 fully_shard

    # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
    #     print(f"wrap module {model.__class__.__name__}")
    with maybe_patch_fsdp_module(model):  # 注释：包裹根模块
        fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def get_shard_placement_fn(fsdp_size):
    """
    选择可被 fsdp_size 整除的维度作为分片维度。（注释：函数用途）

    参数：fsdp_size（int）。（注释：输入说明）
    返回：shard_placement_fn（Callable）。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::get_shard_placement_fn`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用（供 FSDP2 使用）。（注释：调用方）
      - 调用了谁（项目内）：无。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.tensor.Shard`。（注释：外部依赖）
    """

    def shard_placement_fn(param):
        """根据参数形状选择分片维度。（注释：内部函数用途）"""
        shape = list(param.shape)  # 注释：参数形状
        for i in range(len(shape)):  # 注释：遍历维度
            if shape[i] % fsdp_size == 0:  # 注释：可整除则选该维度
                return Shard(i)
        return Shard(0)  # 注释：否则回退到维度 0

    return shard_placement_fn


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """
    FSDP2 下裁剪梯度范数（兼容 CPU DTensor）。（注释：函数用途）

    参数：
      - parameters：参数或参数列表。（注释：输入说明）
      - max_norm：最大范数。（注释：输入说明）
      - norm_type：范数类型。（注释：输入说明）
    返回：total_norm。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::fsdp2_clip_grad_norm_`。（注释：定位）
      - 被谁调用：`verl/workers/actor/dp_actor.py`、`verl/workers/critic/dp_critic.py`。（注释：调用方）
      - 调用了谁（项目内）：`get_device_id`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.nn.utils.clip_grad`。（注释：外部依赖）
    """
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm  # 注释：内部裁剪工具

    if isinstance(parameters, torch.Tensor):  # 注释：统一为列表
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]  # 注释：收集梯度
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)  # 注释：计算总范数
    total_norm = total_norm.to(get_device_id(), non_blocking=True)  # 注释：移动到当前设备
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)  # 注释：裁剪梯度
    return total_norm  # 注释：返回范数


def layered_summon_lora_params(fsdp_module) -> OrderedDict:
    """
    分层召回 LoRA 参数（逐层 summon_full_params）。（注释：函数用途）

    参数：fsdp_module（FSDP 模型）。（注释：输入说明）
    返回：OrderedDict 的 LoRA 参数（CPU）。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::layered_summon_lora_params`。（注释：定位）
      - 被谁调用：`collect_lora_params`。（注释：调用方）
      - 调用了谁（项目内）：`fsdp_version`、`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`peft.utils.save_and_load.get_peft_model_state_dict`。（注释：外部依赖）
    """
    from peft.utils.save_and_load import get_peft_model_state_dict  # 注释：PEFT 工具

    def __prefix_submodules(module, prefix):
        """遍历指定前缀的一级子模块。（注释：内部函数用途）"""
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()  # 注释：收集 LoRA 参数
    prefix_list = [  # 注释：支持 FSDP/FSDP2 的不同前缀
        # fsdp
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)  # 注释：获取 PEFT 模型
    for prefix in prefix_list:  # 注释：遍历前缀
        for name, submodule in __prefix_submodules(fsdp_module, prefix):  # 注释：遍历子模块
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")  # 注释：统一前缀
            if name.endswith(".model") or name.endswith(".layers"):  # 注释：跳过容器节点
                continue
            if fsdp_version(submodule) > 0:  # 注释：仅处理 FSDP 模块
                with FSDP.summon_full_params(submodule, writeback=False):  # 注释：聚合完整参数
                    sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=submodule.state_dict())
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }  # 注释：统一转为 CPU 张量
                    lora_params.update(sub_lora_params)  # 注释：合并参数
                    submodule._is_root = False  # 注释：避免 root 标记影响
                get_torch_device().empty_cache()  # 注释：清理缓存
    return lora_params  # 注释：返回 LoRA 参数


def collect_lora_params(module: FSDP, layered_summon: bool, base_sync_done: bool) -> OrderedDict:
    """
    收集 LoRA 参数；若基础模型未同步则回退为全量参数。（注释：函数用途）

    参数：
      - module (FSDP)：模型。（注释：输入说明）
      - layered_summon (bool)：是否分层召回。（注释：输入说明）
      - base_sync_done (bool)：基础模型是否已同步到 vLLM。（注释：输入说明）
    返回：OrderedDict。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::collect_lora_params`。（注释：定位）
      - 被谁调用：`verl/workers/rollout/*` 与 `verl/workers/engine/*` 中的 LoRA 同步逻辑。（注释：调用方）
      - 调用了谁（项目内）：`layered_summon_lora_params`、`get_device_name`、`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`peft.utils.save_and_load.get_peft_model_state_dict`。（注释：外部依赖）
    """
    from peft.utils.save_and_load import get_peft_model_state_dict  # 注释：PEFT 工具

    lora_params = OrderedDict()  # 注释：输出参数字典
    peft_model = getattr(module, "_fsdp_wrapped_module", module)  # 注释：获取 PEFT 模型
    if fsdp_version(module) > 0:  # 注释：FSDP 分支
        if layered_summon:  # 注释：分层召回
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            lora_params = layered_summon_lora_params(module)
        else:
            with FSDP.summon_full_params(module, writeback=False):  # 注释：聚合完整参数
                if base_sync_done:
                    lora_params = get_peft_model_state_dict(peft_model)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            get_torch_device().empty_cache()
    else:  # 注释：非 FSDP 分支
        if base_sync_done:
            lora_params = get_peft_model_state_dict(peft_model)
        else:
            model = peft_model.base_model.model
            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)
    return lora_params  # 注释：返回参数


def replace_lora_wrapper(k, peft_config):
    """
    将 LoRA 参数名替换为 base_layer 参数名。（注释：函数用途）

    参数：
      - k (str)：原始参数名。（注释：输入说明）
      - peft_config：PEFT 配置。（注释：输入说明）
    返回：
      - str：替换后的参数名。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/fsdp_utils.py::replace_lora_wrapper`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用（供 vLLM 同步使用）。（注释：调用方）
      - 调用了谁（项目内）：`check_exclude_modules`、`check_target_modules`。（注释：内部依赖）
      - 调用了谁（外部依赖）：字符串处理。（注释：外部依赖）
    """
    stacked_params = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 注释：常见投影层
    if k.endswith(".weight"):  # 注释：处理权重
        module_k = k[: -len(".weight")]
        if check_exclude_modules(peft_config, module_k):  # 注释：排除模块
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.weight"
    if k.endswith(".bias"):  # 注释：处理 bias
        module_k = k[: -len(".bias")]
        if check_exclude_modules(peft_config, module_k):
            return k
        elif any([module_k.endswith(s) for s in stacked_params]) or check_target_modules(peft_config, module_k):
            return f"{module_k}.base_layer.bias"
    return k  # 注释：默认返回原名
