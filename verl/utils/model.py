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
模块用途：提供 Hugging Face/ Megatron 相关模型创建、权重加载、并行归一化与多模态输入整理工具。  # 注释：模块用途说明
输入：  # 注释：模块输入说明标题
- 模型名称/路径、配置对象（AutoConfig/PretrainedConfig/DictConfig）。  # 注释：输入含义
- 模型参数字典、并行配置、batch 多模态字段。  # 注释：输入含义
输出：  # 注释：模块输出说明标题
- 构建好的模型/配置对象、权重转换结果、辅助张量。  # 注释：输出说明
依赖：torch、transformers、megatron、trl（可选）。  # 注释：关键依赖说明
典型用法：  # 注释：最小示例标题
- model = create_huggingface_actor(model_name, override_config_kwargs)。  # 注释：示例用法
- position_ids = compute_position_id_with_mask(attention_mask)。  # 注释：示例用法
调用路径概览：  # 注释：调用路径概览标题
- 入口：trainer/worker 初始化模型或数据集构造位置（如 main_ppo.py、fsdp_workers.py）。  # 注释：典型入口说明
- 典型链路：main_ppo.py -> RLHFDataset.__getitem__ -> compute_position_id_with_mask。  # 注释：调用链示例
"""

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tensordict.tensorclass import NonTensorData
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    GenerationConfig,
    MistralForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl.models.registry import ModelRegistry
from verl.utils.import_utils import is_trl_available


class LambdaLayer(nn.Module):
    """
    类用途：将任意函数包装成 nn.Module，便于插入到模型结构中。  # 注释：类用途说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::LambdaLayer。  # 注释：类位置
    - 典型调用路径：create_huggingface_critic -> LambdaLayer(fn=squeeze)。  # 注释：典型调用链
    - 被谁调用：create_huggingface_critic（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：squeeze（本文件，可选）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.nn.Module。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    def __init__(self, fn):
        """
        函数用途：保存待调用的函数句柄。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - fn (Callable)：前向计算时调用的函数。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：无。  # 注释：副作用说明
        异常/边界条件：fn 非可调用对象会在 forward 调用时报错。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：LambdaLayer(fn=torch.relu)。  # 注释：示例输入
        - 输出：可作为 nn.Module 使用的层。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/model.py::LambdaLayer.__init__。  # 注释：函数位置
        - 典型调用路径：create_huggingface_critic -> LambdaLayer。  # 注释：典型调用链
        - 被谁调用：本类构造时调用。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.nn.Module.__init__。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        super().__init__()  # 注释：初始化 nn.Module 基类
        self.fn = fn  # 注释：保存函数句柄

    def forward(self, *args, **kwargs):
        """
        函数用途：前向调用被包装的函数。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - *args/**kwargs：透传给被包装函数的参数。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - fn(*args, **kwargs) 的返回值。  # 注释：返回值语义
        副作用：取决于 fn 的实现。  # 注释：副作用说明
        异常/边界条件：fn 内部异常会直接抛出。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：LambdaLayer(fn=squeeze)(tensor([1.0]))。  # 注释：示例输入
        - 输出：squeeze 后的张量。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/model.py::LambdaLayer.forward。  # 注释：函数位置
        - 典型调用路径：模型前向 -> LambdaLayer.forward。  # 注释：典型调用链
        - 被谁调用：PyTorch 前向传播机制。  # 注释：调用方说明
        - 调用了谁（项目内）：self.fn（传入的函数）。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return self.fn(*args, **kwargs)  # 注释：调用并返回函数结果


def squeeze(x):
    """
    函数用途：沿最后一维做 squeeze，常用于把 [B, 1] 变成 [B]。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - x (torch.Tensor)：待压缩的张量。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - torch.Tensor：压缩后的张量。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：若最后一维不为 1，不会改变形状。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：x.shape=(2,1) -> 输出形状 (2,)。  # 注释：示例说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::squeeze。  # 注释：函数位置
    - 典型调用路径：create_huggingface_critic -> LambdaLayer(fn=squeeze)。  # 注释：典型调用链
    - 被谁调用：LambdaLayer.forward（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.squeeze。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    return torch.squeeze(x, dim=-1)  # 注释：沿最后一维压缩


def update_model_config(module_config, override_config_kwargs):
    """
    函数用途：递归更新 Hugging Face 配置对象中的字段。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - module_config (PretrainedConfig)：待更新的配置对象。  # 注释：参数含义
    - override_config_kwargs (dict)：需要覆盖的配置键值（可嵌套）。  # 注释：参数含义
    返回：无（原地修改）。  # 注释：返回值说明
    副作用：会直接修改 module_config。  # 注释：副作用说明
    异常/边界条件：若字段不存在，setattr 会新增属性。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：override_config_kwargs={\"hidden_size\": 1024}。  # 注释：示例输入
    - 输出：module_config.hidden_size=1024。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::update_model_config。  # 注释：函数位置
    - 典型调用路径：get_huggingface_actor_config -> update_model_config。  # 注释：典型调用链
    - 被谁调用：get_huggingface_actor_config（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：update_model_config（递归自身）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    for key, val in override_config_kwargs.items():  # 注释：遍历覆盖项
        if isinstance(val, dict):  # 注释：若值是嵌套字典
            update_model_config(getattr(module_config, key), val)  # 注释：递归更新子配置
        else:  # 注释：普通字段
            setattr(module_config, key, val)  # 注释：设置字段值


def get_huggingface_actor_config(model_name: str, override_config_kwargs=None, trust_remote_code=False) -> dict:
    """
    函数用途：从预训练模型加载配置并应用覆盖项。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model_name (str)：模型名或本地路径。  # 注释：参数含义
    - override_config_kwargs (dict|None)：需要覆盖的配置字段。  # 注释：参数含义
    - trust_remote_code (bool)：是否信任远程自定义代码。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - module_config (PretrainedConfig)：更新后的配置对象。  # 注释：返回值语义
    副作用：无（返回新配置对象）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - override_config_kwargs 非 dict 时触发 AssertionError。  # 注释：异常说明
    - AutoConfig 加载失败会抛出 transformers 异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_huggingface_actor_config(\"Qwen/Qwen2-7B\", {\"hidden_size\": 4096})。  # 注释：示例输入
    - 输出：更新后的 config 对象。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_huggingface_actor_config。  # 注释：函数位置
    - 典型调用路径：create_huggingface_actor -> get_huggingface_actor_config。  # 注释：典型调用链
    - 被谁调用：create_huggingface_actor、get_generation_config。  # 注释：调用方说明
    - 调用了谁（项目内）：update_model_config。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.AutoConfig.from_pretrained。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if override_config_kwargs is None:  # 注释：默认空覆盖
        override_config_kwargs = {}  # 注释：初始化为空字典
    assert isinstance(override_config_kwargs, dict), (  # 注释：类型校验
        f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"  # 注释：断言信息
    )  # 注释：断言结束
    module_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)  # 注释：加载配置
    update_model_config(module_config, override_config_kwargs)  # 注释：应用覆盖项
    # （分隔说明：返回配置）  # 注释：替代空行，保持逐行注释
    return module_config  # 注释：返回配置对象


def get_generation_config(
    model: str,
    trust_remote_code: bool = False,
) -> Optional[GenerationConfig]:
    """
    函数用途：尝试获取模型的 GenerationConfig，不存在则返回 None。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model (str)：模型名或本地路径。  # 注释：参数含义
    - trust_remote_code (bool)：是否信任远程代码。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - GenerationConfig 或 None。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若模型与配置均不存在，返回 None。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_generation_config(\"Qwen/Qwen2-7B\")。  # 注释：示例输入
    - 输出：GenerationConfig 或 None。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_generation_config。  # 注释：函数位置
    - 典型调用路径：模型加载/推理配置准备 -> get_generation_config。  # 注释：典型调用链
    - 被谁调用：trainer/worker 初始化推理配置（间接）。  # 注释：调用方说明
    - 调用了谁（项目内）：get_huggingface_actor_config。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.GenerationConfig。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    try:  # 注释：优先尝试直接加载 generation config
        return GenerationConfig.from_pretrained(model)  # 注释：返回预训练配置
    except OSError:  # Not found  # 注释：未找到 generation config
        try:  # 注释：尝试从模型 config 构造
            config = get_huggingface_actor_config(  # 注释：加载模型 config
                model,  # 注释：模型名或路径
                trust_remote_code=trust_remote_code,  # 注释：是否信任远程代码
            )  # 注释：get_huggingface_actor_config 结束
            return GenerationConfig.from_model_config(config)  # 注释：从模型配置构建生成配置
        except OSError:  # Not found  # 注释：模型配置也不存在
            return None  # 注释：返回 None


def create_huggingface_actor(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """
    函数用途：根据配置创建 Hugging Face 的 CausalLM 模型实例（actor）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model_name (str)：模型名或本地路径。  # 注释：参数含义
    - override_config_kwargs (dict|None)：覆盖配置字段。  # 注释：参数含义
    - automodel_kwargs (dict|None)：传给 AutoModelForCausalLM.from_config 的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - nn.Module：未加载权重的模型实例。  # 注释：返回值语义
    副作用：无（仅构建结构）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - override_config_kwargs 非 dict 会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：create_huggingface_actor(\"Qwen/Qwen2-7B\", {\"hidden_size\": 4096})。  # 注释：示例输入
    - 输出：AutoModelForCausalLM 实例。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::create_huggingface_actor。  # 注释：函数位置
    - 典型调用路径：trainer/worker 初始化模型 -> create_huggingface_actor。  # 注释：典型调用链
    - 被谁调用：create_huggingface_critic、模型加载工具。  # 注释：调用方说明
    - 调用了谁（项目内）：get_huggingface_actor_config。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.AutoModelForCausalLM.from_config。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if override_config_kwargs is None:  # 注释：默认空覆盖配置
        override_config_kwargs = {}  # 注释：初始化为空字典
    if automodel_kwargs is None:  # 注释：默认空模型参数
        automodel_kwargs = {}  # 注释：初始化为空字典
    assert isinstance(override_config_kwargs, dict), (  # 注释：类型校验
        f"override_config_kwargs must be a dict, got {type(override_config_kwargs)}"  # 注释：断言信息
    )  # 注释：断言结束
    module_config = get_huggingface_actor_config(  # 注释：加载并覆盖配置
        model_name, override_config_kwargs, trust_remote_code=automodel_kwargs.get("trust_remote_code", False)  # 注释：参数
    )  # 注释：get_huggingface_actor_config 结束
    module: nn.Module = AutoModelForCausalLM.from_config(module_config, **automodel_kwargs)  # 注释：构建模型结构
    return module  # 注释：返回模型实例


def create_huggingface_critic(model_name: str, override_config_kwargs=None, automodel_kwargs=None) -> nn.Module:
    """
    函数用途：构建带 value head 的 critic 模型（基于 CausalLM）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model_name (str)：模型名或本地路径。  # 注释：参数含义
    - override_config_kwargs (dict|None)：覆盖配置字段。  # 注释：参数含义
    - automodel_kwargs (dict|None)：传给 AutoModelForCausalLM.from_config 的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - nn.Module：带 value head 的 critic 模型。  # 注释：返回值语义
    副作用：会替换 critic_module.lm_head。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若 hidden_size 不存在会导致线性层构建失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：create_huggingface_critic(\"Qwen/Qwen2-7B\")。  # 注释：示例输入
    - 输出：critic 模型，输出标量 value。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::create_huggingface_critic。  # 注释：函数位置
    - 典型调用路径：critic worker 初始化 -> create_huggingface_critic。  # 注释：典型调用链
    - 被谁调用：训练器/worker 初始化逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：create_huggingface_actor、LambdaLayer、squeeze。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.nn.Linear、torch.nn.Sequential。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    critic_module: nn.Module = create_huggingface_actor(  # 注释：先创建基础 actor 模型
        model_name, override_config_kwargs=override_config_kwargs, automodel_kwargs=automodel_kwargs  # 注释：参数透传
    )  # 注释：创建结束
    if automodel_kwargs is None:  # 注释：默认空模型参数
        automodel_kwargs = {}  # 注释：初始化为空字典
    torch_dtype = automodel_kwargs.get("torch_dtype", torch.float32)  # 注释：读取 dtype
    critic_module.lm_head = nn.Sequential(  # 注释：替换 lm_head 为 value head
        nn.Linear(critic_module.config.hidden_size, 1, dtype=torch_dtype), LambdaLayer(fn=squeeze)  # 注释：线性层+压缩
    )  # 注释：value head 构建结束
    return critic_module  # 注释：返回 critic 模型


def get_model_size(model: nn.Module, scale="auto"):
    """
    函数用途：统计模型参数规模，并按指定单位缩放。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model (nn.Module)：待统计的模型。  # 注释：参数含义
    - scale (str)：\"auto\"/\"B\"/\"M\"/\"K\"/\"\"。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (n_params, scale)：缩放后的数量与单位字符串。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：未知 scale 会抛 NotImplementedError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_model_size(model, scale=\"auto\")。  # 注释：示例输入
    - 输出：(1.23, \"B\")。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_model_size。  # 注释：函数位置
    - 典型调用路径：print_model_size -> get_model_size。  # 注释：典型调用链
    - 被谁调用：print_model_size（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：model.parameters。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    n_params = sum(p.numel() for p in model.parameters())  # 注释：统计总参数量
    # （分隔说明：自动选择单位）  # 注释：替代空行，保持逐行注释
    if scale == "auto":  # 注释：自动单位选择
        if n_params > 1e9:  # 注释：十亿以上
            scale = "B"  # 注释：单位为 B
        elif n_params > 1e6:  # 注释：百万以上
            scale = "M"  # 注释：单位为 M
        elif n_params > 1e3:  # 注释：千以上
            scale = "K"  # 注释：单位为 K
        else:  # 注释：小于千
            scale = ""  # 注释：不缩放
    # （分隔说明：按单位缩放数值）  # 注释：替代空行，保持逐行注释
    if scale == "B":  # 注释：十亿级
        n_params = n_params / 1e9  # 注释：缩放到 B
    elif scale == "M":  # 注释：百万级
        n_params = n_params / 1e6  # 注释：缩放到 M
    elif scale == "K":  # 注释：千级
        n_params = n_params / 1e3  # 注释：缩放到 K
    elif scale == "":  # 注释：不缩放
        pass  # 注释：保持原值
    else:  # 注释：未知单位
        raise NotImplementedError(f"Unknown scale {scale}")  # 注释：抛出异常
    # （分隔说明：返回结果）  # 注释：替代空行，保持逐行注释
    return n_params, scale  # 注释：返回参数量与单位


def print_model_size(model: nn.Module, name: str = None):
    """
    函数用途：打印模型参数规模信息。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model (nn.Module)：待统计的模型。  # 注释：参数含义
    - name (str|None)：输出名称，默认使用类名。  # 注释：参数含义
    返回：无（打印到 stdout）。  # 注释：返回值说明
    副作用：打印日志。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：print_model_size(model, \"Actor\")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::print_model_size。  # 注释：函数位置
    - 典型调用路径：调试/日志输出 -> print_model_size。  # 注释：典型调用链
    - 被谁调用：训练器或脚本工具（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：get_model_size。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：print。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    n_params, scale = get_model_size(model, scale="auto")  # 注释：获取缩放后的参数量
    if name is None:  # 注释：若未指定名字
        name = model.__class__.__name__  # 注释：使用类名作为默认名称
    print(f"{name} contains {n_params:.2f}{scale} parameters")  # 注释：打印参数量


def create_random_mask(
    input_ids: torch.Tensor,
    max_ratio_of_valid_token: float,
    max_ratio_of_left_padding: float,
    min_ratio_of_valid_token: float = 0,
):
    """
    函数用途：基于输入形状生成随机 attention mask（支持左/右 padding）。  # 注释：函数用途说明
    处理流程：  # 注释：流程说明标题
    - 采样有效 token 长度。  # 注释：流程步骤
    - 采样左侧 padding 长度。  # 注释：流程步骤
    - 生成 0/1 mask。  # 注释：流程步骤
    参数：  # 注释：参数说明标题
    - input_ids (torch.Tensor)：形状 (batch_size, seq_len)。  # 注释：参数含义
    - max_ratio_of_valid_token (float)：有效 token 最大比例。  # 注释：参数含义
    - max_ratio_of_left_padding (float)：左 padding 最大比例。  # 注释：参数含义
    - min_ratio_of_valid_token (float)：有效 token 最小比例。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - masks (torch.Tensor)：与 input_ids 同形状的 0/1 mask。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 比例配置不合法会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：input_ids.shape=(2,4), max_ratio_of_valid_token=0.5。  # 注释：示例输入
    - 输出：mask 如 [[0,1,1,0], [1,1,0,0]]。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::create_random_mask。  # 注释：函数位置
    - 典型调用路径：数据增强/预处理逻辑 -> create_random_mask。  # 注释：典型调用链
    - 被谁调用：项目内工具函数（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：numpy.random.randint、torch.ones_like。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    assert max_ratio_of_valid_token > 0 and max_ratio_of_valid_token <= 1.0  # 注释：校验有效 token 比例
    assert max_ratio_of_left_padding >= 0 and max_ratio_of_left_padding < 1.0  # 注释：校验左 padding 比例
    assert min_ratio_of_valid_token <= max_ratio_of_valid_token  # 注释：校验最小比例不超过最大比例
    # （分隔说明：计算长度上限）  # 注释：替代空行，保持逐行注释
    batch_size, sequence_length = input_ids.shape  # 注释：读取 batch 与序列长度
    max_num_valid_tokens = int(sequence_length * max_ratio_of_valid_token)  # 注释：最大有效 token 数
    min_num_valid_tokens = max(1, int(sequence_length * min_ratio_of_valid_token))  # 注释：最小有效 token 数
    max_left_padding = int(sequence_length * max_ratio_of_left_padding)  # 注释：最大左 padding 数
    assert max_num_valid_tokens + max_left_padding <= sequence_length  # 注释：确保长度不溢出
    assert max_num_valid_tokens > 0 and max_ratio_of_valid_token <= sequence_length  # 注释：有效 token 数必须为正
    masks = torch.ones_like(input_ids, dtype=torch.int64)  # 注释：初始化 mask 全为 1
    # TODO: we can make this faster  # 注释：原注释，提示可优化性能
    for i in range(batch_size):  # 注释：逐样本生成随机 mask
        num_left_padding = np.random.randint(low=0, high=max_left_padding + 1, dtype=np.int64)  # 注释：随机左 padding 长度
        num_valid = np.random.randint(low=min_num_valid_tokens, high=max_num_valid_tokens + 1, dtype=np.int64)  # 注释：随机有效 token 数
        # （分隔说明：置零左 padding）  # 注释：替代空行，保持逐行注释
        for index in range(num_left_padding):  # 注释：遍历左 padding 区间
            masks[i, index] = 0  # 注释：置为 0（padding）
        # （分隔说明：置零右侧 padding）  # 注释：替代空行，保持逐行注释
        for index in range(num_left_padding + num_valid, sequence_length):  # 注释：遍历右侧 padding 区间
            masks[i, index] = 0  # 注释：置为 0（padding）
    return masks  # 注释：返回随机 mask


def compute_position_id_with_mask(mask):
    """
    函数用途：根据 attention_mask 生成 position_ids（从 0 开始递增）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - mask (torch.Tensor)：形状 (B, L) 的 0/1 mask。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - position_ids (torch.Tensor)：与 mask 同形状，padding 处为 0。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：mask 全零时返回全零。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：mask=[[1,1,0]] -> 输出：[[0,1,0]]。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::compute_position_id_with_mask。  # 注释：函数位置
    - 典型调用路径：RLHFDataset.__getitem__ -> compute_position_id_with_mask。  # 注释：典型调用链
    - 被谁调用：verl/utils/dataset/rl_dataset.py。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.cumsum、torch.clip。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)  # 注释：累积计数并转为 position ids


def convert_weight_keys(state_dict: dict[str, torch.Tensor], model: PreTrainedModel):
    """
    函数用途：根据 HF 的 checkpoint 转换映射，将权重 key 还原为原始命名。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - state_dict (dict[str, Tensor])：待转换的权重字典。  # 注释：参数含义
    - model (PreTrainedModel)：用于提供转换映射的模型对象。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict[str, Tensor]：转换后的权重字典。  # 注释：返回值语义
    副作用：无（返回新 dict）。  # 注释：副作用说明
    异常/边界条件：若模型无映射属性，直接返回原 dict。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：state_dict 里包含新命名 key。  # 注释：示例输入
    - 输出：转换回旧命名 key。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::convert_weight_keys。  # 注释：函数位置
    - 典型调用路径：权重加载/兼容处理 -> convert_weight_keys。  # 注释：典型调用链
    - 被谁调用：模型权重加载工具（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：re.sub、re.subn。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    # convert state dict keys: https://github.com/huggingface/transformers/pull/38385  # 注释：原注释，说明背景
    if not hasattr(model, "_checkpoint_conversion_mapping"):  # 注释：无映射则直接返回
        return state_dict  # 注释：不做转换
    # （分隔说明：构建反向映射）  # 注释：替代空行，保持逐行注释
    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}  # 注释：反向映射表
    original_weights = {}  # 注释：存放转换后的权重
    for key, value in state_dict.items():  # 注释：遍历权重项
        for pattern, replacement in reverse_key_mapping.items():  # 注释：遍历映射规则
            replacement = replacement.lstrip("^")  # 注释：去掉正则前缀符号
            replacement = re.sub(r"\\(.*\\)", "", replacement)  # 注释：去掉正则括号内容
            key, n_replace = re.subn(pattern, replacement, key)  # 注释：替换 key
            # Early exit of the loop  # 注释：原注释，说明提前退出
            if n_replace > 0:  # 注释：一旦匹配成功
                break  # 注释：跳出内层循环
        original_weights[key] = value  # 注释：保存转换后的权重
    return original_weights  # 注释：返回转换结果


def check_exclude_modules(config, key: str) -> bool:
    """
    函数用途：判断模块名是否命中 adapter 配置的 exclude_modules。  # 注释：函数用途说明
    说明：逻辑改编自 PEFT tuners_utils。  # 注释：来源说明
    参数：  # 注释：参数说明标题
    - config (LoraConfig|LycorisConfig)：包含 exclude_modules 的配置对象。  # 注释：参数含义
    - key (str)：待匹配的模块名。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：命中为 True，否则 False。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：exclude_modules 为空时直接返回 False。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：exclude_modules=[\"q_proj\"], key=\"model.layers.0.q_proj\" -> True。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::check_exclude_modules。  # 注释：函数位置
    - 典型调用路径：LoRA/adapter 初始化 -> check_exclude_modules。  # 注释：典型调用链
    - 被谁调用：adapter 工具逻辑（本文件或外部）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：re.fullmatch。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if hasattr(config, "exclude_modules") and config.exclude_modules:  # 注释：确认配置存在
        if isinstance(config.exclude_modules, str):  # 注释：正则字符串形式
            if re.fullmatch(config.exclude_modules, key):  # 注释：正则匹配
                return True  # 注释：命中
        elif key in config.exclude_modules:  # 注释：直接匹配
            return True  # 注释：命中
        elif any(key.endswith(f".{exclude_key}") for exclude_key in config.exclude_modules):  # 注释：后缀匹配
            return True  # 注释：命中
    return False  # 注释：未命中


def check_target_modules(config, key: str) -> bool:
    """
    函数用途：判断模块名是否命中 adapter 配置的 target_modules。  # 注释：函数用途说明
    说明：逻辑改编自 PEFT tuners_utils。  # 注释：来源说明
    参数：  # 注释：参数说明标题
    - config (LoraConfig|LycorisConfig)：包含 target_modules 的配置对象。  # 注释：参数含义
    - key (str)：待匹配的模块名。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：命中为 True，否则 False。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：layers_pattern 与 layers_to_transform 为空时放宽匹配。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：target_modules=[\"q_proj\"], key=\"model.layers.0.q_proj\" -> True。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::check_target_modules。  # 注释：函数位置
    - 典型调用路径：LoRA/adapter 初始化 -> check_target_modules。  # 注释：典型调用链
    - 被谁调用：adapter 工具逻辑（本文件或外部）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：re.fullmatch、re.match。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if isinstance(config.target_modules, str):  # 注释：正则字符串形式
        target_module_found = re.fullmatch(config.target_modules, key)  # 注释：正则匹配
    elif key in config.target_modules:  # 注释：直接包含
        # this module is specified directly in target_modules  # 注释：原注释，直接命中
        target_module_found = True  # 注释：命中
    else:  # 注释：后缀匹配
        target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)  # 注释：后缀匹配
        layer_indexes = getattr(config, "layers_to_transform", None)  # 注释：可选层索引
        layers_pattern = getattr(config, "layers_pattern", None)  # 注释：可选层模式
        is_using_layer_indexes = layer_indexes is not None and (  # 注释：是否启用层索引过滤
            len(layer_indexes) != 0 if isinstance(layer_indexes, list) else True  # 注释：兼容 list/int
        )  # 注释：条件结束
        if is_using_layer_indexes and target_module_found:  # 注释：命中且需要层过滤
            layer_index = None  # 注释：初始化匹配结果
            # TODO: It's still unclear how empty layers_pattern (None, [], or "") should behave  # 注释：原 TODO
            # For now, empty layers_pattern means any layer pattern is ok  # 注释：原 TODO
            if layers_pattern is None or len(layers_pattern) == 0:  # 注释：未指定 pattern
                layer_index = re.match(r".*\.[^.]*\.(\d+)\.", key)  # 注释：默认匹配层号
            else:  # 注释：指定 pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern  # 注释：标准化为列表
                for pattern in layers_pattern:  # 注释：遍历 pattern
                    layer_index = re.match(rf".*\.{pattern}\.(\d+)\.", key)  # 注释：匹配层号
                    if layer_index is not None:  # 注释：匹配成功
                        break  # 注释：退出循环
            if layer_index is None:  # 注释：未匹配到层号
                target_module_found = False  # 注释：视为未命中
            else:  # 注释：匹配到层号
                layer_index = int(layer_index.group(1))  # 注释：解析层号
                if isinstance(layer_indexes, int):  # 注释：单一层号
                    target_module_found = layer_index == layer_indexes  # 注释：比较是否等于指定层
                else:  # 注释：多层列表
                    target_module_found = layer_index in layer_indexes  # 注释：判断是否在列表中
    return target_module_found  # 注释：返回匹配结果


def normalize_model_name(name, pp_rank, vpp_rank, transformer_config, layer_name="layers"):
    """
    函数用途：将并行分片的权重名归一化为推理引擎中的全局层编号。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - name (str)：原始权重名。  # 注释：参数含义
    - pp_rank (int)：pipeline parallel rank。  # 注释：参数含义
    - vpp_rank (int)：virtual pipeline parallel rank。  # 注释：参数含义
    - transformer_config：并行配置（用于计算层偏移）。  # 注释：参数含义
    - layer_name (str)：层命名关键字，默认 \"layers\"。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - name (str)：归一化后的权重名。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：层号字段不合法会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：name=\"model.layers.0.attn\"，pp_rank=1 -> 输出层号增加偏移。  # 注释：示例说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::normalize_model_name。  # 注释：函数位置
    - 典型调用路径：normalize_pp_vpp_params -> normalize_model_name。  # 注释：典型调用链
    - 被谁调用：normalize_pp_vpp_params（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.megatron_utils.get_transformer_layer_offset。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from verl.utils.megatron_utils import get_transformer_layer_offset  # 注释：计算层偏移的工具
    # （分隔说明：计算层偏移）  # 注释：替代空行，保持逐行注释
    layer_offset = get_transformer_layer_offset(pp_rank, vpp_rank, transformer_config)  # 注释：得到偏移量
    if layer_name in name:  # 注释：仅处理包含层名的权重
        split_name = name.split(".")  # 注释：按点拆分名称
        # find the num next to split_name  # 注释：原注释，寻找层号
        for i, name in enumerate(split_name):  # 注释：遍历字段
            if name == layer_name:  # 注释：找到层名字段
                break  # 注释：停止搜索
        layer_num_idx = i + 1  # 注释：层号索引位置
        # check the name  # 注释：原注释，校验层号
        assert len(split_name) >= layer_num_idx + 1, f"split_name = {split_name}"  # 注释：确保存在层号
        assert split_name[layer_num_idx].isdigit(), f"split_name = {split_name}"  # 注释：层号必须是数字
        # increment layer_num_idx by layer_offset  # 注释：原注释，应用偏移
        split_name[layer_num_idx] = str(int(split_name[layer_num_idx]) + layer_offset)  # 注释：叠加偏移
        name = ".".join(split_name)  # 注释：重组权重名
    return name  # 注释：返回归一化后的名称


def normalize_pp_vpp_params(params, num_hidden_layers, layer_name="layers"):
    """
    函数用途：将 pp/vpp 分片的参数名归一化为完整模型参数名。  # 注释：函数用途说明
    说明：适用于从多 pp rank 收集参数后还原完整命名。  # 注释：使用场景说明
    参数：  # 注释：参数说明标题
    - params (Iterable[List[Dict[str, param]]])：pp->vpp->参数字典的嵌套结构。  # 注释：参数含义
    - num_hidden_layers (int)：模型层数，用于计算偏移。  # 注释：参数含义
    - layer_name (str)：层名关键字，默认 \"layers\"。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 迭代器（yield）：(normalized_name, param) 对。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：参数结构不符合预期时可能抛 KeyError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：params=[[{\"layers.0.w\": ...}], ...]。  # 注释：示例输入
    - 输出：yield 归一化后的参数名。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::normalize_pp_vpp_params。  # 注释：函数位置
    - 典型调用路径：模型权重汇总 -> normalize_pp_vpp_params。  # 注释：典型调用链
    - 被谁调用：模型合并/权重加载逻辑（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：normalize_model_name。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    pp_size = len(params)  # 注释：pipeline parallel 数量
    for pp_rank in range(len(params)):  # 注释：遍历 pp rank
        vpp_size = len(params[pp_rank])  # 注释：当前 pp 的 vpp 数量
        for vpp_rank in range(vpp_size):  # 注释：遍历 vpp rank
            for name, param in params[pp_rank][vpp_rank].items():  # 注释：遍历参数
                normalized_name = normalize_model_name(  # 注释：归一化参数名
                    name, pp_rank, vpp_rank, pp_size, vpp_size, num_hidden_layers, layer_name=layer_name  # 注释：参数
                )  # 注释：normalize_model_name 结束
                yield normalized_name, param  # 注释：产出归一化参数


def get_parallel_model_from_config(
    config, megatron_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    """
    函数用途：根据配置创建 Megatron 并行模型实例。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - config (PretrainedConfig)：HF 模型配置。  # 注释：参数含义
    - megatron_config (ModelParallelConfig)：Megatron 并行配置。  # 注释：参数含义
    - pre_process/post_process：是否启用前/后处理阶段。  # 注释：参数含义
    - share_embeddings_and_output_weights (bool)：是否共享词嵌入与输出权重。  # 注释：参数含义
    - value (bool)：是否 value 模型。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - nn.Module：并行模型实例。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：megatron_config 类型错误会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_parallel_model_from_config(hf_config, mp_config)。  # 注释：示例输入
    - 输出：并行模型实例。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_parallel_model_from_config。  # 注释：函数位置
    - 典型调用路径：模型并行初始化 -> get_parallel_model_from_config。  # 注释：典型调用链
    - 被谁调用：并行训练/推理初始化流程（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：_get_parallel_model_architecture_from_config。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：megatron.core.ModelParallelConfig。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from megatron.core import ModelParallelConfig  # 注释：导入 Megatron 并行配置类型
    assert isinstance(megatron_config, ModelParallelConfig)  # 注释：类型校验
    model_class = _get_parallel_model_architecture_from_config(config, value)  # 注释：解析模型类
    model = model_class(  # 注释：实例化模型
        config,  # 注释：HF 配置
        megatron_config,  # 注释：并行配置
        pre_process=pre_process,  # 注释：前处理开关
        post_process=post_process,  # 注释：后处理开关
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,  # 注释：共享权重开关
    )  # 注释：实例化结束
    return model  # 注释：返回模型


def _get_parallel_model_architecture_from_config(config: PretrainedConfig, value=False) -> type[nn.Module]:
    """
    函数用途：根据 HF 配置中的 architectures 查找并行模型类。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - config (PretrainedConfig)：HF 模型配置。  # 注释：参数含义
    - value (bool)：是否为 value 模型。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - nn.Module 子类：匹配的模型类。  # 注释：返回值语义
    副作用：打印日志。  # 注释：副作用说明
    异常/边界条件：未找到支持架构会抛 ValueError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：architectures=[\"LlamaForCausalLM\"]。  # 注释：示例输入
    - 输出：对应并行模型类。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::_get_parallel_model_architecture_from_config。  # 注释：函数位置
    - 典型调用路径：get_parallel_model_from_config -> _get_parallel_model_architecture_from_config。  # 注释：典型调用链
    - 被谁调用：get_parallel_model_from_config（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：ModelRegistry.load_model_cls。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    architectures = getattr(config, "architectures", [])  # 注释：读取架构列表
    for arch in architectures:  # 注释：遍历架构
        model_cls = ModelRegistry.load_model_cls(arch, value)  # 注释：从注册表加载模型类
        print("after load model cls")  # 注释：打印加载状态
        if model_cls is not None:  # 注释：找到匹配类
            return model_cls  # 注释：返回模型类
    raise ValueError(  # 注释：未找到支持架构
        f"Model architectures {architectures} are not supported for now. Supported architectures: "  # 注释：错误信息
        f"{ModelRegistry.get_supported_archs()}"  # 注释：错误信息
    )  # 注释：异常结束


def _load_hf_model(config, model_config, is_value_model):
    """
    函数用途：加载 HF 模型并返回 state_dict（用于并行权重加载）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - config：训练/推理配置（含 model.path）。  # 注释：参数含义
    - model_config (PretrainedConfig)：HF 模型配置。  # 注释：参数含义
    - is_value_model (bool)：是否为 value 模型。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (architectures, model, state_dict, is_value_model)：模型架构、模型实例、权重、是否 value。  # 注释：返回值语义
    副作用：可能下载模型到本地、打印日志。  # 注释：副作用说明
    异常/边界条件：模型路径无效会触发异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：_load_hf_model(cfg, hf_config, False)。  # 注释：示例输入
    - 输出：architectures, model, state_dict, is_value_model。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::_load_hf_model。  # 注释：函数位置
    - 典型调用路径：load_megatron_model_weights -> _load_hf_model。  # 注释：典型调用链
    - 被谁调用：load_megatron_model_weights、load_megatron_gptmodel_weights。  # 注释：调用方说明
    - 调用了谁（项目内）：get_hf_auto_model_class、_megatron_calc_global_rank。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.*.from_pretrained、accelerate.init_empty_weights。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from accelerate import init_empty_weights
    from megatron.core import parallel_state as mpu

    from verl.models.mcore.saver import _megatron_calc_global_rank

    assert hasattr(model_config, "architectures"), "architectures cannot be empty when load weight!"
    architectures = getattr(model_config, "architectures", [])

    # get auto class
    auto_cls = get_hf_auto_model_class(model_config)

    if config.model.path.startswith("hdfs:"):
        from verl.utils.fs import copy_to_local

        print(f"start download from {config.model.path}")
        local_model_path = copy_to_local(src=config.model.path, use_shm=config.model.get("use_shm", False))
        print("finish download")
    else:
        local_model_path = config.model.path
        print(f"load from local dir {local_model_path}")

    src_rank = _megatron_calc_global_rank(tp_rank=0, dp_rank=0, pp_rank=0, cp_rank=mpu.get_context_parallel_rank())
    cpu_init_weights = lambda: torch.device("cpu")
    init_context = init_empty_weights if torch.distributed.get_rank() != src_rank else cpu_init_weights
    with init_context(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # TODO: to find a better way to load mistral7b-rm lm_head
        if "mistral7b-rm" in config.model.path:
            model = MistralForSequenceClassification.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto",  # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )  # use score head instead of lm_head
            state_dict = model.state_dict()
            state_dict["lm_head.weight"] = state_dict["score.weight"]
            state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"][
                :32000
            ]  # workaround, 32001 -> 32000
            is_value_model = True
        else:
            model = auto_cls.from_pretrained(
                local_model_path,
                torch_dtype="auto",
                # device_map="auto", # disable auto device_map, the HF weight is only loaded to CPU in src_rank
                # low_cpu_mem_usage=True
            )
            state_dict = model.state_dict()

    return architectures, model, state_dict, is_value_model


def get_hf_model_path(config):
    """
    函数用途：返回可用的本地模型路径（必要时从 HDFS 拷贝）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - config：包含 model.path 与 use_shm 的配置对象。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - local_model_path (str)：本地可访问路径。  # 注释：返回值语义
    副作用：可能触发拷贝操作。  # 注释：副作用说明
    异常/边界条件：拷贝失败会抛出 I/O 异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_hf_model_path(cfg)（cfg.model.path=\"hdfs://...\"）。  # 注释：示例输入
    - 输出：本地路径。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_hf_model_path。  # 注释：函数位置
    - 典型调用路径：模型加载流程 -> get_hf_model_path。  # 注释：典型调用链
    - 被谁调用：模型合并/权重加载逻辑（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.fs.copy_to_local。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if config.model.path.startswith("hdfs:"):  # 注释：HDFS 路径需拷贝
        from verl.utils.fs import copy_to_local  # 注释：导入拷贝工具
        local_model_path = copy_to_local(src=config.model.path, use_shm=config.model.get("use_shm", False))  # 注释：拷贝到本地
    else:  # 注释：本地路径直接使用
        local_model_path = config.model.path  # 注释：本地路径
    return local_model_path  # 注释：返回本地路径


def load_megatron_model_weights(config, model_config, parallel_model, params_dtype, is_value_model=False):
    """
    函数用途：加载权重到 VERL 自定义并行模型。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - config：训练配置（含模型路径）。  # 注释：参数含义
    - model_config (PretrainedConfig)：HF 模型配置。  # 注释：参数含义
    - parallel_model：并行模型实例列表/封装。  # 注释：参数含义
    - params_dtype：参数 dtype（如 torch.float16）。  # 注释：参数含义
    - is_value_model (bool)：是否为 value 模型。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - model.config：加载权重后的模型配置。  # 注释：返回值语义
    副作用：会加载权重并打印日志。  # 注释：副作用说明
    异常/边界条件：权重加载失败会抛出异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：load_megatron_model_weights(cfg, hf_config, model, torch.float16)。  # 注释：示例输入
    - 输出：model.config。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::load_megatron_model_weights。  # 注释：函数位置
    - 典型调用路径：worker 初始化 -> load_megatron_model_weights。  # 注释：典型调用链
    - 被谁调用：并行训练初始化逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：_load_hf_model、get_weight_loader。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：HF 模型 load_state_dict 相关。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    architectures, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model)

    from verl.models.weight_loader_registry import get_weight_loader

    print(f"before weight loader: architectures = {architectures}...")
    for arch in architectures:
        print(f"call weight loader arch = {arch}, model config = {model.config}")
        weight_loader = get_weight_loader(arch)
        weight_loader(
            state_dict=state_dict,
            wrapped_models=parallel_model,
            config=model.config,
            params_dtype=params_dtype,
            is_value_model=is_value_model,
            tie_word_embeddings=model_config.tie_word_embeddings,
        )
    return model.config


def load_megatron_gptmodel_weights(config, model_config, parallel_model, params_dtype, is_value_model=False):
    """
    函数用途：加载权重到 mcore GPT 并行模型。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - config：训练配置。  # 注释：参数含义
    - model_config (PretrainedConfig)：HF 模型配置。  # 注释：参数含义
    - parallel_model：并行模型实例。  # 注释：参数含义
    - params_dtype：参数 dtype。  # 注释：参数含义
    - is_value_model (bool)：是否为 value 模型。  # 注释：参数含义
    返回：无。  # 注释：返回值说明
    副作用：会加载权重到模型并释放 state_dict。  # 注释：副作用说明
    异常/边界条件：权重加载失败会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：load_megatron_gptmodel_weights(cfg, hf_config, model, torch.float16)。  # 注释：示例输入
    - 输出：模型权重被填充。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::load_megatron_gptmodel_weights。  # 注释：函数位置
    - 典型调用路径：mcore 模型初始化 -> load_megatron_gptmodel_weights。  # 注释：典型调用链
    - 被谁调用：并行训练初始化逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：_load_hf_model、load_state_dict_to_megatron_gptmodel。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    _, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model)

    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    load_state_dict_to_megatron_gptmodel(
        state_dict=state_dict,
        wrapped_models=parallel_model,
        config=model.config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )
    del state_dict, model


# pad input_ids_rmpad, cu_seqlens and max_seqlen_in_batch to be divisible by tp
def pad_packed_inputs(unpad_tokens: torch.Tensor, cu_seqlens, max_seqlen_in_batch, size):
    """
    函数用途：将打包后的 token 序列补齐到 size 的整数倍。  # 注释：函数用途说明
    说明：适用于 sequence/context parallel 的长度对齐。  # 注释：使用场景说明
    参数：  # 注释：参数说明标题
    - unpad_tokens (Tensor)：形状 (total_nnz, ...) 的非 padding token。  # 注释：参数含义
    - cu_seqlens (Tensor)：累积序列长度，形状 (total_nnz+1,)。  # 注释：参数含义
    - max_seqlen_in_batch (int)：当前 batch 最大长度。  # 注释：参数含义
    - size (int)：对齐单位。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (unpad_tokens, cu_seqlens, max_seqlen_in_batch)：补齐后的结果。  # 注释：返回值语义
    副作用：无（返回新张量）。  # 注释：副作用说明
    异常/边界条件：不支持 ndim>2 时抛 NotImplementedError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：total_nnz=5, size=4 -> pad_size=3。  # 注释：示例输入
    - 输出：长度补齐为 8。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::pad_packed_inputs。  # 注释：函数位置
    - 典型调用路径：并行训练输入打包 -> pad_packed_inputs。  # 注释：典型调用链
    - 被谁调用：并行训练/推理工具（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.nn.functional.pad。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    F = nn.functional  # 注释：便捷引用 functional
    total_nnz = unpad_tokens.shape[0]  # 注释：总有效 token 数
    pad_size = 0 if total_nnz % size == 0 else size - total_nnz % size  # 注释：计算需要补齐的长度
    # we assume adding a new data in the batch with seqlen pad_size  # 注释：原注释，补齐视作新序列
    if pad_size > 0:  # 注释：仅在需要补齐时处理
        if unpad_tokens.ndim == 1:  # 注释：一维 token
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))  # 注释：右侧补齐
        elif unpad_tokens.ndim == 2:  # 注释：二维 token（batch, hidden）
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))  # 注释：在 batch 维度补齐
        else:  # 注释：不支持更高维度
            raise NotImplementedError(f"Padding dim {unpad_tokens.ndim()} is not supported")  # 注释：抛异常
        cu_seqlens = F.pad(cu_seqlens, (0, 1), value=pad_size + cu_seqlens[-1])  # 注释：更新累积长度
        max_seqlen_in_batch = max(max_seqlen_in_batch, pad_size)  # 注释：更新最大长度
    return unpad_tokens, cu_seqlens, max_seqlen_in_batch  # 注释：返回补齐结果


def load_mcore_dist_weights(parallel_model, dist_weight_path, is_value_model=False, prefix=""):
    """
    函数用途：从 Megatron dist checkpoint 加载权重到并行模型。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - parallel_model：并行模型列表/封装。  # 注释：参数含义
    - dist_weight_path (str)：分布式权重路径。  # 注释：参数含义
    - is_value_model (bool)：是否为 value 模型（影响过滤输出层）。  # 注释：参数含义
    - prefix (str)：权重前缀。  # 注释：参数含义
    返回：无。  # 注释：返回值说明
    副作用：会就地加载权重。  # 注释：副作用说明
    异常/边界条件：路径无效会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：load_mcore_dist_weights(models, \"/path/ckpt\")。  # 注释：示例输入
    - 输出：模型权重被更新。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::load_mcore_dist_weights。  # 注释：函数位置
    - 典型调用路径：并行模型恢复 -> load_mcore_dist_weights。  # 注释：典型调用链
    - 被谁调用：并行权重恢复逻辑（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.megatron_utils.unwrap_model。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：megatron.core.dist_checkpointing.load。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.serialization import StrictHandling

    from verl.utils.megatron_utils import unwrap_model

    # strict = StrictHandling.IGNORE_ALL if is_value_model else StrictHandling.ASSUME_OK_UNEXPECTED
    strict = StrictHandling.ASSUME_OK_UNEXPECTED
    for model in parallel_model:
        ssd = unwrap_model(model).sharded_state_dict(prefix=prefix)
        if is_value_model:
            for k in list(ssd.keys()):
                if "output_layer" in k:
                    ssd.pop(k)
        dist_checkpointing.load(ssd, dist_weight_path, strict=strict)

    return


def get_parallel_gptmodel_from_config(
    tfconfig, hf_config, pre_process=None, post_process=None, share_embeddings_and_output_weights=False, value=False
):
    """
    函数用途：根据 Megatron 配置构建 GPT 并行模型。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - tfconfig：Megatron transformer 配置。  # 注释：参数含义
    - hf_config：HF 模型配置（提供词表、rope 等）。  # 注释：参数含义
    - pre_process/post_process：是否启用前/后处理阶段。  # 注释：参数含义
    - share_embeddings_and_output_weights (bool)：是否共享词嵌入与输出权重。  # 注释：参数含义
    - value (bool)：是否为 value 模型（决定输出层）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - GPTModel 实例（并行版）。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：只支持 RMSNorm 与 linear rope scaling。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_parallel_gptmodel_from_config(tfconfig, hf_config)。  # 注释：示例输入
    - 输出：并行 GPTModel。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_parallel_gptmodel_from_config。  # 注释：函数位置
    - 典型调用路径：并行模型初始化 -> get_parallel_gptmodel_from_config。  # 注释：典型调用链
    - 被谁调用：并行训练/推理初始化逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：LinearForLastLayer（value 模式）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：megatron.core.models.gpt.GPTModel。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    use_te = True
    assert tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
    transformer_layer_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
    rope_scaling_args = {}
    if hf_config.rope_scaling is not None:
        assert hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
        rope_scaling_args["seq_len_interpolation_factor"] = hf_config.rope_scaling["factor"]
    parallel_model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=hf_config.vocab_size,
        max_sequence_length=hf_config.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        position_embedding_type="rope",
        rotary_base=hf_config.rope_theta,
        **rope_scaling_args,
    )
    # # for layer in parallel_model.decoder.layers:
    # layer.self_attention.core_attention.flash_attention.softmax_scale = None
    if post_process and value:
        from verl.models.llama.megatron.layers.parallel_linear import LinearForLastLayer

        parallel_model.output_layer = LinearForLastLayer(
            input_size=tfconfig.hidden_size, output_size=1, config=tfconfig
        )
    return parallel_model


def patch_valuehead_model(model) -> None:
    """
    函数用途：为 TRL 的 value head 模型补齐常用接口（tie_weights 等）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - model：AutoModelForCausalLMWithValueHead 实例。  # 注释：参数含义
    返回：无。  # 注释：返回值说明
    副作用：会修改 model 的方法和忽略保存键。  # 注释：副作用说明
    异常/边界条件：model 结构异常可能导致属性访问错误。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：patch_valuehead_model(model)。  # 注释：示例输入
    - 输出：model 增加 tie_weights/get_input_embeddings 等方法。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::patch_valuehead_model。  # 注释：函数位置
    - 典型调用路径：load_valuehead_model -> patch_valuehead_model。  # 注释：典型调用链
    - 被谁调用：load_valuehead_model（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：types.MethodType。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from types import MethodType

    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead

    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        """函数用途：将权重绑定到 pretrained_model（适配 TRL）。"""  # 注释：内嵌函数用途说明
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        """函数用途：返回 pretrained_model 的输入嵌入层。"""  # 注释：内嵌函数用途说明
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        """函数用途：返回 pretrained_model 的输出嵌入层。"""  # 注释：内嵌函数用途说明
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def can_generate(self):
        """函数用途：显式声明该模型不可直接 generate。"""  # 注释：内嵌函数用途说明
        return False

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    model._keys_to_ignore_on_save = ignore_modules
    model.tie_weights = MethodType(tie_weights, model)
    model.get_input_embeddings = MethodType(get_input_embeddings, model)
    model.get_output_embeddings = MethodType(get_output_embeddings, model)
    model.can_generate = MethodType(can_generate, model)
    model._no_split_modules = getattr(model.pretrained_model, "_no_split_modules", [])


def load_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code):
    """
    函数用途：加载带 value head 的模型（优先尝试 TokenClassification，再回退 TRL）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - local_path (str)：本地模型路径。  # 注释：参数含义
    - torch_dtype：加载权重的 dtype。  # 注释：参数含义
    - model_config (PretrainedConfig)：模型配置。  # 注释：参数含义
    - trust_remote_code (bool)：是否信任远程代码。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - model：带 value head 的模型实例。  # 注释：返回值语义
    副作用：可能触发模型下载/加载。  # 注释：副作用说明
    异常/边界条件：若 trl 未安装且非 value head，会抛 RuntimeError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：load_valuehead_model(\"/path\", torch.float16, cfg, True)。  # 注释：示例输入
    - 输出：value head 模型实例。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::load_valuehead_model。  # 注释：函数位置
    - 典型调用路径：reward/critic 模型初始化 -> load_valuehead_model。  # 注释：典型调用链
    - 被谁调用：训练/评估初始化逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：patch_valuehead_model、is_trl_available。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.AutoModelForTokenClassification、trl.AutoModelForCausalLMWithValueHead。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq

    try:
        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )
        return model
    except BaseException as e:
        if not is_trl_available():
            raise RuntimeError(
                f"model({local_path}) is not a value head model, please install trl to make it valid"
            ) from e

    assert is_trl_available()

    from trl import AutoModelForCausalLMWithValueHead

    if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
        module_class = AutoModelForVision2Seq
    else:
        module_class = AutoModelForCausalLM
    ori_model = module_class.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
    patch_valuehead_model(model)
    return model


_architecture_to_auto_class = {
    "ForCausalLM": AutoModelForCausalLM,
    "ForVision2Seq": AutoModelForVision2Seq,
    "ForTokenClassification": AutoModelForTokenClassification,
    "ForSequenceClassification": AutoModelForSequenceClassification,
}


def get_hf_auto_model_class(hf_config):
    """
    函数用途：根据 HF 配置选择合适的 AutoModel 类。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - hf_config (PretrainedConfig)：HF 配置对象。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - actor_module_class：AutoModel* 类。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：auto_map 不含预期字段时走默认分支。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：hf_config.architectures=[\"Qwen2ForCausalLM\"]。  # 注释：示例输入
    - 输出：AutoModelForCausalLM。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_hf_auto_model_class。  # 注释：函数位置
    - 典型调用路径：_load_hf_model -> get_hf_auto_model_class。  # 注释：典型调用链
    - 被谁调用：_load_hf_model（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：_architecture_to_auto_class。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.AutoModel*。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    has_remote_code = hasattr(hf_config, "auto_map") and any(
        hf_config.architectures[0] in val for val in hf_config.auto_map.values()
    )
    if has_remote_code:
        auto_class = next(k for k, v in hf_config.auto_map.items() if hf_config.architectures[0] in v)
        match auto_class:
            case "AutoModelForVision2Seq":
                actor_module_class = AutoModelForVision2Seq
            case "AutoModelForCausalLM":
                actor_module_class = AutoModelForCausalLM
            case "AutoModelForImageTextToText":
                actor_module_class = AutoModelForImageTextToText
            case _:
                actor_module_class = AutoModel
    else:
        actor_module_class = AutoModel
        # For VLM models, we use type to check instead of architecture
        if type(hf_config) in AutoModelForImageTextToText._model_mapping.keys():
            actor_module_class = AutoModelForImageTextToText
        else:
            for key, cls in _architecture_to_auto_class.items():
                if key in hf_config.architectures[0]:
                    actor_module_class = cls
                    break

    return actor_module_class


def extract_multi_modal_inputs(
    batch_data: list[dict[str, torch.Tensor]],
    indices: Optional[list[int]] = None,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """
    函数用途：从 batch 中提取并拼接多模态输入字段。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - batch_data (list[dict])：batch 中的样本输入（可能包含图像/视频字段）。  # 注释：参数含义
    - indices (list[int]|None)：若提供，仅处理指定索引。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict[str, Tensor|list[Tensor]]：整理后的多模态输入。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：batch_data 中混合 None 会被跳过。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：batch_data=[{\"image\": tensor}, {\"image\": tensor}]。  # 注释：示例输入
    - 输出：{\"image\": torch.cat(...)}。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::extract_multi_modal_inputs。  # 注释：函数位置
    - 典型调用路径：worker/rollout -> extract_multi_modal_inputs。  # 注释：典型调用链
    - 被谁调用：多模态输入整理流程（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：NonTensorData 解包逻辑。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.cat。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    multi_modal_inputs = {}
    multi_modal_inputs_collected = {}
    has_image_bound = False

    selected_batch_data = batch_data
    if indices is not None:
        selected_batch_data = [batch_data[i] for i in indices if i < len(batch_data)]

    for inputs in selected_batch_data:
        inputs = inputs.data if isinstance(inputs, NonTensorData) else inputs
        # Mixed pure text and multi-modal dataset.
        if inputs is None:
            continue
        if "image_bound" in inputs:
            has_image_bound = True
        for key, value in inputs.items():
            if value is not None:
                if key not in multi_modal_inputs_collected:
                    multi_modal_inputs_collected[key] = []
                multi_modal_inputs_collected[key].append(value)

    for key, values in multi_modal_inputs_collected.items():
        if has_image_bound:  # minicpm-o logic
            multi_modal_inputs[key] = values
        else:
            multi_modal_inputs[key] = torch.cat(values, dim=0)

    return multi_modal_inputs


def get_lora_rank_from_adapter(adapter_path: str | os.PathLike) -> int:
    """
    函数用途：从 LoRA adapter_config.json 中提取 rank（r）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - adapter_path (str|PathLike)：LoRA adapter 目录。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - int：LoRA rank 值。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 路径不存在抛 FileNotFoundError。  # 注释：异常说明
    - JSON 解析失败或缺失 r 字段抛 ValueError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：get_lora_rank_from_adapter(\"/path/to/adapter\")。  # 注释：示例输入
    - 输出：如 8。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::get_lora_rank_from_adapter。  # 注释：函数位置
    - 典型调用路径：LoRA 权重加载 -> get_lora_rank_from_adapter。  # 注释：典型调用链
    - 被谁调用：模型合并或 adapter 工具（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：json.load、os.path。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    adapter_path = os.path.abspath(os.path.expanduser(str(adapter_path)))

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter path not found: {adapter_path}")

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
            if "r" not in config:
                raise ValueError(f"LoRA rank 'r' not found in {config_path}")
            return int(config["r"])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}") from e
    except (KeyError, ValueError) as e:
        raise ValueError(f"Cannot parse LoRA rank from {config_path}: {e}") from e


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    """
    类用途：扩展 CausalLMOutput，增加 PPO 需要的 log_probs 与 entropy 字段。  # 注释：类用途说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/model.py::CausalLMOutputForPPO。  # 注释：类位置
    - 典型调用路径：模型前向 -> 返回 CausalLMOutputForPPO。  # 注释：典型调用链
    - 被谁调用：PPO/GRPO 训练逻辑（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.modeling_outputs.CausalLMOutputWithPast。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
