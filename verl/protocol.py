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
模块用途：定义 VERL 内部的数据传输协议（DataProto），用于跨模块/进程传递批数据。  # 注释：模块用途说明
输入：  # 注释：模块输入说明标题
- TensorDict / numpy / Python dict 形式的 batch 数据。  # 注释：输入含义
- 可选 meta_info 与非张量字段。  # 注释：输入含义
输出：  # 注释：模块输出说明标题
- DataProto / DataProtoItem 等协议对象，支持合并、切分、序列化。  # 注释：输出说明
依赖：torch、tensordict、ray、numpy。  # 注释：关键依赖说明
典型用法：  # 注释：最小示例标题
- data = DataProto.from_dict({\"input_ids\": tensor(...)})。  # 注释：示例用法
- data = data.select(batch_keys=[\"input_ids\"]).chunk(2)。  # 注释：示例用法
调用路径概览：  # 注释：调用路径概览标题
- 入口：训练循环/worker 在组 batch 与跨进程通信时构造 DataProto。  # 注释：典型入口说明
- 典型链路：ray_trainer -> DataProto -> DataProto.make_iterator/concat。  # 注释：调用链示例
"""

import contextlib
import copy
import logging
import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import ray
import tensordict
import torch
import torch.distributed
from packaging import version
from packaging.version import parse as parse_version
from tensordict import TensorDict
from torch.utils.data import DataLoader

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.py_functional import union_two_dict
from verl.utils.torch_functional import allgather_dict_tensors

__all__ = ["DataProto", "union_tensor_dict"]

with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()
    if parse_version(tensordict.__version__) < parse_version("0.10.0"):
        tensordict.set_list_to_stack(True).set()


class _DataProtoConfigMeta(type):
    """
    类用途：为 DataProtoConfig 提供全局配置（如 auto_padding）。  # 注释：类用途说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::_DataProtoConfigMeta。  # 注释：类位置
    - 典型调用路径：DataProtoConfig.auto_padding 读取/设置。  # 注释：典型调用链
    - 被谁调用：DataProtoConfig（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.getenv。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    _config = {}

    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        """
        函数用途：读取是否开启自动 padding（环境变量或配置）。  # 注释：函数用途说明
        参数：无（类属性）。  # 注释：参数说明
        返回：bool，是否开启。  # 注释：返回值说明
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：DataProtoConfig.auto_padding -> True/False。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::_DataProtoConfigMeta.auto_padding。  # 注释：函数位置
        - 典型调用路径：训练/数据迭代 -> DataProtoConfig.auto_padding。  # 注释：典型调用链
        - 被谁调用：DataProto/数据加载逻辑（可选）。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：os.getenv。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        """
        函数用途：设置是否启用自动 padding。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - enabled (bool)：开关值。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：更新类级配置字典。  # 注释：副作用说明
        异常/边界条件：enabled 不是 bool 时触发 AssertionError。  # 注释：异常说明
        最小示例：DataProtoConfig.auto_padding = True。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::_DataProtoConfigMeta.auto_padding.setter。  # 注释：函数位置
        - 典型调用路径：配置初始化 -> DataProtoConfig.auto_padding = True。  # 注释：典型调用链
        - 被谁调用：用户/框架配置代码。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    """
    类用途：作为 DataProto 的全局配置入口（当前支持 auto_padding）。  # 注释：类用途说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::DataProtoConfig。  # 注释：类位置
    - 典型调用路径：DataProtoConfig.auto_padding 读取/设置。  # 注释：典型调用链
    - 被谁调用：DataProto/训练逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：_DataProtoConfigMeta。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    pass


_padding_size_key = "_padding_size_key_x123d"


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """
    函数用途：将 DataProto 补齐到 size_divisor 的整数倍。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - data (DataProto)：待补齐的数据。  # 注释：参数含义
    - size_divisor (int)：目标整除因子。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - data_padded (DataProto)：补齐后的数据。  # 注释：返回值语义
    - pad_size (int)：补齐的样本数。  # 注释：返回值语义
    副作用：无（返回新对象）。  # 注释：副作用说明
    异常/边界条件：data 不是 DataProto 会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：len(data)=6, size_divisor=4。  # 注释：示例输入
    - 输出：pad_size=2，data_padded 长度=8。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::pad_dataproto_to_divisor。  # 注释：函数位置
    - 典型调用路径：DataProto.make_iterator -> pad_dataproto_to_divisor。  # 注释：典型调用链
    - 被谁调用：DataProto.make_iterator / 分布式对齐逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：DataProto.concat。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：logging.warning。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    assert isinstance(data, DataProto), "data must be a DataProto"  # 注释：类型校验
    if len(data) % size_divisor != 0:  # 注释：若长度不可整除
        pad_size = size_divisor - len(data) % size_divisor  # 注释：计算需要补齐的数量
        padding_protos = []  # 注释：保存补齐用的切片
        remaining_pad = pad_size  # 注释：剩余待补齐数量
        while remaining_pad > 0:  # 注释：循环补齐
            take_size = min(remaining_pad, len(data))  # 注释：本轮取样大小
            padding_protos.append(data[:take_size])  # 注释：追加切片
            remaining_pad -= take_size  # 注释：更新剩余数量
        data_padded = DataProto.concat([data] + padding_protos)  # 注释：拼接为补齐后的 DataProto
    else:  # 注释：无需补齐
        if len(data) == 0:  # 注释：空数据集
            logging.warning("padding a DataProto with no item, no changed made")  # 注释：提示空 padding
        pad_size = 0  # 注释：补齐数量为 0
        data_padded = data  # 注释：直接返回原数据
    return data_padded, pad_size  # 注释：返回补齐结果与 pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    """
    函数用途：根据 pad_size 去除补齐样本（等价于 data[:-pad_size]）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - data (DataProto)：待裁剪数据。  # 注释：参数含义
    - pad_size (int)：需要去除的末尾样本数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - DataProto：裁剪后的数据。  # 注释：返回值语义
    副作用：无（返回新切片）。  # 注释：副作用说明
    异常/边界条件：pad_size=0 时直接返回原数据。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：pad_size=2 -> 返回 data[:-2]。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::unpad_dataproto。  # 注释：函数位置
    - 典型调用路径：pad_dataproto_to_divisor -> unpad_dataproto（训练后裁剪）。  # 注释：典型调用链
    - 被谁调用：训练/评估流程中的对齐逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：DataProto.__getitem__。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if pad_size != 0:  # 注释：只有 pad_size 非 0 时裁剪
        data = data[:-pad_size]  # 注释：去除末尾补齐样本
    return data  # 注释：返回裁剪后的数据


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """
    函数用途：合并两个 TensorDict（相同 key 必须值相等）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - tensor_dict1 (TensorDict)：主 TensorDict。  # 注释：参数含义
    - tensor_dict2 (TensorDict)：待合并 TensorDict。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - TensorDict：合并后的 tensor_dict1（原地修改）。  # 注释：返回值语义
    副作用：会原地更新 tensor_dict1。  # 注释：副作用说明
    异常/边界条件：batch_size 不一致或重复 key 值不等会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：td1={\"a\":...}, td2={\"b\":...} -> 输出包含 a,b。  # 注释：示例说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::union_tensor_dict。  # 注释：函数位置
    - 典型调用路径：DataProto.union -> union_tensor_dict。  # 注释：典型调用链
    - 被谁调用：DataProto.union（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：tensordict.TensorDict。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (  # 注释：要求 batch_size 相同
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"  # 注释：断言信息
    )  # 注释：断言结束
    for key in tensor_dict2.keys():  # 注释：遍历待合并 keys
        if key not in tensor_dict1.keys():  # 注释：新 key 直接加入
            tensor_dict1[key] = tensor_dict2[key]  # 注释：写入字段
        else:  # 注释：重复 key 需校验
            assert tensor_dict1[key].equal(tensor_dict2[key]), (  # 注释：确保值相同
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"  # 注释：断言信息
            )  # 注释：断言结束
    return tensor_dict1  # 注释：返回合并结果


def _array_equal(array1: np.ndarray, array2: np.ndarray, visited: set[int]) -> bool:
    """
    函数用途：递归比较两个 NumPy 数组是否严格相等（支持 object/NaN/循环引用）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - array1 (np.ndarray)：第一个数组。  # 注释：参数含义
    - array2 (np.ndarray)：第二个数组。  # 注释：参数含义
    - visited (set[int])：已访问对象 id，用于检测循环引用。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：dtypes、shape 与元素均相等时为 True。  # 注释：返回值语义
    副作用：会更新 visited 集合。  # 注释：副作用说明
    异常/边界条件：类型或形状不一致直接返回 False。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：array1=[1, np.nan], array2=[1, np.nan] -> True。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::_array_equal。  # 注释：函数位置
    - 典型调用路径：_deep_equal -> _array_equal。  # 注释：典型调用链
    - 被谁调用：_deep_equal（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：_deep_equal。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：numpy.array_equal。  # 注释：外部依赖说明
    """
    # Check dtype and shape first, as this is the fastest failure path.
    if array1.dtype != array2.dtype or array1.shape != array2.shape:
        return False

    # For non-object dtypes, use NumPy's implementation with equal_nan=True.
    if array1.dtype != "object":
        return np.array_equal(array1, array2, equal_nan=True)

    # For object-dtype arrays, we must recursively compare each element.
    # We delegate to _deep_equal to handle elements, as they could be any
    # type, including other nested arrays or NaNs.
    return all(_deep_equal(x, y, visited) for x, y in zip(array1.flat, array2.flat, strict=False))


def _deep_equal(a: Any, b: Any, visited: set[int]) -> bool:
    """
    函数用途：递归比较任意 Python 对象是否相等。  # 注释：函数用途说明
    规则：  # 注释：规则说明标题
    - NaN 与 NaN 视为相等。  # 注释：规则说明
    - 支持循环引用检测。  # 注释：规则说明
    - NumPy 数组转交 _array_equal 处理。  # 注释：规则说明
    参数：  # 注释：参数说明标题
    - a (Any)：对象 a。  # 注释：参数含义
    - b (Any)：对象 b。  # 注释：参数含义
    - visited (set[int])：访问过的对象 id 集合。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：深度相等返回 True。  # 注释：返回值语义
    副作用：会更新 visited 集合。  # 注释：副作用说明
    异常/边界条件：类型不同直接返回 False。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：_deep_equal({\"a\":1},{\"a\":1},set()) -> True。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::_deep_equal。  # 注释：函数位置
    - 典型调用路径：union_numpy_dict/_array_equal -> _deep_equal。  # 注释：典型调用链
    - 被谁调用：_array_equal、union_numpy_dict。  # 注释：调用方说明
    - 调用了谁（项目内）：_array_equal。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：math.isnan。  # 注释：外部依赖说明
    """
    if type(a) is not type(b):
        return False

    # If we have seen this object ID before on this path, it's a cycle.
    # Since we already know the types match, we can safely assume this part
    # of the structure is equal.
    obj_id = id(a)
    if obj_id in visited:
        return True

    visited.add(obj_id)

    # Perform the specific comparison based on type
    result = False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        result = True
    elif isinstance(a, np.ndarray):
        # We know b is also an ndarray due to the initial type check
        result = _array_equal(a, b, visited)
    else:
        # Standard equality for all other types
        result = a == b

    # Clean up the visited set on the way out of the recursion
    visited.remove(obj_id)
    return result


def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    函数用途：合并两个 numpy 字典，重复 key 要求内容相等。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - tensor_dict1 (dict[str, np.ndarray])：主字典（原地更新）。  # 注释：参数含义
    - tensor_dict2 (dict[str, np.ndarray])：待合并字典。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict[str, np.ndarray]：合并后的 tensor_dict1。  # 注释：返回值语义
    副作用：会原地修改 tensor_dict1。  # 注释：副作用说明
    异常/边界条件：重复 key 内容不一致会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：{\"a\":np.array([1])} + {\"b\":np.array([2])}。  # 注释：示例输入
    - 输出：包含 a,b 的字典。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::union_numpy_dict。  # 注释：函数位置
    - 典型调用路径：DataProto.union -> union_numpy_dict。  # 注释：典型调用链
    - 被谁调用：DataProto.union（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：_deep_equal。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：numpy。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # to properly deal with nan and object type
            assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (
                f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."
            )
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """
    函数用途：将 list[dict] 转换为 dict[list]（按 key 聚合）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - list_of_dict (list[dict])：样本列表，每个元素为字典。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict：key -> list 的聚合结果。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：空列表返回空 dict；键不一致会触发 AssertionError。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：[{\"a\":1},{\"a\":2}] -> 输出：{\"a\":[1,2]}。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::list_of_dict_to_dict_of_list。  # 注释：函数位置
    - 典型调用路径：批处理聚合 -> list_of_dict_to_dict_of_list。  # 注释：典型调用链
    - 被谁调用：DataProto.from_dict/构造逻辑（可选）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()
    output = {key: [] for key in keys}
    for data in list_of_dict:
        for key, item in data.items():
            assert key in output
            output[key].append(item)
    return output


def fold_batch_dim(data: "DataProto", new_batch_size):
    """
    函数用途：将 batch 维度从 [bsz, ...] 折叠为 [new_bsz, bsz//new_bsz, ...]。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - data (DataProto)：待折叠的数据。  # 注释：参数含义
    - new_batch_size (int)：新的 batch size。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - DataProto：折叠后的数据。  # 注释：返回值语义
    副作用：无（返回新对象）。  # 注释：副作用说明
    异常/边界条件：new_batch_size 必须整除原 batch_size，否则断言失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：bsz=8, new_batch_size=4 -> 输出 batch_size=4，额外维度=2。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::fold_batch_dim。  # 注释：函数位置
    - 典型调用路径：DataProto.make_iterator -> fold_batch_dim。  # 注释：典型调用链
    - 被谁调用：DataProto.make_iterator（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：TensorDict.view/reshape。  # 注释：外部依赖说明
    """
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims=2):
    """
    函数用途：将前 n 个维度展开为新的 batch 维度。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - data (DataProto)：待展开的数据。  # 注释：参数含义
    - batch_dims (int)：需要展开的维度数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - DataProto：展开后的数据。  # 注释：返回值语义
    副作用：无（返回新对象）。  # 注释：副作用说明
    异常/边界条件：batch_dims 超过实际维度可能报错。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：batch_dims=2，将 (B1,B2,...) 展开为 (B1*B2,...)。  # 注释：示例说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::unfold_batch_dim。  # 注释：函数位置
    - 典型调用路径：fold_batch_dim -> unfold_batch_dim（反向操作）。  # 注释：典型调用链
    - 被谁调用：DataProto.make_iterator（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：TensorDict.view/reshape。  # 注释：外部依赖说明
    """
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    tensor.auto_batch_size_(batch_dims=batch_dims)
    tensor = tensor.view(-1)

    batch_size = tensor.batch_size[0]

    non_tensor_new = {}

    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def serialize_single_tensor(obj: torch.Tensor) -> tuple[str, tuple[int, ...], int | memoryview]:
    """
    函数用途：将单个 Tensor 序列化为 (dtype, shape, raw_bytes)。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - obj (torch.Tensor)：待序列化的张量。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (dtype, shape, data)：dtype 字符串、形状、字节视图。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：非连续张量会被 contiguous 处理。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：tensor([1,2]) -> (\"int64\", (2,), bytes)。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::serialize_single_tensor。  # 注释：函数位置
    - 典型调用路径：serialize_tensordict -> serialize_single_tensor。  # 注释：典型调用链
    - 被谁调用：serialize_tensordict（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.Tensor.flatten/contiguous。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    data = obj.flatten().contiguous().view(torch.uint8).numpy()
    dtype = str(obj.dtype).removeprefix("torch.")
    return dtype, obj.shape, data


def serialize_tensordict(batch: TensorDict) -> tuple[tuple[int, ...], Optional[str], dict[str, tuple[str, Any]]]:
    """
    函数用途：将 TensorDict 序列化为可传输的 Python 结构。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - batch (TensorDict)：待序列化的 TensorDict。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - (batch_size, device, encoded_items)：批大小、设备字符串与编码后的字段。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：NestedTensor 会被逐个序列化。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：TensorDict({\"x\": tensor}) -> 输出包含 dtype/shape/data。  # 注释：示例说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::serialize_tensordict。  # 注释：函数位置
    - 典型调用路径：DataProto.__getstate__ -> serialize_tensordict。  # 注释：典型调用链
    - 被谁调用：DataProto 序列化逻辑（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：serialize_single_tensor。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：TensorDict.items。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    encoded_items: dict[str, tuple[Any]] = {}
    for k, v in batch.items():
        if not v.is_nested:
            encoded_items[k] = serialize_single_tensor(v)
        else:
            layout = str(v.layout).removeprefix("torch.")
            data = [serialize_single_tensor(tensor) for tensor in v.unbind()]
            encoded_items[k] = (layout, data)

    batch_size = tuple(batch.batch_size)
    device = str(batch.device) if batch.device is not None else None
    return batch_size, device, encoded_items


def deserialize_single_tensor(arr: Any) -> torch.Tensor:
    """
    函数用途：将 (dtype, shape, data) 反序列化为 Tensor。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - arr (Any)：序列化后的三元组。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - torch.Tensor：恢复的张量。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：数据格式不符会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：(\"int64\", (2,), bytes) -> tensor([..])。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::deserialize_single_tensor。  # 注释：函数位置
    - 典型调用路径：deserialize_tensordict -> deserialize_single_tensor。  # 注释：典型调用链
    - 被谁调用：deserialize_tensordict（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.from_numpy。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    dtype, shape, data = arr

    torch_dtype = getattr(torch, dtype)
    assert isinstance(torch_dtype, torch.dtype)

    buffer = bytearray(data)
    # Create uint8 array
    arr = torch.frombuffer(buffer, dtype=torch.uint8)
    # Convert back to proper shape & type
    return arr.view(torch_dtype).view(shape)


def deserialize_tensordict(arr: Any) -> TensorDict:
    """
    函数用途：将序列化结构反序列化回 TensorDict。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - arr (Any)：serialize_tensordict 的输出。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - TensorDict：恢复后的 TensorDict。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：编码格式不匹配会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：serialize_tensordict(...) 的输出。  # 注释：示例输入
    - 输出：TensorDict。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::deserialize_tensordict。  # 注释：函数位置
    - 典型调用路径：DataProto.__setstate__ -> deserialize_tensordict。  # 注释：典型调用链
    - 被谁调用：DataProto 反序列化逻辑（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：deserialize_single_tensor。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：TensorDict。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    batch_size, device, encoded_items = arr
    decoded_items: dict[str, Any] = {}

    for k, v in encoded_items.items():
        if len(v) == 3:
            # decode single tensor
            decoded_items[k] = deserialize_single_tensor(v)
        elif len(v) == 2:
            # decode nested tensor
            layout, data = v
            torch_layout = getattr(torch, layout)
            decoded_items[k] = torch.nested.as_nested_tensor(
                [deserialize_single_tensor(tensor) for tensor in data], layout=torch_layout
            )
        else:
            raise ValueError(f"Invalid tensor encoding format, expected length 2 or 3, got {len(v)}")

    return TensorDict(source=decoded_items, batch_size=batch_size, device=device)


def collate_fn(x: list["DataProtoItem"]):
    """
    函数用途：将 DataProtoItem 列表合并为 DataProto。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - x (list[DataProtoItem])：待合并的样本列表。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - DataProto：合并后的批数据。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：空列表会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：[DataProtoItem(...), DataProtoItem(...)] -> 输出 DataProto。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::collate_fn。  # 注释：函数位置
    - 典型调用路径：DataLoader(collate_fn) -> collate_fn。  # 注释：典型调用链
    - 被谁调用：DataProto.make_iterator（内部 DataLoader）。  # 注释：调用方说明
    - 调用了谁（项目内）：DataProto.from_dict。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.utils.data.DataLoader。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    batch = []
    non_tensor_batch = []
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    """
    类用途：表示单条样本的 DataProto 片段（batch + 非张量 + meta）。  # 注释：类用途说明
    字段：  # 注释：字段说明标题
    - batch (TensorDict)：张量字段。  # 注释：字段含义
    - non_tensor_batch (dict)：非张量字段。  # 注释：字段含义
    - meta_info (dict)：元信息。  # 注释：字段含义
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::DataProtoItem。  # 注释：类位置
    - 典型调用路径：DataProto.make_iterator -> DataLoader -> collate_fn。  # 注释：典型调用链
    - 被谁调用：collate_fn（本文件）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：dataclasses.dataclass。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    # TODO(zhangchi.usc1992) add consistency check
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    类用途：提供统一的数据交换协议（batch 张量 + 非张量字段 + 元信息）。  # 注释：类用途说明
    结构说明：  # 注释：结构说明标题
    - batch (TensorDict)：所有张量字段，要求 batch_size 一致。  # 注释：字段说明
    - non_tensor_batch (dict)：非张量字段（如字符串、对象）。  # 注释：字段说明
    - meta_info (dict)：元信息（如 padding、来源等）。  # 注释：字段说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::DataProto。  # 注释：类位置
    - 典型调用路径：ray_trainer/worker -> DataProto.from_dict -> DataProto.make_iterator。  # 注释：典型调用链
    - 被谁调用：训练/评估流程、worker 通信。  # 注释：调用方说明
    - 调用了谁（项目内）：union_tensor_dict/union_numpy_dict 等工具函数。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：tensordict.TensorDict、torch.utils.data.DataLoader。  # 注释：外部依赖说明
    """

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        函数用途：dataclass 初始化后进行一致性检查。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：无。  # 注释：返回值说明
        副作用：可能触发断言错误。  # 注释：副作用说明
        异常/边界条件：batch/non_tensor 不一致会触发 AssertionError。  # 注释：异常说明
        最小示例：DataProto(batch=..., non_tensor_batch=...).  # 注释：示例说明
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.__post_init__。  # 注释：函数位置
        - 典型调用路径：DataProto 构造 -> __post_init__。  # 注释：典型调用链
        - 被谁调用：dataclasses 自动调用。  # 注释：调用方说明
        - 调用了谁（项目内）：check_consistency。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        # perform necessary checking
        self.check_consistency()

    def __len__(self):
        """
        函数用途：返回 DataProto 的 batch 大小。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：int，样本数。  # 注释：返回值说明
        副作用：无。  # 注释：副作用说明
        异常/边界条件：batch 与 non_tensor_batch 均为空时返回 0。  # 注释：边界说明
        最小示例：len(data_proto) -> 32。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.__len__。  # 注释：函数位置
        - 典型调用路径：训练循环/评估 -> len(DataProto)。  # 注释：典型调用链
        - 被谁调用：PyTorch/自定义逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        else:
            return 0

    def __getitem__(self, item):
        """
        函数用途：支持多种索引方式访问 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - item：支持 int/slice/list/np.ndarray/torch.Tensor。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：slice/列表索引返回 DataProto。  # 注释：返回值语义
        - DataProtoItem：单个 int 索引返回 DataProtoItem。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：不支持的索引类型抛 TypeError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：data[0] -> DataProtoItem；data[:2] -> DataProto。  # 注释：示例说明
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.__getitem__。  # 注释：函数位置
        - 典型调用路径：训练循环/采样 -> data[idx]。  # 注释：典型调用链
        - 被谁调用：用户代码或 DataLoader。  # 注释：调用方说明
        - 调用了谁（项目内）：slice、select_idxs、DataProtoItem。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：numpy/torch 索引。  # 注释：外部依赖说明
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, list | np.ndarray | torch.Tensor):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, int | np.integer):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        """
        函数用途：自定义序列化逻辑，用于 pickle/ray 传输。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：tuple：序列化后的 (batch, non_tensor_batch, meta_info) 或字节流。  # 注释：返回值说明
        副作用：可能触发 tensordict consolidate。  # 注释：副作用说明
        异常/边界条件：batch 为空时跳过 consolidate。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：pickle.dumps(DataProto)。  # 注释：示例输入
        - 输出：可序列化的数据结构。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.__getstate__。  # 注释：函数位置
        - 典型调用路径：pickle/ray -> __getstate__。  # 注释：典型调用链
        - 被谁调用：pickle/ray 序列化。  # 注释：调用方说明
        - 调用了谁（项目内）：serialize_tensordict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.save。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
            # Check if batch is empty to avoid torch.cat error in consolidate
            if len(self.batch.keys()) > 0:
                batch = self.batch.contiguous().consolidate()
            else:
                batch = self.batch
        else:
            batch = self.batch

        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            if batch is not None:
                batch = serialize_tensordict(self.batch)

            return (
                batch,
                self.non_tensor_batch,
                self.meta_info,
            )
        else:
            import io

            buffer = io.BytesIO()
            torch.save(batch, buffer)
            buffer_bytes = buffer.getvalue()
            return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        """
        函数用途：自定义反序列化逻辑，将存储结构恢复为 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - data：__getstate__ 生成的数据。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：会恢复 batch/non_tensor/meta_info。  # 注释：副作用说明
        异常/边界条件：数据格式不匹配会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：pickle.loads(...)。  # 注释：示例输入
        - 输出：DataProto 状态恢复。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.__setstate__。  # 注释：函数位置
        - 典型调用路径：pickle/ray -> __setstate__。  # 注释：典型调用链
        - 被谁调用：pickle/ray 反序列化。  # 注释：调用方说明
        - 调用了谁（项目内）：deserialize_tensordict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.load。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        batch_deserialized_bytes, non_tensor_batch, meta_info = data

        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            if batch_deserialized_bytes is not None:
                self.batch = deserialize_tensordict(batch_deserialized_bytes)
            else:
                self.batch = None
        else:
            import io

            batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
            batch = torch.load(
                batch_deserialized,
                weights_only=False,
                map_location="cpu" if not get_torch_device().is_available() else None,
            )
            self.batch = batch

        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        """
        函数用途：将 DataProto 保存为 pickle 文件。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - filepath (str)：保存路径。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：写入磁盘文件。  # 注释：副作用说明
        异常/边界条件：路径不可写会抛异常。  # 注释：异常说明
        最小示例：data.save_to_disk(\"/tmp/data.pkl\")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.save_to_disk。  # 注释：函数位置
        - 典型调用路径：调试/缓存 -> save_to_disk。  # 注释：典型调用链
        - 被谁调用：用户代码或工具脚本。  # 注释：调用方说明
        - 调用了谁（项目内）：__getstate__。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：pickle.dump。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        """
        函数用途：从磁盘读取 pickle 文件恢复 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - filepath (str)：文件路径。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：恢复的数据对象。  # 注释：返回值语义
        副作用：读取文件。  # 注释：副作用说明
        异常/边界条件：文件不存在/损坏会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：DataProto.load_from_disk(\"/tmp/data.pkl\")。  # 注释：示例输入
        - 输出：DataProto 实例。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.load_from_disk。  # 注释：函数位置
        - 典型调用路径：调试/缓存 -> load_from_disk。  # 注释：典型调用链
        - 被谁调用：用户代码或工具脚本。  # 注释：调用方说明
        - 调用了谁（项目内）：__setstate__。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：pickle.load。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        """
        函数用途：打印 batch 与 non_tensor_batch 的内存占用估计。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - prefix (str)：可选前缀。  # 注释：参数含义
        返回：无（打印）。  # 注释：返回值说明
        副作用：打印日志。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：data.print_size(\"train\")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.print_size。  # 注释：函数位置
        - 典型调用路径：调试/日志 -> print_size。  # 注释：典型调用链
        - 被谁调用：用户代码或调试脚本。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：print。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        size_of_tensordict = 0
        if self.batch is not None:
            for _, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        size_of_numpy_array = 0
        for _, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        if prefix:
            message = f"{prefix}, " + message
        print(message)

    def check_consistency(self):
        """
        函数用途：校验 batch 与 non_tensor_batch 的一致性。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：无。  # 注释：返回值说明
        副作用：可能抛出断言错误。  # 注释：副作用说明
        异常/边界条件：batch_size 不一致会触发 AssertionError。  # 注释：异常说明
        最小示例：data.check_consistency()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.check_consistency。  # 注释：函数位置
        - 典型调用路径：__post_init__ -> check_consistency。  # 注释：典型调用链
        - 被谁调用：__post_init__ 或用户显式调用。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：numpy。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), (
                    f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for "
                    f"{key=}, got {type(val)=}"
                )
                assert val.shape[0] == batch_size, (
                    f"key {key} length {len(val)} is not equal to batch size {batch_size}"
                )

    @classmethod
    def from_single_dict(cls, data: dict[str, torch.Tensor | np.ndarray], meta_info=None, auto_padding=False):
        """
        函数用途：从单个混合 dict 构建 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - data (dict)：值为 torch.Tensor 或 np.ndarray。  # 注释：参数含义
        - meta_info (dict|None)：元信息。  # 注释：参数含义
        - auto_padding (bool)：是否启用自动 padding。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：构建后的对象。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：遇到不支持类型会抛 ValueError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：{\"input_ids\": tensor, \"text\": np.array(...)}。  # 注释：示例输入
        - 输出：DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.from_single_dict。  # 注释：函数位置
        - 典型调用路径：collate_fn -> DataProto.from_dict/from_single_dict。  # 注释：典型调用链
        - 被谁调用：collate_fn 或用户代码。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.from_dict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：numpy。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors=None,
        meta_info=None,
        num_batch_dims=1,
        auto_padding=False,
    ):
        """
        函数用途：从张量字典与非张量字典构建 DataProto。  # 注释：函数用途说明
        约束：  # 注释：约束说明标题
        - tensors 中所有张量 batch 维度一致。  # 注释：约束说明
        - non_tensors 存在时 num_batch_dims 必须为 1。  # 注释：约束说明
        参数：  # 注释：参数说明标题
        - tensors (dict[str, Tensor]|None)：张量字段。  # 注释：参数含义
        - non_tensors (dict|None)：非张量字段。  # 注释：参数含义
        - meta_info (dict|None)：元信息。  # 注释：参数含义
        - num_batch_dims (int)：批维度数。  # 注释：参数含义
        - auto_padding (bool)：是否启用自动 padding。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：构造好的对象。  # 注释：返回值语义
        副作用：可能将 non_tensors 转为 np.array(dtype=object)。  # 注释：副作用说明
        异常/边界条件：batch 维度不一致会触发 AssertionError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：tensors={\"x\":tensor([..])} -> DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.from_dict。  # 注释：函数位置
        - 典型调用路径：collate_fn -> DataProto.from_dict。  # 注释：典型调用链
        - 被谁调用：collate_fn / 用户代码。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProtoConfig.auto_padding_key。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：tensordict.TensorDict。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. "
                    f"Got {pivot_key} has {batch_size}, {key} has {current_batch}"
                )

        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    @classmethod
    def from_tensordict(
        cls,
        tensor_dict: TensorDict = None,
        meta_info=None,
        num_batch_dims=1,
    ):
        """
        函数用途：从 TensorDict 构建 DataProto（支持 NonTensorData/NonTensorStack）。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - tensor_dict (TensorDict)：输入 TensorDict。  # 注释：参数含义
        - meta_info (dict|None)：元信息。  # 注释：参数含义
        - num_batch_dims (int)：批维度数。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：构建后的对象。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：tensordict 版本 <0.10.0 会断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：DataProto.from_tensordict(tensor_dict)。  # 注释：示例输入
        - 输出：DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.from_tensordict。  # 注释：函数位置
        - 典型调用路径：TensorDict -> DataProto.from_tensordict。  # 注释：典型调用链
        - 被谁调用：用户代码或内部转换逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProtoConfig.auto_padding_key。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：tensordict.NonTensorData/NonTensorStack。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        assert version.parse(tensordict.__version__) >= version.parse("0.10.0"), (
            "Build DataProto from TensorDict at least requires tensordict version 0.10.0"
        )
        from tensordict import NonTensorData, NonTensorStack

        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if not all(isinstance(val, torch.Tensor) for val in tensor_dict.values()):
            assert num_batch_dims == 1, "only support num_batch_dims=1 when tensor_dict contains non tensor data."

        if meta_info is None:
            meta_info = {}
        batch = {}
        non_tensor_batch = {}
        batch_size = None
        for key, val in tensor_dict.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val
                if batch_size is None:
                    batch_size = val.shape[:num_batch_dims]
            elif isinstance(val, NonTensorStack):
                non_tensor_batch[key] = np.array([elem.data for elem in val], dtype=object)
            elif isinstance(val, NonTensorData):
                meta_info[key] = val.data

        return cls(
            batch=TensorDict(batch, batch_size=batch_size),
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def to(self, device) -> "DataProto":
        """
        函数用途：将 batch 张量迁移到指定设备。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - device (torch.device|str)：目标设备。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：自身（batch 已迁移）。  # 注释：返回值语义
        副作用：修改 self.batch 的设备。  # 注释：副作用说明
        异常/边界条件：batch 为空时直接返回。  # 注释：边界说明
        最小示例：data.to(\"cuda\")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.to。  # 注释：函数位置
        - 典型调用路径：worker/训练 -> DataProto.to(device)。  # 注释：典型调用链
        - 被谁调用：训练/推理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：TensorDict.to。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """
        函数用途：按字段选择 DataProto 子集（支持张量/非张量/meta）。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - batch_keys (list|None)：选择 batch 字段。  # 注释：参数含义
        - non_tensor_batch_keys (list|None)：选择非张量字段。  # 注释：参数含义
        - meta_info_keys (list|None)：选择元信息字段。  # 注释：参数含义
        - deepcopy (bool)：是否深拷贝非张量与 meta。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：筛选后的对象。  # 注释：返回值语义
        副作用：无（返回新对象）。  # 注释：副作用说明
        异常/边界条件：batch_keys 为空时返回完整 batch。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.select(batch_keys=[\"input_ids\"])。  # 注释：示例输入
        - 输出：仅包含 input_ids 的 DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.select。  # 注释：函数位置
        - 典型调用路径：训练/评估 -> data.select(... )。  # 注释：典型调用链
        - 被谁调用：用户代码或内部处理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict.select。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：copy.deepcopy。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        函数用途：根据索引选择子集样本。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - idxs (torch.Tensor|np.ndarray|list)：索引或布尔 mask。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：仅包含 выбран索引的对象。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：idxs 类型不支持会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.select_idxs([0,2]) -> 选取第 0、2 条。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.select_idxs。  # 注释：函数位置
        - 典型调用路径：__getitem__ -> select_idxs。  # 注释：典型调用链
        - 被谁调用：DataProto.__getitem__ 或用户代码。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict.__getitem__。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch/numpy 索引。  # 注释：外部依赖说明
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()},
                batch_size=(batch_size,),
                device=self.batch.device,
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        函数用途：按 slice 规则裁剪 DataProto，并返回新的 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - start (int|None)：起始索引。  # 注释：参数含义
        - end (int|None)：结束索引（不含）。  # 注释：参数含义
        - step (int|None)：步长。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：裁剪后的数据。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：slice 超界时返回可用范围内数据。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.slice(10, 20) 或 data[10:20]。  # 注释：示例说明
        - 输出：包含索引 10~19 的 DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.slice。  # 注释：函数位置
        - 典型调用路径：DataProto.__getitem__ -> slice。  # 注释：典型调用链
        - 被谁调用：DataProto.__getitem__ 或用户代码。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict.__getitem__。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：slice。  # 注释：外部依赖说明
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """
        函数用途：从当前 DataProto 弹出指定字段并返回新 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - batch_keys (list|None)：要弹出的 batch 字段。  # 注释：参数含义
        - non_tensor_batch_keys (list|None)：要弹出的非张量字段。  # 注释：参数含义
        - meta_info_keys (list|None)：要弹出的 meta 字段。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：包含被弹出字段的新对象。  # 注释：返回值语义
        副作用：会原地移除 self 中对应字段。  # 注释：副作用说明
        异常/边界条件：key 不存在会触发 AssertionError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.pop(batch_keys=[\"input_ids\"])。  # 注释：示例输入
        - 输出：返回仅包含 input_ids 的 DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.pop。  # 注释：函数位置
        - 典型调用路径：数据清理 -> pop。  # 注释：典型调用链
        - 被谁调用：用户代码或内部处理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.from_dict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：dict.pop。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        函数用途：重命名 batch 中的字段名。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - old_keys (list|str|None)：原字段名。  # 注释：参数含义
        - new_keys (list|str|None)：新字段名。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：自身（就地修改）。  # 注释：返回值语义
        副作用：会原地修改 batch 字段名。  # 注释：副作用说明
        异常/边界条件：新旧键数量不一致会抛 ValueError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：rename(\"a\", \"b\") -> batch 中 a 改为 b。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.rename。  # 注释：函数位置
        - 典型调用路径：字段整理 -> rename。  # 注释：典型调用链
        - 被谁调用：用户代码或内部处理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict.rename_key_。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """

        def validate_input(keys):
            """函数用途：校验并标准化 keys 为列表形式。"""  # 注释：内嵌函数用途说明
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if

        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        参数：  # 注释：参数说明标题
        - other (DataProto)：待合并的 DataProto。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：合并后的对象（self）。  # 注释：返回值语义
        副作用：会原地更新 self.batch/non_tensor/meta_info。  # 注释：副作用说明
        异常/边界条件：batch size 或字段冲突时会断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.union(other)。  # 注释：示例输入
        - 输出：self 中包含合并字段。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.union。  # 注释：函数位置
        - 典型调用路径：批处理合并 -> DataProto.union。  # 注释：典型调用链
        - 被谁调用：训练/评估逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：union_tensor_dict、union_numpy_dict、union_two_dict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        r"""
        函数用途：基于 DataProto 构造可迭代的 mini-batch 迭代器。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - mini_batch_size (int)：mini-batch 大小，要求 batch_size 能整除。  # 注释：参数含义
        - epochs (int)：迭代轮数。  # 注释：参数含义
        - seed (int|None)：随机种子（可选）。  # 注释：参数含义
        - dataloader_kwargs (dict|None)：传给 DataLoader 的参数。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - Iterator：按 mini-batch 产出 DataProto 的迭代器。  # 注释：返回值语义
        副作用：内部创建 DataLoader，并在迭代时注入 meta_info。  # 注释：副作用说明
        异常/边界条件：batch_size 不能整除 mini_batch_size 时断言失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.make_iterator(32, epochs=2)。  # 注释：示例输入
        - 输出：每次 yield 一个 mini-batch DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.make_iterator。  # 注释：函数位置
        - 典型调用路径：训练循环 -> make_iterator。  # 注释：典型调用链
        - 被谁调用：trainer/worker 训练逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：collate_fn。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.utils.data.DataLoader。  # 注释：外部依赖说明
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            """函数用途：按 epoch 迭代 DataLoader 并注入 meta_info。"""  # 注释：内嵌函数用途说明
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def is_padding_enabled(self):
        """
        函数用途：判断 DataProto 是否启用自动 padding。  # 注释：函数用途说明
        返回：bool，是否启用。  # 注释：返回值说明
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：data.is_padding_enabled() -> True/False。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.is_padding_enabled。  # 注释：函数位置
        - 典型调用路径：padding 流程 -> is_padding_enabled。  # 注释：典型调用链
        - 被谁调用：DataProto.padding。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProtoConfig.auto_padding。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        """
        函数用途：按指定候选样本进行 padding（重复拼接）。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - padding_size (int)：需要补齐的数量。  # 注释：参数含义
        - padding_candidate (str)：\"first\"/\"last\" 或空，指定从首/尾取样。  # 注释：参数含义
        返回：无（原地修改）。  # 注释：返回值说明
        副作用：会扩展 batch 与 non_tensor_batch。  # 注释：副作用说明
        异常/边界条件：padding_size=0 时直接返回。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：padding_size=2, padding_candidate=\"last\"。  # 注释：示例输入
        - 输出：追加 2 条样本。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.padding。  # 注释：函数位置
        - 典型调用路径：pad_dataproto_to_divisor -> padding。  # 注释：典型调用链
        - 被谁调用：训练/分布式对齐逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.concat。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：numpy/torch 追加。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if padding_size == 0:
            return
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        padding_part = padding_candidate.repeat(padding_size)
        padded_dp = DataProto.concat([self, padding_part])
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    def chunk(self, chunks: int) -> list["DataProto"]:
        """
        函数用途：按 dim=0 将 batch 拆分为若干块。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - chunks (int)：切分块数量。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - list[DataProto]：切分后的 DataProto 列表。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：若未启用 padding，要求 len(self) 能整除 chunks。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：len=8, chunks=2 -> 输出 2 个 DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.chunk。  # 注释：函数位置
        - 典型调用路径：分布式拆分 -> chunk。  # 注释：典型调用链
        - 被谁调用：训练/推理并行逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.is_padding_enabled。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：TensorDict.chunk、numpy.array_split。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, (
                f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
            )

        bsz_in_batch = None
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            )

        return output

    def split(self, split_size: int) -> list["DataProto"]:
        """
        函数用途：按固定大小切分 DataProto。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - split_size (int)：每块大小。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - list[DataProto]：切分结果列表。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：split_size<=0 会导致异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：len=10, split_size=4 -> 输出 3 个 DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.split。  # 注释：函数位置
        - 典型调用路径：批处理切片 -> split。  # 注释：典型调用链
        - 被谁调用：用户代码或内部处理逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.__getitem__。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return [self[i : i + split_size] for i in range(0, len(self), split_size)]

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        """
        函数用途：拼接多个 DataProto（沿 dim=0）。  # 注释：函数用途说明
        说明：meta_info 会合并，指标字段会特殊处理。  # 注释：说明
        参数：  # 注释：参数说明标题
        - data (list[DataProto])：待拼接列表。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：拼接后的对象。  # 注释：返回值语义
        副作用：无（返回新对象）。  # 注释：副作用说明
        异常/边界条件：空列表会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：[dp1, dp2] -> 输出合并的 DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.concat。  # 注释：函数位置
        - 典型调用路径：pad_dataproto_to_divisor/分布式合并 -> concat。  # 注释：典型调用链
        - 被谁调用：训练/评估聚合逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：union_two_dict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.cat。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        # Merge meta_info with special handling for metrics
        merged_meta_info = {}
        if data:
            # Merge non-metric meta_info and aggregate metrics from all workers.
            all_metrics = []
            for d in data:
                for k, v in d.meta_info.items():
                    if k == "metrics":
                        if v is not None:
                            if isinstance(v, list):
                                all_metrics.extend(v)
                            else:
                                all_metrics.append(v)
                    else:
                        if k in merged_meta_info:
                            # Ensure consistency for overlapping non-metric keys
                            assert merged_meta_info[k] == v, f"Conflicting values for meta_info key '{k}'"
                        else:
                            merged_meta_info[k] = v

            # Flatten list of dicts to dict of lists for consistent metrics structure
            if all_metrics:
                merged_meta_info["metrics"] = list_of_dict_to_dict_of_list(all_metrics)

        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=merged_meta_info)

    def reorder(self, indices):
        """
        函数用途：按索引重排 DataProto（原地）。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - indices (torch.Tensor)：索引顺序。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：原地修改 batch 与 non_tensor_batch。  # 注释：副作用说明
        异常/边界条件：indices 需为 torch.Tensor。  # 注释：异常说明
        最小示例：data.reorder(torch.tensor([1,0]))。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.reorder。  # 注释：函数位置
        - 典型调用路径：采样/排序逻辑 -> reorder。  # 注释：典型调用链
        - 被谁调用：训练/评估逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict.__getitem__。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：numpy 索引。  # 注释：外部依赖说明
        """
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        """
        函数用途：重复 batch 数据若干次（可交错/堆叠）。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - repeat_times (int)：重复次数。  # 注释：参数含义
        - interleave (bool)：是否交错重复。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：重复后的新对象。  # 注释：返回值语义
        副作用：无（返回新对象）。  # 注释：副作用说明
        异常/边界条件：repeat_times<=0 将导致空输出。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：repeat_times=2, interleave=True。  # 注释：示例输入
        - 输出：样本顺序交错重复。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.repeat。  # 注释：函数位置
        - 典型调用路径：数据扩增 -> repeat。  # 注释：典型调用链
        - 被谁调用：训练/采样逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.repeat_interleave、numpy.repeat。  # 注释：外部依赖说明
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def unfold_column_chunks(self, n_split: int, split_keys: Optional[list[str]] = None):
        """
        函数用途：沿第二维分块并展开到 batch 维（用于分组张量）。  # 注释：函数用途说明
        说明：split_keys 指定需要分块的字段，其余字段按 batch 重复。  # 注释：说明
        参数：  # 注释：参数说明标题
        - n_split (int)：第二维分块数。  # 注释：参数含义
        - split_keys (list[str]|None)：需要分块的字段名列表。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：展开后的对象。  # 注释：返回值语义
        副作用：无（返回新对象）。  # 注释：副作用说明
        异常/边界条件：split_keys 为空时默认对所有字段重复。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：n_split=2，split_keys=[\"grouped\"]。  # 注释：示例输入
        - 输出：batch 大小扩大 2 倍。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.unfold_column_chunks。  # 注释：函数位置
        - 典型调用路径：数据重排 -> unfold_column_chunks。  # 注释：典型调用链
        - 被谁调用：数据集/采样逻辑（可选）。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.repeat_interleave、numpy.repeat。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # locate the `unfolded_batch` as a TensorDict on the same device as the original batch
            unfolded_batch = TensorDict(
                source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device
            )
        else:
            unfolded_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def sample_level_repeat(self, repeat_times):
        """
        函数用途：按样本级别重复数据，每行可有不同重复次数。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - repeat_times (torch.Tensor|list|tuple|ndarray)：每条样本的重复次数。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：重复后的新对象。  # 注释：返回值语义
        副作用：无（返回新对象）。  # 注释：副作用说明
        异常/边界条件：repeat_times 维度必须为 1。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：repeat_times=[1,3] -> 第 2 条重复 3 次。  # 注释：示例说明
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.sample_level_repeat。  # 注释：函数位置
        - 典型调用路径：数据扩增 -> sample_level_repeat。  # 注释：典型调用链
        - 被谁调用：训练/采样逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：TensorDict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.repeat_interleave、numpy.repeat。  # 注释：外部依赖说明
        """
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), (
                f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
            )
        repeat_times = torch.tensor(repeat_times)

        if self.batch is not None:
            # Interleave the data
            repeated_tensors = {
                key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
            }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def to_tensordict(self) -> TensorDict:
        """
        函数用途：将 DataProto 转换为 TensorDict（包含非张量与 meta）。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：  # 注释：返回值说明标题
        - TensorDict：转换后的结构。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：tensordict 版本 <0.10 会断言失败。  # 注释：异常说明
        最小示例：data.to_tensordict()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.to_tensordict。  # 注释：函数位置
        - 典型调用路径：对外接口/调试 -> to_tensordict。  # 注释：典型调用链
        - 被谁调用：用户代码或内部转换逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：verl.utils.tensordict_utils.get_tensordict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：tensordict.NonTensorStack。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        assert parse_version(tensordict.__version__) >= parse_version("0.10"), (
            "Convert DataProto to TensorDict at least requires tensordict version 0.10"
        )
        tensor_batch = self.batch.to_dict()
        non_tensor_batch = self.non_tensor_batch

        from tensordict.tensorclass import NonTensorData, NonTensorStack

        from verl.utils import tensordict_utils as tu

        common_keys = set(tensor_batch.keys()) & set(non_tensor_batch.keys())
        assert len(common_keys) == 0, f"tensor_batch and non_tensor_batch have common keys {common_keys}"

        for key, val in non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            # Convert to NonTensorStack instead of plain list to handle nested structures
            tensor_batch[key] = NonTensorStack.from_list([NonTensorData(item) for item in val])
        output = tu.get_tensordict(tensor_dict=tensor_batch, non_tensor_dict=self.meta_info)
        return output

    def get_data_info(self) -> str:
        """
        函数用途：生成当前 DataProto 的字段信息摘要（含类型/形状）。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：  # 注释：返回值说明标题
        - str：格式化的字段信息字符串。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：batch 为空时仅输出非张量/meta 信息。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：data.get_data_info() -> \"batch\\n  input_ids: ...\"。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto.get_data_info。  # 注释：函数位置
        - 典型调用路径：调试/日志 -> get_data_info。  # 注释：典型调用链
        - 被谁调用：用户代码或监控逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：_get_type_info。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        info = ["batch"]

        for key, tensor in self.batch.items():
            if hasattr(tensor, "shape") and hasattr(tensor, "dtype") and hasattr(tensor, "device"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype}) {tensor.device}")
            elif hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
            else:
                info.append(f"  {key}: {type(tensor).__name__}")

        info.append("non_tensor_batch")
        for key, array in self.non_tensor_batch.items():
            info.append(f"  {key}: ndarray{array.shape} ({array.dtype})")

        info.append("meta_info")
        for k, v in self.meta_info.items():
            type_info = self._get_type_info(v)
            info.append(f"  {k}: {type_info}")

        return "\n".join(info)

    def _get_type_info(self, value):
        """
        函数用途：递归生成嵌套结构的类型描述字符串。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - value (Any)：待分析的值。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - str：类型描述（含列表/字典嵌套）。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：空 dict 返回 \"dict\"。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：[{\"a\":1}] -> \"list[dict[str: int]]\"。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProto._get_type_info。  # 注释：函数位置
        - 典型调用路径：get_data_info -> _get_type_info。  # 注释：典型调用链
        - 被谁调用：get_data_info（本文件）。  # 注释：调用方说明
        - 调用了谁（项目内）：自身递归。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：numpy。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if isinstance(value, list):
            elem_types = {self._get_type_info(v) for v in value[:3]}
            return f"list[{'|'.join(elem_types) if elem_types else '...'}]"
        if isinstance(value, tuple):
            elem_types = [self._get_type_info(v) for v in value]
            return f"tuple({', '.join(elem_types)})"
        if isinstance(value, dict):
            if not value:
                return "dict"
            k, v = next(iter(value.items()))
            return f"dict[{self._get_type_info(k)}: {self._get_type_info(v)}]"
        if isinstance(value, np.ndarray):
            return f"ndarray{value.shape} ({value.dtype})"
        return type(value).__name__


@dataclass
class DataProtoFuture:
    """
    类用途：封装 DataProto 的异步引用，避免 driver 等待数据。  # 注释：类用途说明
    机制说明：  # 注释：机制说明标题
    - futures 保存各 worker 的 ObjectRef。  # 注释：说明
    - collect_fn 将 futures 汇聚为 DataProto。  # 注释：说明
    - dispatch_fn 将 DataProto 切分并选择。  # 注释：说明
    限制：  # 注释：限制说明标题
    - 仅支持在 worker 间传递，driver 侧不做直接数据操作。  # 注释：限制说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::DataProtoFuture。  # 注释：类位置
    - 典型调用路径：worker RPC -> DataProtoFuture -> get。  # 注释：典型调用链
    - 被谁调用：异步 worker 调度逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：DataProto.concat。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：ray.ObjectRef。  # 注释：外部依赖说明
    """

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "DataProtoFuture":
        """
        函数用途：将多个 ObjectRef 合并为一个 DataProtoFuture。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - data (list[ray.ObjectRef])：各 worker 的 future 列表。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProtoFuture：封装后的 future。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：空列表会导致后续 concat 出错。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：DataProtoFuture.concat(futures)。  # 注释：示例输入
        - 输出：DataProtoFuture。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProtoFuture.concat。  # 注释：函数位置
        - 典型调用路径：异步 worker 汇聚 -> concat。  # 注释：典型调用链
        - 被谁调用：DataProtoFuture.get 或上层调度逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.concat。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：ray.ObjectRef。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> list["DataProtoFuture"]:
        """
        函数用途：将 DataProtoFuture 拆分为多个子 future。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - chunks (int)：拆分数量。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - list[DataProtoFuture]：拆分后的 future 列表。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：chunks<=0 会导致错误。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：future.chunk(2) -> 2 个 DataProtoFuture。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProtoFuture.chunk。  # 注释：函数位置
        - 典型调用路径：并行处理 -> chunk。  # 注释：典型调用链
        - 被谁调用：异步调度逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.chunk。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：functools.partial。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                """函数用途：从 DataProto 中取出指定 chunk。"""  # 注释：内嵌函数用途说明
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        """
        函数用途：拉取 futures 并合并为实际数据。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：  # 注释：返回值说明标题
        - DataProto 或 TensorDict：合并后的结果。  # 注释：返回值语义
        副作用：触发 ray.get 拉取数据。  # 注释：副作用说明
        异常/边界条件：未知类型会抛 TypeError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：future.get() -> DataProto。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/protocol.py::DataProtoFuture.get。  # 注释：函数位置
        - 典型调用路径：异步调度 -> get。  # 注释：典型调用链
        - 被谁调用：上层训练/调度逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：DataProto.concat、tensordict_utils.concat_tensordict。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：ray.get。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        output = ray.get(self.futures)  # dp_size.
        for o in output:
            assert isinstance(o, DataProto | TensorDict)

        if isinstance(output[0], DataProto):
            output = DataProto.concat(output)  # select dp, concat
        elif isinstance(output[0], TensorDict):
            from verl.utils.tensordict_utils import concat_tensordict

            output = concat_tensordict(output)
        else:
            raise TypeError(f"Unknown type {type(o[0])} in DataProtoFuture")

        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # split in batch dim, select using dp
        return output


def all_gather_data_proto(data: DataProto, process_group):
    """
    函数用途：在分布式组内 all_gather DataProto（张量与非张量）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - data (DataProto)：待 all_gather 的数据（原地修改）。  # 注释：参数含义
    - process_group：torch.distributed 进程组。  # 注释：参数含义
    返回：无（原地更新 data）。  # 注释：返回值说明
    副作用：会修改 data.batch 与 data.non_tensor_batch。  # 注释：副作用说明
    异常/边界条件：data 必须为 DataProto，group 未初始化会报错。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：all_gather_data_proto(data, group) -> data 聚合所有 rank。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/protocol.py::all_gather_data_proto。  # 注释：函数位置
    - 典型调用路径：分布式训练同步 -> all_gather_data_proto。  # 注释：典型调用链
    - 被谁调用：分布式 worker/训练逻辑。  # 注释：调用方说明
    - 调用了谁（项目内）：allgather_dict_tensors、get_device_id。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.distributed.all_gather_object。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    # Note that this is an inplace operator just like torch.distributed.all_gather  # 注释：原注释，说明原地操作
    group_size = torch.distributed.get_world_size(group=process_group)
    assert isinstance(data, DataProto)
    prev_device = data.batch.device
    data = data.to(get_device_id())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    data = data.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
