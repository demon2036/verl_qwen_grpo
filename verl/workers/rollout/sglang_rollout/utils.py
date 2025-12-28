# Copyright 2023-2024 SGLang Team  # 注释：版权声明（SGLang）
# Copyright 2025 ModelBest Inc. and/or its affiliates  # 注释：版权声明（ModelBest）
#  # 注释：空行占位
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：许可证获取提示
#  # 注释：空行占位
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无担保声明
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明

"""
模块用途：提供 SGLang rollout 需要的广播与张量分桶工具函数。  # 注释：模块用途
输入：Python 对象列表、张量迭代器、分桶大小等。  # 注释：输入说明
输出：广播后的对象或按大小分桶的张量列表。  # 注释：输出说明
关键依赖：torch.distributed、pickle、numpy。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- broadcast_pyobj(data, rank=0, dist_group=pg)  # 注释：广播示例
- for bucket in get_named_tensor_buckets(named_tensors, bucket_bytes=1<<20): ...  # 注释：分桶示例
调用路径概览：  # 注释：调用路径标题
- sglang_rollout.py / http_server_engine.py 中的权重同步使用本模块。  # 注释：调用链
"""  # 注释：模块 docstring 结束

import pickle  # 注释：标准库，序列化 Python 对象
from typing import Any, Iterator, Optional  # 注释：类型注解

import numpy as np  # 注释：第三方库，构造字节数组
import torch  # 注释：第三方库，张量与分布式通信
import torch.distributed as dist  # 注释：PyTorch 分布式通信

from verl.utils.device import get_device_name  # 注释：项目内工具，获取设备名称


def broadcast_pyobj(  # 注释：广播 Python 对象列表
    data: list[Any],  # 注释：待广播的数据
    rank: int,  # 注释：当前进程 rank
    dist_group: Optional[torch.distributed.ProcessGroup] = None,  # 注释：分布式通信组
    src: int = 0,  # 注释：源 rank
    force_cpu_device: bool = False,  # 注释：是否强制使用 CPU
):  # 注释：函数签名结束
    """
    from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).

    功能：在分布式进程组中将 Python 对象列表从 src 广播到其他 rank。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data (list[Any])：需要广播的数据。  # 注释：参数含义
    - rank (int)：当前进程的 rank。  # 注释：参数含义
    - dist_group (ProcessGroup)：通信组（可选）。  # 注释：参数含义
    - src (int)：源 rank。  # 注释：参数含义
    - force_cpu_device (bool)：是否强制使用 CPU 设备。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - list[Any]：广播后的数据。  # 注释：返回值语义
    副作用：分布式通信（broadcast）。  # 注释：副作用说明
    异常/边界条件：通信组未初始化会报错。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - broadcast_pyobj([1,2], rank=0, src=0) -> [1,2]。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/utils.py::broadcast_pyobj。  # 注释：函数位置
    - 典型调用路径：SGLang rollout 权重同步 -> broadcast_pyobj。  # 注释：调用链
    - 被谁调用：sglang_rollout.py / http_server_engine.py。  # 注释：调用方说明
    - 调用了谁（项目内）：get_device_name。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：torch.distributed.broadcast、pickle、numpy。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    device = torch.device(get_device_name() if not force_cpu_device else "cpu")  # 注释：选择设备

    if rank == src:  # 注释：源 rank 发送数据
        if len(data) == 0:  # 注释：空列表情况
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)  # 注释：构造 size 张量
            dist.broadcast(tensor_size, src=src, group=dist_group)  # 注释：广播 size
        else:  # 注释：非空数据
            serialized_data = pickle.dumps(data)  # 注释：序列化 Python 对象
            size = len(serialized_data)  # 注释：序列化字节长度

            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)  # 注释：构造字节张量
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)  # 注释：构造 size 张量

            dist.broadcast(tensor_size, src=src, group=dist_group)  # 注释：广播 size
            dist.broadcast(tensor_data, src=src, group=dist_group)  # 注释：广播数据
        return data  # 注释：源 rank 返回原数据
    else:  # 注释：非源 rank 接收数据
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)  # 注释：初始化 size 张量
        dist.broadcast(tensor_size, src=src, group=dist_group)  # 注释：接收 size
        size = tensor_size.item()  # 注释：获取 size 数值

        if size == 0:  # 注释：空数据情况
            return []  # 注释：返回空列表

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)  # 注释：创建接收缓冲
        dist.broadcast(tensor_data, src=src, group=dist_group)  # 注释：接收数据

        serialized_data = bytes(tensor_data.cpu().numpy())  # 注释：转换为 bytes
        data = pickle.loads(serialized_data)  # 注释：反序列化
        return data  # 注释：返回接收数据


def get_named_tensor_buckets(  # 注释：按字节大小分桶张量
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int  # 注释：输入张量迭代器与桶大小
) -> Iterator[list[tuple[str, torch.Tensor]]]:  # 注释：返回桶迭代器
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    功能：按 bucket_bytes 将张量分组，便于分批同步或传输。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - iterable (Iterator[(str, Tensor)])：带名称的张量迭代器。  # 注释：参数含义
    - bucket_bytes (int)：每个桶的最大字节数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - Iterator[list[(str, Tensor)]]：分桶后的张量列表迭代器。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：bucket_bytes<=0 时抛 ValueError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - list(get_named_tensor_buckets([("w", tensor)], 1024))。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/sglang_rollout/utils.py::get_named_tensor_buckets。  # 注释：函数位置
    - 典型调用路径：权重同步/广播逻辑 -> get_named_tensor_buckets。  # 注释：调用链
    - 被谁调用：sglang_rollout.py / http_server_engine.py（需要分桶时）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：torch.Tensor.element_size/numel。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    if bucket_bytes <= 0:  # 注释：参数合法性检查
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")  # 注释：抛出异常

    current_bucket = []  # 注释：当前桶缓存
    current_size = 0  # 注释：当前桶大小
    for name, tensor in iterable:  # 注释：遍历张量
        tensor_size = tensor.element_size() * tensor.numel()  # 注释：计算张量字节大小
        if current_size + tensor_size > bucket_bytes:  # 注释：超过桶大小
            if current_bucket:  # 注释：当前桶非空
                yield current_bucket  # 注释：输出当前桶
            current_bucket = [(name, tensor)]  # 注释：新桶初始化
            current_size = tensor_size  # 注释：重置桶大小
        else:  # 注释：未超过桶大小
            current_bucket.append((name, tensor))  # 注释：追加到当前桶
            current_size += tensor_size  # 注释：更新桶大小

    if current_bucket:  # 注释：最后一个桶非空
        yield current_bucket  # 注释：输出最后一个桶
