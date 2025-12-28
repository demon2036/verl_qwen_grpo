# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
模块用途：提供 TransferQueue 与 DataProto 之间的桥接工具（tqbridge），并维护全局客户端实例。（注释：模块功能概述）
输入：BatchMeta/DataProto、TransferQueue 配置、调度 Dispatch 信息。（注释：输入形态说明）
输出：在装饰器内完成 BatchMeta<->DataProto 转换、回写输出与可选的 BatchMeta 返回。（注释：输出形态说明）
关键依赖：transfer_queue（可选）、tensordict、asyncio、verl.protocol.DataProto。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - create_transferqueue_client(client_id="Trainer", config=cfg.transfer_queue, sync=True)
  - @tqbridge(put_data=False)\n    def compute_reward(...): ...
调用路径概览：（注释：全局调用关系）
  - recipe/transfer_queue/ray_trainer.py -> create_transferqueue_client / tqbridge
  - verl/trainer/ppo/reward.py -> @tqbridge -> compute_reward
"""

import asyncio  # 注释：异步事件循环与协程调度
import functools  # 注释：partial/函数包装等工具
import inspect  # 注释：判断函数是否为协程
import logging  # 注释：日志记录
import os  # 注释：读取环境变量与进程信息
import threading  # 注释：在独立线程中运行事件循环
from functools import wraps  # 注释：保留被装饰函数元信息
from typing import TYPE_CHECKING, Any, Callable  # 注释：类型提示与条件导入

if TYPE_CHECKING:  # 注释：仅类型检查时导入，避免运行时循环依赖
    from verl.single_controller.base.decorator import Dispatch  # 注释：调度模式类型

from tensordict import TensorDict  # 注释：用于构造/传递张量字典

try:  # 注释：transfer_queue 为可选依赖
    from transfer_queue import (  # 注释：TransferQueue 客户端与 BatchMeta
        AsyncTransferQueueClient,  # 注释：异步客户端
        BatchMeta,  # 注释：批次元信息
        TransferQueueClient,  # 注释：同步客户端
    )

except ImportError:  # 注释：未安装 transfer_queue 时回退
    # TODO: Use a hacky workaround for ImportError since  # 注释：原注释保留
    # transfer_queue isn't a default verl dependency.  # 注释：说明 transfer_queue 非默认依赖
    class BatchMeta:  # 注释：占位 BatchMeta，避免类型引用错误
        pass  # 注释：空实现，仅作占位


from verl.protocol import DataProto  # 注释：统一数据容器

logger = logging.getLogger(__name__)  # 注释：模块级日志器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：读取环境变量设置日志级别

_TRANSFER_QUEUE_CLIENT = None  # 注释：全局缓存的 TransferQueue 客户端实例

is_transferqueue_enabled = os.environ.get("TRANSFER_QUEUE_ENABLE", False)  # 注释：是否启用 TQ（环境变量开关）


def create_transferqueue_client(  # 注释：创建/缓存 TransferQueue 客户端
    client_id: str,
    config,
    sync: bool = False,
) -> "AsyncTransferQueueClient | TransferQueueClient":
    """
    功能：创建并缓存 TransferQueue 客户端（同步或异步），并初始化存储后端。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - client_id (str): 客户端标识，用于区分不同角色。（注释：客户端 ID）
      - config: TransferQueue 配置对象，需含 controller_info/storage_backend。（注释：配置来源）
      - sync (bool): True 表示使用同步客户端；False 表示使用异步客户端。（注释：同步/异步开关）
    返回：（注释：返回值说明）
      - AsyncTransferQueueClient | TransferQueueClient：全局复用的客户端实例。（注释：返回类型）
    副作用：（注释：副作用说明）
      - 写入全局 _TRANSFER_QUEUE_CLIENT 缓存。（注释：全局状态变更）
      - 可能建立与 TransferQueue 的网络连接。（注释：网络副作用）
      - 初始化存储管理器（可能创建本地/远端资源）。（注释：资源副作用）
    异常/边界条件：（注释：异常与边界）
      - transfer_queue 未安装时，TransferQueueClient 名称不存在将触发 NameError。（注释：依赖缺失）
      - config 缺少 controller_info/storage_backend 时会抛 AttributeError。（注释：配置缺失）
    最小示例：（注释：最小可理解示例）
      - 输入：create_transferqueue_client("Trainer", cfg.transfer_queue, sync=True)
      - 输出：同步 TransferQueueClient 实例（全局缓存）
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::create_transferqueue_client`
      - 典型调用路径：`recipe/transfer_queue/ray_trainer.py` -> `create_transferqueue_client`
      - 被谁调用：`verl/single_controller/base/worker.py`、`recipe/transfer_queue/agent_loop.py`、
        `verl/experimental/agent_loop/agent_loop.py`
      - 调用了谁（项目内）：无（仅操作全局变量）
      - 调用了谁（外部依赖）：`TransferQueueClient`/`AsyncTransferQueueClient`、`initialize_storage_manager`
    """
    global _TRANSFER_QUEUE_CLIENT  # 注释：声明将写入全局缓存
    if _TRANSFER_QUEUE_CLIENT is None:  # 注释：仅在首次调用时创建客户端
        if sync:  # 注释：同步客户端分支
            _TRANSFER_QUEUE_CLIENT = TransferQueueClient(client_id, config.controller_info)  # 注释：实例化同步客户端
        else:  # 注释：异步客户端分支
            _TRANSFER_QUEUE_CLIENT = AsyncTransferQueueClient(client_id, config.controller_info)  # 注释：实例化异步客户端
        _TRANSFER_QUEUE_CLIENT.initialize_storage_manager(  # 注释：初始化存储管理器
            manager_type=config.storage_backend,
            config=config,
        )

    return _TRANSFER_QUEUE_CLIENT  # 注释：返回缓存的客户端实例


def get_transferqueue_client() -> "AsyncTransferQueueClient | TransferQueueClient":  # 注释：获取全局客户端实例
    """
    功能：获取全局缓存的 TransferQueue 客户端实例。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - 无。（注释：无参数）
    返回：（注释：返回值说明）
      - AsyncTransferQueueClient | TransferQueueClient | None：未创建时可能为 None。（注释：返回类型）
    副作用：（注释：副作用说明）
      - 无（只读全局变量）。（注释：无副作用）
    异常/边界条件：（注释：异常与边界）
      - 若未调用 create_transferqueue_client，则返回 None。（注释：未初始化边界）
    最小示例：（注释：最小可理解示例）
      - 输入：get_transferqueue_client()
      - 输出：已创建的 TransferQueueClient 或 None
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::get_transferqueue_client`
      - 典型调用路径：`recipe/transfer_queue/ray_trainer.py` -> `get_transferqueue_client`
      - 被谁调用：`recipe/transfer_queue/ray_trainer.py`
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：无
    """
    return _TRANSFER_QUEUE_CLIENT  # 注释：直接返回全局缓存实例


# TODO (TQ): verl will make all actor async, so this can be cleanup later.  # 注释：待迁移到全异步后的清理点
def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:  # 注释：在临时事件循环里运行协程
    """
    功能：为异步函数临时创建事件循环并在独立线程中执行，避免已有事件循环冲突。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - async_func (Callable[..., Any]): 需要执行的异步函数。（注释：协程函数）
      - *args/**kwargs: 透传给 async_func 的参数。（注释：位置与关键字参数）
    返回：（注释：返回值说明）
      - Any：async_func 执行结果。（注释：返回值）
    副作用：（注释：副作用说明）
      - 创建新的事件循环与后台线程。（注释：资源副作用）
    异常/边界条件：（注释：异常与边界）
      - async_func 内部异常会在 result() 时抛出。（注释：异常传播）
    最小示例：（注释：最小可理解示例）
      - 输入：async_func=async lambda x: x+1, args=(1,)
      - 输出：2
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_run_async_in_temp_loop`
      - 典型调用路径：`_batchmeta_to_dataproto` / `_update_batchmeta_with_output`
      - 被谁调用：仅本文件内部（BatchMeta<->DataProto 转换与回写）
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：`asyncio.run_coroutine_threadsafe`、`threading.Thread`
    """
    # Use a temporary event loop in a new thread because event  # 注释：原注释保留：避免与已有事件循环冲突
    # loop may already exist in server mode  # 注释：服务模式可能已有事件循环
    tmp_event_loop = asyncio.new_event_loop()  # 注释：创建临时事件循环
    thread = threading.Thread(  # 注释：启动线程运行事件循环
        target=tmp_event_loop.run_forever,
        name="batchmeta dataproto converter",
        daemon=True,
    )

    def run_coroutine(coroutine):  # 注释：在线程事件循环中执行协程并阻塞获取结果
        """
        功能：将协程提交到临时事件循环并同步等待结果。（注释：内部辅助函数说明）
        参数：（注释：函数参数说明）
          - coroutine: 需要执行的协程对象。（注释：协程对象）
        返回：（注释：返回值说明）
          - Any：协程执行结果。（注释：返回结果）
        副作用：（注释：副作用说明）
          - 可能启动线程并阻塞当前线程直到结果返回。（注释：线程/阻塞副作用）
        异常/边界条件：（注释：异常与边界）
          - 协程内部异常会在 future.result() 时抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：run_coroutine(async_func())
          - 输出：async_func 的返回值
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/utils/transferqueue_utils.py::_run_async_in_temp_loop.run_coroutine`
          - 典型调用路径：`_run_async_in_temp_loop` -> `run_coroutine`
          - 被谁调用：仅本函数 `_run_async_in_temp_loop`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`asyncio.run_coroutine_threadsafe`
        """
        if not thread.is_alive():  # 注释：确保事件循环线程已启动
            thread.start()  # 注释：启动线程
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)  # 注释：提交协程到线程事件循环
        return future.result()  # 注释：同步等待并返回结果

    async def stop_loop():  # 注释：异步停止事件循环
        """
        功能：停止临时事件循环以便线程退出。（注释：内部辅助函数说明）
        参数：（注释：函数参数说明）
          - 无。（注释：无参数）
        返回：（注释：返回值说明）
          - None。（注释：无返回）
        副作用：（注释：副作用说明）
          - 请求停止事件循环。（注释：事件循环副作用）
        异常/边界条件：（注释：异常与边界）
          - 无显式异常。（注释：边界条件）
        最小示例：（注释：最小可理解示例）
          - 输入：await stop_loop()
          - 输出：None
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/utils/transferqueue_utils.py::_run_async_in_temp_loop.stop_loop`
          - 典型调用路径：`_run_async_in_temp_loop` -> `stop_loop`
          - 被谁调用：仅本函数 `_run_async_in_temp_loop`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：无
        """
        tmp_event_loop.stop()  # 注释：请求停止事件循环

    try:  # 注释：执行异步函数并返回结果
        return run_coroutine(async_func(*args, **kwargs))  # 注释：提交协程并等待结果
    finally:  # 注释：确保线程与事件循环正确清理
        if thread.is_alive():  # 注释：仅在线程存活时清理
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)  # 注释：请求停止事件循环
            thread.join()  # 注释：等待线程结束


def _find_batchmeta(*args, **kwargs):  # 注释：在参数中查找 BatchMeta
    """
    功能：在位置参数与关键字参数中查找第一个 BatchMeta 实例。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - *args/**kwargs: 任意参数集合。（注释：待扫描参数）
    返回：（注释：返回值说明）
      - BatchMeta | None：找到则返回对象，否则 None。（注释：返回形态）
    副作用：（注释：副作用说明）
      - 无（只读扫描）。（注释：无副作用）
    异常/边界条件：（注释：异常与边界）
      - 无显式异常。（注释：边界条件）
    最小示例：（注释：最小可理解示例）
      - 输入：_find_batchmeta(1, BatchMeta(), k=2)
      - 输出：BatchMeta 实例
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_find_batchmeta`
      - 典型调用路径：`tqbridge` -> `inner/async_inner`
      - 被谁调用：仅本文件 `tqbridge` 内部包装函数
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：`isinstance`
    """
    for arg in args:  # 注释：扫描位置参数
        if isinstance(arg, BatchMeta):  # 注释：命中 BatchMeta
            return arg  # 注释：返回找到的 BatchMeta
    for v in kwargs.values():  # 注释：扫描关键字参数
        if isinstance(v, BatchMeta):  # 注释：命中 BatchMeta
            return v  # 注释：返回找到的 BatchMeta
    return None  # 注释：未找到则返回 None


async def _async_batchmeta_to_dataproto(batchmeta: "BatchMeta") -> DataProto:  # 注释：异步将 BatchMeta 转为 DataProto
    """
    功能：异步拉取 BatchMeta 对应数据并构造 DataProto。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - batchmeta (BatchMeta): TransferQueue 批次元信息。（注释：输入批次）
    返回：（注释：返回值说明）
      - DataProto：包含张量 batch 与 meta_info 的数据容器。（注释：返回形态）
    副作用：（注释：副作用说明）
      - 通过 TransferQueue 客户端进行数据拉取（网络 I/O）。（注释：I/O 副作用）
    异常/边界条件：（注释：异常与边界）
      - batchmeta.samples 为空时返回空 DataProto。（注释：空批次处理）
      - _TRANSFER_QUEUE_CLIENT 未初始化时会抛 AttributeError。（注释：客户端未就绪）
    最小示例：（注释：最小可理解示例）
      - 输入：BatchMeta(samples=[...], extra_info={})
      - 输出：DataProto.from_tensordict(...)
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_async_batchmeta_to_dataproto`
      - 典型调用路径：`tqbridge` -> `async_inner`
      - 被谁调用：本文件 `tqbridge` 的异步包装器与 `_batchmeta_to_dataproto`
      - 调用了谁（项目内）：`DataProto.from_tensordict`
      - 调用了谁（外部依赖）：`_TRANSFER_QUEUE_CLIENT.async_get_data`
    """
    if batchmeta.samples == [] or batchmeta.samples is None:  # 注释：空样本直接返回空 DataProto
        return DataProto(  # 注释：构造空批次 DataProto
            batch=TensorDict({}, batch_size=(0,)),  # 注释：空 TensorDict
            non_tensor_batch={},  # 注释：空非张量字段
            meta_info=batchmeta.extra_info.copy(),  # 注释：保留元信息
        )

    tensordict = await _TRANSFER_QUEUE_CLIENT.async_get_data(batchmeta)  # 注释：异步拉取数据
    return DataProto.from_tensordict(  # 注释：转换为 DataProto
        tensordict, meta_info=batchmeta.extra_info.copy()
    )


def _batchmeta_to_dataproto(batchmeta: "BatchMeta") -> DataProto:  # 注释：同步将 BatchMeta 转为 DataProto
    """
    功能：同步包装 _async_batchmeta_to_dataproto，用临时事件循环执行异步拉取。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - batchmeta (BatchMeta): TransferQueue 批次元信息。（注释：输入批次）
    返回：（注释：返回值说明）
      - DataProto：转换后的数据容器。（注释：返回形态）
    副作用：（注释：副作用说明）
      - 创建临时事件循环与线程（由 _run_async_in_temp_loop 完成）。（注释：资源副作用）
    异常/边界条件：（注释：异常与边界）
      - _TRANSFER_QUEUE_CLIENT 未初始化时会抛异常。（注释：客户端未就绪）
    最小示例：（注释：最小可理解示例）
      - 输入：BatchMeta(...)
      - 输出：DataProto(...)
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_batchmeta_to_dataproto`
      - 典型调用路径：`tqbridge` -> `inner`
      - 被谁调用：本文件 `tqbridge` 的同步包装器
      - 调用了谁（项目内）：`_run_async_in_temp_loop`、`_async_batchmeta_to_dataproto`
      - 调用了谁（外部依赖）：无
    """
    return _run_async_in_temp_loop(_async_batchmeta_to_dataproto, batchmeta)  # 注释：临时事件循环执行异步转换


async def _async_update_batchmeta_with_output(  # 注释：异步回写输出到 BatchMeta
    output: DataProto, batchmeta: "BatchMeta", func_name=None
) -> "BatchMeta":
    """
    功能：将 DataProto 输出写入 TransferQueue，并更新 BatchMeta 的 extra_info。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - output (DataProto): 函数输出数据。（注释：输出 DataProto）
      - batchmeta (BatchMeta): 原始 BatchMeta 元信息。（注释：元信息载体）
      - func_name (str|None): 调用函数名，用于日志。（注释：日志辅助）
    返回：（注释：返回值说明）
      - BatchMeta：更新后的 BatchMeta（可能包含新数据引用）。（注释：返回形态）
    副作用：（注释：副作用说明）
      - 写入 TransferQueue 存储（网络/存储 I/O）。（注释：I/O 副作用）
      - 更新 batchmeta.extra_info。（注释：元信息副作用）
      - 记录日志。（注释：日志副作用）
    异常/边界条件：（注释：异常与边界）
      - _TRANSFER_QUEUE_CLIENT 未初始化会抛异常。（注释：客户端未就绪）
      - output 为空（len==0）时直接返回原 batchmeta。（注释：空输出处理）
    最小示例：（注释：最小可理解示例）
      - 输入：output=DataProto(...), batchmeta=BatchMeta(...)
      - 输出：updated BatchMeta（指向新输出数据）
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_async_update_batchmeta_with_output`
      - 典型调用路径：`tqbridge` -> `async_inner`
      - 被谁调用：本文件 `tqbridge` 的异步包装器与 `_update_batchmeta_with_output`
      - 调用了谁（项目内）：`DataProto.to_tensordict`
      - 调用了谁（外部依赖）：`_TRANSFER_QUEUE_CLIENT.async_put`
    """
    pid = os.getpid()  # 注释：记录进程号用于日志

    for k, v in output.meta_info.items():  # 注释：将 meta_info 写入 BatchMeta extra_info
        batchmeta.set_extra_info(k, v)  # 注释：更新元信息

    if len(output) > 0:  # 注释：仅在非空输出时写入 TransferQueue
        tensordict = output.to_tensordict()  # 注释：转换为 TensorDict
        # pop meta_info  # 注释：去除 meta_info 字段，避免重复存储
        for key in output.meta_info.keys():  # 注释：遍历 meta_info 的键
            tensordict.pop(key)  # 注释：从 tensordict 中移除 meta 字段

        logger.info(  # 注释：记录写入日志
            f"Task {func_name} (pid={pid}) putting output data to TransferQueue with "
            f"batch_size={tensordict.batch_size},\n"
            f"tensordict keys={list(tensordict.keys())}"
        )

        updated_batch_meta = await _TRANSFER_QUEUE_CLIENT.async_put(  # 注释：异步写入 TransferQueue
            data=tensordict, metadata=batchmeta
        )
        return updated_batch_meta  # 注释：返回更新后的 BatchMeta
    else:  # 注释：空输出直接返回原 BatchMeta
        return batchmeta  # 注释：不写入 TransferQueue


def _update_batchmeta_with_output(  # 注释：同步回写输出到 BatchMeta
    output: DataProto, batchmeta: "BatchMeta", func_name=None
) -> "BatchMeta":
    """
    功能：同步包装 _async_update_batchmeta_with_output。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - output (DataProto): 函数输出数据。（注释：输出 DataProto）
      - batchmeta (BatchMeta): 原始 BatchMeta。（注释：元信息载体）
      - func_name (str|None): 调用函数名，用于日志。（注释：日志辅助）
    返回：（注释：返回值说明）
      - BatchMeta：更新后的 BatchMeta。（注释：返回形态）
    副作用：（注释：副作用说明）
      - 创建临时事件循环执行异步写入。（注释：资源副作用）
    异常/边界条件：（注释：异常与边界）
      - 同 _async_update_batchmeta_with_output。（注释：异常传播）
    最小示例：（注释：最小可理解示例）
      - 输入：_update_batchmeta_with_output(output, batchmeta, "fn")
      - 输出：updated BatchMeta
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_update_batchmeta_with_output`
      - 典型调用路径：`tqbridge` -> `inner`
      - 被谁调用：本文件 `tqbridge` 的同步包装器
      - 调用了谁（项目内）：`_run_async_in_temp_loop`、`_async_update_batchmeta_with_output`
      - 调用了谁（外部依赖）：无
    """
    updated_batch_meta = _run_async_in_temp_loop(  # 注释：使用临时事件循环执行异步写入
        _async_update_batchmeta_with_output, output, batchmeta, func_name
    )
    return updated_batch_meta  # 注释：返回更新后的 BatchMeta


def _compute_need_collect(dispatch_mode: "dict | Dispatch", args: list) -> bool:  # 注释：判断当前进程是否需要收集数据
    """
    功能：根据调度模式与 Worker 信息判断当前进程是否负责数据收集。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - dispatch_mode (dict|Dispatch|None): 调度配置；含 collect_fn 时可做惰性收集判断。（注释：调度信息）
      - args (list): 被装饰函数的参数列表（期望 args[0] 为 Worker）。（注释：参数列表）
    返回：（注释：返回值说明）
      - bool：True 表示当前进程需要收集数据。（注释：返回含义）
    副作用：（注释：副作用说明）
      - 无（仅查询）。（注释：无副作用）
    异常/边界条件：（注释：异常与边界）
      - dispatch_mode 为 dict 但缺少 collect_fn 时触发断言。（注释：配置校验）
      - collect_fn 不是 partial 或参数不匹配时默认返回 True。（注释：回退策略）
    最小示例：（注释：最小可理解示例）
      - 输入：dispatch_mode=None, args=[worker]
      - 输出：True
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_compute_need_collect`
      - 典型调用路径：`tqbridge` -> `inner/async_inner`
      - 被谁调用：本文件 `tqbridge`
      - 调用了谁（项目内）：`Worker.query_collect_info`
      - 调用了谁（外部依赖）：`functools.partial`
    """
    from verl.single_controller.base.decorator import Dispatch  # 注释：延迟导入避免循环依赖
    from verl.single_controller.base.worker import Worker  # 注释：Worker 类型用于惰性收集判断

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):  # 注释：未配置或直接 Dispatch 时默认收集
        return True  # 注释：当前进程需要收集数据

    assert "collect_fn" in dispatch_mode.keys(), "collect_fn should be in dispatch_mode."  # 注释：确保存在收集函数

    collect_fn = dispatch_mode["collect_fn"]  # 注释：提取 collect_fn

    # Check if collect_fn is a functools.partial and handle gracefully  # 注释：仅对 partial 可读取参数
    if isinstance(collect_fn, functools.partial):  # 注释：partial 情况可解析 mesh 名称
        collect_fn_name = collect_fn.func.__name__  # 注释：获取被 partial 的函数名
        if (  # 注释：函数名/参数/类型不匹配则回退默认行为
            collect_fn_name != "collect_lazy_compute_data_proto"
            or len(args) < 1
            or not isinstance(args[0], Worker)
        ):
            return True  # 注释：不满足惰性收集条件，默认收集

        collect_mesh_name = collect_fn.args[0] if collect_fn.args else None  # 注释：读取 mesh 名称
        if collect_mesh_name is None:  # 注释：缺失 mesh 名称则回退
            return True  # 注释：默认收集

        return args[0].query_collect_info(collect_mesh_name)  # 注释：由 Worker 判断是否收集
    else:  # 注释：非 partial 无法解析 mesh 信息
        # If collect_fn is not a partial, we can't extract mesh_name information  # 注释：保留原说明
        # Fall back to default behavior (collect data)  # 注释：回退默认收集
        return True  # 注释：默认收集


def _postprocess_common(output, put_data, need_collect):  # 注释：统一处理 tqbridge 输出
    """
    功能：根据 put_data/need_collect 决定是否返回空占位结果。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - output: 被装饰函数的原始输出。（注释：原始输出）
      - put_data (bool): 是否需要将输出写回 TransferQueue。（注释：写回开关）
      - need_collect (bool): 当前进程是否需要收集数据。（注释：收集开关）
    返回：（注释：返回值说明）
      - BatchMeta.empty()：需要 BatchMeta 结构但不收集数据时。（注释：空 BatchMeta）
      - DataProto()：不写回且不收集且 output 为 DataProto 时。（注释：空 DataProto）
      - output：其他情况返回原输出。（注释：原样返回）
    副作用：（注释：副作用说明）
      - 无（纯函数式处理）。（注释：无副作用）
    异常/边界条件：（注释：异常与边界）
      - BatchMeta.empty 依赖 transfer_queue；若缺失可能抛异常。（注释：依赖边界）
    最小示例：（注释：最小可理解示例）
      - 输入：output=DataProto(...), put_data=False, need_collect=False
      - 输出：DataProto()（空）
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::_postprocess_common`
      - 典型调用路径：`tqbridge` -> `inner/async_inner`
      - 被谁调用：本文件 `tqbridge`
      - 调用了谁（项目内）：`DataProto`
      - 调用了谁（外部依赖）：`BatchMeta.empty`
    """
    if put_data and not need_collect:  # 注释：需要 BatchMeta 结构但当前不收集
        return BatchMeta.empty()  # 注释：返回空 BatchMeta
    elif not put_data and not need_collect and isinstance(output, DataProto):  # 注释：不写回且 output 为 DataProto
        return DataProto()  # 注释：返回空 DataProto 避免冗余通信
    else:  # 注释：其他情况保留原输出
        return output  # 注释：原样返回


def tqbridge(dispatch_mode: "dict | Dispatch" = None, put_data: bool = True):  # 注释：BatchMeta/DataProto 桥接装饰器
    """
    功能：构造装饰器，实现 BatchMeta 与 DataProto 的自动转换与回写。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - dispatch_mode (dict|Dispatch|None): 用于判断是否需要收集数据。（注释：调度配置）
      - put_data (bool): 是否将输出写回 TransferQueue。（注释：写回开关）
    返回：（注释：返回值说明）
      - decorator：可装饰同步/异步函数的装饰器。（注释：返回装饰器）
    副作用：（注释：副作用说明）
      - 根据配置可能触发 TransferQueue 读写。（注释：I/O 副作用）
      - 记录日志。（注释：日志副作用）
    异常/边界条件：（注释：异常与边界）
      - transfer_queue 未安装但启用时，BatchMeta/Client 调用可能失败。（注释：依赖边界）
      - dispatch_mode 配置不完整会触发断言。（注释：配置校验）
    最小示例：（注释：最小可理解示例）
      - 输入：@tqbridge(put_data=False)\n        def compute_reward(data): ...
      - 输出：compute_reward 自动支持 BatchMeta 输入
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/transferqueue_utils.py::tqbridge`
      - 典型调用路径：`verl/trainer/ppo/reward.py::compute_reward` -> `@tqbridge`
      - 被谁调用：`recipe/transfer_queue/ray_trainer.py`、`verl/experimental/agent_loop/agent_loop.py`、
        `verl/single_controller/base/decorator.py`
      - 调用了谁（项目内）：`_find_batchmeta`、`_batchmeta_to_dataproto`、`_async_batchmeta_to_dataproto`、
        `_update_batchmeta_with_output`、`_async_update_batchmeta_with_output`、`_postprocess_common`
      - 调用了谁（外部依赖）：`inspect.iscoroutinefunction`
    """

    def decorator(func):  # 注释：返回具体的函数包装器
        """
        功能：根据函数类型生成同步/异步包装器，并选择是否启用 TransferQueue 逻辑。（注释：内部装饰器说明）
        参数：（注释：函数参数说明）
          - func (Callable): 被装饰的原函数。（注释：原函数）
        返回：（注释：返回值说明）
          - Callable：包装后的函数（同步或异步）。（注释：返回包装器）
        副作用：（注释：副作用说明）
          - 读取环境变量决定是否启用 TQ。（注释：环境读取）
        异常/边界条件：（注释：异常与边界）
          - 无显式异常，内部异常由包装函数抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：decorator(func)
          - 输出：wrapper(func)
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/utils/transferqueue_utils.py::tqbridge.decorator`
          - 典型调用路径：`tqbridge` -> `decorator`
          - 被谁调用：`tqbridge`
          - 调用了谁（项目内）：`inner`/`async_inner`/`dummy_inner`/`dummy_async_inner`
          - 调用了谁（外部依赖）：`inspect.iscoroutinefunction`
        """
        pid = os.getpid()  # 注释：记录进程号用于日志

        @wraps(func)
        def inner(*args, **kwargs):  # 注释：同步包装器（含 BatchMeta 转换）
            """
            功能：同步执行并在需要时完成 BatchMeta 回写。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - *args/**kwargs: 被装饰函数的参数。（注释：透传参数）
            返回：（注释：返回值说明）
              - BatchMeta 或原函数输出。（注释：根据 put_data 决定）
            副作用：（注释：副作用说明）
              - 可能触发 TransferQueue 读写与日志。（注释：I/O 与日志）
            异常/边界条件：（注释：异常与边界）
              - TransferQueue 未初始化时可能抛异常。（注释：依赖边界）
            最小示例：（注释：最小可理解示例）
              - 输入：inner(BatchMeta(...))
              - 输出：BatchMeta 或 DataProto
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`verl/utils/transferqueue_utils.py::tqbridge.inner`
              - 典型调用路径：`tqbridge` -> `decorator` -> `inner`
              - 被谁调用：`decorator` 返回的包装器
              - 调用了谁（项目内）：`_find_batchmeta`、`_batchmeta_to_dataproto`、`_update_batchmeta_with_output`
              - 调用了谁（外部依赖）：无
            """
            batchmeta = _find_batchmeta(*args, **kwargs)  # 注释：尝试获取 BatchMeta
            if batchmeta is None:  # 注释：无 BatchMeta 则直接执行原函数
                return func(*args, **kwargs)  # 注释：原样调用
            else:  # 注释：存在 BatchMeta 需进行转换
                logger.info(  # 注释：记录批次信息
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batchmeta.size}, "
                    f"global_idx={batchmeta.global_indexes}"
                )
                args = [  # 注释：将位置参数中的 BatchMeta 转换为 DataProto
                    _batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args
                ]
                kwargs = {  # 注释：将关键字参数中的 BatchMeta 转换为 DataProto
                    k: _batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()
                }
                output = func(*args, **kwargs)  # 注释：执行原函数
                need_collect = _compute_need_collect(dispatch_mode, args)  # 注释：判断是否需要收集
                if put_data and need_collect:  # 注释：需要写回时更新 BatchMeta
                    updated_batch_meta = _update_batchmeta_with_output(output, batchmeta, func.__name__)  # 注释：回写输出
                    return updated_batch_meta  # 注释：返回更新后的 BatchMeta
                return _postprocess_common(output, put_data, need_collect)  # 注释：统一后处理

        @wraps(func)
        async def async_inner(*args, **kwargs):  # 注释：异步包装器（含 BatchMeta 转换）
            """
            功能：异步执行并在需要时完成 BatchMeta 回写。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - *args/**kwargs: 被装饰函数的参数。（注释：透传参数）
            返回：（注释：返回值说明）
              - BatchMeta 或原函数输出。（注释：根据 put_data 决定）
            副作用：（注释：副作用说明）
              - 可能触发 TransferQueue 读写与日志。（注释：I/O 与日志）
            异常/边界条件：（注释：异常与边界）
              - TransferQueue 未初始化时可能抛异常。（注释：依赖边界）
            最小示例：（注释：最小可理解示例）
              - 输入：await async_inner(BatchMeta(...))
              - 输出：BatchMeta 或 DataProto
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`verl/utils/transferqueue_utils.py::tqbridge.async_inner`
              - 典型调用路径：`tqbridge` -> `decorator` -> `async_inner`
              - 被谁调用：`decorator` 返回的包装器
              - 调用了谁（项目内）：`_find_batchmeta`、`_async_batchmeta_to_dataproto`、`_async_update_batchmeta_with_output`
              - 调用了谁（外部依赖）：无
            """
            batchmeta = _find_batchmeta(*args, **kwargs)  # 注释：尝试获取 BatchMeta
            if batchmeta is None:  # 注释：无 BatchMeta 则直接执行原函数
                return await func(*args, **kwargs)  # 注释：await 原函数
            else:  # 注释：存在 BatchMeta 需进行转换
                logger.info(  # 注释：记录批次信息
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batchmeta.size}, "
                    f"global_idx={batchmeta.global_indexes}"
                )
                args = [  # 注释：异步转换位置参数中的 BatchMeta
                    await _async_batchmeta_to_dataproto(arg) if isinstance(arg, BatchMeta) else arg for arg in args
                ]
                kwargs = {  # 注释：异步转换关键字参数中的 BatchMeta
                    k: await _async_batchmeta_to_dataproto(v) if isinstance(v, BatchMeta) else v
                    for k, v in kwargs.items()
                }
                output = await func(*args, **kwargs)  # 注释：await 原函数
                need_collect = _compute_need_collect(dispatch_mode, args)  # 注释：判断是否需要收集
                if put_data and need_collect:  # 注释：需要写回时更新 BatchMeta
                    updated_batchmeta = await _async_update_batchmeta_with_output(  # 注释：异步回写输出
                        output, batchmeta, func.__name__
                    )
                    return updated_batchmeta  # 注释：返回更新后的 BatchMeta
                return _postprocess_common(output, put_data, need_collect)  # 注释：统一后处理

        @wraps(func)
        def dummy_inner(*args, **kwargs):  # 注释：TQ 未启用时的同步直通包装器
            """
            功能：在未启用 TransferQueue 时直接调用原函数。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - *args/**kwargs: 原函数参数。（注释：透传参数）
            返回：（注释：返回值说明）
              - 原函数返回值。（注释：原样返回）
            副作用：（注释：副作用说明）
              - 无。（注释：无副作用）
            异常/边界条件：（注释：异常与边界）
              - 原函数异常原样抛出。（注释：异常传播）
            最小示例：（注释：最小可理解示例）
              - 输入：dummy_inner(x)
              - 输出：func(x)
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`verl/utils/transferqueue_utils.py::tqbridge.dummy_inner`
              - 典型调用路径：`tqbridge` -> `decorator` -> `dummy_inner`
              - 被谁调用：`decorator` 返回的包装器（TQ 未启用）
              - 调用了谁（项目内）：无
              - 调用了谁（外部依赖）：无
            """
            output = func(*args, **kwargs)  # 注释：直接执行原函数
            return output  # 注释：原样返回

        @wraps(func)
        async def dummy_async_inner(*args, **kwargs):  # 注释：TQ 未启用时的异步直通包装器
            """
            功能：未启用 TransferQueue 时直接 await 原函数。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - *args/**kwargs: 原函数参数。（注释：透传参数）
            返回：（注释：返回值说明）
              - 原函数返回值。（注释：原样返回）
            副作用：（注释：副作用说明）
              - 无。（注释：无副作用）
            异常/边界条件：（注释：异常与边界）
              - 原函数异常原样抛出。（注释：异常传播）
            最小示例：（注释：最小可理解示例）
              - 输入：await dummy_async_inner(x)
              - 输出：await func(x)
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`verl/utils/transferqueue_utils.py::tqbridge.dummy_async_inner`
              - 典型调用路径：`tqbridge` -> `decorator` -> `dummy_async_inner`
              - 被谁调用：`decorator` 返回的包装器（TQ 未启用）
              - 调用了谁（项目内）：无
              - 调用了谁（外部依赖）：无
            """
            output = await func(*args, **kwargs)  # 注释：直接 await 原函数
            return output  # 注释：原样返回

        wrapper_inner = inner if is_transferqueue_enabled else dummy_inner  # 注释：根据开关选择同步包装器
        wrapper_async_inner = async_inner if is_transferqueue_enabled else dummy_async_inner  # 注释：选择异步包装器

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner  # 注释：按函数类型分派
        return wrapper  # 注释：返回最终包装器

    return decorator  # 注释：返回装饰器本体
