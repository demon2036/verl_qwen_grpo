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
模块用途：在 FSDP + Ulysses 序列并行场景中提供数据重分片（resharding）管理器。（注释：说明模块职责）
输入/输出：
  - 输入：DeviceMesh（包含 sp 维度）、DataProto 批数据。（注释：说明输入类型）
  - 输出：与 SP 组一致的数据切分/聚合后的 DataProto。（注释：说明输出形态）
关键依赖：torch.distributed.device_mesh.DeviceMesh、verl.protocol.all_gather_data_proto、verl.utils.ulysses。（注释：列出关键依赖）
典型用法（最小示例）：
  - `with FSDPUlyssesShardingManager(device_mesh): data = mgr.preprocess_data(data)`。（注释：最小调用示意）
调用路径概览：
  - `verl/trainer/fsdp_sft_trainer.py`/`verl/workers/fsdp_workers.py` -> `FSDPUlyssesShardingManager`。（注释：典型入口）
"""  # 注释：模块级 docstring 结束

# ===== 第三方依赖导入 =====
from torch.distributed.device_mesh import DeviceMesh  # 注释：设备网格抽象

# ===== 项目内依赖导入 =====
from verl import DataProto  # 注释：统一数据结构
from verl.protocol import all_gather_data_proto  # 注释：跨进程收集 DataProto
from verl.utils.ulysses import (  # 注释：Ulysses 序列并行组管理
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)

from .base import BaseShardingManager  # 注释：分片管理器基类


class FSDPUlyssesShardingManager(BaseShardingManager):
    """
    Sharding manager to support data resharding when using FSDP + Ulysses。（注释：类用途）

    参数：
      - device_mesh (DeviceMesh)：包含 "sp" 维度的设备网格。（注释：输入说明）
    返回：实例对象。（注释：构造返回）
    副作用：
      - 在上下文管理器中临时替换全局 SP 组设置。（注释：说明副作用）
    异常/边界条件：
      - device_mesh 为 None 时所有操作为 no-op。（注释：边界条件）
    最小示例：
      - 输入：device_mesh 含 sp 维度，data 为 DataProto。（注释：示例输入）
      - 关键中间结果：进入上下文后设置 SP group。（注释：中间步骤）
      - 输出：data 被 all_gather 后再按 sp 切分。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager`。（注释：类定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `FSDPUlyssesShardingManager(...)`。（注释：典型路径）
      - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py`、
        `recipe/prime/prime_fsdp_workers.py`。（注释：主要调用方）
      - 调用了谁（项目内）：`all_gather_data_proto`、`set_ulysses_sequence_parallel_group`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`DeviceMesh.get_group`/`get_local_rank`。（注释：外部依赖）
    """

    def __init__(self, device_mesh: DeviceMesh):
        super().__init__()  # 注释：初始化基类状态
        self.device_mesh = device_mesh  # 注释：保存设备网格
        self.seed_offset = 12345  # 注释：预留随机种子偏移（暂未使用）

    def __enter__(self):
        """
        进入上下文：切换到模型特定的 SP 进程组。（注释：上下文入口说明）

        参数：无。（注释：无输入）
        返回：None。（注释：不返回上下文对象）
        副作用：修改全局 Ulysses SP 组引用。（注释：全局状态变化）
        异常/边界条件：device_mesh 为 None 时不操作。（注释：边界条件）
        最小示例：
          - 输入：device_mesh 含 sp 维度。（注释：示例输入）
          - 输出：全局 SP group 被替换。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager.__enter__`。（注释：定位）
          - 典型调用路径：`with FSDPUlyssesShardingManager(...) as mgr:`。（注释：典型用法）
          - 被谁调用：上下文管理器语法糖调用。（注释：调用方式）
          - 调用了谁（项目内）：`get_ulysses_sequence_parallel_group`、`set_ulysses_sequence_parallel_group`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`DeviceMesh.__getitem__`/`get_group`。（注释：外部依赖）
        """
        if self.device_mesh is not None:  # 注释：仅在提供 mesh 时生效
            # We have a global SP group
            # so we have to change to use model-specific sp group
            self.prev_sp_group = get_ulysses_sequence_parallel_group()  # 注释：保存原来的 SP group
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())  # 注释：切换为模型 SP group
            # TODO: check how to set seed for each model

    def __exit__(self, exc_type, exc_value, traceback):
        """
        退出上下文：恢复原始 SP 进程组设置。（注释：上下文退出说明）

        参数：
          - exc_type/exc_value/traceback：异常信息（若有）。（注释：异常上下文参数）
        返回：None。（注释：不抑制异常）
        副作用：恢复全局 Ulysses SP 组引用。（注释：全局状态变化）
        异常/边界条件：device_mesh 为 None 时不操作。（注释：边界条件）
        最小示例：
          - 输入：prev_sp_group 已保存。（注释：示例输入）
          - 输出：全局 SP group 恢复。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/workers/sharding_manager/fsdp_ulysses.py::FSDPUlyssesShardingManager.__exit__`。（注释：定位）
          - 典型调用路径：`with ...` 语句块结束自动调用。（注释：典型路径）
          - 被谁调用：上下文管理器协议。（注释：调用方式）
          - 调用了谁（项目内）：`set_ulysses_sequence_parallel_group`。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        # restore random states
        if self.device_mesh is not None:  # 注释：仅在提供 mesh 时恢复
            # revert to previous sp group
            set_ulysses_sequence_parallel_group(self.prev_sp_group)  # 注释：恢复原 SP group
            # TODO: check how to set seed for each model

    def preprocess_data(self, data: DataProto) -> DataProto:
        """
        预处理：在 SP 组内 AllGather 数据，保证组内样本一致。（注释：函数目的）

        参数：
          - data (DataProto)：输入批数据。（注释：输入说明）
        返回：
          - DataProto：AllGather 后的完整数据。（注释：返回说明）
        副作用：在分布式进程间执行通信。（注释：副作用说明）
        异常/边界条件：device_mesh 为 None 时直接返回原数据。（注释：边界条件）
        最小示例：
          - 输入：sp_size=2 时 rank0/1 各持一半数据。（注释：示例输入）
          - 关键中间结果：`all_gather_data_proto` 聚合数据。（注释：中间步骤）
          - 输出：每个 rank 得到完整 data。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/workers/sharding_manager/fsdp_ulysses.py::preprocess_data`。（注释：定位）
          - 典型调用路径：`FSDPUlyssesShardingManager.__enter__` -> `preprocess_data`。（注释：典型路径）
          - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py`。（注释：调用方）
          - 调用了谁（项目内）：`all_gather_data_proto`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`DeviceMesh.get_group`。（注释：外部依赖）
        """
        if self.device_mesh is not None:  # 注释：仅在使用 mesh 时执行通信
            group = self.device_mesh["sp"].get_group()  # 注释：获取 SP 进程组

            all_gather_data_proto(data=data, process_group=group)  # 注释：在 SP 组内聚合 data
        return data  # 注释：返回处理后的 data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """
        后处理：按 SP 组切分数据，使其与 FSDP 分片一致。（注释：函数目的）

        参数：
          - data (DataProto)：待切分的完整数据。（注释：输入说明）
        返回：
          - DataProto：当前 rank 对应的切片。（注释：返回说明）
        副作用：无。（注释：纯函数行为）
        异常/边界条件：device_mesh 为 None 时直接返回原数据。（注释：边界条件）
        最小示例：
          - 输入：sp_size=2，data 含 4 条样本。（注释：示例输入）
          - 关键中间结果：`data.chunk(2)` 得到 2 份。（注释：中间步骤）
          - 输出：当前 sp_rank 对应的那份数据。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/workers/sharding_manager/fsdp_ulysses.py::postprocess_data`。（注释：定位）
          - 典型调用路径：`preprocess_data` -> 模型计算 -> `postprocess_data`。（注释：典型路径）
          - 被谁调用：`verl/workers/fsdp_workers.py`、`verl/workers/engine/fsdp/transformer_impl.py`。（注释：调用方）
          - 调用了谁（项目内）：`DataProto.chunk`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`DeviceMesh.size`/`get_local_rank`。（注释：外部依赖）
        """
        if self.device_mesh is not None:  # 注释：仅在使用 mesh 时进行切分
            sp_size = self.device_mesh["sp"].size()  # 注释：SP 维度大小
            sp_rank = self.device_mesh["sp"].get_local_rank()  # 注释：SP 维度上的本地 rank
            data = data.chunk(chunks=sp_size)[sp_rank]  # 注释：按 sp_size 切分后取本 rank 的分片
        return data  # 注释：返回处理后的 data
