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
模块用途：定义通用 checkpoint 管理接口与工具函数。（注释：模块职责）
输入/输出：
  - 输入：模型/优化器/调度器/随机状态及保存路径。（注释：输入说明）
  - 输出：checkpoint 目录与恢复状态。（注释：输出说明）
关键依赖：torch、numpy、omegaconf、transformers。（注释：依赖说明）
典型用法（最小示例）：
  - `ckpt_path = find_latest_ckpt_path(output_dir)`。（注释：查找最新 ckpt）
调用路径概览：
  - 训练入口 -> `BaseCheckpointManager` 子类（如 FSDPCheckpointManager）-> 本模块工具函数。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import os  # 注释：路径操作
import random  # 注释：Python 随机数
import shutil  # 注释：删除目录

# ===== 第三方依赖导入 =====
import numpy as np  # 注释：NumPy 随机数状态
import torch  # 注释：PyTorch 状态与分布式
import torch.distributed  # 注释：分布式通信
from omegaconf import DictConfig  # 注释：配置类型
from transformers import PreTrainedTokenizer, ProcessorMixin  # 注释：HF tokenizer/processor

# ===== 项目内依赖导入 =====
from verl.trainer.config import CheckpointConfig  # 注释：训练配置
from verl.utils.device import get_device_name, get_torch_device  # 注释：设备工具


class BaseCheckpointManager:
    """
    通用 checkpoint 管理器基类，负责保存/加载模型与训练状态。（注释：类用途）

    保存/加载内容：
      - model、optimizer、lr_scheduler、extra_states（如 RNG）。（注释：保存内容）
      - HF tokenizer/config（用于恢复或合并）。（注释：补充说明）
    返回：实例对象。（注释：构造返回）
    副作用：初始化分布式 rank/world_size 读取。（注释：副作用）
    异常/边界条件：子类必须实现 load/save。（注释：边界条件）
    最小示例：
      - 输入：model/optimizer 及 checkpoint_config。（注释：示例输入）
      - 输出：可调用的 checkpoint manager。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `FSDPCheckpointManager` -> `BaseCheckpointManager`。（注释：典型路径）
      - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py` 等子类。（注释：调用方）
      - 调用了谁（项目内）：`get_device_name`、`get_torch_device`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`torch.distributed.get_rank`/`get_world_size`。（注释：外部依赖）
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig | CheckpointConfig = None,
    ):
        """
        初始化 checkpoint 管理器，记录保存/加载策略与训练组件。（注释：方法用途）

        参数：
          - model：训练模型。（注释：输入说明）
          - optimizer：优化器。（注释：输入说明）
          - lr_scheduler：学习率调度器。（注释：输入说明）
          - processing_class：HF tokenizer/processor。（注释：输入说明）
          - checkpoint_config：保存/加载内容配置。（注释：输入说明）
        返回：无。（注释：构造函数）
        副作用：读取分布式 rank/world_size 并初始化内部状态。（注释：副作用）
        异常/边界条件：checkpoint_config 为空时使用默认内容列表。（注释：边界条件）
        最小示例：
          - 输入：checkpoint_config.save_contents=["model"]。（注释：示例输入）
          - 输出：只保存模型。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.__init__`。（注释：定位）
          - 典型调用路径：`FSDPCheckpointManager.__init__` -> `BaseCheckpointManager.__init__`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：`torch.distributed.get_rank/get_world_size`。（注释：外部依赖）
        """
        # 解析保存/加载内容配置（可来自 DictConfig 或默认值）  # 注释：配置解析
        self.checkpoint_config = checkpoint_config  # 注释：保存原始配置
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None  # 注释：加载内容列表
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None  # 注释：保存内容列表
        if checkpoint_load_contents is None:  # 注释：默认加载内容
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:  # 注释：默认保存内容
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None  # 注释：记录上次保存步数
        self.previous_saved_paths = []  # 注释：已保存路径列表

        # 保存训练组件引用  # 注释：组件绑定
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()  # 注释：当前 rank
        self.world_size = torch.distributed.get_world_size()  # 注释：世界大小

    @property
    def should_save_model(self) -> bool:
        """
        是否需要保存模型状态。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_save_model`。（注释：定位）
          - 典型调用路径：子类 `save_checkpoint` -> `should_save_model`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "model" in self.checkpoint_save_contents  # 注释：检查保存内容

    @property
    def should_save_optimizer(self) -> bool:
        """
        是否需要保存优化器状态。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_save_optimizer`。（注释：定位）
          - 典型调用路径：子类 `save_checkpoint` -> `should_save_optimizer`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "optimizer" in self.checkpoint_save_contents  # 注释：检查保存内容

    @property
    def should_save_extra(self) -> bool:
        """
        是否需要保存额外状态（如 RNG/调度器）。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_save_extra`。（注释：定位）
          - 典型调用路径：子类 `save_checkpoint` -> `should_save_extra`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "extra" in self.checkpoint_save_contents  # 注释：检查保存内容

    @property
    def should_save_hf_model(self) -> bool:
        """
        是否需要额外保存 HF 格式模型。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_save_hf_model`。（注释：定位）
          - 典型调用路径：子类 `save_checkpoint` -> `should_save_hf_model`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "hf_model" in self.checkpoint_save_contents  # 注释：检查保存内容

    @property
    def should_load_model(self) -> bool:
        """
        是否需要加载模型状态。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_load_model`。（注释：定位）
          - 典型调用路径：子类 `load_checkpoint` -> `should_load_model`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "model" in self.checkpoint_load_contents  # 注释：检查加载内容

    @property
    def should_load_optimizer(self) -> bool:
        """
        是否需要加载优化器状态。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_load_optimizer`。（注释：定位）
          - 典型调用路径：子类 `load_checkpoint` -> `should_load_optimizer`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "optimizer" in self.checkpoint_load_contents  # 注释：检查加载内容

    @property
    def should_load_extra(self) -> bool:
        """
        是否需要加载额外状态。（注释：属性用途）

        返回：bool。（注释：返回说明）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.should_load_extra`。（注释：定位）
          - 典型调用路径：子类 `load_checkpoint` -> `should_load_extra`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        return "extra" in self.checkpoint_load_contents  # 注释：检查加载内容

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        """
        加载 checkpoint 的抽象接口。（注释：方法用途）

        参数：
          - local_path (str)：本地路径。（注释：输入说明）
          - hdfs_path (str, optional)：远端路径（如 HDFS）。（注释：输入说明）
          - del_local_after_load (bool)：加载后是否删除本地文件。（注释：输入说明）
        返回：无。（注释：子类实现）
        副作用：加载模型/优化器/调度器/随机状态。（注释：副作用）
        异常/边界条件：子类必须实现。（注释：边界条件）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.load_checkpoint`。（注释：定位）
          - 典型调用路径：训练恢复 -> 子类 `load_checkpoint`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        raise NotImplementedError  # 注释：由子类实现

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        """
        保存 checkpoint 的抽象接口。（注释：方法用途）

        参数：
          - local_path (str)：本地保存路径。（注释：输入说明）
          - hdfs_path (str, optional)：远端保存路径。（注释：输入说明）
          - global_step (int)：当前训练步。（注释：输入说明）
          - max_ckpt_to_keep (int, optional)：最多保留多少个 ckpt。（注释：输入说明）
        返回：无。（注释：子类实现）
        副作用：写入模型/优化器/调度器/随机状态。（注释：副作用）
        异常/边界条件：子类必须实现。（注释：边界条件）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.save_checkpoint`。（注释：定位）
          - 典型调用路径：训练过程 -> 子类 `save_checkpoint`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：无。（注释：外部依赖）
        """
        raise NotImplementedError  # 注释：由子类实现

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        """
        校验本地/远端路径并返回可用路径。（注释：方法用途）

        参数：
          - local_path/hdfs_path：二者至少有一个不为空。（注释：输入说明）
        返回：
          - (bool, str)：是否使用本地路径，以及实际路径。（注释：返回说明）
        异常/边界条件：二者都为空则断言失败。（注释：边界条件）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.checkpath`。（注释：定位）
          - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
          - 调用了谁（外部依赖）：断言机制。（注释：外部依赖）
        """
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"  # 注释：基本校验
        return local_path is not None, local_path if local_path is not None else hdfs_path  # 注释：返回可用路径

    def remove_previous_save_local_path(self, path):
        """
        删除历史 checkpoint 目录。（注释：方法用途）

        参数：
          - path (str|list)：路径或路径列表。（注释：输入说明）
        返回：无。（注释：仅副作用）
        副作用：删除目录。（注释：副作用）
        异常/边界条件：路径不存在则跳过。（注释：边界条件）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.remove_previous_save_local_path`。（注释：定位）
          - 典型调用路径：子类 `save_checkpoint` -> 旧 ckpt 清理。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：`shutil.rmtree`。（注释：外部依赖）
        """
        if isinstance(path, str):  # 注释：统一为列表
            path = [path]
        for p in path:  # 注释：遍历路径列表
            abs_path = os.path.abspath(p)  # 注释：转换为绝对路径
            print(f"Checkpoint manager remove previous save local path: {abs_path}")  # 注释：输出提示
            if not os.path.exists(abs_path):  # 注释：路径不存在则跳过
                continue
            shutil.rmtree(abs_path, ignore_errors=True)  # 注释：递归删除目录

    @staticmethod
    def get_rng_state():
        """
        获取 CPU/设备 的随机数状态。（注释：方法用途）

        返回：dict，包含 cpu/numpy/random/设备 RNG 状态。（注释：返回说明）
        副作用：无。（注释：纯查询）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.get_rng_state`。（注释：定位）
          - 典型调用路径：子类 `save_checkpoint` -> `get_rng_state`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：`get_device_name`、`get_torch_device`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`torch.get_rng_state`、`numpy.random.get_state`。（注释：外部依赖）
        """
        rng_state = {  # 注释：收集 CPU/NumPy/Python 随机状态
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        if get_device_name() != "cpu":  # 注释：若存在加速设备，记录其 RNG
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state  # 注释：返回 RNG 状态字典

    @staticmethod
    def load_rng_state(rng_state):
        """
        恢复 CPU/设备 的随机数状态。（注释：方法用途）

        参数：
          - rng_state (dict)：由 `get_rng_state` 产生的状态字典。（注释：输入说明）
        返回：无。（注释：仅副作用）
        副作用：覆盖当前 RNG 状态。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::BaseCheckpointManager.load_rng_state`。（注释：定位）
          - 典型调用路径：子类 `load_checkpoint` -> `load_rng_state`。（注释：典型路径）
          - 被谁调用：`verl/utils/checkpoint/fsdp_checkpoint_manager.py`。（注释：调用方）
          - 调用了谁（项目内）：`get_device_name`、`get_torch_device`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`torch.set_rng_state`、`numpy.random.set_state`。（注释：外部依赖）
        """
        torch.set_rng_state(rng_state["cpu"])  # 注释：恢复 CPU RNG
        np.random.set_state(rng_state["numpy"])  # 注释：恢复 NumPy RNG
        random.setstate(rng_state["random"])  # 注释：恢复 Python RNG

        if get_device_name() != "cpu":  # 注释：若有加速设备则恢复其 RNG
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    """
    根据 tracker 文件定位最新 checkpoint 目录。（注释：函数用途）

    参数：
      - path (str)：checkpoint 根目录。（注释：输入说明）
      - directory_format (str)：子目录格式，默认 "global_step_{}"。（注释：输入说明）
    返回：
      - str|None：最新 ckpt 路径；不存在则 None。（注释：返回说明）
    副作用：可能打印提示信息。（注释：副作用）
    异常/边界条件：tracker 文件或目录缺失则返回 None。（注释：边界条件）
    最小示例：
      - 输入：path="/tmp/ckpt"，tracker 记录 step=100。（注释：示例输入）
      - 输出："/tmp/ckpt/global_step_100"。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::find_latest_ckpt_path`。（注释：定位）
      - 典型调用路径：训练恢复 -> `find_latest_ckpt_path`。（注释：典型路径）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`verl/trainer/ppo/ray_trainer.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`get_checkpoint_tracker_filename`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`os.path.exists`、`torch.distributed.get_rank`。（注释：外部依赖）
    """
    if path is None:  # 注释：无路径直接返回
        return None

    tracker_file = get_checkpoint_tracker_filename(path)  # 注释：获取 tracker 文件路径
    if not os.path.exists(tracker_file):  # 注释：tracker 不存在
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:  # 注释：仅 rank0 打印
            print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None

    with open(tracker_file, "rb") as f:  # 注释：读取 tracker 文件
        iteration = int(f.read().decode())  # 注释：解析最新步数
    ckpt_path = os.path.join(path, directory_format.format(iteration))  # 注释：拼接目录路径
    if not os.path.exists(ckpt_path):  # 注释：目录不存在
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)  # 注释：提示找到的 ckpt
    return ckpt_path  # 注释：返回路径


def get_checkpoint_tracker_filename(root_path: str):
    """
    获取 tracker 文件路径（记录最新 checkpoint 迭代）。（注释：函数用途）

    参数：
      - root_path (str)：checkpoint 根目录。（注释：输入说明）
    返回：
      - str：tracker 文件路径。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::get_checkpoint_tracker_filename`。（注释：定位）
      - 被谁调用：`find_latest_ckpt_path`、`verl/utils/checkpoint/checkpoint_handler.py`。（注释：调用方）
      - 调用了谁（外部依赖）：`os.path.join`。（注释：外部依赖）
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")  # 注释：固定文件名


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    """
    根据 ESI/容量过期时间判断是否需要提前保存 checkpoint。（注释：函数用途）

    参数：
      - max_steps_duration (float)：一步训练预计耗时（秒）。（注释：输入说明）
      - save_ckpt_duration (float)：保存 ckpt 预计耗时（秒）。（注释：输入说明）
      - redundant_time (float)：冗余缓冲时间（秒）。（注释：输入说明）
    返回：
      - bool：True 表示应立即保存。（注释：返回说明）
    副作用：无。（注释：纯逻辑）
    异常/边界条件：环境变量不存在则返回 False。（注释：边界条件）
    最小示例：
      - 输入：remaining=30s，save_ckpt_duration=60s。（注释：示例输入）
      - 输出：True（时间不足需保存）。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/checkpoint/checkpoint_manager.py::should_save_ckpt_esi`。（注释：定位）
      - 典型调用路径：训练循环 -> `should_save_ckpt_esi`。（注释：典型路径）
      - 被谁调用：`verl/trainer/ppo/ray_trainer.py`、`recipe/fully_async_policy/fully_async_trainer.py`。（注释：调用方）
      - 调用了谁（外部依赖）：`os.getenv`、`time.time`/`datetime`。（注释：外部依赖）
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp  # 注释：火山 MLP 过期时间
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws  # 注释：AWS 过期时间
    if exp_ts_mlp:  # 注释：优先使用 MLP 变量
        try:
            import time  # 注释：延迟导入 time

            remaining = float(exp_ts_mlp) - time.time()  # 注释：剩余时间（秒）
        except ValueError:
            return False  # 注释：时间戳非法则不保存
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )  # 注释：判断是否接近到期
    elif exp_ts_aws:  # 注释：AWS 时间戳逻辑
        from datetime import datetime, timedelta  # 注释：时间处理

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))  # 注释：过期时间
        time_difference = expiration_time - datetime.now()  # 注释：剩余时间差
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60  # 注释：阈值分钟
        return time_difference < timedelta(minutes=threshold_minutes)  # 注释：是否需要保存
    else:
        return False  # 注释：无过期信息则不保存
