# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：分隔说明，保持逐行注释
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：获取许可证地址提示
#  # 注释：空行占位，保持逐行注释
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位，保持逐行注释
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无明示或暗示担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：定义 rollout 抽象基类与注册表，用于选择不同后端（vLLM/SGLang）。  # 注释：模块用途
输入/输出：输入配置/模型参数，输出为具体 rollout 类对象或推理输出 DataProto。  # 注释：输入输出概览
关键依赖：torch、DeviceMesh、verl.workers.config、DataProto。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- rollout_cls = get_rollout_class("vllm", "async"); rollout = rollout_cls(...).  # 注释：示例用法
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/engine_workers.py 在 init_model 中选择 rollout。  # 注释：上层入口举例
- 典型链路：trainer -> ActorRolloutRefWorker.init_model -> get_rollout_class -> 实例化 BaseRollout 子类。  # 注释：调用链路
"""  # 注释：模块 docstring 结束
# （分隔说明：标准库导入）  # 注释：替代空行，保持逐行注释
import importlib  # 注释：动态导入模块
from abc import ABC, abstractmethod  # 注释：抽象基类
from typing import Generator  # 注释：类型注解
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import torch  # 注释：张量类型与类型注解
from torch.distributed.device_mesh import DeviceMesh  # 注释：设备网格（并行拓扑）
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl import DataProto  # 注释：数据协议对象
from verl.utils.config import omega_conf_to_dataclass  # 注释：配置转换工具
from verl.workers.config import HFModelConfig, RolloutConfig  # 注释：配置数据类
# （分隔说明：模块导出）  # 注释：替代空行，保持逐行注释
__all__ = ["BaseRollout"]  # 注释：公开导出基类
# （分隔说明：抽象基类定义）  # 注释：替代空行，保持逐行注释
class BaseRollout(ABC):  # 注释：rollout 抽象基类
    """Base class for rollout.  # 注释：保留英文简述

    功能：定义 rollout 生命周期接口（resume/update/release/generate）。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - config (RolloutConfig)：rollout 配置。  # 注释：参数含义
    - model_config (HFModelConfig)：模型配置。  # 注释：参数含义
    - device_mesh (DeviceMesh)：设备拓扑。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - BaseRollout 实例（子类实现）。  # 注释：返回值语义
    副作用：初始化中可能触发配置转换。  # 注释：副作用说明
    异常/边界条件：子类必须实现抽象方法，否则实例化报错。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - class MyRollout(BaseRollout): ...  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/base.py::BaseRollout。  # 注释：类位置
    - 典型调用路径：ActorRolloutRefWorker.init_model -> get_rollout_class -> BaseRollout 子类。  # 注释：典型调用链
    - 被谁调用：verl/workers/engine_workers.py、rollout/replica.py。  # 注释：调用方说明
    - 调用了谁（项目内）：omega_conf_to_dataclass。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：torch、DeviceMesh。  # 注释：外部依赖
    """  # 注释：类 docstring 结束
    def __init__(  # 注释：构造函数
        self,  # 注释：实例本身
        config: RolloutConfig,  # 注释：rollout 配置
        model_config: HFModelConfig,  # 注释：模型配置
        device_mesh: DeviceMesh,  # 注释：设备网格
    ):  # 注释：参数列表结束
        """初始化 BaseRollout（完成配置转换与保存）。  # 注释：函数用途

        参数：  # 注释：参数说明标题
        - config (RolloutConfig)：rollout 配置。  # 注释：参数含义
        - model_config (HFModelConfig)：模型配置。  # 注释：参数含义
        - device_mesh (DeviceMesh)：设备网格。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None。  # 注释：返回值语义
        副作用：将 OmegaConf 转换为 dataclass。  # 注释：副作用说明
        异常/边界条件：config/model_config 类型不匹配时可能抛错。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - rollout = BaseRollout(config, model_config, mesh)  # 注释：示例（抽象类不直接实例化）
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/base.py::BaseRollout.__init__。  # 注释：函数位置
        - 典型调用路径：get_rollout_class -> 子类 __init__ -> super().__init__。  # 注释：典型调用链
        - 被谁调用：BaseRollout 子类构造函数。  # 注释：调用方说明
        - 调用了谁（项目内）：omega_conf_to_dataclass。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        self.config = omega_conf_to_dataclass(config)  # 注释：将配置转为数据类
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)  # 注释：转换模型配置
        self.device_mesh = device_mesh  # 注释：保存设备网格
# （分隔说明：抽象方法）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法装饰器
    async def resume(self, tags: list[str]):  # 注释：恢复权重或 KV cache
        """Resume rollout weights or kv cache in GPU memory.  # 注释：保留英文简述

        参数：  # 注释：参数说明标题
        - tags (list[str])：需要恢复的标签，如 ["weights", "kv_cache"]。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None（异步完成恢复）。  # 注释：返回值语义
        副作用：占用 GPU 显存，可能加载权重或 cache。  # 注释：副作用说明
        异常/边界条件：子类实现中可能因资源不足失败。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await rollout.resume(["weights"])。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/base.py::BaseRollout.resume。  # 注释：函数位置
        - 典型调用路径：ActorRolloutRefWorker.wake_up -> rollout.resume。  # 注释：典型调用链
        - 被谁调用：verl/workers/engine_workers.py。  # 注释：调用方说明
        - 调用了谁（项目内）：由子类实现决定。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：由子类实现决定。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        pass  # 注释：抽象方法占位
# （分隔说明：抽象方法）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法装饰器
    async def update_weights(  # 注释：更新 rollout 权重
        self,  # 注释：实例本身
        weights: Generator[tuple[str, torch.Tensor], None, None],  # 注释：权重迭代器
        **kwargs,  # 注释：扩展参数
    ):  # 注释：参数列表结束
        """Update the weights of the rollout model.  # 注释：保留英文简述

        参数：  # 注释：参数说明标题
        - weights：生成器，yield (name, tensor)。  # 注释：参数含义
        - kwargs：如 peft_config、base_sync_done。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None（异步完成更新）。  # 注释：返回值语义
        副作用：修改推理引擎权重。  # 注释：副作用说明
        异常/边界条件：权重不匹配或通信失败时可能异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await rollout.update_weights(weights_iter)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/base.py::BaseRollout.update_weights。  # 注释：函数位置
        - 典型调用路径：ActorRolloutRefWorker.wake_up -> rollout.update_weights。  # 注释：典型调用链
        - 被谁调用：verl/workers/engine_workers.py。  # 注释：调用方说明
        - 调用了谁（项目内）：由子类实现决定。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：由子类实现决定。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        pass  # 注释：抽象方法占位
# （分隔说明：抽象方法）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法装饰器
    async def release(self):  # 注释：释放 GPU 中的权重与 cache
        """Release weights and kv cache in GPU memory.  # 注释：保留英文简述

        参数：无。  # 注释：参数说明
        返回：None。  # 注释：返回值语义
        副作用：释放显存。  # 注释：副作用说明
        异常/边界条件：子类实现中可能依赖引擎状态。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await rollout.release()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/base.py::BaseRollout.release。  # 注释：函数位置
        - 典型调用路径：ActorRolloutRefWorker.sleep -> rollout.release。  # 注释：典型调用链
        - 被谁调用：verl/workers/engine_workers.py。  # 注释：调用方说明
        - 调用了谁（项目内）：由子类实现决定。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：由子类实现决定。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        pass  # 注释：抽象方法占位
# （分隔说明：同步推理接口）  # 注释：替代空行，保持逐行注释
    def generate_sequences(self, prompts: DataProto) -> DataProto:  # 注释：同步批量生成接口
        """Batch generate sequences in sync mode.  # 注释：保留英文简述

        参数：  # 注释：参数说明标题
        - prompts (DataProto)：包含输入 prompt 的数据结构。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - DataProto：包含生成结果的结构。  # 注释：返回值语义
        副作用：无（默认实现直接抛异常）。  # 注释：副作用说明
        异常/边界条件：未实现时抛 NotImplementedError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - outputs = rollout.generate_sequences(prompts)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/base.py::BaseRollout.generate_sequences。  # 注释：函数位置
        - 典型调用路径：某些同步推理工具 -> rollout.generate_sequences。  # 注释：典型调用链
        - 被谁调用：当前仓库默认异步模式，可能无直接调用。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        raise NotImplementedError  # 注释：基类未实现
# （分隔说明：rollout 注册表）  # 注释：替代空行，保持逐行注释
_ROLLOUT_REGISTRY = {  # 注释：rollout 名称与类路径映射
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout",  # 注释：vLLM 异步模式
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",  # 注释：SGLang 异步模式
}  # 注释：映射结束
# （分隔说明：工厂方法）  # 注释：替代空行，保持逐行注释
def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:  # 注释：根据名称获取 rollout 类
    """Get the rollout class by name.  # 注释：保留英文简述

    参数：  # 注释：参数说明标题
    - rollout_name (str)：rollout 名称（如 vllm/sglang）。  # 注释：参数含义
    - mode (str)：rollout 模式（默认 async）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - type[BaseRollout]：rollout 类对象。  # 注释：返回值语义
    副作用：动态 import 对应模块。  # 注释：副作用说明
    异常/边界条件：未注册的 (rollout, mode) 将触发断言。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - cls = get_rollout_class("vllm"); rollout = cls(...).  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/base.py::get_rollout_class。  # 注释：函数位置
    - 典型调用路径：ActorRolloutRefWorker.init_model -> get_rollout_class。  # 注释：典型调用链
    - 被谁调用：verl/workers/engine_workers.py。  # 注释：调用方说明
    - 调用了谁（项目内）：importlib.import_module。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"  # 注释：校验映射存在
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]  # 注释：获取全限定类名
    module_name, class_name = fqdn.rsplit(".", 1)  # 注释：拆分模块路径与类名
    rollout_module = importlib.import_module(module_name)  # 注释：动态导入模块
    return getattr(rollout_module, class_name)  # 注释：返回类对象
