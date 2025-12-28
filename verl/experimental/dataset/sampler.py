# Copyright 2025 Amazon.com Inc and/or its affiliates  # 注释：版权声明
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：许可证声明
# you may not use this file except in compliance with the License.  # 注释：需遵守许可证
# You may obtain a copy of the License at  # 注释：提示可获取许可证
#  # 注释：保留注释符号，保证此行也有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：保留注释符号，保证此行也有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开始
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制
"""
模块用途：定义 RL 数据集采样器的抽象接口（含课程学习扩展）。  # 注释：模块用途说明
输入：  # 注释：模块输入说明标题
- data_source（实现 Sized 接口的数据集）。  # 注释：输入含义
- data_config（OmegaConf 配置，控制采样策略）。  # 注释：输入含义
输出：  # 注释：模块输出说明标题
- 抽象 Sampler/课程采样接口，供具体实现继承。  # 注释：输出说明
依赖：torch.utils.data.Sampler、omegaconf.DictConfig。  # 注释：关键依赖说明
典型用法：  # 注释：最小示例标题
- class MySampler(AbstractSampler): ...  # 注释：继承示例
调用路径概览：  # 注释：调用路径概览标题
- 入口：verl/trainer/main_ppo.py 的 create_rl_sampler。  # 注释：典型入口
- 典型链路：main_ppo.py -> create_rl_sampler -> 具体 Sampler 实现。  # 注释：调用链说明
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
from abc import abstractmethod  # 注释：定义抽象方法装饰器
from collections.abc import Sized  # 注释：Sized 接口，用于要求数据集可计算长度
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
from omegaconf import DictConfig  # 注释：OmegaConf 配置类型
from torch.utils.data import Sampler  # 注释：PyTorch Sampler 基类
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl import DataProto  # 注释：数据协议类型，用于 curriculum sampler 的 update
# （分隔说明：抽象采样器定义）  # 注释：替代空行，保持逐行注释
class AbstractSampler(Sampler[int]):  # 注释：定义抽象采样器基类
    """
    类用途：提供自定义采样器统一接口（构造参数统一）。  # 注释：类用途说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/dataset/sampler.py::AbstractSampler。  # 注释：类位置
    - 典型调用路径：main_ppo.py -> create_rl_sampler -> Sampler 实例。  # 注释：典型调用链
    - 被谁调用：verl/trainer/main_ppo.py（通过配置动态导入）。  # 注释：调用方说明
    - 调用了谁（项目内）：无（抽象接口）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.utils.data.Sampler。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    @abstractmethod  # 注释：声明抽象构造方法
    def __init__(  # 注释：抽象初始化方法签名
        self,  # 注释：实例本身
        data_source: Sized,  # 注释：数据源，需支持 __len__
        data_config: DictConfig,  # 注释：采样配置
    ):  # 注释：参数列表结束
        """
        函数用途：定义采样器必须实现的初始化接口。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - data_source (Sized)：可计算长度的数据集对象。  # 注释：参数含义
        - data_config (DictConfig)：采样相关配置。  # 注释：参数含义
        返回：无（构造方法）。  # 注释：返回值说明
        副作用：由子类决定。  # 注释：副作用说明
        异常/边界条件：由子类决定。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：MySampler(dataset, cfg.data)。  # 注释：示例输入
        - 输出：MySampler 实例。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/dataset/sampler.py::AbstractSampler.__init__。  # 注释：函数位置
        - 典型调用路径：main_ppo.py -> create_rl_sampler -> Sampler.__init__。  # 注释：典型调用链
        - 被谁调用：子类构造函数调用 super().__init__。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        pass  # 注释：抽象方法占位
# （分隔说明：课程学习采样器定义）  # 注释：替代空行，保持逐行注释
class AbstractCurriculumSampler(AbstractSampler):  # 注释：定义课程学习采样器抽象类
    """
    类用途：扩展采样器接口，支持基于训练反馈的采样更新。  # 注释：类用途说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/dataset/sampler.py::AbstractCurriculumSampler。  # 注释：类位置
    - 典型调用路径：trainer -> sampler.update(batch)。  # 注释：典型调用链
    - 被谁调用：课程学习采样器的具体实现与训练循环。  # 注释：调用方说明
    - 调用了谁（项目内）：无（抽象接口）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    @abstractmethod  # 注释：声明抽象更新方法
    def update(self, batch: DataProto) -> None:  # 注释：根据 batch 更新采样策略
        """
        函数用途：接收训练 batch 反馈，更新采样器内部状态。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - batch (DataProto)：包含当前 batch 的数据与元信息。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：更新采样权重或难度分布（由子类实现）。  # 注释：副作用说明
        异常/边界条件：由子类实现决定。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：sampler.update(batch) 更新难样本权重。  # 注释：示例说明
        - 输出：采样器内部状态变化。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/dataset/sampler.py::AbstractCurriculumSampler.update。  # 注释：函数位置
        - 典型调用路径：trainer -> sampler.update(batch)。  # 注释：典型调用链
        - 被谁调用：课程学习训练循环或自定义 trainer。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        pass  # 注释：抽象方法占位
