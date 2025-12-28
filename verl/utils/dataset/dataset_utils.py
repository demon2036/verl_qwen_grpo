# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明，说明该文件的归属与年份
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行可读
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明使用 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用文件需遵守许可证条款
# You may obtain a copy of the License at  # 注释：提示可在下方链接获取许可证全文
#  # 注释：保留注释符号，保证此行也有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证链接
#  # 注释：保留注释符号，保证此行也有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明的起始说明
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按“原样”提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供任何担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款信息
# limitations under the License.  # 注释：许可限制说明
#  # 注释：分隔行，提示许可证段结束
"""
模块用途：提供数据集 padding 模式枚举与 SFT/RL 通用的批处理 collate 工具。  # 注释：模块功能概述
输入：  # 注释：模块级输入说明标题
- DatasetPadMode 枚举值（pad_mode）。  # 注释：模块输入类型说明
- DataLoader 的 batch 样本列表（list[dict]）。  # 注释：批处理输入说明
输出：  # 注释：模块级输出说明标题
- 统一的 batch 字典，包含张量与非张量字段。  # 注释：输出结构说明
依赖：torch、tensordict.tensorclass.NonTensorData（封装非张量字段）。  # 注释：关键外部依赖说明
典型用法：  # 注释：最小示例标题
- collate_fn = SFTTensorCollator(DatasetPadMode.NO_PADDING)；DataLoader(..., collate_fn=collate_fn)。  # 注释：示例使用方式
调用路径概览：  # 注释：调用路径概览标题
- 入口：训练器（如 verl/trainer/sft_trainer.py）创建 DataLoader 时设置 collate_fn。  # 注释：典型入口说明
- 典型链路：sft_trainer.py -> DataLoader -> SFTTensorCollator.__call__ -> collate_variable_batch。  # 注释：调用链说明
"""  # 注释：模块 docstring 结束
# （分隔说明：模块说明结束，下面导入依赖）  # 注释：替代空行，保持逐行注释
from enum import Enum  # 注释：导入枚举基类，用于定义 padding 模式
# （分隔说明：第三方依赖导入）  # 注释：分隔标准库与第三方库
import torch  # 注释：张量与 NestedTensor 操作
from tensordict.tensorclass import NonTensorData  # 注释：封装非张量数据，便于 stack
# （分隔说明：依赖导入结束）  # 注释：替代空行，保持逐行注释
class DatasetPadMode(str, Enum):  # 注释：定义数据集 padding 模式枚举
    """
    类用途：枚举数据集 batch 的 padding 策略。  # 注释：类用途说明
    成员：  # 注释：成员说明标题
    - RIGHT：右侧 padding（常见的左对齐）。  # 注释：枚举值说明
    - LEFT_RIGHT：左右混合 padding（多轮对话场景）。  # 注释：枚举值说明
    - NO_PADDING：不 padding，保留变长序列（NestedTensor）。  # 注释：枚举值说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/dataset_utils.py::DatasetPadMode。  # 注释：类定义位置
    - 典型调用路径：sft_trainer.py -> SFTTensorCollator(pad_mode)。  # 注释：典型调用链
    - 被谁调用：verl/trainer/sft_trainer.py、verl/utils/dataset/multiturn_sft_dataset.py 等。  # 注释：外部引用说明
    - 调用了谁（项目内）：无（枚举定义）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：enum.Enum。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    RIGHT = "right"  # 注释：右侧 padding 模式
    LEFT_RIGHT = "left_right"  # 注释：左右混合 padding 模式
    NO_PADDING = "no_padding"  # 注释：不做 padding（变长）模式
# （分隔说明：枚举定义结束）  # 注释：替代空行，保持逐行注释
class SFTTensorCollator:  # 注释：定义 SFT/RL 批处理 collate 工具类
    """
    类用途：根据 pad_mode 对样本批次进行整理与堆叠。  # 注释：类用途说明
    处理逻辑：  # 注释：处理逻辑说明标题
    - NO_PADDING：将变长序列转为 NestedTensor；非张量字段用 NonTensorData 包装。  # 注释：变长场景说明
    - RIGHT/LEFT_RIGHT：交给 default_collate 进行常规堆叠。  # 注释：常规堆叠说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/dataset_utils.py::SFTTensorCollator。  # 注释：类位置
    - 典型调用路径：sft_trainer.py -> DataLoader(collate_fn) -> SFTTensorCollator.__call__。  # 注释：典型调用链
    - 被谁调用：verl/trainer/sft_trainer.py、tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py。  # 注释：调用方说明
    - 调用了谁（项目内）：collate_variable_batch（本类方法）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.utils.data.default_collate、torch.nested。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    def __init__(self, pad_mode: DatasetPadMode = DatasetPadMode.LEFT_RIGHT):  # 注释：初始化 collator，并设置 padding 模式
        """
        函数用途：创建批处理整理器，并记录 padding 策略。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - pad_mode (DatasetPadMode)：padding 模式，默认 LEFT_RIGHT。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：无（仅保存配置）。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：SFTTensorCollator(DatasetPadMode.NO_PADDING)。  # 注释：示例输入
        - 输出：collator 实例，可传给 DataLoader。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/dataset_utils.py::SFTTensorCollator.__init__。  # 注释：函数位置
        - 典型调用路径：sft_trainer.py -> SFTTensorCollator(...)。  # 注释：典型调用链
        - 被谁调用：同上（训练器/测试代码构造）。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        self.pad_mode = pad_mode  # 注释：保存 padding 模式到实例
    # （分隔说明：初始化结束）  # 注释：替代空行，保持逐行注释
    def __call__(self, batch: list[dict[str, any]]) -> dict[str, any]:  # 注释：使实例可作为 collate_fn 直接调用
        """
        函数用途：按 pad_mode 选择合适的批处理策略。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - batch (list[dict[str, Any]])：DataLoader 提供的样本列表。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - dict[str, Any]：整理后的批数据。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 若 pad_mode 不在已支持的枚举中，抛出 NotImplementedError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：batch=[{"input_ids": tensor([1,2])}, {"input_ids": tensor([3])}]。  # 注释：示例输入
        - 输出：NO_PADDING 模式下返回 NestedTensor。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/dataset_utils.py::SFTTensorCollator.__call__。  # 注释：函数位置
        - 典型调用路径：DataLoader -> collate_fn -> __call__。  # 注释：典型调用链
        - 被谁调用：torch.utils.data.DataLoader（运行期回调）。  # 注释：调用方说明
        - 调用了谁（项目内）：collate_variable_batch（本类方法）。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.utils.data.default_collate。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if self.pad_mode == DatasetPadMode.NO_PADDING:  # 注释：不做 padding，进入变长批处理
            return self.collate_variable_batch(batch)  # 注释：调用变长序列批处理
        elif self.pad_mode in [DatasetPadMode.RIGHT, DatasetPadMode.LEFT_RIGHT]:  # 注释：右侧或左右 padding
            from torch.utils.data import default_collate  # 注释：按需导入默认 collate
            # （分隔说明：调用默认 collate）  # 注释：替代空行，保持逐行注释
            return default_collate(batch)  # 注释：使用默认堆叠行为
        else:  # 注释：未知 padding 模式
            raise NotImplementedError(f"pad_mode {self.pad_mode} not implemented")  # 注释：明确抛出未实现错误
    # （分隔说明：__call__ 结束）  # 注释：替代空行，保持逐行注释
    def collate_variable_batch(self, batch: list[dict[str, any]]) -> dict[str, any]:  # 注释：处理变长序列的批处理函数
        """
        函数用途：将变长序列样本整理成可训练的 NestedTensor/非张量堆叠结构。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - batch (list[dict[str, Any]])：数据集输出的样本列表。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - dict[str, Any]：包含 NestedTensor/堆叠 NonTensorData 的批次字典。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 若 batch 为空或键不一致，可能导致 KeyError/IndexError。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：batch=[{"input_ids": tensor([1,2])}, {"input_ids": tensor([3])}]。  # 注释：示例输入
        - 输出：{"input_ids": NestedTensor([[1,2],[3]])}。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/dataset_utils.py::collate_variable_batch。  # 注释：函数位置
        - 典型调用路径：SFTTensorCollator.__call__ -> collate_variable_batch。  # 注释：典型调用链
        - 被谁调用：SFTTensorCollator.__call__。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.nested.as_nested_tensor、torch.stack、NonTensorData。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        final_batch = {}  # 注释：初始化最终批次字典
        # （分隔说明：收集批次中所有键）  # 注释：替代空行，保持逐行注释
        tensor_keys = set().union(*(d.keys() for d in batch))  # 注释：合并所有样本的键集合
        # （分隔说明：根据键类型分别处理）  # 注释：替代空行，保持逐行注释
        # Handle tensor values by creating a NestedTensor.  # 注释：说明张量字段将转为 NestedTensor
        for key in tensor_keys:  # 注释：遍历每个字段
            if isinstance(batch[0][key], torch.Tensor):  # 注释：判断该字段是否为张量
                tensors = [item[key] for item in batch]  # 注释：收集该字段的所有样本张量
                final_batch[key] = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)  # 注释：转为 NestedTensor
            else:  # 注释：非张量字段
                tensors = [NonTensorData(item.get(key)) for item in batch]  # 注释：包装为 NonTensorData 列表
                final_batch[key] = torch.stack(tensors, dim=0)  # 注释：沿 batch 维度堆叠非张量数据
        # （分隔说明：批次整理完成）  # 注释：替代空行，保持逐行注释
        return final_batch  # 注释：返回整理后的批次字典
