# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行说明
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：提示许可证链接
#  # 注释：保留注释符号，保证该行有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证地址
#  # 注释：保留注释符号，保证该行有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
Adapted from Cruise.  # 注释：保留英文说明
模块用途：提供精度类型（fp16/fp32/bf16）判断与转换工具。  # 注释：模块用途
输入/输出：输入精度标记（字符串/数字/torch dtype），输出 dtype 或字符串。  # 注释：模块输入输出概览
关键依赖：torch。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- PrecisionType.is_fp16("fp16") -> True  # 注释：判断精度类型
- PrecisionType.to_dtype("bf16") -> torch.bfloat16  # 注释：转换为 dtype
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：模型/训练配置解析时判断精度。  # 注释：上层入口举例
- 典型链路：配置 -> PrecisionType.* -> torch dtype。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import torch  # 注释：提供 dtype 常量
# （分隔说明：常量列表）  # 注释：替代空行，保持逐行注释
HALF_LIST = [16, "16", "fp16", "float16", torch.float16]  # 注释：fp16 等价写法集合
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]  # 注释：fp32 等价写法集合
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]  # 注释：bf16 等价写法集合
# （分隔说明：精度类型工具类）  # 注释：替代空行，保持逐行注释
class PrecisionType:  # 注释：精度类型枚举与工具方法集合
    """Type of precision used.  # 注释：保留英文说明

    说明：提供字符串常量与多种判断/转换方法。  # 注释：类用途
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/torch_dtypes.py::PrecisionType。  # 注释：类位置
    - 典型调用路径：配置解析 -> PrecisionType.is_* / to_dtype。  # 注释：典型调用链
    - 被谁调用：项目内各类训练/worker 配置解析逻辑。  # 注释：调用方概述
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch dtype 常量。  # 注释：外部依赖说明

    >>> PrecisionType.HALF == 16  # 注释：示例：比较字符串与数值
    True  # 注释：示例输出
    >>> PrecisionType.HALF in (16, "16")  # 注释：示例：集合判断
    True  # 注释：示例输出
    """  # 注释：类 docstring 结束

    HALF = "16"  # 注释：半精度标记
    FLOAT = "32"  # 注释：单精度标记
    FULL = "64"  # 注释：双精度标记
    BFLOAT = "bf16"  # 注释：bf16 标记
    MIXED = "mixed"  # 注释：混合精度标记

    @staticmethod  # 注释：静态方法装饰器
    def supported_type(precision: str | int) -> bool:  # 注释：判断精度是否在已知类型中
        """
        功能：判断传入精度标记是否为支持的类型。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - precision (str|int)：精度标记。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - bool：是否支持。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.supported_type("fp16") -> True/False。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.supported_type。  # 注释：函数位置
        - 典型调用路径：配置解析 -> supported_type。  # 注释：典型调用链
        - 被谁调用：训练/worker 配置校验逻辑。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return any(x == precision for x in PrecisionType)  # 注释：遍历类属性判断

    @staticmethod  # 注释：静态方法装饰器
    def supported_types() -> list[str]:  # 注释：返回所有支持的精度字符串
        """
        功能：返回所有支持的精度字符串列表。  # 注释：函数用途
        参数：无。  # 注释：参数说明标题
        返回：  # 注释：返回值说明标题
        - list[str]：可用精度列表。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.supported_types() -> ["16", "32", "64", "bf16", "mixed"]。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.supported_types。  # 注释：函数位置
        - 典型调用路径：配置校验/日志输出。  # 注释：典型调用链
        - 被谁调用：项目内精度参数检查。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return [x.value for x in PrecisionType]  # 注释：遍历枚举值返回列表

    @staticmethod  # 注释：静态方法装饰器
    def is_fp16(precision):  # 注释：判断是否为 fp16
        """
        功能：判断输入是否属于 fp16 类型集合。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - precision：精度标记（数字/字符串/torch dtype）。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - bool：是否为 fp16。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.is_fp16(torch.float16) -> True。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.is_fp16。  # 注释：函数位置
        - 典型调用路径：配置解析 -> is_fp16。  # 注释：典型调用链
        - 被谁调用：精度分支逻辑。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return precision in HALF_LIST  # 注释：是否在 fp16 列表中

    @staticmethod  # 注释：静态方法装饰器
    def is_fp32(precision):  # 注释：判断是否为 fp32
        """
        功能：判断输入是否属于 fp32 类型集合。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - precision：精度标记（数字/字符串/torch dtype）。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - bool：是否为 fp32。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.is_fp32("fp32") -> True。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.is_fp32。  # 注释：函数位置
        - 典型调用路径：配置解析 -> is_fp32。  # 注释：典型调用链
        - 被谁调用：精度分支逻辑。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return precision in FLOAT_LIST  # 注释：是否在 fp32 列表中

    @staticmethod  # 注释：静态方法装饰器
    def is_bf16(precision):  # 注释：判断是否为 bf16
        """
        功能：判断输入是否属于 bf16 类型集合。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - precision：精度标记（数字/字符串/torch dtype）。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - bool：是否为 bf16。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.is_bf16("bf16") -> True。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.is_bf16。  # 注释：函数位置
        - 典型调用路径：配置解析 -> is_bf16。  # 注释：典型调用链
        - 被谁调用：精度分支逻辑。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return precision in BFLOAT_LIST  # 注释：是否在 bf16 列表中

    @staticmethod  # 注释：静态方法装饰器
    def to_dtype(precision):  # 注释：精度标记转 torch dtype
        """
        功能：将精度标记转换为 torch.dtype。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - precision：精度标记（数字/字符串/torch dtype）。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - torch.dtype：对应 dtype。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 非法精度会抛 RuntimeError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.to_dtype("fp16") -> torch.float16。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.to_dtype。  # 注释：函数位置
        - 典型调用路径：配置解析 -> to_dtype。  # 注释：典型调用链
        - 被谁调用：模型/优化器 dtype 设置。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.float16/float32/bfloat16。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if precision in HALF_LIST:  # 注释：匹配 fp16
            return torch.float16  # 注释：返回 float16
        elif precision in FLOAT_LIST:  # 注释：匹配 fp32
            return torch.float32  # 注释：返回 float32
        elif precision in BFLOAT_LIST:  # 注释：匹配 bf16
            return torch.bfloat16  # 注释：返回 bfloat16
        else:  # 注释：未知精度
            raise RuntimeError(f"unexpected precision: {precision}")  # 注释：抛出异常

    @staticmethod  # 注释：静态方法装饰器
    def to_str(precision):  # 注释：torch dtype 转字符串
        """
        功能：将 torch.dtype 转换为字符串标记。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - precision：torch dtype。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - str：精度字符串（fp16/fp32/bf16）。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 非法 dtype 会抛 RuntimeError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - PrecisionType.to_str(torch.bfloat16) -> "bf16"。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/torch_dtypes.py::PrecisionType.to_str。  # 注释：函数位置
        - 典型调用路径：日志/配置导出 -> to_str。  # 注释：典型调用链
        - 被谁调用：配置/日志工具。  # 注释：调用方概述
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch dtype 比较。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if precision == torch.float16:  # 注释：dtype 为 float16
            return "fp16"  # 注释：返回 fp16
        elif precision == torch.float32:  # 注释：dtype 为 float32
            return "fp32"  # 注释：返回 fp32
        elif precision == torch.bfloat16:  # 注释：dtype 为 bfloat16
            return "bf16"  # 注释：返回 bf16
        else:  # 注释：未知 dtype
            raise RuntimeError(f"unexpected precision: {precision}")  # 注释：抛出异常
