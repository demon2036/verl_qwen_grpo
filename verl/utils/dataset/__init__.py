# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明，标记文件归属
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行可读
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明使用 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：说明使用需遵守许可证
# You may obtain a copy of the License at  # 注释：提示可通过链接获取许可证
#  # 注释：保留注释符号，保证此行也有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：保留注释符号，保证此行也有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
#  # 注释：分隔行，许可证段结束
"""
模块用途：集中导出数据集类，提供统一的 import 入口。  # 注释：模块用途说明
输入：无（纯导出模块）。  # 注释：模块输入说明
输出：RLHFDataset、RMDataset、SFTDataset 三个类的公共导出。  # 注释：模块输出说明
依赖：verl.utils.dataset.rl_dataset / rm_dataset / sft_dataset。  # 注释：关键依赖说明
典型用法：from verl.utils.dataset import RLHFDataset。  # 注释：最小示例
调用路径概览：  # 注释：调用路径概览标题
- 入口：训练器或数据加载模块导入本包。  # 注释：典型入口
- 典型链路：trainer -> from verl.utils.dataset import RLHFDataset -> rl_dataset.RLHFDataset。  # 注释：典型调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：开始导入需要公开的类）  # 注释：替代空行，保持逐行注释
from .rl_dataset import RLHFDataset  # 注释：导出 RLHF 数据集类
from .rm_dataset import RMDataset  # 注释：导出 Reward Model 数据集类
from .sft_dataset import SFTDataset  # 注释：导出 SFT 数据集类
# （分隔说明：定义模块对外导出列表）  # 注释：替代空行，保持逐行注释
__all__ = ["RLHFDataset", "RMDataset", "SFTDataset"]  # 注释：声明对外可用符号
