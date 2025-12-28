# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
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
模块用途：verl.utils.metric 包入口，导出指标聚合函数 reduce_metrics。  # 注释：模块用途
输入/输出：输入为指标字典，输出为聚合后的标量指标字典。  # 注释：模块输入输出概览
关键依赖：verl.utils.metric.utils、numpy（间接依赖）。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- from verl.utils.metric import reduce_metrics  # 注释：导入聚合函数
- reduce_metrics({"loss": [1.0, 2.0]}) -> {"loss": 1.5}  # 注释：最小示例
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/trainer/ppo/ray_trainer.py 收集 worker 指标。  # 注释：上层入口举例
- 典型链路：ray_trainer -> reduce_metrics -> numpy 聚合。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入实现）  # 注释：替代空行，保持逐行注释
from .utils import reduce_metrics  # 注释：导入核心聚合函数
# （分隔说明：导出公共接口）  # 注释：替代空行，保持逐行注释
__all__ = ["reduce_metrics"]  # 注释：限制包导出范围，避免泄露内部实现
