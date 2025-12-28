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
模块用途：调试工具包的向后兼容入口，重定向旧的 debug API 到 profiler。  # 注释：模块用途
输入/输出：输入为 import 行为，输出为 profiler 中的公共 API。  # 注释：模块输入输出概览
关键依赖：verl.utils.profiler（实际实现）。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- from verl.utils.debug import PerformanceProfiler  # 注释：旧路径仍可用
- from verl.utils.profiler import PerformanceProfiler  # 注释：推荐新路径
调用路径概览：  # 注释：调用路径说明标题
- 用户代码 -> verl.utils.debug -> 本文件 -> verl.utils.profiler。  # 注释：兼容重定向链路
"""  # 注释：模块 docstring 结束
# （分隔说明：向后兼容导入）  # 注释：替代空行，保持逐行注释
# APIs kept for backward compatibility purpose  # 注释：保留旧 API 以兼容历史代码
# For new features please develop in verl/utils/profiler/  # 注释：新功能请放在 profiler 目录
from ..profiler import *  # noqa: F401  # 注释：导出 profiler 的公共 API，忽略未使用告警
