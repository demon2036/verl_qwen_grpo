# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
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
模块用途：聚合 vLLM 相关工具的对外导出，提供版本判断与 hijack 能力。  # 注释：模块用途
输入/输出：无显式输入；导出工具类/函数供其他模块 import。  # 注释：输入输出说明
关键依赖：verl.utils.vllm.utils（TensorLoRARequest/VLLMHijack/is_version_ge）。  # 注释：关键依赖
典型用法：  # 注释：最小用法示例标题
- from verl.utils.vllm import VLLMHijack, is_version_ge  # 注释：导入示例
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/rollout/vllm_rollout/vllm_rollout.py。  # 注释：上层入口举例
- 典型链路：rollout 初始化 -> import 本模块 -> 使用 hijack/版本判断。  # 注释：调用链路
"""  # 注释：模块 docstring 结束
# （分隔说明：导入导出符号）  # 注释：替代空行，保持逐行注释
from .utils import TensorLoRARequest, VLLMHijack, is_version_ge  # 注释：导出工具符号
# （分隔说明：延迟导入说明）  # 注释：替代空行，保持逐行注释
# The contents of vllm/patch.py should not be imported here, because the contents of  # 注释：原注释保留（说明延迟导入原因）
# patch.py should be imported after the vllm LLM instance is created. Therefore,  # 注释：原注释保留（说明时序）
# wait until you actually start using it before importing the contents of  # 注释：原注释保留（使用时再导入）
# patch.py separately.  # 注释：原注释保留（单独导入）
# （分隔说明：公开导出列表）  # 注释：替代空行，保持逐行注释
__all__ = [  # 注释：模块对外导出
    "TensorLoRARequest",  # 注释：张量化 LoRA 请求结构
    "VLLMHijack",  # 注释：vLLM 行为注入工具
    "is_version_ge",  # 注释：版本比较工具
]  # 注释：导出列表结束
