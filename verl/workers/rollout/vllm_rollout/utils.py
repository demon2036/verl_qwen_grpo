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
模块用途：提供 vLLM rollout 中 LoRA 相关的常量与工具函数。  # 注释：模块用途
输入/输出：输入 LoRA rank 等参数，输出合规的 max_lora_rank。  # 注释：输入输出概览
关键依赖：无第三方强依赖（纯 Python 常量与函数）。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- max_rank = get_vllm_max_lora_rank(lora_rank=12)  # 注释：示例用法
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/rollout/vllm_rollout/vllm_rollout.py。  # 注释：上层入口举例
- 典型链路：rollout 初始化 -> 读取 LoRA 配置 -> get_vllm_max_lora_rank。  # 注释：调用链路
"""  # 注释：模块 docstring 结束
# （分隔说明：LoRA 常量）  # 注释：替代空行，保持逐行注释
# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process  # 注释：原注释保留（说明常量用途）
VLLM_LORA_INT_ID = 123  # 注释：LoRA 适配器整数 ID
VLLM_LORA_NAME = "123"  # 注释：LoRA 适配器名称
VLLM_LORA_PATH = "simon_lora_path"  # 注释：LoRA 适配器路径占位
# （分隔说明：工具函数）  # 注释：替代空行，保持逐行注释
def get_vllm_max_lora_rank(lora_rank: int):  # 注释：将 LoRA rank 调整到 vLLM 允许范围
    """
    For vLLM, the smallest `max_lora_rank` is 8, and allowed values are (8, 16, 32, 64, 128, 256, 320, 512)
    This function automatically adjusts the `max_lora_rank` to the nearest allowed value.

    Reference: https://github.com/vllm-project/vllm/blob/8a297115e2367d463b781adb86b55ac740594cf6/vllm/config/lora.py#L27

    功能：将任意 LoRA rank 向上取整到 vLLM 支持的档位。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - lora_rank (int)：期望的 LoRA rank（>0）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - int：vLLM 允许的最小上界 rank。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：lora_rank<=0 触发断言；超过最大档位抛 ValueError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入 12 -> 输出 16。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/vllm_rollout/utils.py::get_vllm_max_lora_rank。  # 注释：函数位置
    - 典型调用路径：vLLMAsyncRollout.__init__ -> get_vllm_max_lora_rank。  # 注释：典型调用链
    - 被谁调用：verl/workers/rollout/vllm_rollout/vllm_rollout.py。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    assert lora_rank > 0, f"lora_rank must be greater than 0 to invoke this function, get {lora_rank}"  # 注释：基本合法性检查
    vllm_max_lora_ranks = [8, 16, 32, 64, 128, 256, 320, 512]  # 注释：vLLM 支持的 rank 档位
    for rank in vllm_max_lora_ranks:  # 注释：遍历档位
        if lora_rank <= rank:  # 注释：找到第一个 >= 的档位
            return rank  # 注释：返回最近上界
# （分隔说明：异常分支）  # 注释：替代空行，保持逐行注释
    raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")  # 注释：超出最大档位时抛错
