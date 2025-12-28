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
模块用途：统一 CUDA/NPU 的 flash-attn padding/unpadding 接口。  # 注释：模块用途
输入/输出：输入张量与索引等参数，输出 pad/unpad 后张量或索引。  # 注释：模块输入输出概览
关键依赖：flash_attn.bert_padding、NPU 兼容实现（verl.utils.npu_flash_attn_utils）。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- from verl.utils.attention_utils import unpad_input, pad_input  # 注释：导入统一接口
- x_rmpad, indices, *_ = unpad_input(x.unsqueeze(-1), attention_mask=mask)  # 注释：去除 padding
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/trainer/fsdp_sft_trainer.py、verl/workers/actor/dp_actor.py。  # 注释：上层入口举例
- 典型链路：训练/推理模块 -> attention_utils.* -> CUDA/NPU 实现。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
from typing import Callable  # 注释：类型注解
# （分隔说明：缓存全局函数引用）  # 注释：替代空行，保持逐行注释
_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None  # 注释：缓存设备相关实现
# （分隔说明：内部函数：加载具体实现）  # 注释：替代空行，保持逐行注释
def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:  # 注释：动态加载函数
    """Dynamically import attention functions based on available hardware.  # 注释：保留英文说明

    功能：根据设备类型（NPU/CUDA）选择对应的 pad/unpad/rearrange 实现。  # 注释：函数用途
    参数：无。  # 注释：参数说明标题
    返回：  # 注释：返回值说明标题
    - tuple[Callable, Callable, Callable, Callable]：四个函数引用。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 会更新模块级缓存变量 _index_first_axis/_pad_input/...。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若 flash-attn 未安装且非 NPU 可能 ImportError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - index_first_axis, pad_input, rearrange, unpad_input = _get_attention_functions()。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/attention_utils.py::_get_attention_functions。  # 注释：函数位置
    - 典型调用路径：index_first_axis/pad_input/... -> _get_attention_functions。  # 注释：典型调用链
    - 被谁调用：本文件内的四个对外接口函数。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.device.is_npu_available、verl.utils.npu_flash_attn_utils。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：flash_attn.bert_padding。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束

    from verl.utils.device import is_npu_available  # 注释：检查是否可用 NPU

    global _index_first_axis, _pad_input, _rearrange, _unpad_input  # 注释：声明更新模块级缓存

    if is_npu_available:  # 注释：NPU 环境
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input  # 注释：导入 NPU 版本
    else:  # 注释：CUDA/CPU 环境
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # 注释：导入 flash-attn 版本

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input  # 注释：更新缓存

    return _index_first_axis, _pad_input, _rearrange, _unpad_input  # 注释：返回函数引用
# （分隔说明：统一接口：index_first_axis）  # 注释：替代空行，保持逐行注释
def index_first_axis(*args, **kwargs):  # 注释：统一的 index_first_axis 接口
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.  # 注释：保留英文说明

    功能：隐藏设备差异，选择合适的 index_first_axis 实现。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - *args/**kwargs：透传给底层实现的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 与底层实现一致的张量输出。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 底层实现 import 失败会抛 ImportError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - y = index_first_axis(x, indices)  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/attention_utils.py::index_first_axis。  # 注释：函数位置
    - 典型调用路径：fsdp_sft_trainer -> index_first_axis -> _get_attention_functions。  # 注释：典型调用链
    - 被谁调用：verl/trainer/fsdp_sft_trainer.py、verl/workers/actor/dp_actor.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：_get_attention_functions。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：flash_attn.bert_padding.index_first_axis 或 NPU 版本。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    func, *_ = _get_attention_functions()  # 注释：获取 index_first_axis 实现
    return func(*args, **kwargs)  # 注释：执行并返回
# （分隔说明：统一接口：pad_input）  # 注释：替代空行，保持逐行注释
def pad_input(*args, **kwargs):  # 注释：统一的 pad_input 接口
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.  # 注释：保留英文说明

    功能：隐藏设备差异，选择合适的 pad_input 实现。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - *args/**kwargs：透传给底层实现的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 与底层实现一致的张量输出。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 底层实现 import 失败会抛 ImportError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - padded = pad_input(x, indices, batch, seqlen)  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/attention_utils.py::pad_input。  # 注释：函数位置
    - 典型调用路径：dp_actor/critic -> pad_input -> _get_attention_functions。  # 注释：典型调用链
    - 被谁调用：verl/workers/actor/dp_actor.py、verl/workers/critic/dp_critic.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：_get_attention_functions。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：flash_attn.bert_padding.pad_input 或 NPU 版本。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    _, func, *_ = _get_attention_functions()  # 注释：获取 pad_input 实现
    return func(*args, **kwargs)  # 注释：执行并返回
# （分隔说明：统一接口：rearrange）  # 注释：替代空行，保持逐行注释
def rearrange(*args, **kwargs):  # 注释：统一的 rearrange 接口
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.  # 注释：保留英文说明

    功能：隐藏设备差异，选择合适的 rearrange 实现。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - *args/**kwargs：透传给底层实现的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 与底层实现一致的张量输出。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 底层实现 import 失败会抛 ImportError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - y = rearrange(x, "b s h d -> (b s) h d")  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/attention_utils.py::rearrange。  # 注释：函数位置
    - 典型调用路径：fsdp_workers -> rearrange -> _get_attention_functions。  # 注释：典型调用链
    - 被谁调用：verl/trainer/fsdp_sft_trainer.py、verl/workers/fsdp_workers.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：_get_attention_functions。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：flash_attn.bert_padding.rearrange 或 NPU 版本。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    *_, func, _ = _get_attention_functions()  # 注释：获取 rearrange 实现
    return func(*args, **kwargs)  # 注释：执行并返回
# （分隔说明：统一接口：unpad_input）  # 注释：替代空行，保持逐行注释
def unpad_input(*args, **kwargs):  # 注释：统一的 unpad_input 接口
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.  # 注释：保留英文说明

    功能：隐藏设备差异，选择合适的 unpad_input 实现。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - *args/**kwargs：透传给底层实现的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 与底层实现一致的张量输出（去 padding 后的张量、索引等）。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 底层实现 import 失败会抛 ImportError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - x_rmpad, indices, *_ = unpad_input(x.unsqueeze(-1), attention_mask=mask)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/attention_utils.py::unpad_input。  # 注释：函数位置
    - 典型调用路径：dp_actor/critic -> unpad_input -> _get_attention_functions。  # 注释：典型调用链
    - 被谁调用：verl/workers/actor/dp_actor.py、verl/workers/critic/dp_critic.py、verl/trainer/fsdp_sft_trainer.py。  # 注释：调用方示例
    - 调用了谁（项目内）：_get_attention_functions。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：flash_attn.bert_padding.unpad_input 或 NPU 版本。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    *_, func = _get_attention_functions()  # 注释：获取 unpad_input 实现
    return func(*args, **kwargs)  # 注释：执行并返回
# （分隔说明：对外导出符号）  # 注释：替代空行，保持逐行注释
__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]  # 注释：模块对外导出列表
