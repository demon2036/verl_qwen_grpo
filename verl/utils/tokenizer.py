# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
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
模块用途：封装 Hugging Face tokenizer/processor 的创建与兼容性修正逻辑。  # 注释：模块用途说明
输入：  # 注释：模块输入说明标题
- 模型名称或本地路径（name_or_path）。  # 注释：输入含义
- 兼容性开关（correct_pad_token/correct_gemma2）。  # 注释：输入含义
输出：  # 注释：模块输出说明标题
- PreTrainedTokenizer 或 ProcessorMixin（可能为 None）。  # 注释：输出类型
依赖：transformers.AutoTokenizer/AutoProcessor、warnings。  # 注释：关键依赖说明
典型用法：  # 注释：最小示例标题
- tokenizer = hf_tokenizer("Qwen/Qwen2-7B-Instruct")。  # 注释：示例用法
- processor = hf_processor("Qwen/Qwen2-VL-2B-Instruct")。  # 注释：示例用法
调用路径概览：  # 注释：调用路径概览标题
- 入口：训练入口脚本（如 verl/trainer/main_ppo.py）或 worker 初始化。  # 注释：典型入口说明
- 典型链路：main_ppo.py -> hf_tokenizer/hf_processor -> 数据集或 rollout 使用。  # 注释：调用链说明
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import warnings  # 注释：用于发出兼容性警告
# （分隔说明：公开导出列表）  # 注释：替代空行，保持逐行注释
__all__ = ["hf_tokenizer", "hf_processor"]  # 注释：声明对外暴露的函数
# （分隔说明：工具函数定义）  # 注释：替代空行，保持逐行注释
def set_pad_token_id(tokenizer):  # 注释：定义修正 pad_token_id 的工具函数
    """
    函数用途：当 tokenizer 未设置 pad token 时，用 eos token 兜底。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - tokenizer (transformers.PreTrainedTokenizer)：待修正的 tokenizer。  # 注释：参数含义
    返回：无。  # 注释：返回值说明
    副作用：会原地修改 tokenizer 的 pad_token/pad_token_id，并触发 warnings。  # 注释：副作用说明
    异常/边界条件：无显式异常；若 tokenizer 缺失 eos_token 可能导致下游报错。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：tokenizer.pad_token_id=None, tokenizer.eos_token_id=2。  # 注释：示例输入
    - 输出：tokenizer.pad_token_id=2, tokenizer.pad_token=tokenizer.eos_token。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/tokenizer.py::set_pad_token_id。  # 注释：函数位置
    - 典型调用路径：hf_tokenizer -> set_pad_token_id。  # 注释：典型调用链
    - 被谁调用：verl/utils/tokenizer.py::hf_tokenizer。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：warnings.warn。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if tokenizer.pad_token_id is None:  # 注释：若 pad_token_id 为空
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 注释：用 eos_token_id 兜底
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)  # 注释：发出提示警告
    if tokenizer.pad_token is None:  # 注释：若 pad_token 为空
        tokenizer.pad_token = tokenizer.eos_token  # 注释：用 eos_token 兜底
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)  # 注释：发出提示警告
# （分隔说明：主要创建函数）  # 注释：替代空行，保持逐行注释
def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):  # 注释：创建 Hugging Face tokenizer
    """
    函数用途：创建 Hugging Face 预训练 tokenizer，并修正 pad/eos 兼容性。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - name_or_path (str|Path)：模型名或本地路径。  # 注释：参数含义
    - correct_pad_token (bool)：是否补齐 pad_token_id。  # 注释：参数含义
    - correct_gemma2 (bool)：是否对 gemma2 eos 进行修正。  # 注释：参数含义
    - **kwargs：透传给 AutoTokenizer.from_pretrained。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - transformers.PreTrainedTokenizer：可直接用于编码/解码。  # 注释：返回值语义
    副作用：可能修改 kwargs（如 gemma2 的 eos 设置），并触发 warnings。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 模型路径无效时会抛出 transformers 相关异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入：hf_tokenizer("Qwen/Qwen2-7B-Instruct")。  # 注释：示例输入
    - 输出：可用的 tokenizer 实例。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/tokenizer.py::hf_tokenizer。  # 注释：函数位置
    - 典型调用路径：verl/trainer/main_ppo.py -> hf_tokenizer -> create_rl_dataset。  # 注释：典型调用链
    - 被谁调用：verl/trainer/main_ppo.py、verl/workers/fsdp_workers.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：set_pad_token_id。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.AutoTokenizer.from_pretrained。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from transformers import AutoTokenizer  # 注释：延迟导入 transformers 的自动 tokenizer
    # （分隔说明：对 gemma2 做兼容性修正）  # 注释：替代空行，保持逐行注释
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:  # 注释：仅针对 gemma2 特例
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.  # 注释：原注释，说明 eos 歧义问题
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a  # 注释：参考链接（保留原意）
        warnings.warn(  # 注释：发出修正提示警告
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1  # 注释：警告内容
        )  # 注释：结束 warnings.warn 调用
        kwargs["eos_token"] = "<end_of_turn>"  # 注释：修正 eos_token
        kwargs["eos_token_id"] = 107  # 注释：修正 eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)  # 注释：加载预训练 tokenizer
    if correct_pad_token:  # 注释：若需要修正 pad token
        set_pad_token_id(tokenizer)  # 注释：执行 pad token 修正
    return tokenizer  # 注释：返回 tokenizer 实例
# （分隔说明：多模态 processor 创建函数）  # 注释：替代空行，保持逐行注释
def hf_processor(name_or_path, **kwargs):  # 注释：创建 Hugging Face Processor（多模态预处理）
    """
    函数用途：创建 Hugging Face processor，用于多模态输入预处理。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - name_or_path (str|Path)：模型名或本地路径。  # 注释：参数含义
    - **kwargs：透传给 AutoProcessor.from_pretrained。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - transformers.ProcessorMixin 或 None：失败时返回 None。  # 注释：返回值语义
    副作用：可能触发 warnings（捕获异常后提示）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - AutoProcessor 创建失败时返回 None，避免中断流程。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：hf_processor("Qwen/Qwen2-VL-2B-Instruct")。  # 注释：示例输入
    - 输出：ProcessorMixin 实例或 None。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/tokenizer.py::hf_processor。  # 注释：函数位置
    - 典型调用路径：main_ppo.py -> hf_processor -> RLHFDataset(processor=...)。  # 注释：典型调用链
    - 被谁调用：verl/trainer/main_ppo.py、verl/workers/fsdp_workers.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：transformers.AutoProcessor.from_pretrained。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    from transformers import AutoProcessor  # 注释：延迟导入 AutoProcessor
    try:  # 注释：尝试创建 processor
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)  # 注释：从预训练模型加载 processor
    except Exception as e:  # 注释：捕获创建失败异常
        processor = None  # 注释：失败时置为 None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid  # 注释：原 TODO 说明
        # silent failure  # 注释：原 TODO 说明（静默失败）
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)  # 注释：提示创建失败
    # Avoid load tokenizer, see:  # 注释：说明为何要过滤非 Processor 类型
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344  # 注释：参考链接
    if processor is not None and "Processor" not in processor.__class__.__name__:  # 注释：若对象不是 Processor 子类
        processor = None  # 注释：置空，避免误用
    return processor  # 注释：返回 processor 或 None
