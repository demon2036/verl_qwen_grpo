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
The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.  # 注释：保留英文说明
模块用途：定义混合引擎（vLLM 等）所需的 tokenizer 抽象接口。  # 注释：模块用途
输入/输出：输入文本或 token id，输出 token id 或文本字符串。  # 注释：输入输出概览
关键依赖：numpy、torch。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- class MyTokenizer(HybridEngineBaseTokenizer): ...  # 注释：自定义实现
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：vLLM rollout 初始化时构造 tokenizer 适配器。  # 注释：上层入口举例
- 典型链路：rollout -> tokenizer.encode/decode -> vLLM 推理。  # 注释：调用链路
"""  # 注释：模块 docstring 结束
# （分隔说明：抽象基类依赖）  # 注释：替代空行，保持逐行注释
from abc import ABC, abstractmethod  # 注释：抽象基类工具
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import numpy as np  # 注释：数组类型
import torch  # 注释：张量类型
# （分隔说明：对外导出）  # 注释：替代空行，保持逐行注释
__all__ = ["HybridEngineBaseTokenizer"]  # 注释：公开导出
# （分隔说明：抽象 tokenizer）  # 注释：替代空行，保持逐行注释
class HybridEngineBaseTokenizer(ABC):  # 注释：混合引擎 tokenizer 抽象类
    """the tokenizer property and function name should align with HF's to meet vllm requirement  # 注释：保留英文说明

    功能：规定 vLLM 期望的 tokenizer 属性/方法签名。  # 注释：类用途
    参数：无（抽象基类）。  # 注释：参数说明
    返回：子类实例。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：未实现抽象方法的子类不可实例化。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - class MyTok(HybridEngineBaseTokenizer): ...  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer。  # 注释：类位置
    - 典型调用路径：rollout 初始化 -> tokenizer 适配 -> vLLM 调用。  # 注释：典型调用链
    - 被谁调用：各 rollout/推理模块的 tokenizer 适配器。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
    """  # 注释：类 docstring 结束
    @property  # 注释：属性访问器
    @abstractmethod  # 注释：抽象属性
    def vocab_size(self):  # 注释：基础词表大小
        """
        `int`: Size of the base vocabulary (without the added tokens).

        功能：返回基础词表大小（不含新增 token）。  # 注释：功能说明
        参数：无。  # 注释：参数说明
        返回：int。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：子类未实现会报错。  # 注释：异常说明
        最小示例：vocab = tokenizer.vocab_size。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.vocab_size。  # 注释：位置
        - 典型调用路径：vLLM 初始化 -> tokenizer.vocab_size。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：无。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：pad_token_id）  # 注释：替代空行，保持逐行注释
    @property  # 注释：属性访问器
    @abstractmethod  # 注释：抽象属性
    def pad_token_id(self):  # 注释：padding token id
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.

        功能：返回 padding token id。  # 注释：功能说明
        参数：无。  # 注释：参数说明
        返回：Optional[int]。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：子类未实现会报错。  # 注释：异常说明
        最小示例：pad_id = tokenizer.pad_token_id。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.pad_token_id。  # 注释：位置
        - 典型调用路径：vLLM 处理 padding -> tokenizer.pad_token_id。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：无。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：eos_token_id）  # 注释：替代空行，保持逐行注释
    @property  # 注释：属性访问器
    @abstractmethod  # 注释：抽象属性
    def eos_token_id(self):  # 注释：eos token id
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.

        功能：返回 EOS token id。  # 注释：功能说明
        参数：无。  # 注释：参数说明
        返回：Optional[int]。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：子类未实现会报错。  # 注释：异常说明
        最小示例：eos_id = tokenizer.eos_token_id。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.eos_token_id。  # 注释：位置
        - 典型调用路径：生成停止条件 -> tokenizer.eos_token_id。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：无。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：all_special_ids）  # 注释：替代空行，保持逐行注释
    @property  # 注释：属性访问器
    @abstractmethod  # 注释：抽象属性
    def all_special_ids(self) -> list[int]:  # 注释：特殊 token id 列表
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.

        功能：返回特殊 token 的 id 列表。  # 注释：功能说明
        参数：无。  # 注释：参数说明
        返回：List[int]。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：子类未实现会报错。  # 注释：异常说明
        最小示例：special_ids = tokenizer.all_special_ids。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.all_special_ids。  # 注释：位置
        - 典型调用路径：vLLM 初始化 -> tokenizer.all_special_ids。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：无。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：all_special_tokens）  # 注释：替代空行，保持逐行注释
    @property  # 注释：属性访问器
    @abstractmethod  # 注释：抽象属性
    def all_special_tokens(self) -> list[str]:  # 注释：特殊 token 字符串列表
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.

        功能：返回特殊 token 字符串列表。  # 注释：功能说明
        参数：无。  # 注释：参数说明
        返回：List[str]。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：子类未实现会报错。  # 注释：异常说明
        最小示例：special_tokens = tokenizer.all_special_tokens。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.all_special_tokens。  # 注释：位置
        - 典型调用路径：vLLM 初始化 -> tokenizer.all_special_tokens。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：无。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：encode）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法
    def encode(self, text):  # 注释：文本编码为 token id
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.

        功能：将文本编码为 token id 序列。  # 注释：功能说明
        参数：text：输入文本或 token 列表。  # 注释：参数说明
        返回：token id 列表或数组。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：输入类型不支持时由子类决定。  # 注释：异常说明
        最小示例：ids = tokenizer.encode("hi")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.encode。  # 注释：位置
        - 典型调用路径：rollout.generate -> tokenizer.encode。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：由子类实现决定。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：decode）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法
    def decode(  # 注释：token id 解码为文本
        self,  # 注释：实例本身
        token_ids: int | list[int] | np.ndarray | torch.Tensor,  # 注释：token id 输入
        skip_special_tokens: bool = False,  # 注释：是否跳过特殊 token
        clean_up_tokenization_spaces: bool = None,  # 注释：是否清理空白
        **kwargs,  # 注释：扩展参数
    ) -> str:  # 注释：返回类型
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.

        功能：将 token id 序列解码为字符串。  # 注释：功能说明
        参数：token_ids/skip_special_tokens/clean_up_tokenization_spaces。  # 注释：参数说明
        返回：str。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：由子类具体实现决定。  # 注释：异常说明
        最小示例：text = tokenizer.decode([1,2,3])。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.decode。  # 注释：位置
        - 典型调用路径：rollout.generate -> tokenizer.decode。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：由子类实现决定。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：convert_ids_to_tokens）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法
    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:  # 注释：id 转 token
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).

        功能：将 id 转为 token 字符串。  # 注释：功能说明
        参数：ids/skip_special_tokens。  # 注释：参数说明
        返回：str 或 List[str]。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：由子类具体实现决定。  # 注释：异常说明
        最小示例：token = tokenizer.convert_ids_to_tokens(1)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.convert_ids_to_tokens。  # 注释：位置
        - 典型调用路径：decode -> convert_ids_to_tokens。  # 注释：调用链
        - 被谁调用：decode/外部调用。  # 注释：调用方
        - 调用了谁：由子类实现决定。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：get_added_vocab）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法
    def get_added_vocab(self) -> dict[str, int]:  # 注释：新增词表获取
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.

        功能：返回新增 token 的字典映射。  # 注释：功能说明
        参数：无。  # 注释：参数说明
        返回：dict[str, int]。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：由子类具体实现决定。  # 注释：异常说明
        最小示例：added = tokenizer.get_added_vocab()。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.get_added_vocab。  # 注释：位置
        - 典型调用路径：vLLM 初始化 -> tokenizer.get_added_vocab。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：由子类实现决定。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：convert_tokens_to_string）  # 注释：替代空行，保持逐行注释
    @abstractmethod  # 注释：抽象方法
    def convert_tokens_to_string(self, tokens: list[str]) -> str:  # 注释：tokens 拼接成字符串
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.

        功能：将 token 列表还原为文本字符串。  # 注释：功能说明
        参数：tokens：token 列表。  # 注释：参数说明
        返回：str。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：由子类具体实现决定。  # 注释：异常说明
        最小示例：" ".join(tokens)（示意）。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.convert_tokens_to_string。  # 注释：位置
        - 典型调用路径：decode -> convert_tokens_to_string。  # 注释：调用链
        - 被谁调用：decode/外部调用。  # 注释：调用方
        - 调用了谁：由子类实现决定。  # 注释：依赖
        """  # 注释：docstring 结束
        pass  # 注释：抽象占位
# （分隔说明：is_fast 属性）  # 注释：替代空行，保持逐行注释
    @property  # 注释：属性访问器
    def is_fast(self):  # 注释：是否 fast tokenizer
        """判断是否为 fast tokenizer（默认 False）。  # 注释：函数用途

        参数：无。  # 注释：参数说明
        返回：bool。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：if tokenizer.is_fast: ...  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/workers/rollout/tokenizer.py::HybridEngineBaseTokenizer.is_fast。  # 注释：位置
        - 典型调用路径：vLLM/rollout 适配逻辑 -> tokenizer.is_fast。  # 注释：调用链
        - 被谁调用：vLLM/rollout 适配逻辑。  # 注释：调用方
        - 调用了谁：无。  # 注释：依赖
        """  # 注释：docstring 结束
        return False  # 注释：默认非 fast tokenizer
