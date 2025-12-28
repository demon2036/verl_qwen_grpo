# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# Copyright 2025 ModelBest Inc. and/or its affiliates  # 注释：版权声明（新增贡献方）
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
模块用途：多轮 SFT 数据集，支持对话、工具调用与多模态输入。  # 注释：模块用途
输入/输出：输入 parquet 文件与 tokenizer/processor；输出多轮拼接后的训练张量。  # 注释：模块输入输出概览
关键依赖：pandas、numpy、torch、transformers、Qwen2-VL get_rope_index、chat_template 工具。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- dataset = MultiTurnSFTDataset("data.parquet", tokenizer, cfg, processor=processor)  # 注释：构建多轮数据集
- sample = dataset[0]  # 注释：获取单样本（包含 multi_modal_inputs 可选字段）
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/trainer/fsdp_sft_trainer.py（当 sft_trainer.yaml 中 multiturn.enable=true）。  # 注释：上层入口举例
- 典型链路：fsdp_sft_trainer -> MultiTurnSFTDataset -> __getitem__ -> _process_single_message/_build_messages。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import logging  # 注释：日志输出
import os  # 注释：读取环境变量
import re  # 注释：正则拆分 <image>/<video> 占位符
from typing import Any, Optional  # 注释：类型注解
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import numpy as np  # 注释：随机采样与数组处理
import pandas as pd  # 注释：读取 parquet
import torch  # 注释：张量运算
import torch.nn.functional as F  # 注释：padding 等函数
from omegaconf import DictConfig, ListConfig  # 注释：OmegaConf 配置类型
from torch.utils.data import Dataset  # 注释：PyTorch Dataset 基类
from transformers import PreTrainedTokenizer, ProcessorMixin  # 注释：Tokenizer/Processor 类型
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl.models.transformers.qwen2_vl import get_rope_index  # 注释：Qwen2-VL 位置编码索引
from verl.utils import hf_tokenizer  # 注释：加载 HF tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation  # 注释：解析系统/生成 prompt
from verl.utils.dataset.dataset_utils import DatasetPadMode  # 注释：padding 模式枚举
from verl.utils.dataset.vision_utils import process_image, process_video  # 注释：图像/视频预处理
from verl.utils.fs import copy_local_path_from_hdfs  # 注释：从 HDFS 拷贝到本地
# （分隔说明：日志器初始化）  # 注释：替代空行，保持逐行注释
logger = logging.getLogger(__file__)  # 注释：以文件路径作为 logger 名称
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：读取环境变量设置日志级别
# （分隔说明：辅助函数）  # 注释：替代空行，保持逐行注释
def convert_nested_value_to_list_recursive(data_item):  # 注释：递归将嵌套值转换为 list
    """
    功能：将嵌套结构中的 numpy 数组递归转换为 Python list。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data_item：可能是 dict/list/np.ndarray/原子类型。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 结构等价但 numpy 数组已转为 list 的对象。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无（递归终止于基础类型）。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - convert_nested_value_to_list_recursive(np.array([1,2])) -> [1,2]。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::convert_nested_value_to_list_recursive。  # 注释：函数位置
    - 典型调用路径：_read_files_and_process -> convert_nested_value_to_list_recursive。  # 注释：典型调用链
    - 被谁调用：MultiTurnSFTDataset._read_files_and_process。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：numpy.ndarray.tolist。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if isinstance(data_item, dict):  # 注释：若为字典
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}  # 注释：递归处理值
    elif isinstance(data_item, list):  # 注释：若为列表
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]  # 注释：递归处理元素
    elif isinstance(data_item, np.ndarray):  # 注释：若为 numpy 数组
        # Convert to list, then recursively process the elements of the new list  # 注释：原注释保留
        return convert_nested_value_to_list_recursive(data_item.tolist())  # 注释：先转 list 再递归
    else:  # 注释：基础类型
        # Base case: item is already a primitive type (int, str, float, bool, etc.)  # 注释：原注释保留
        return data_item  # 注释：直接返回
# （分隔说明：多轮 SFT 数据集）  # 注释：替代空行，保持逐行注释
class MultiTurnSFTDataset(Dataset):  # 注释：多轮对话 SFT 数据集
    """
    功能：支持多轮对话、多模态、工具调用的 SFT 数据集。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - data_files (str|list)：parquet 文件路径。  # 注释：参数含义
    - tokenizer (PreTrainedTokenizer)：文本 tokenizer。  # 注释：参数含义
    - config (DictConfig)：配置项（messages_key/max_length 等）。  # 注释：参数含义
    - processor (ProcessorMixin, optional)：多模态处理器。  # 注释：参数含义
    - max_samples (int)：最大样本数，-1 表示不限制。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - Dataset 子类实例。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 读取 parquet 并可能执行图像/视频预处理。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - pad_mode/truncation 非法触发 assert；多模态缺少 processor 会断言失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - dataset = MultiTurnSFTDataset("data.parquet", tokenizer, cfg, processor=processor)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::MultiTurnSFTDataset。  # 注释：类位置
    - 典型调用路径：fsdp_sft_trainer -> MultiTurnSFTDataset -> DataLoader。  # 注释：典型调用链
    - 被谁调用：verl/trainer/fsdp_sft_trainer.py（multiturn.enable=true）、tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py。  # 注释：调用方示例
    - 调用了谁（项目内）：extract_system_prompt_and_generation、process_image/process_video、get_rope_index。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：pandas.read_parquet、tokenizer.apply_chat_template。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束

    def __init__(  # 注释：初始化多轮数据集
        self,
        parquet_files: str | list[str],  # 注释：parquet 路径
        tokenizer: PreTrainedTokenizer,  # 注释：文本 tokenizer
        config: DictConfig,  # 注释：配置对象
        processor: Optional[ProcessorMixin] = None,  # 注释：多模态 processor
        max_samples: int = -1,  # 注释：最大样本数
    ):
        """
        功能：读取配置并加载 parquet，解析对话与工具信息。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - parquet_files：parquet 路径或列表。  # 注释：参数含义
        - tokenizer：文本 tokenizer。  # 注释：参数含义
        - config：多轮数据集配置。  # 注释：参数含义
        - processor：多模态 processor（可选）。  # 注释：参数含义
        - max_samples：最大样本数。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None（初始化对象）。  # 注释：返回值语义
        副作用：  # 注释：副作用说明标题
        - 读取 parquet；可能加载图像/视频。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - pad_mode/truncation 非法将触发 assert。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - MultiTurnSFTDataset("data.parquet", tok, cfg, processor=None)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::MultiTurnSFTDataset.__init__。  # 注释：函数位置
        - 典型调用路径：fsdp_sft_trainer -> MultiTurnSFTDataset(...)。  # 注释：典型调用链
        - 被谁调用：verl/trainer/fsdp_sft_trainer.py。  # 注释：调用方示例
        - 调用了谁（项目内）：self._download、self._read_files_and_process、hf_tokenizer。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        # Set defaults and extract parameters from config if provided  # 注释：原注释保留
        config = config or {}  # 注释：确保 config 有默认值
        self.pad_mode = config.get("pad_mode", "right")  # 注释：padding 模式
        assert self.pad_mode in ["right", "no_padding"], (  # 注释：校验 pad_mode
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"  # 注释：错误信息
        )  # 注释：assert 结束
        self.truncation = config.get("truncation", "error")  # 注释：截断策略
        # for right padding  # 注释：原注释保留
        self.max_length = config.get("max_length", 1024)  # 注释：最大长度
        # Get messages_key from the new multiturn config structure  # 注释：原注释保留
        self.messages_key = config.get("messages_key", "messages")  # 注释：消息字段名
        self.image_key = config.get("image_key", "images")  # 注释：图像字段名
        self.video_key = config.get("video_key", "videos")  # 注释：视频字段名
        self.image_patch_size = config.get(  # 注释：图像 patch 大小
            "image_patch_size", processor.image_processor.patch_size if processor else None  # 注释：默认取 processor 配置
        )  # 注释：image_patch_size 设置结束
        self.tools_key = config.get("tools_key", "tools")  # 注释：工具字段名
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")  # 注释：思考模式字段名
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})  # 注释：chat template 参数
        self.shuffle = config.get("shuffle", False)  # 注释：是否随机采样
        self.seed = config.get("seed")  # 注释：随机种子
        self.max_samples = max_samples  # 注释：最大样本数
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)  # 注释：是否忽略 input_ids 不一致
        assert self.truncation in ["error", "left", "right"]  # 注释：校验截断策略

        if not isinstance(parquet_files, list | ListConfig):  # 注释：规范化 parquet_files
            parquet_files = [parquet_files]  # 注释：单路径转列表

        self.parquet_files = parquet_files  # 注释：保存路径列表
        if isinstance(tokenizer, str):  # 注释：tokenizer 为路径
            tokenizer = hf_tokenizer(tokenizer)  # 注释：加载 tokenizer
        self.tokenizer: PreTrainedTokenizer = tokenizer  # 注释：保存 tokenizer
        self.processor = processor  # 注释：保存 processor

        self._download()  # 注释：下载/拷贝 parquet
        self._read_files_and_process()  # 注释：读取并处理数据

    def _download(self):  # 注释：内部函数：下载/拷贝 parquet
        """
        功能：将 parquet 文件复制到本地。  # 注释：函数用途
        参数：无。  # 注释：参数说明标题
        返回：  # 注释：返回值说明标题
        - None（更新 self.parquet_files）。  # 注释：返回值语义
        副作用：触发文件复制。  # 注释：副作用说明
        异常/边界条件：拷贝失败将抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - self._download() -> 本地路径列表更新。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::_download。  # 注释：函数位置
        - 典型调用路径：__init__ -> _download。  # 注释：典型调用链
        - 被谁调用：仅在本文件内 __init__ 调用。  # 注释：调用方说明
        - 调用了谁（项目内）：copy_local_path_from_hdfs。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：文件系统。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        for i, parquet_file in enumerate(self.parquet_files):  # 注释：遍历路径
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)  # 注释：拷贝到本地

    def _read_files_and_process(self):  # 注释：读取 parquet 并处理字段
        """
        功能：读取 parquet，提取 messages/tools/enable_thinking 等字段。  # 注释：函数用途
        参数：无。  # 注释：参数说明标题
        返回：  # 注释：返回值说明标题
        - None（缓存 messages/tools 等）。  # 注释：返回值语义
        副作用：读取 parquet 并可能打印样本数。  # 注释：副作用说明
        异常/边界条件：字段缺失或 parquet 读取失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - self._read_files_and_process() -> self.messages/self.tools 填充。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::_read_files_and_process。  # 注释：函数位置
        - 典型调用路径：__init__ -> _read_files_and_process。  # 注释：典型调用链
        - 被谁调用：仅在本文件内 __init__ 调用。  # 注释：调用方说明
        - 调用了谁（项目内）：convert_nested_value_to_list_recursive、extract_system_prompt_and_generation。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：pandas.read_parquet。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        def series_to_item(ls):  # 注释：内部工具：展开单元素 Series/ndarray
            import numpy  # 注释：局部导入 numpy
            import pandas  # 注释：局部导入 pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:  # 注释：逐层展开
                ls = ls[0]  # 注释：取出单元素
            return ls  # 注释：返回展开结果

        dataframes = []  # 注释：保存 DataFrame 列表
        for parquet_file in self.parquet_files:  # 注释：遍历 parquet 路径
            dataframe = pd.read_parquet(parquet_file)  # 注释：读取 parquet
            dataframes.append(dataframe)  # 注释：加入列表
        self.dataframe = pd.concat(dataframes)  # 注释：拼接所有 DataFrame

        total = len(self.dataframe)  # 注释：样本总数
        print(f"dataset len: {len(self.dataframe)}")  # 注释：打印数据集长度

        if self.max_samples > 0 and self.max_samples < total:  # 注释：需要子采样
            if self.shuffle:  # 注释：随机采样
                rngs_args = (self.seed,) if self.seed is not None else ()  # 注释：种子参数
                rng = np.random.default_rng(*rngs_args)  # 注释：构造 RNG
                indices = rng.choice(total, size=self.max_samples, replace=False)  # 注释：无放回采样
            else:  # 注释：顺序采样
                indices = np.arange(self.max_samples)  # 注释：生成顺序索引
            self.dataframe = self.dataframe.iloc[indices.tolist()]  # 注释：裁剪 DataFrame
            print(f"selected {self.max_samples} random samples out of {total}")  # 注释：打印采样信息

        # Extract messages list from dataframe  # 注释：原注释保留
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()  # 注释：提取 messages

        # Extract tools list from dataframe  # 注释：原注释保留
        if self.tools_key in self.dataframe.columns:  # 注释：存在工具列
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()  # 注释：提取 tools
        else:  # 注释：无工具列
            self.tools = None  # 注释：置为 None
        # Extract enable_thinking list from dataframe  # 注释：原注释保留
        if self.enable_thinking_key in self.dataframe.columns:  # 注释：存在思考标志
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()  # 注释：提取 enable_thinking
        else:  # 注释：无该字段
            self.enable_thinking = None  # 注释：置为 None

        # system prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n  # 注释：原注释保留
        # generation prompt: <|im_start|>assistant\n  # 注释：原注释保留
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)  # 注释：解析系统/生成提示词

    def __len__(self):  # 注释：返回数据集长度
        """
        功能：返回样本数量。  # 注释：函数用途
        参数：无。  # 注释：参数说明标题
        返回：  # 注释：返回值说明标题
        - int：样本数。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - len(dataset) -> N。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::__len__。  # 注释：函数位置
        - 典型调用路径：DataLoader -> __len__。  # 注释：典型调用链
        - 被谁调用：PyTorch DataLoader。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return len(self.messages)  # 注释：返回 messages 数量

    def _process_single_message(  # 注释：处理单条消息并 token 化
        self,
        index: int,  # 注释：消息序号
        message: dict[str, Any],  # 注释：消息字典
        tools: Optional[list[dict[str, Any]]] = None,  # 注释：工具列表（仅首轮传入）
        enable_thinking: Optional[bool] = None,  # 注释：是否启用思考模式
    ) -> tuple[list[int], list[int], list[int]]:  # 注释：返回 token 与 mask
        """
        Process a single message and return its tokenized representation.  # 注释：保留英文说明

        功能：对单条消息应用 chat template，并生成 input_ids/attention_mask/loss_mask。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - index：对话轮次索引。  # 注释：参数含义
        - message：单条消息字典。  # 注释：参数含义
        - tools：工具列表（通常仅第一轮）。  # 注释：参数含义
        - enable_thinking：是否启用思考模式。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - tuple(input_ids, loss_mask, attention_mask, inputs)：包含多模态输入的字典。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - tokenizer/processor 不支持模板会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - _process_single_message(0, {"role":"user","content":"hi"}) -> (ids, mask, attn, inputs)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::_process_single_message。  # 注释：函数位置
        - 典型调用路径：__getitem__ -> _process_single_message。  # 注释：典型调用链
        - 被谁调用：仅在本文件 __getitem__ 内调用。  # 注释：调用方说明
        - 调用了谁（项目内）：tokenizer/processor.apply_chat_template。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：transformers Processor/Tokenizer。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        processor = self.processor if self.processor is not None else self.tokenizer  # 注释：优先使用 processor
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}  # 注释：复制模板参数
        if enable_thinking is not None:  # 注释：若指定思考模式
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking  # 注释：写入参数

        inputs = processor.apply_chat_template(  # 注释：应用 chat template
            [message],  # 注释：单条消息列表
            tools=tools,  # 注释：工具定义
            add_generation_prompt=False,  # 注释：不添加生成提示词
            tokenize=True,  # 注释：直接 token 化
            return_dict=True,  # 注释：返回字典
            return_tensors="pt",  # 注释：返回 PyTorch 张量
            **apply_chat_template_kwargs,  # 注释：附加参数
        )  # 注释：apply_chat_template 结束

        inputs = dict(inputs)  # 注释：转为普通 dict
        input_ids = inputs.pop("input_ids")[0]  # 注释：取 input_ids
        attention_mask = inputs.pop("attention_mask")[0]  # 注释：取 attention_mask

        # remove system prompt if exists  # 注释：原注释保留
        if index != 0 and message["role"] != "system":  # 注释：非首轮且非 system 消息
            input_ids = input_ids[len(self.system_prompt) :]  # 注释：去掉 system prompt 前缀
            attention_mask = attention_mask[len(self.system_prompt) :]  # 注释：同步裁剪 mask

        if message["role"] == "assistant":  # 注释：assistant 消息需要计算 loss
            loss_mask = torch.ones_like(attention_mask)  # 注释：初始 loss_mask 为 1
            # mask out generation prompt if assistant message  # 注释：原注释保留
            loss_mask[: len(self.generation_prompt)] = 0  # 注释：屏蔽生成提示词
        else:  # 注释：非 assistant 不计算 loss
            loss_mask = torch.zeros_like(attention_mask)  # 注释：loss_mask 全 0

        return input_ids, loss_mask, attention_mask, inputs  # 注释：返回单条消息结果

    def _build_messages(self, example: dict):  # 注释：替换 <image>/<video> 占位符
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", "video": video}

        功能：将消息中的 <image>/<video> 文本替换为 processor 需要的结构化内容。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - example：DataFrame 行字典。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - messages：替换后消息列表。  # 注释：返回值语义
        副作用：会原地修改 messages 中的 content 字段。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 若包含多模态但缺少 processor，会触发 assert。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入："请看 <image>" -> 输出：[{"type":"image", ...}]。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::_build_messages。  # 注释：函数位置
        - 典型调用路径：__getitem__ -> _build_messages。  # 注释：典型调用链
        - 被谁调用：仅在本文件 __getitem__ 内调用。  # 注释：调用方说明
        - 调用了谁（项目内）：process_image、process_video。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：re.split。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        messages: list = example[self.messages_key]  # 注释：取 messages 列
        images = example[self.image_key] if self.image_key in example else []  # 注释：取图像列表
        videos = example[self.video_key] if self.video_key in example else []  # 注释：取视频列表

        image_offset, video_offset = 0, 0  # 注释：图像/视频索引偏移
        for message in messages:  # 注释：遍历消息
            if self.image_key not in example and self.video_key not in example:  # 注释：无多模态字段
                continue  # 注释：跳过
            assert self.processor is not None, "processor is needed to process image and video"  # 注释：多模态需要 processor

            content = message["content"]  # 注释：消息内容
            if not isinstance(content, str):  # 注释：非字符串（可能已是结构化内容）
                continue  # 注释：跳过

            content_list = []  # 注释：构造结构化内容列表
            segments = re.split("(<image>|<video>)", content)  # 注释：按占位符切分
            segments = [item for item in segments if item != ""]  # 注释：移除空段
            for segment in segments:  # 注释：遍历每个片段
                if segment == "<image>":  # 注释：图像占位符
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)  # 注释：处理图像
                    content_list.append({"type": "image", "image": image})  # 注释：追加图像段
                    image_offset += 1  # 注释：图像索引递增
                elif segment == "<video>":  # 注释：视频占位符
                    video = process_video(videos[video_offset], image_patch_size=self.image_patch_size)  # 注释：处理视频
                    content_list.append({"type": "video", "video": video})  # 注释：追加视频段
                    video_offset += 1  # 注释：视频索引递增
                else:  # 注释：文本片段
                    content_list.append({"type": "text", "text": segment})  # 注释：追加文本段
            message["content"] = content_list  # 注释：回写结构化内容

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"  # 注释：校验图像数量
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"  # 注释：校验视频数量
        return messages  # 注释：返回替换后的消息

    def __getitem__(self, item):  # 注释：根据索引构建多轮样本
        """
        功能：将多轮 messages 拼接为单序列，并生成 loss_mask/position_ids 等。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - item (int)：样本索引。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - dict：包含 input_ids/attention_mask/position_ids/loss_mask/multi_modal_inputs。  # 注释：返回值语义
        副作用：无（仅计算）。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - pad_mode/truncation 非法抛 ValueError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - sample = dataset[0]; sample["input_ids"].shape -> (L,)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::__getitem__。  # 注释：函数位置
        - 典型调用路径：DataLoader -> __getitem__ -> _process_single_message/_build_messages。  # 注释：典型调用链
        - 被谁调用：PyTorch DataLoader。  # 注释：调用方说明
        - 调用了谁（项目内）：_build_messages、_process_single_message、get_rope_index。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.cat、torch.arange。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        row_dict: dict = self.dataframe.iloc[item].to_dict()  # 注释：取出一行数据
        messages = self._build_messages(row_dict)  # 注释：处理多模态占位符
        tools = self.tools[item] if self.tools is not None else None  # 注释：取工具列表
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None  # 注释：取思考标志

        # 1. tokenize each message  # 注释：原注释保留
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}  # 注释：初始化累积容器
        for i, message in enumerate(messages):  # 注释：逐条消息处理
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(  # 注释：处理单条消息
                index=i,  # 注释：消息索引
                message=message,  # 注释：消息内容
                tools=tools if i == 0 else None,  # 注释：仅首轮传 tools
                enable_thinking=enable_thinking,  # 注释：思考模式
            )  # 注释：单条处理结束
            input_ids.append(_input_ids)  # 注释：收集 input_ids
            loss_mask.append(_loss_mask)  # 注释：收集 loss_mask
            attention_mask.append(_attention_mask)  # 注释：收集 attention_mask
            for k, v in _inputs.items():  # 注释：收集多模态输入
                multi_modal_inputs.setdefault(k, []).append(v)  # 注释：按 key 聚合

        input_ids = torch.cat(input_ids, dim=0)  # 注释：拼接所有轮次 input_ids
        loss_mask = torch.cat(loss_mask, dim=0)  # 注释：拼接所有轮次 loss_mask
        attention_mask = torch.cat(attention_mask, dim=0)  # 注释：拼接所有轮次 attention_mask
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (  # 注释：校验形状一致
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"  # 注释：错误信息
        )  # 注释：assert 结束
        self.sanity_check(input_ids, messages, tools, enable_thinking)  # 注释：一致性检查

        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes  # 注释：原注释保留
        keys_to_remove = []  # 注释：待移除的键列表
        for k, v in multi_modal_inputs.items():  # 注释：遍历多模态输入
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):  # 注释：确保是张量列表
                # Check if all tensors in the list have the same shape  # 注释：原注释保留
                first_shape = v[0].shape[1:]  # 注释：基准形状（忽略 batch 维）
                if not all(tensor.shape[1:] == first_shape for tensor in v):  # 注释：形状不一致
                    keys_to_remove.append(k)  # 注释：标记移除

        for k in keys_to_remove:  # 注释：删除不一致的键
            del multi_modal_inputs[k]  # 注释：移除键

        for k, v in multi_modal_inputs.items():  # 注释：拼接多模态输入
            multi_modal_inputs[k] = torch.concat(v, dim=0)  # 注释：沿 batch 维拼接

        # 2. handle position_ids for Qwen-VL series models  # 注释：原注释保留
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:  # 注释：Qwen2-VL 特例
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)  # 注释：图像 grid
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)  # 注释：视频 grid
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)  # 注释：视频时间尺度

            vision_position_ids = get_rope_index(  # 注释：计算视觉位置编码
                self.processor,  # 注释：processor
                input_ids=input_ids,  # 注释：输入 token ids
                image_grid_thw=image_grid_thw,  # 注释：图像 grid
                video_grid_thw=video_grid_thw,  # 注释：视频 grid
                second_per_grid_ts=second_per_grid_ts,  # 注释：视频时间尺度
                attention_mask=attention_mask,  # 注释：attention mask
            )  # (3, seq_len)  # 注释：输出形状说明
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)  # 注释：文本位置编码
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)  # 注释：拼接文本+视觉位置编码
        else:  # 注释：非 Qwen2-VL
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)  # 注释：默认位置编码

        # 3. handle padding  # 注释：原注释保留
        sequence_length = input_ids.shape[0]  # 注释：序列长度
        # Handle sequence length  # 注释：原注释保留
        if self.pad_mode == DatasetPadMode.RIGHT:  # 注释：右侧 padding
            if sequence_length < self.max_length:  # 注释：需要 padding
                # Pad sequences  # 注释：原注释保留
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0  # 注释：pad token id
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)  # 注释：构造 input_ids padding
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)  # 注释：构造 attention_mask padding
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)  # 注释：构造 loss_mask padding

                input_ids = torch.cat((input_ids, padded_input_ids))  # 注释：拼接 input_ids
                attention_mask = torch.cat((attention_mask, padded_attention_mask))  # 注释：拼接 attention_mask
                loss_mask = torch.cat((loss_mask, padded_loss_mask))  # 注释：拼接 loss_mask
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)  # 注释：补齐 position_ids
            elif sequence_length > self.max_length:  # 注释：需要截断
                if self.truncation == "left":  # 注释：左截断
                    input_ids = input_ids[-self.max_length :]  # 注释：取末尾
                    attention_mask = attention_mask[-self.max_length :]  # 注释：截断 mask
                    loss_mask = loss_mask[-self.max_length :]  # 注释：截断 loss_mask
                    position_ids = position_ids[..., -self.max_length :]  # 注释：截断 position_ids
                elif self.truncation == "right":  # 注释：右截断
                    input_ids = input_ids[: self.max_length]  # 注释：取开头
                    attention_mask = attention_mask[: self.max_length]  # 注释：截断 mask
                    loss_mask = loss_mask[: self.max_length]  # 注释：截断 loss_mask
                    position_ids = position_ids[..., : self.max_length]  # 注释：截断 position_ids
                elif self.truncation == "error":  # 注释：超长报错
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")  # 注释：抛异常
                else:  # 注释：未知截断方式
                    raise ValueError(f"Unknown truncation method {self.truncation}")  # 注释：抛异常

            res = {  # 注释：构造返回字典
                "input_ids": input_ids,  # 注释：input_ids
                "attention_mask": attention_mask,  # 注释：attention_mask
                "position_ids": position_ids,  # 注释：position_ids
                "loss_mask": loss_mask,  # 注释：loss_mask
            }  # 注释：字典构造结束
            if len(multi_modal_inputs) > 0:  # 注释：存在多模态输入
                res["multi_modal_inputs"] = multi_modal_inputs  # 注释：添加多模态字段
            return res  # 注释：返回结果
        elif self.pad_mode == DatasetPadMode.NO_PADDING:  # 注释：不做 padding
            # truncate input_ids if it is longer than max_length  # 注释：原注释保留
            if len(input_ids) > self.max_length:  # 注释：超长截断
                input_ids = input_ids[: self.max_length]  # 注释：截断 input_ids
                loss_mask = loss_mask[: self.max_length]  # 注释：截断 loss_mask
                position_ids = position_ids[..., : self.max_length]  # 注释：截断 position_ids

            # return nested tensor with out padding  # 注释：原注释保留
            res = {  # 注释：构造返回字典
                "input_ids": input_ids,  # 注释：input_ids
                "position_ids": position_ids,  # 注释：position_ids
                "loss_mask": loss_mask,  # 注释：loss_mask
            }  # 注释：字典构造结束
            if len(multi_modal_inputs) > 0:  # 注释：存在多模态输入
                res["multi_modal_inputs"] = multi_modal_inputs  # 注释：添加多模态字段
            return res  # 注释：返回结果
        else:  # 注释：未知 pad_mode
            raise ValueError(f"Unknown pad mode {self.pad_mode}")  # 注释：抛异常

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):  # 注释：一致性检查
        """Check concatenated input_ids of apply_chat_template to each turn equals
        apply_chat_template to whole messages.

        功能：验证逐轮 apply_chat_template 的拼接结果与整体 apply_chat_template 一致。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - input_ids：逐轮拼接后的 input_ids。  # 注释：参数含义
        - messages：对话消息列表。  # 注释：参数含义
        - tools：工具列表。  # 注释：参数含义
        - enable_thinking：是否启用思考模式。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None（不返回，若不一致可能告警或抛错）。  # 注释：返回值语义
        副作用：  # 注释：副作用说明标题
        - 可能输出 warning 或抛 AssertionError。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 当 ignore_input_ids_mismatch=False 且不一致时抛 AssertionError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - sanity_check(ids, messages, tools, False) -> 无异常。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/multiturn_sft_dataset.py::sanity_check。  # 注释：函数位置
        - 典型调用路径：__getitem__ -> sanity_check。  # 注释：典型调用链
        - 被谁调用：仅在本文件 __getitem__ 内调用。  # 注释：调用方说明
        - 调用了谁（项目内）：tokenizer/processor.apply_chat_template、logger.warning_once。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：torch.equal。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        processor = self.processor if self.processor is not None else self.tokenizer  # 注释：优先使用 processor
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}  # 注释：复制模板参数
        if enable_thinking is not None:  # 注释：设置思考模式参数
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking  # 注释：写入参数
        inputs = processor.apply_chat_template(  # 注释：整体 apply_chat_template
            messages,  # 注释：消息列表
            tools=tools,  # 注释：工具列表
            add_generation_prompt=False,  # 注释：不添加生成提示词
            tokenize=True,  # 注释：直接 token 化
            return_dict=True,  # 注释：返回字典
            return_tensors="pt",  # 注释：返回 PyTorch 张量
            **apply_chat_template_kwargs,  # 注释：附加参数
        )  # 注释：apply_chat_template 结束

        error_message = (  # 注释：错误提示信息
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "  # 注释：提示内容1
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"  # 注释：提示内容2
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "  # 注释：提示内容3
            "your tokenizer chat template settings.\n"  # 注释：提示内容4
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "  # 注释：提示内容5
            "input_ids as the final input_ids. "  # 注释：提示内容6
        )  # 注释：错误信息构造结束

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):  # 注释：若拼接结果不一致
            if self.ignore_input_ids_mismatch:  # 注释：允许忽略
                logger.warning_once(error_message)  # 注释：记录一次告警
            else:  # 注释：不允许忽略
                raise AssertionError(error_message)  # 注释：抛异常
