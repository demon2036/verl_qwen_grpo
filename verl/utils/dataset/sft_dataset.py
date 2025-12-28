# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：保留注释符号，保证该行有中文说明
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
模块用途：单轮 SFT 数据集，读取 parquet 并在内存中完成 tokenization。  # 注释：模块用途
输入/输出：输入 parquet 文件路径与 tokenizer，输出 __getitem__ 返回训练所需张量。  # 注释：模块输入输出概览
关键依赖：pandas、numpy、torch、transformers.PreTrainedTokenizer、compute_position_id_with_mask。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- dataset = SFTDataset("data.parquet", tokenizer, config)  # 注释：构建数据集
- batch = dataset[0]  # 注释：获取样本（input_ids/attention_mask/position_ids/loss_mask）
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/trainer/fsdp_sft_trainer.py。  # 注释：上层入口举例
- 典型链路：fsdp_sft_trainer -> SFTDataset -> __getitem__ -> compute_position_id_with_mask。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import numpy as np  # 注释：随机采样与数组处理
import pandas as pd  # 注释：读取 parquet 数据
import torch  # 注释：张量拼接与 padding
from omegaconf.listconfig import ListConfig  # 注释：OmegaConf 列表配置类型
from torch.utils.data import Dataset  # 注释：PyTorch Dataset 基类
from transformers import PreTrainedTokenizer  # 注释：Tokenizer 类型注解
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl.utils import hf_tokenizer  # 注释：加载 HF tokenizer
from verl.utils.fs import copy_to_local  # 注释：支持 HDFS/本地的文件拷贝
from verl.utils.model import compute_position_id_with_mask  # 注释：根据 attention mask 计算 position_ids
# （分隔说明：SFTDataset 定义）  # 注释：替代空行，保持逐行注释
class SFTDataset(Dataset):  # 注释：单轮 SFT 数据集
    """
    功能：将单轮 prompt/response 数据读入内存并构造训练样本。  # 注释：类用途
    参数：  # 注释：参数说明标题
    - config (OmegaConf): 数据相关配置。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - Dataset 子类实例。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 读取 parquet 并占用内存；可能打印数据集长度。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - truncation 非法会触发 assert；parquet 读取失败会抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - dataset = SFTDataset("data.parquet", tokenizer, cfg, max_samples=100)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/sft_dataset.py::SFTDataset。  # 注释：类位置
    - 典型调用路径：fsdp_sft_trainer -> SFTDataset -> DataLoader。  # 注释：典型调用链
    - 被谁调用：verl/trainer/fsdp_sft_trainer.py、verl/trainer/sft_trainer.py、tests/utils/dataset/test_sft_dataset_on_cpu.py。  # 注释：调用方示例
    - 调用了谁（项目内）：hf_tokenizer、copy_to_local、compute_position_id_with_mask。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：pandas.read_parquet、tokenizer.apply_chat_template。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, max_samples: int = -1):  # 注释：初始化数据集
        """
        功能：读取配置与 parquet 路径，完成下载/读取与缓存。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - parquet_files (str | ListConfig)：单个或多个 parquet 路径。  # 注释：参数含义
        - tokenizer (str|PreTrainedTokenizer)：tokenizer 或其路径。  # 注释：参数含义
        - config：数据配置（键如 prompt_key/response_key/max_length）。  # 注释：参数含义
        - max_samples (int)：最多采样样本数，-1 表示全量。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None（初始化对象）。  # 注释：返回值语义
        副作用：  # 注释：副作用说明标题
        - 读取 parquet 文件到内存。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - truncation 非法触发 assert；读取失败抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - SFTDataset("data.parquet", tokenizer, cfg, max_samples=10)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/sft_dataset.py::SFTDataset.__init__。  # 注释：函数位置
        - 典型调用路径：fsdp_sft_trainer.create_dataset -> SFTDataset(...)。  # 注释：典型调用链
        - 被谁调用：verl/trainer/fsdp_sft_trainer.py、verl/trainer/sft_trainer.py。  # 注释：调用方示例
        - 调用了谁（项目内）：self._download、self._read_files_and_tokenize、hf_tokenizer。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：None（主要为内部方法）。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        prompt_key = config.get("prompt_key", "prompt")  # 注释：读取 prompt 字段名
        prompt_dict_keys = config.get("prompt_dict_keys", None)  # 注释：读取 prompt 字典嵌套键
        response_key = config.get("response_key", "response")  # 注释：读取 response 字段名
        response_dict_keys = config.get("response_dict_keys", None)  # 注释：读取 response 字典嵌套键
        max_length = config.get("max_length", 1024)  # 注释：最大序列长度
        truncation = config.get("truncation", "error")  # 注释：截断策略
        use_shm = config.get("use_shm", False)  # 注释：是否使用共享内存
        self.shuffle = config.get("shuffle", False)  # 注释：是否随机采样
        self.seed = config.get("seed")  # 注释：随机种子
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})  # 注释：chat template 参数

        assert truncation in ["error", "left", "right"]  # 注释：校验截断策略合法
        self.truncation = truncation  # 注释：保存截断策略
        self.use_shm = use_shm  # 注释：保存共享内存开关

        if not isinstance(parquet_files, ListConfig):  # 注释：确保为列表配置类型
            parquet_files = [parquet_files]  # 注释：单文件转列表

        self.parquet_files = parquet_files  # 注释：保存 parquet 路径列表
        self.max_samples = max_samples  # 注释：保存最大样本数
        if isinstance(tokenizer, str):  # 注释：tokenizer 为路径时
            tokenizer = hf_tokenizer(tokenizer)  # 注释：加载 HF tokenizer
        self.tokenizer: PreTrainedTokenizer = tokenizer  # 注释：保存 tokenizer 实例

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]  # 注释：规范化 prompt_key 列表
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]  # 注释：规范化 response_key 列表
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []  # 注释：规范化 prompt_dict_keys
        self.response_dict_keys = response_dict_keys if response_dict_keys else []  # 注释：规范化 response_dict_keys

        self.max_length = max_length  # 注释：保存最大长度

        self._download()  # 注释：下载/拷贝 parquet 到本地
        self._read_files_and_tokenize()  # 注释：读取 parquet 并缓存 prompt/response

    def _download(self):  # 注释：内部函数：下载/拷贝数据文件
        """
        功能：将 parquet 文件复制到本地（支持 HDFS/共享内存）。  # 注释：函数用途
        参数：无（使用 self.parquet_files）。  # 注释：参数说明标题
        返回：  # 注释：返回值说明标题
        - None（原地更新 self.parquet_files）。  # 注释：返回值语义
        副作用：  # 注释：副作用说明标题
        - 触发文件系统复制操作。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 拷贝失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - self._download() -> 本地路径列表更新。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/sft_dataset.py::_download。  # 注释：函数位置
        - 典型调用路径：SFTDataset.__init__ -> _download。  # 注释：典型调用链
        - 被谁调用：仅在本文件内 SFTDataset.__init__ 调用。  # 注释：调用方说明
        - 调用了谁（项目内）：copy_to_local。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：文件系统操作。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        for i, parquet_file in enumerate(self.parquet_files):  # 注释：遍历 parquet 路径
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)  # 注释：拷贝到本地

    def _read_files_and_tokenize(self):  # 注释：读取 parquet 并准备 prompt/response
        """
        功能：读取 parquet 文件，抽取 prompt/response 列并缓存为列表。  # 注释：函数用途
        参数：无（使用 self.parquet_files）。  # 注释：参数说明标题
        返回：  # 注释：返回值说明标题
        - None（在对象内缓存 dataframe/prompts/responses）。  # 注释：返回值语义
        副作用：  # 注释：副作用说明标题
        - 读取数据集到内存，并可能打印样本数。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - parquet 读取失败；字段缺失导致 KeyError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - self._read_files_and_tokenize() -> self.prompts/self.responses 填充。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/sft_dataset.py::_read_files_and_tokenize。  # 注释：函数位置
        - 典型调用路径：SFTDataset.__init__ -> _read_files_and_tokenize。  # 注释：典型调用链
        - 被谁调用：仅在本文件内 SFTDataset.__init__ 调用。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：pandas.read_parquet、pandas.concat。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        def series_to_item(ls):  # 注释：内部工具：展开单元素 Series/ndarray
            """
            功能：将嵌套的单元素 Series/ndarray 展开为实际对象。  # 注释：函数用途
            参数：  # 注释：参数说明标题
            - ls：可能为 Series/ndarray 的对象。  # 注释：参数含义
            返回：  # 注释：返回值说明标题
            - 展开后的元素。  # 注释：返回值语义
            副作用：无。  # 注释：副作用说明
            异常/边界条件：无。  # 注释：异常说明
            最小示例：  # 注释：最小示例标题
            - series_to_item(np.array([{"a":1}])) -> {"a":1}。  # 注释：示例
            调用路径依赖：  # 注释：调用路径说明标题
            - 所在位置：verl/utils/dataset/sft_dataset.py::_read_files_and_tokenize.series_to_item。  # 注释：函数位置
            - 典型调用路径：_read_files_and_tokenize -> series_to_item。  # 注释：典型调用链
            - 被谁调用：仅在本函数内部使用。  # 注释：调用方说明
            - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
            - 调用了谁（关键外部依赖）：numpy.ndarray、pandas.core.series.Series。  # 注释：外部依赖说明
            """  # 注释：函数 docstring 结束
            import numpy  # 注释：局部导入 numpy
            import pandas  # 注释：局部导入 pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:  # 注释：单元素序列持续展开
                ls = ls[0]  # 注释：取出单元素
            return ls  # 注释：返回展开后的对象

        dataframes = []  # 注释：收集所有 parquet 的 DataFrame
        for parquet_file in self.parquet_files:  # 注释：遍历文件路径
            # read parquet files and cache  # 注释：原注释保留并补充说明
            dataframe = pd.read_parquet(parquet_file)  # 注释：读取 parquet 为 DataFrame
            dataframes.append(dataframe)  # 注释：加入列表
        self.dataframe = pd.concat(dataframes)  # 注释：拼接所有 DataFrame

        total = len(self.dataframe)  # 注释：样本总数
        print(f"dataset len: {len(self.dataframe)}")  # 注释：打印数据集长度

        if self.max_samples > 0 and self.max_samples < total:  # 注释：需要子采样
            if self.shuffle:  # 注释：随机采样
                rngs_args = (self.seed,) if self.seed is not None else ()  # 注释：设置随机种子参数
                rng = np.random.default_rng(*rngs_args)  # 注释：构造随机数生成器
                indices = rng.choice(total, size=self.max_samples, replace=False)  # 注释：无放回采样索引
            else:  # 注释：不随机，取前 max_samples
                indices = np.arange(self.max_samples)  # 注释：生成顺序索引
            self.dataframe = self.dataframe.iloc[indices.tolist()]  # 注释：裁剪 DataFrame
            print(f"selected {self.max_samples} random samples out of {total}")  # 注释：打印采样信息

        self.prompts = self.dataframe[self.prompt_key]  # 注释：读取 prompt 列
        for key in self.prompt_dict_keys:  # 注释：处理 prompt 字典嵌套键
            # type(x): pandas.core.series.Series  # 注释：原注释保留
            # type(x[0]): numpy.ndarray  # 注释：原注释保留
            # type(x[0][0]): dict  # 注释：原注释保留
            try:  # 注释：尝试按 key 展开
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023  # 注释：从嵌套字典取值
            except Exception:  # 注释：捕获异常以打印调试
                print(f"self.prompts={self.prompts}")  # 注释：输出当前 prompts
                raise  # 注释：重新抛出异常
        if isinstance(self.prompts, pd.DataFrame):  # 注释：若仍为 DataFrame
            self.prompts = self.prompts.squeeze()  # 注释：压缩为 Series
        self.prompts = self.prompts.tolist()  # 注释：转为 Python 列表
        self.responses = self.dataframe[self.response_key]  # 注释：读取 response 列
        for key in self.response_dict_keys:  # 注释：处理 response 字典嵌套键
            try:  # 注释：尝试按 key 展开
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023  # 注释：从嵌套字典取值
            except Exception:  # 注释：捕获异常以打印调试
                print(f"self.responses={self.responses}")  # 注释：输出当前 responses
                raise  # 注释：重新抛出异常
        if isinstance(self.responses, pd.DataFrame):  # 注释：若仍为 DataFrame
            self.responses = self.responses.squeeze()  # 注释：压缩为 Series
        self.responses = self.responses.tolist()  # 注释：转为 Python 列表

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
        - 所在位置：verl/utils/dataset/sft_dataset.py::__len__。  # 注释：函数位置
        - 典型调用路径：PyTorch DataLoader -> __len__。  # 注释：典型调用链
        - 被谁调用：DataLoader 或训练逻辑。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return len(self.prompts)  # 注释：返回 prompt 列表长度

    def __getitem__(self, item):  # 注释：根据索引构建训练样本
        """
        功能：根据索引返回模型训练所需的 input_ids/attention_mask/position_ids/loss_mask。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - item (int)：样本索引。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - dict：包含 input_ids、attention_mask、position_ids、loss_mask。  # 注释：返回值语义
        副作用：无（仅计算张量）。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - truncation=="error" 且超长会抛 NotImplementedError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - dataset[0]["input_ids"].shape -> (max_length,)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/sft_dataset.py::__getitem__。  # 注释：函数位置
        - 典型调用路径：DataLoader -> __getitem__ -> compute_position_id_with_mask。  # 注释：典型调用链
        - 被谁调用：PyTorch DataLoader（训练迭代）。  # 注释：调用方说明
        - 调用了谁（项目内）：compute_position_id_with_mask。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：tokenizer.apply_chat_template、torch.cat。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        tokenizer = self.tokenizer  # 注释：取出 tokenizer

        prompt = self.prompts[item]  # 注释：取 prompt 文本
        response = self.responses[item]  # 注释：取 response 文本

        # apply chat template  # 注释：原注释保留并补充说明
        prompt_chat = [{"role": "user", "content": prompt}]  # 注释：构造单轮对话格式

        # string  # 注释：原注释保留并补充说明
        prompt_chat_str = tokenizer.apply_chat_template(  # 注释：渲染 prompt chat template
            prompt_chat, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs  # 注释：添加生成提示词但不 token 化
        )  # 注释：模板渲染结束
        response_chat_str = response + tokenizer.eos_token  # 注释：在回复末尾加 eos

        # tokenize  # 注释：原注释保留并补充说明
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)  # 注释：tokenize prompt
        prompt_ids = prompt_ids_output["input_ids"][0]  # 注释：取 input_ids
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]  # 注释：取 attention_mask

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)  # 注释：tokenize response
        response_ids = response_ids_output["input_ids"][0]  # 注释：取 response input_ids
        response_attention_mask = response_ids_output["attention_mask"][0]  # 注释：取 response attention_mask

        prompt_length = prompt_ids.shape[0]  # 注释：prompt 长度
        response_length = response_ids.shape[0]  # 注释：response 长度

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)  # 注释：拼接 prompt 与 response
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)  # 注释：拼接 attention mask

        # padding to max length  # 注释：原注释保留并补充说明
        sequence_length = input_ids.shape[0]  # 注释：当前序列长度
        if sequence_length < self.max_length:  # 注释：需要 padding
            padded_input_ids = (  # 注释：构造 pad token 张量
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)  # 注释：填充长度
                * self.tokenizer.pad_token_id  # 注释：使用 pad_token_id
            )  # 注释：padded_input_ids 构造结束
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)  # 注释：padding 的 attention_mask 为 0

            input_ids = torch.cat((input_ids, padded_input_ids))  # 注释：拼接 pad 到 input_ids
            attention_mask = torch.cat((attention_mask, padded_attention_mask))  # 注释：拼接 pad 到 attention_mask
        elif sequence_length > self.max_length:  # 注释：需要截断
            if self.truncation == "left":  # 注释：左侧截断
                # actually, left truncation may not be reasonable  # 注释：原注释保留
                input_ids = input_ids[-self.max_length :]  # 注释：取末尾 max_length
                attention_mask = attention_mask[-self.max_length :]  # 注释：同步截断 mask
            elif self.truncation == "right":  # 注释：右侧截断
                input_ids = input_ids[: self.max_length]  # 注释：取开头 max_length
                attention_mask = attention_mask[: self.max_length]  # 注释：同步截断 mask
            elif self.truncation == "error":  # 注释：超长直接报错
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")  # 注释：抛出异常
            else:  # 注释：未知策略
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")  # 注释：抛出异常

        position_ids = compute_position_id_with_mask(attention_mask)  # 注释：根据 mask 计算 position_ids

        loss_mask = attention_mask.clone()  # 注释：初始化 loss_mask
        if prompt_length > 1:  # 注释：若 prompt 长度 >1
            # mask out prompt for SFT.  # 注释：原注释保留
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0  # 注释：屏蔽 prompt 部分 loss
        # mask out the last token in response  # 注释：原注释保留
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0  # 注释：屏蔽 response 最后一 token 的 loss

        return {  # 注释：返回训练所需字段
            "input_ids": input_ids,  # 注释：输入 token ids
            "attention_mask": attention_mask,  # 注释：注意力 mask
            "position_ids": position_ids,  # 注释：位置编码
            "loss_mask": loss_mask,  # 注释：loss 计算 mask
        }  # 注释：字典返回结束
