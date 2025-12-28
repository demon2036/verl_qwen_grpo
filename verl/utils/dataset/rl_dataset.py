# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明，标记文件归属
# Copyright 2023-2024 SGLang Team  # 注释：版权声明，说明贡献团队
# Copyright 2025 ModelBest Inc. and/or its affiliates  # 注释：版权声明，说明贡献方
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行可读
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
模块用途：加载 RLHF/GRPO 使用的 Parquet 数据集，构造模型可用的输入张量与元信息。  # 注释：模块用途说明
输入：  # 注释：模块输入说明标题
- Parquet 文件路径（本地或远端），配置项（prompt_key、max_prompt_length 等）。  # 注释：输入含义
- tokenizer（文本）与 processor（多模态，可选）。  # 注释：输入含义
输出：  # 注释：模块输出说明标题
- RLHFDataset：每条样本输出 input_ids/attention_mask/position_ids 等字段。  # 注释：输出说明
- collate_fn：将样本列表合并为 batch。  # 注释：输出说明
依赖：datasets、torch、transformers、verl.utils.torch_functional。  # 注释：关键依赖说明
典型用法：  # 注释：最小示例标题
- dataset = RLHFDataset(["/path/train.parquet"], tokenizer, cfg.data)；DataLoader(..., collate_fn=collate_fn)。  # 注释：示例用法
调用路径概览：  # 注释：调用路径概览标题
- 入口：verl/trainer/main_ppo.py::create_rl_dataset。  # 注释：典型入口
- 典型链路：main_ppo.py -> create_rl_dataset -> RLHFDataset -> DataLoader/rollout。  # 注释：调用链说明
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import copy  # 注释：深拷贝数据路径列表
import logging  # 注释：日志记录
import os  # 注释：路径处理与环境读取
import re  # 注释：正则切分多模态占位符
import traceback  # 注释：异常打印栈
from collections import defaultdict  # 注释：用于按字段聚合数据
from typing import Optional  # 注释：类型提示
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
import datasets  # 注释：Hugging Face datasets 读取 parquet
import numpy as np  # 注释：数组处理与随机采样
import torch  # 注释：张量操作
from omegaconf import DictConfig, ListConfig  # 注释：配置类型
from torch.utils.data import Dataset  # 注释：PyTorch 数据集基类
from transformers import PreTrainedTokenizer, ProcessorMixin  # 注释：tokenizer/processor 基类
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
import verl.utils.torch_functional as verl_F  # 注释：封装的张量后处理函数
from verl.utils.model import compute_position_id_with_mask  # 注释：根据 attention_mask 生成 position_ids
# （分隔说明：日志器）  # 注释：替代空行，保持逐行注释
logger = logging.getLogger(__name__)  # 注释：模块级 logger
# （分隔说明：批处理函数）  # 注释：替代空行，保持逐行注释
def collate_fn(data_list: list[dict]) -> dict:  # 注释：将样本列表聚合为 batch
    """
    函数用途：将样本字典列表整理为批量张量/数组。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - data_list (list[dict])：每条样本为字段->值的字典。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict：张量字段堆叠为 (B, ...) 的 torch.Tensor，非张量字段转为 object ndarray。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若 data_list 为空，torch.stack 会报错。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - 输入：[{"input_ids": tensor([1,2]), "id": "a"}, {"input_ids": tensor([3,4]), "id": "b"}]。  # 注释：示例输入
    - 输出：{"input_ids": tensor([[1,2],[3,4]]), "id": np.array(["a","b"], dtype=object)}。  # 注释：示例输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/rl_dataset.py::collate_fn。  # 注释：函数位置
    - 典型调用路径：main_ppo.py -> DataLoader(collate_fn=collate_fn)。  # 注释：典型调用链
    - 被谁调用：DataLoader（训练/评测/测试）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.stack、numpy.fromiter。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    tensors = defaultdict(list)  # 注释：按字段收集张量值
    non_tensors = defaultdict(list)  # 注释：按字段收集非张量值
    # （分隔说明：遍历样本并分类）  # 注释：替代空行，保持逐行注释
    for data in data_list:  # 注释：遍历每条样本
        for key, val in data.items():  # 注释：遍历样本字段
            if isinstance(val, torch.Tensor):  # 注释：张量字段
                tensors[key].append(val)  # 注释：收集张量
            else:  # 注释：非张量字段
                non_tensors[key].append(val)  # 注释：收集非张量
    # （分隔说明：堆叠张量字段）  # 注释：替代空行，保持逐行注释
    for key, val in tensors.items():  # 注释：遍历张量字段
        tensors[key] = torch.stack(val, dim=0)  # 注释：沿 batch 维度堆叠
    # （分隔说明：处理非张量字段）  # 注释：替代空行，保持逐行注释
    for key, val in non_tensors.items():  # 注释：遍历非张量字段
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))  # 注释：转为 object 数组
    # （分隔说明：返回合并字典）  # 注释：替代空行，保持逐行注释
    return {**tensors, **non_tensors}  # 注释：合并并返回 batch
# （分隔说明：RLHFDataset 定义）  # 注释：替代空行，保持逐行注释
class RLHFDataset(Dataset):  # 注释：定义 RLHF/RL 数据集类
    """
    类用途：读取 parquet 数据并生成可供 RL rollout 的模型输入字段。  # 注释：类用途说明

    功能概览：  # 注释：功能概览标题
    - 缓存文件到本地（支持 shm）。  # 注释：功能点说明
    - 使用 HF datasets 读取 parquet 并拼接。  # 注释：功能点说明
    - 进行 chat_template 渲染与 tokenizer/processor 编码。  # 注释：功能点说明
    - 支持多模态图像/视频与工具 schema。  # 注释：功能点说明
    - 可过滤超长 prompt 并支持断点恢复。  # 注释：功能点说明

    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/dataset/rl_dataset.py::RLHFDataset。  # 注释：类位置
    - 典型调用路径：main_ppo.py -> create_rl_dataset -> RLHFDataset。  # 注释：典型调用链
    - 被谁调用：verl/trainer/main_ppo.py、tests/utils/dataset/test_rl_dataset_on_cpu.py。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.model.compute_position_id_with_mask、verl.utils.torch_functional.postprocess_data 等。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：datasets.load_dataset、tokenizer.apply_chat_template、processor(...)。  # 注释：外部依赖说明
    """  # 注释：类 docstring 结束
    def __init__(  # 注释：初始化数据集
        self,  # 注释：实例本身
        data_files: str | list[str],  # 注释：parquet 文件路径（单个或列表）
        tokenizer: PreTrainedTokenizer,  # 注释：文本 tokenizer
        config: DictConfig,  # 注释：数据集配置
        processor: Optional[ProcessorMixin] = None,  # 注释：多模态 processor（可选）
        max_samples: int = -1,  # 注释：最大采样数量（-1 表示全部）
    ):  # 注释：参数列表结束
        """
        函数用途：创建 RL 数据集实例并加载数据。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - data_files (str|list[str])：parquet 文件路径。  # 注释：参数含义
        - tokenizer (PreTrainedTokenizer)：用于文本编码。  # 注释：参数含义
        - config (DictConfig)：数据集配置（prompt_key/max_prompt_length 等）。  # 注释：参数含义
        - processor (ProcessorMixin|None)：多模态 processor，可选。  # 注释：参数含义
        - max_samples (int)：截取样本数，-1 表示不截取。  # 注释：参数含义
        返回：无（构造方法）。  # 注释：返回值说明
        副作用：  # 注释：副作用说明标题
        - 可能下载/缓存 parquet 文件到 cache_dir。  # 注释：副作用说明
        - 会读取并缓存 datasets.Dataset 到内存。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - parquet 读取失败会抛出 datasets 异常。  # 注释：异常说明
        - 配置项缺失可能导致 KeyError/AssertionError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：RLHFDataset(["train.parquet"], tokenizer, cfg.data)。  # 注释：示例输入
        - 输出：dataset 实例，可用于 DataLoader。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::RLHFDataset.__init__。  # 注释：函数位置
        - 典型调用路径：main_ppo.py -> create_rl_dataset -> RLHFDataset(...)。  # 注释：典型调用链
        - 被谁调用：verl/trainer/main_ppo.py。  # 注释：调用方说明
        - 调用了谁（项目内）：self._download、self._read_files_and_tokenize。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：datasets.load_dataset。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if not isinstance(data_files, list | ListConfig):  # 注释：若输入不是列表
            data_files = [data_files]  # 注释：统一转为列表
        # （分隔说明：保存配置与依赖）  # 注释：替代空行，保持逐行注释
        self.data_files = copy.deepcopy(data_files)  # 注释：保存可变列表（后续可能修改）
        self.original_data_files = copy.deepcopy(data_files)  # 注释：保留原始路径用于 resume
        self.tokenizer = tokenizer  # 注释：保存 tokenizer
        self.processor = processor  # 注释：保存 processor（可为 None）
        self.max_samples = max_samples  # 注释：保存最大样本数限制
        self.config = config  # 注释：保存配置
        # （分隔说明：解析配置项）  # 注释：替代空行，保持逐行注释
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))  # 注释：缓存目录
        self.prompt_key = config.get("prompt_key", "prompt")  # 注释：样本中 prompt 字段名
        self.image_key = config.get("image_key", "images")  # 注释：样本中图像字段名
        self.video_key = config.get("video_key", "videos")  # 注释：样本中视频字段名
        self.image_patch_size = config.get("image_patch_size", 14)  # 注释：图像 patch 大小
        self.max_prompt_length = config.get("max_prompt_length", 1024)  # 注释：prompt 最大长度
        self.return_raw_chat = config.get("return_raw_chat", False)  # 注释：是否返回原始对话结构
        self.return_full_prompt = config.get("return_full_prompt", False)  # 注释：是否返回完整 prompt 字符串
        self.truncation = config.get("truncation", "error")  # 注释：截断策略（left/right/middle/error）
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)  # 注释：是否过滤过长 prompt
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})  # 注释：chat_template 的额外参数
        # （分隔说明：工具 schema 配置）  # 注释：替代空行，保持逐行注释
        self.tool_config_path = config.get("tool_config_path", None)  # 注释：工具配置文件路径
        self.tool_schemas = None  # 注释：工具 schema 列表（初始化为空）
        if self.tool_config_path:  # 注释：若配置了工具 schema
            try:  # 注释：尝试初始化工具
                from verl.tools.utils.tool_registry import initialize_tools_from_config  # 注释：导入工具注册函数
                # （分隔说明：加载工具列表）  # 注释：替代空行，保持逐行注释
                tool_list = initialize_tools_from_config(self.tool_config_path)  # 注释：根据配置初始化工具
                # match ToolAgentLoop behaviour: model_dump to plain dicts  # 注释：保持与 ToolAgentLoop 一致的输出格式
                self.tool_schemas = [  # 注释：生成工具 schema 列表
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list  # 注释：转为 dict
                ]  # 注释：工具 schema 列表结束
            except Exception as e:  # 注释：工具初始化失败时
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)  # 注释：记录警告
                self.tool_schemas = None  # 注释：回退为空
        # （分隔说明：过滤与采样相关配置）  # 注释：替代空行，保持逐行注释
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))  # 注释：过滤线程数
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None  # 注释：限制最大线程数
        self.use_shm = config.get("use_shm", False)  # 注释：是否使用共享内存缓存
        self.chat_template_func = config.get("chat_template_func", None)  # 注释：可选的模板函数名（当前未使用）
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)  # 注释：是否需要 tools_kwargs
        self.filter_prompts = config.get("filter_prompts", True)  # 注释：是否启用 prompt 过滤
        self.serialize_dataset = False  # 注释：是否序列化 dataset（断点恢复用）
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)  # 注释：是否返回多模态 inputs
        self.shuffle = config.get("shuffle", False)  # 注释：是否对样本随机抽样
        self.seed = config.get("seed")  # 注释：随机种子
        # （分隔说明：执行下载与读取）  # 注释：替代空行，保持逐行注释
        self._download()  # 注释：下载/缓存 parquet 文件
        self._read_files_and_tokenize()  # 注释：读取 parquet 并过滤/截取
    # （分隔说明：内部下载方法）  # 注释：替代空行，保持逐行注释
    def _download(self, use_origin_parquet=False):  # 注释：将 parquet 文件缓存到本地
        """
        函数用途：把远端 parquet 拷贝到本地缓存目录（支持 shm）。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - use_origin_parquet (bool)：是否使用原始路径（用于 resume）。  # 注释：参数含义
        返回：无。  # 注释：返回值说明
        副作用：  # 注释：副作用说明标题
        - 会在 cache_dir 中创建/写入缓存文件。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 远端路径不可达会抛出 I/O 异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：_download() 将 data_files 缓存至 ~/.cache/verl/rlhf。  # 注释：示例输入
        - 输出：self.data_files 替换为本地路径。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::_download。  # 注释：函数位置
        - 典型调用路径：RLHFDataset.__init__ -> _download。  # 注释：典型调用链
        - 被谁调用：RLHFDataset.__init__、resume_dataset_state。  # 注释：调用方说明
        - 调用了谁（项目内）：verl.utils.fs.copy_to_local。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        from verl.utils.fs import copy_to_local  # 注释：本地/远端通用拷贝
        # （分隔说明：选择使用的 parquet 列表）  # 注释：替代空行，保持逐行注释
        data_files = self.data_files if not use_origin_parquet else self.original_data_files  # 注释：选择路径来源
        for i, parquet_file in enumerate(data_files):  # 注释：遍历每个 parquet 文件
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)  # 注释：拷贝并更新路径
    # （分隔说明：读取 parquet 并处理）  # 注释：替代空行，保持逐行注释
    def _read_files_and_tokenize(self):  # 注释：读取 parquet 文件并构建 datasets.Dataset
        """
        函数用途：读取 parquet 文件、拼接成统一 Dataset，并按配置过滤样本。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：无（结果保存在 self.dataframe）。  # 注释：返回值说明
        副作用：  # 注释：副作用说明标题
        - 会修改 self.dataframe 并可能进行随机抽样。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - parquet 文件损坏会导致 datasets.load_dataset 抛错。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：self.data_files=["a.parquet","b.parquet"]。  # 注释：示例输入
        - 输出：self.dataframe 为拼接后的 Dataset。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::_read_files_and_tokenize。  # 注释：函数位置
        - 典型调用路径：RLHFDataset.__init__ -> _read_files_and_tokenize。  # 注释：典型调用链
        - 被谁调用：RLHFDataset.__init__、resume_dataset_state。  # 注释：调用方说明
        - 调用了谁（项目内）：maybe_filter_out_long_prompts。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：datasets.load_dataset、datasets.concatenate_datasets。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        dataframes = []  # 注释：缓存每个 parquet 的 Dataset
        for parquet_file in self.data_files:  # 注释：遍历 parquet 文件
            # read parquet files and cache  # 注释：原注释，说明读取 parquet
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]  # 注释：读取 parquet 并取 train split
            dataframes.append(dataframe)  # 注释：追加到列表
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)  # 注释：拼接多个 Dataset
        # （分隔说明：统计样本数）  # 注释：替代空行，保持逐行注释
        total = len(self.dataframe)  # 注释：总样本数
        print(f"dataset len: {len(self.dataframe)}")  # 注释：打印样本数
        # （分隔说明：按 max_samples 截取）  # 注释：替代空行，保持逐行注释
        if self.max_samples > 0 and self.max_samples < total:  # 注释：若需要截取样本
            if self.shuffle:  # 注释：若启用随机抽样
                rngs_args = (self.seed,) if self.seed is not None else ()  # 注释：随机种子参数
                rng = np.random.default_rng(*rngs_args)  # 注释：创建随机数生成器
                indices = rng.choice(total, size=self.max_samples, replace=False)  # 注释：随机选择索引
            else:  # 注释：不随机，取前 max_samples
                indices = np.arange(self.max_samples)  # 注释：生成顺序索引
            self.dataframe = self.dataframe.select(indices.tolist())  # 注释：选择子集
            print(f"selected {self.max_samples} random samples out of {total}")  # 注释：打印抽样信息
        # （分隔说明：过滤超长 prompt）  # 注释：替代空行，保持逐行注释
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)  # 注释：过滤超长样本
    # （分隔说明：过滤超长 prompt 方法）  # 注释：替代空行，保持逐行注释
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):  # 注释：过滤超长 prompt 样本
        """
        函数用途：根据 max_prompt_length 过滤超长 prompt。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - dataframe (datasets.Dataset|None)：待过滤的 Dataset。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - datasets.Dataset：过滤后的 Dataset。  # 注释：返回值语义
        副作用：无（返回新 Dataset）。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - 若样本解析异常，将被跳过并打印 traceback。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：max_prompt_length=4，某样本 token 长度为 6。  # 注释：示例输入
        - 输出：该样本被过滤掉。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::maybe_filter_out_long_prompts。  # 注释：函数位置
        - 典型调用路径：_read_files_and_tokenize -> maybe_filter_out_long_prompts。  # 注释：典型调用链
        - 被谁调用：RLHFDataset._read_files_and_tokenize。  # 注释：调用方说明
        - 调用了谁（项目内）：self._build_messages、verl.utils.dataset.vision_utils.process_image/process_video（可选）。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：datasets.Dataset.filter、tokenizer.apply_chat_template。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        # filter out too long prompts  # 注释：原注释，说明过滤逻辑
        if self.filter_overlong_prompts:  # 注释：仅在配置允许时过滤
            tokenizer = self.tokenizer  # 注释：本地变量引用 tokenizer
            processor = self.processor  # 注释：本地变量引用 processor
            prompt_key = self.prompt_key  # 注释：本地变量引用 prompt_key
            image_key = self.image_key  # 注释：本地变量引用 image_key
            video_key = self.video_key  # 注释：本地变量引用 video_key
            # （分隔说明：多模态处理路径）  # 注释：替代空行，保持逐行注释
            if processor is not None:  # 注释：若为多模态处理器
                from verl.utils.dataset.vision_utils import process_image, process_video  # 注释：导入图像/视频预处理
                # （分隔说明：定义样本长度计算函数）  # 注释：替代空行，保持逐行注释
                def doc2len(doc) -> int:  # 注释：计算单条样本 token 长度
                    try:  # 注释：捕获异常以避免中断过滤
                        messages = self._build_messages(doc)  # 注释：构造消息列表
                        # pass tool schemas if available so the processor can format prompts  # 注释：原注释，传入工具 schema
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)  # 注释：复制模板参数
                        if self.tool_schemas is not None:  # 注释：若有工具 schema
                            apply_kwargs["tools"] = self.tool_schemas  # 注释：注入 tools 参数
                        # （分隔说明：生成原始 prompt 文本）  # 注释：替代空行，保持逐行注释
                        raw_prompt = self.processor.apply_chat_template(  # 注释：生成 prompt 字符串
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs  # 注释：模板参数
                        )  # 注释：结束 apply_chat_template
                        if image_key in doc and doc[image_key]:  # 注释：若样本包含图像
                            images = [  # 注释：处理图像列表
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]  # 注释：逐个处理图像
                            ]  # 注释：图像列表结束
                        else:  # 注释：无图像
                            images = None  # 注释：置空图像
                        # （分隔说明：处理视频输入）  # 注释：替代空行，保持逐行注释
                        if video_key in doc and doc[video_key]:  # 注释：若样本包含视频
                            videos, video_metadata = zip(  # 注释：处理视频并收集元信息
                                *[  # 注释：展开生成器
                                    process_video(  # 注释：处理单个视频
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True  # 注释：返回元信息
                                    )  # 注释：process_video 调用结束
                                    for video in doc[video_key]  # 注释：遍历视频列表
                                ],  # 注释：列表推导结束
                                strict=True,  # 注释：确保 zip 等长
                            )  # 注释：zip 结束
                            videos = list(videos)  # 注释：转换为列表
                            video_metadata = list(video_metadata)  # 注释：转换为列表
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}  # 注释：视频额外参数
                        else:  # 注释：无视频
                            videos = None  # 注释：置空视频
                            videos_kwargs = {}  # 注释：空参数
                        # （分隔说明：计算 token 长度）  # 注释：替代空行，保持逐行注释
                        return len(  # 注释：返回 token 长度
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[  # 注释：调用 processor
                                "input_ids"  # 注释：取 input_ids 字段
                            ][0]  # 注释：取第一条样本
                        )  # 注释：len 结束
                    except Exception:  # 注释：捕获任何异常
                        print("Error processing one of the samples, skipping...")  # 注释：打印提示
                        traceback.print_exc()  # 注释：输出异常堆栈
                        return self.max_prompt_length + 1  # 注释：返回超长以便过滤
            # （分隔说明：纯文本处理路径）  # 注释：替代空行，保持逐行注释
            else:  # 注释：无 processor（纯文本）
                def doc2len(doc) -> int:  # 注释：计算纯文本 prompt 长度
                    try:  # 注释：捕获异常以避免中断
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)  # 注释：复制模板参数
                        if self.tool_schemas is not None:  # 注释：若有工具 schema
                            apply_kwargs["tools"] = self.tool_schemas  # 注释：注入 tools 参数
                        return len(  # 注释：返回 token 长度
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)  # 注释：模板渲染并 tokenize
                        )  # 注释：len 结束
                    except Exception:  # 注释：捕获异常
                        print("Error processing one of the samples, skipping...")  # 注释：打印提示
                        traceback.print_exc()  # 注释：输出异常堆栈
                        return self.max_prompt_length + 1  # 注释：返回超长以便过滤
            # （分隔说明：执行过滤）  # 注释：替代空行，保持逐行注释
            dataframe = dataframe.filter(  # 注释：调用 datasets 过滤
                lambda doc: doc2len(doc) <= self.max_prompt_length,  # 注释：过滤条件
                num_proc=self.num_workers,  # 注释：并行进程数
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",  # 注释：进度条描述
            )  # 注释：filter 结束
            # （分隔说明：打印过滤结果）  # 注释：替代空行，保持逐行注释
            print(f"filter dataset len: {len(dataframe)}")  # 注释：输出过滤后样本数
        return dataframe  # 注释：返回过滤后的数据集
    # （分隔说明：断点恢复方法）  # 注释：替代空行，保持逐行注释
    def resume_dataset_state(self):  # 注释：从断点恢复数据集
        """
        函数用途：根据序列化状态恢复 dataset（用于训练恢复）。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：无。  # 注释：返回值说明
        副作用：可能重新下载 parquet 并重建 dataframe。  # 注释：副作用说明
        异常/边界条件：若旧 checkpoint 仅保存 data.pt，则提示从头训练。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：resume_dataset_state()。  # 注释：示例输入
        - 输出：self.dataframe 被恢复或打印提示。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::resume_dataset_state。  # 注释：函数位置
        - 典型调用路径：trainer -> dataset.resume_dataset_state。  # 注释：典型调用链
        - 被谁调用：训练恢复流程（如 checkpoint manager）。  # 注释：调用方说明
        - 调用了谁（项目内）：self._download、self._read_files_and_tokenize。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        self.serialize_dataset = not hasattr(self, "original_data_files")  # 注释：判断是否有原始路径
        # resume dataframe if not it's serialized in data.pt  # 注释：原注释，说明恢复逻辑
        if not self.serialize_dataset:  # 注释：若可恢复原 parquet
            self._download(use_origin_parquet=True)  # 注释：从原 parquet 重新下载
            self._read_files_and_tokenize()  # 注释：重新构建 dataframe
        else:  # 注释：仅有序列化数据
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")  # 注释：提示从头训练
    # （分隔说明：长度接口）  # 注释：替代空行，保持逐行注释
    def __len__(self):  # 注释：返回数据集长度
        """
        函数用途：返回数据集样本数。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：int，样本数。  # 注释：返回值说明
        副作用：无。  # 注释：副作用说明
        异常/边界条件：self.dataframe 未初始化会报错。  # 注释：边界说明
        最小示例：len(dataset) -> 1000。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::__len__。  # 注释：函数位置
        - 典型调用路径：DataLoader -> len(dataset)。  # 注释：典型调用链
        - 被谁调用：PyTorch DataLoader / 训练循环。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        return len(self.dataframe)  # 注释：返回 dataframe 长度
    # （分隔说明：构造消息列表）  # 注释：替代空行，保持逐行注释
    def _build_messages(self, example: dict):  # 注释：将样本转换为消息列表
        """
        函数用途：从样本中取出 prompt 并处理图像/视频占位符。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - example (dict)：包含 prompt/image/video 字段的样本。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - messages (list)：符合 chat_template 输入格式的消息列表。  # 注释：返回值语义
        副作用：会从 example 中 pop prompt_key 对应字段。  # 注释：副作用说明
        异常/边界条件：若 prompt 结构异常可能抛 KeyError。  # 注释：边界说明
        最小示例：  # 注释：最小示例标题
        - 输入：{"prompt": [{"role":"user","content":"hi <image>"}], "images": [...]}。  # 注释：示例输入
        - 输出：content 被拆成 [{"type":"text"},{"type":"image"}] 列表。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::_build_messages。  # 注释：函数位置
        - 典型调用路径：__getitem__ / maybe_filter_out_long_prompts -> _build_messages。  # 注释：典型调用链
        - 被谁调用：RLHFDataset.__getitem__、RLHFDataset.maybe_filter_out_long_prompts。  # 注释：调用方说明
        - 调用了谁（项目内）：re.split。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        messages: list = example.pop(self.prompt_key)  # 注释：弹出 prompt 字段作为消息列表
        # （分隔说明：处理多模态占位符）  # 注释：替代空行，保持逐行注释
        if self.image_key in example or self.video_key in example:  # 注释：若样本含图像或视频
            for message in messages:  # 注释：遍历每条消息
                content = message["content"]  # 注释：原始 content 字符串
                content_list = []  # 注释：拆分后的 content 列表
                segments = re.split("(<image>|<video>)", content)  # 注释：按占位符切分
                segments = [item for item in segments if item != ""]  # 注释：过滤空片段
                for segment in segments:  # 注释：遍历片段
                    if segment == "<image>":  # 注释：图像占位符
                        content_list.append({"type": "image"})  # 注释：添加图像类型标记
                    elif segment == "<video>":  # 注释：视频占位符
                        content_list.append({"type": "video"})  # 注释：添加视频类型标记
                    else:  # 注释：普通文本片段
                        content_list.append({"type": "text", "text": segment})  # 注释：添加文本片段
                # （分隔说明：写回内容列表）  # 注释：替代空行，保持逐行注释
                message["content"] = content_list  # 注释：替换为结构化 content
        return messages  # 注释：返回消息列表
    # （分隔说明：取样本接口）  # 注释：替代空行，保持逐行注释
    def __getitem__(self, item):  # 注释：根据索引返回样本
        """
        函数用途：读取单条样本并构造模型输入字段。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - item (int)：样本索引。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - dict：包含 input_ids/attention_mask/position_ids/raw_prompt 等字段。  # 注释：返回值语义
        副作用：  # 注释：副作用说明标题
        - 会 pop 掉样本中的 image/video/prompt 字段。  # 注释：副作用说明
        异常/边界条件：  # 注释：异常说明标题
        - prompt 超过 max_prompt_length 且 truncation=error 会抛出 RuntimeError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：dataset[0]，prompt="hi"。  # 注释：示例输入
        - 输出：{"input_ids": tensor(...), "attention_mask": tensor(...), ...}。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::__getitem__。  # 注释：函数位置
        - 典型调用路径：DataLoader -> RLHFDataset.__getitem__。  # 注释：典型调用链
        - 被谁调用：torch.utils.data.DataLoader。  # 注释：调用方说明
        - 调用了谁（项目内）：_build_messages、verl_F.postprocess_data、compute_position_id_with_mask。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：tokenizer/processor.apply_chat_template、tokenizer.encode。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """  # 注释：保留原英文注释（说明返回 raw_input_ids 用途）
        row_dict: dict = self.dataframe[item]  # 注释：取出样本字典
        messages = self._build_messages(row_dict)  # 注释：构造消息列表
        model_inputs = {}  # 注释：初始化模型输入字典
        # （分隔说明：多模态处理路径）  # 注释：替代空行，保持逐行注释
        if self.processor is not None:  # 注释：若有多模态 processor
            from verl.utils.dataset.vision_utils import process_image, process_video  # 注释：导入图像/视频处理函数
            # （分隔说明：生成 raw_prompt）  # 注释：替代空行，保持逐行注释
            raw_prompt = self.processor.apply_chat_template(  # 注释：生成 prompt 字符串
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs  # 注释：模板参数
            )  # 注释：apply_chat_template 结束
            multi_modal_data = {}  # 注释：多模态原始数据容器
            # （分隔说明：处理图像）  # 注释：替代空行，保持逐行注释
            images = None  # 注释：默认无图像
            row_dict_images = row_dict.pop(self.image_key, None)  # 注释：弹出图像字段
            if row_dict_images:  # 注释：若图像存在
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]  # 注释：处理图像列表
                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here  # 注释：原注释，说明字段名差异
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # 注释：参考链接
                multi_modal_data["image"] = images  # 注释：保存到 multi_modal_data（vLLM 期望的 key）
            # （分隔说明：处理视频）  # 注释：替代空行，保持逐行注释
            videos = None  # 注释：默认无视频
            videos_kwargs = {}  # 注释：视频额外参数
            row_dict_videos = row_dict.pop(self.video_key, None)  # 注释：弹出视频字段
            if row_dict_videos:  # 注释：若视频存在
                videos, video_metadata = zip(  # 注释：处理视频并收集元数据
                    *[  # 注释：展开视频列表
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)  # 注释：处理单个视频
                        for video in row_dict_videos  # 注释：遍历视频列表
                    ],  # 注释：列表推导结束
                    strict=True,  # 注释：确保 zip 等长
                )  # 注释：zip 结束
                videos = list(videos)  # 注释：转换为列表
                video_metadata = list(video_metadata)  # 注释：转换为列表
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}  # 注释：视频处理参数
                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here  # 注释：原注释，说明字段名差异
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # 注释：参考链接
                multi_modal_data["video"] = [  # 注释：保存视频数据（vLLM 期望的 key）
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)  # 注释：转换为 numpy 并带元数据
                ]  # 注释：视频列表结束
            # （分隔说明：调用 processor 编码）  # 注释：替代空行，保持逐行注释
            model_inputs = self.processor(  # 注释：调用 processor 编码
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"  # 注释：输入与输出格式
            )  # 注释：processor 调用结束
            input_ids = model_inputs.pop("input_ids")  # 注释：取出 input_ids
            attention_mask = model_inputs.pop("attention_mask")  # 注释：取出 attention_mask
            # （分隔说明：清理不需要的字段）  # 注释：替代空行，保持逐行注释
            if "second_per_grid_ts" in model_inputs:  # 注释：若存在 second_per_grid_ts
                model_inputs.pop("second_per_grid_ts")  # 注释：移除该字段
            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature  # 注释：原注释，提醒类型问题
            row_dict["multi_modal_data"] = multi_modal_data  # 注释：保存原始多模态数据
            # We will do batch.union() in the trainer,  # 注释：原注释，说明后续合并逻辑
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs  # 注释：原注释，避免覆盖
            if self.return_multi_modal_inputs:  # 注释：若需要返回多模态输入
                row_dict["multi_modal_inputs"] = dict(model_inputs)  # 注释：转换为普通 dict
                # second_per_grid_ts isn't used for training, just for mrope  # 注释：原注释，说明字段用途
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)  # 注释：移除该字段
        # （分隔说明：纯文本处理路径）  # 注释：替代空行，保持逐行注释
        else:  # 注释：无 processor（纯文本）
            if self.apply_chat_template_kwargs.get("chat_template") is None:  # 注释：若未传入 chat_template
                assert hasattr(self.tokenizer, "chat_template"), (  # 注释：要求 tokenizer 自带模板
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "  # 注释：断言信息
                    "models like GLM can copy chat_template.jinja from instruct models"  # 注释：断言信息
                )  # 注释：断言结束
            raw_prompt = self.tokenizer.apply_chat_template(  # 注释：生成 prompt 字符串
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs  # 注释：模板参数
            )  # 注释：apply_chat_template 结束
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)  # 注释：tokenizer 编码
            input_ids = model_inputs.pop("input_ids")  # 注释：取出 input_ids
            attention_mask = model_inputs.pop("attention_mask")  # 注释：取出 attention_mask
        # （分隔说明：统一后处理）  # 注释：替代空行，保持逐行注释
        input_ids, attention_mask = verl_F.postprocess_data(  # 注释：统一 padding/截断处理
            input_ids=input_ids,  # 注释：输入 ids
            attention_mask=attention_mask,  # 注释：输入 mask
            max_length=self.max_prompt_length,  # 注释：最大长度
            pad_token_id=self.tokenizer.pad_token_id,  # 注释：padding token id
            left_pad=True,  # 注释：左侧 padding
            truncation=self.truncation,  # 注释：截断策略
        )  # 注释：postprocess_data 结束
        # （分隔说明：计算 position_ids）  # 注释：替代空行，保持逐行注释
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:  # 注释：Qwen2/3 VL 处理器
            # qwen-vl mrope  # 注释：原注释，说明 mrope 处理
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:  # 注释：区分 Qwen3
                from verl.models.transformers.qwen3_vl import get_rope_index  # 注释：导入 Qwen3 VL rope
            else:  # 注释：Qwen2 VL
                from verl.models.transformers.qwen2_vl import get_rope_index  # 注释：导入 Qwen2 VL rope
            # （分隔说明：计算视觉 position_ids）  # 注释：替代空行，保持逐行注释
            vision_position_ids = get_rope_index(  # 注释：计算视觉 position ids
                self.processor,  # 注释：processor
                input_ids=input_ids[0],  # 注释：单条 input_ids
                image_grid_thw=model_inputs.get("image_grid_thw"),  # 注释：图像网格
                video_grid_thw=model_inputs.get("video_grid_thw"),  # 注释：视频网格
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),  # 注释：视频时间信息
                attention_mask=attention_mask[0],  # 注释：单条 mask
            )  # 注释：get_rope_index 结束
            valid_mask = attention_mask[0].bool()  # 注释：有效 token mask
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)  # 注释：初始化文本 position ids
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())  # 注释：填充有效位置
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # 注释：拼接文本+视觉 position ids
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:  # 注释：GLM4V 处理器
            from verl.models.transformers.glm4v import get_rope_index  # 注释：导入 GLM4V rope
            vision_position_ids = get_rope_index(  # 注释：计算视觉 position ids
                self.processor,  # 注释：processor
                input_ids=input_ids[0],  # 注释：单条 input_ids
                image_grid_thw=model_inputs.get("image_grid_thw"),  # 注释：图像网格
                video_grid_thw=model_inputs.get("video_grid_thw"),  # 注释：视频网格
                attention_mask=attention_mask[0],  # 注释：单条 mask
            )  # 注释：get_rope_index 结束
            valid_mask = attention_mask[0].bool()  # 注释：有效 token mask
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)  # 注释：初始化文本 position ids
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())  # 注释：填充有效位置
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # 注释：拼接 position ids
        else:  # 注释：纯文本或非特殊处理器
            position_ids = compute_position_id_with_mask(attention_mask)  # 注释：根据 mask 生成 position ids
        # （分隔说明：写回核心字段）  # 注释：替代空行，保持逐行注释
        row_dict["input_ids"] = input_ids[0]  # 注释：保存 input_ids
        row_dict["attention_mask"] = attention_mask[0]  # 注释：保存 attention_mask
        row_dict["position_ids"] = position_ids[0]  # 注释：保存 position_ids
        # （分隔说明：计算 raw_prompt_ids）  # 注释：替代空行，保持逐行注释
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)  # 注释：编码 raw_prompt
        if len(raw_prompt_ids) > self.max_prompt_length:  # 注释：若超长需要截断
            if self.truncation == "left":  # 注释：左截断
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]  # 注释：保留尾部
            elif self.truncation == "right":  # 注释：右截断
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]  # 注释：保留头部
            elif self.truncation == "middle":  # 注释：中间截断
                left_half = self.max_prompt_length // 2  # 注释：左半长度
                right_half = self.max_prompt_length - left_half  # 注释：右半长度
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]  # 注释：拼接两端
            elif self.truncation == "error":  # 注释：错误模式
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")  # 注释：抛出异常
        # （分隔说明：保存原始 prompt 信息）  # 注释：替代空行，保持逐行注释
        row_dict["raw_prompt_ids"] = raw_prompt_ids  # 注释：保存 raw_prompt_ids
        # encode prompts without chat template  # 注释：原注释，说明原始对话返回
        if self.return_raw_chat:  # 注释：若需要返回 raw chat
            row_dict["raw_prompt"] = messages  # 注释：保存消息列表
        # get prompts with chat template  # 注释：原注释，说明返回完整 prompt
        if self.return_full_prompt:  # 注释：若需要返回完整 prompt
            row_dict["full_prompts"] = raw_prompt  # 注释：保存完整 prompt 字符串
        # add index for each prompt  # 注释：原注释，说明索引与额外信息
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:  # 注释：若无 extra_info 字段
            row_dict["extra_info"] = dict()  # 注释：初始化空字典
        index = row_dict.get("extra_info", {}).get("index", 0)  # 注释：读取样本索引
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})  # 注释：读取工具参数
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})  # 注释：读取交互参数
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)  # 注释：是否需要 tools_kwargs
        if need_tools_kwargs and not tools_kwargs:  # 注释：需要工具参数但为空
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])  # 注释：记录警告
        row_dict["index"] = index  # 注释：写回索引字段
        row_dict["tools_kwargs"] = tools_kwargs  # 注释：写回工具参数
        row_dict["interaction_kwargs"] = interaction_kwargs  # 注释：写回交互参数
        return row_dict  # 注释：返回样本字典
    # （分隔说明：序列化支持）  # 注释：替代空行，保持逐行注释
    def __getstate__(self):  # 注释：pickle 时控制序列化内容
        """
        函数用途：自定义 pickle 序列化，避免保存大 dataframe。  # 注释：函数用途说明
        参数：无。  # 注释：参数说明
        返回：dict，序列化状态。  # 注释：返回值说明
        副作用：无。  # 注释：副作用说明
        异常/边界条件：无。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：pickle.dumps(dataset)。  # 注释：示例输入
        - 输出：state 中不包含 dataframe。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/utils/dataset/rl_dataset.py::__getstate__。  # 注释：函数位置
        - 典型调用路径：pickle -> __getstate__。  # 注释：典型调用链
        - 被谁调用：pickle 序列化流程。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：pickle。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        if not self.serialize_dataset:  # 注释：若不序列化 dataframe
            state = self.__dict__.copy()  # 注释：复制实例字典
            # （分隔说明：移除 dataframe）  # 注释：替代空行，保持逐行注释
            if "dataframe" in state:  # 注释：若包含 dataframe
                del state["dataframe"]  # 注释：删除 dataframe 以减小体积
            return state  # 注释：返回裁剪后的状态
        # （分隔说明：序列化完整状态）  # 注释：替代空行，保持逐行注释
        return self.__dict__.copy()  # 注释：返回完整状态
