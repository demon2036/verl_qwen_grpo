# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# Copyright 2023-2024 SGLang Team  # 注释：版权声明（SGLang）
# Copyright 2025 ModelBest Inc. and/or its affiliates  # 注释：版权声明（ModelBest）
#  # 注释：空行占位
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：许可证获取提示
#  # 注释：空行占位
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无担保声明
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：将 AIME-2024 数据集补充为“多轮 + 工具调用”格式，并保存为 parquet。  # 注释：模块用途
输入：AIME-2024 原始数据（本地或 HF Hub）与命令行参数。  # 注释：输入说明
输出：train.parquet（含 extra_info.tools_kwargs 等字段）。  # 注释：输出说明
关键依赖：datasets、verl.utils.hdfs_io、os。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- python examples/data_preprocess/aime2024_multiturn_w_tool.py --local_save_dir ~/data/retool_aime2024  # 注释：最小命令
调用路径概览：  # 注释：调用链标题
- 手动运行脚本 -> load_dataset -> map -> to_parquet -> (可选) copy 到 HDFS。  # 注释：调用链
"""  # 注释：模块 docstring 结束

import argparse  # 注释：标准库，解析命令行参数
import os  # 注释：标准库，路径处理

import datasets  # 注释：第三方库，加载 HF 数据集

from verl.utils.hdfs_io import copy, makedirs  # 注释：项目内工具，HDFS 操作

if __name__ == "__main__":  # 注释：脚本入口
    parser = argparse.ArgumentParser()  # 注释：创建参数解析器
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")  # 注释：旧参数（保存目录）
    parser.add_argument("--hdfs_dir", default=None)  # 注释：可选 HDFS 目录
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")  # 注释：本地原始数据路径
    parser.add_argument(  # 注释：添加保存目录参数
        "--local_save_dir", default="~/data/retool_aime2024", help="The save directory for the preprocessed dataset."  # 注释：默认保存目录
    )  # 注释：参数定义结束

    args = parser.parse_args()  # 注释：解析参数
    local_dataset_path = args.local_dataset_path  # 注释：读取本地数据路径

    data_path = "BytedTsinghua-SIA/AIME-2024"  # 注释：默认数据集路径

    if local_dataset_path is not None:  # 注释：优先使用本地路径
        dataset = datasets.load_dataset(local_dataset_path, "default")  # 注释：加载本地数据集
    else:  # 注释：否则使用远端路径
        dataset = datasets.load_dataset(data_path, "default")  # 注释：加载 HF 数据集

    train_dataset = dataset["train"]  # 注释：取训练集切分

    # add a row to each data item that represents a unique id  # 注释：原注释保留（map 处理说明）
    def make_map_fn(split):  # 注释：构造 map 处理函数
        """
        功能：构造样本后处理函数，补充工具调用参数。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - split (str)：数据切分名称。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - callable：处理单条样本的函数。  # 注释：返回值语义
        副作用：修改 example 的 extra_info 字段。  # 注释：副作用说明
        异常/边界条件：缺少 extra_info 字段会抛 KeyError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - fn = make_map_fn("train"); fn(example, 0)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：examples/data_preprocess/aime2024_multiturn_w_tool.py::make_map_fn。  # 注释：函数位置
        - 典型调用路径：__main__ -> dataset.map(function=make_map_fn(...))。  # 注释：调用链
        - 被谁调用：当前文件内。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：datasets.Dataset.map。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        def process_fn(example, idx):  # 注释：处理单条样本
            """
            功能：在 extra_info 中添加 tools_kwargs（code_interpreter）。  # 注释：函数用途
            参数：  # 注释：参数说明标题
            - example (dict)：样本字典。  # 注释：参数含义
            - idx (int)：样本索引。  # 注释：参数含义
            返回：  # 注释：返回值说明标题
            - dict：补充工具字段后的样本。  # 注释：返回值语义
            副作用：会修改 example 与 extra_info。  # 注释：副作用说明
            异常/边界条件：缺少字段会抛 KeyError。  # 注释：异常说明
            最小示例：  # 注释：最小示例标题
            - 输入包含 extra_info 的样本 -> 输出附带 tools_kwargs。  # 注释：示例
            调用路径依赖：  # 注释：调用路径说明标题
            - 所在位置：examples/data_preprocess/aime2024_multiturn_w_tool.py::process_fn。  # 注释：函数位置
            - 典型调用路径：make_map_fn -> datasets.map -> process_fn。  # 注释：调用链
            - 被谁调用：datasets.map 内部。  # 注释：调用方说明
            - 调用了谁（项目内）：无。  # 注释：项目内依赖
            - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
            """  # 注释：函数 docstring 结束
            orig_extra_info = example.pop("extra_info")  # 注释：取出原始 extra_info
            extra_info = orig_extra_info.copy()  # 注释：复制避免原地修改
            extra_info["need_tools_kwargs"] = True  # 注释：标记需要工具参数
            extra_info["tools_kwargs"] = {  # 注释：设置 tools_kwargs
                "code_interpreter": {  # 注释：工具名称
                    "create_kwargs": {  # 注释：创建参数
                        "ground_truth": example["reward_model"]["ground_truth"],  # 注释：传入 ground_truth
                    },  # 注释：create_kwargs 结束
                },  # 注释：code_interpreter 配置结束
            }  # 注释：tools_kwargs 结束
            example["extra_info"] = extra_info  # 注释：回填 extra_info
            return example  # 注释：返回处理后样本

        return process_fn  # 注释：返回处理函数

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)  # 注释：处理训练集

    hdfs_dir = args.hdfs_dir  # 注释：读取 HDFS 目录
    local_save_dir = args.local_dir  # 注释：兼容旧参数 local_dir
    if local_save_dir is not None:  # 注释：旧参数存在时
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")  # 注释：打印弃用提示
    else:  # 注释：旧参数未设置
        local_save_dir = args.local_save_dir  # 注释：使用新参数

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))  # 注释：保存训练 parquet

    if hdfs_dir is not None:  # 注释：若需要写入 HDFS
        makedirs(hdfs_dir)  # 注释：创建 HDFS 目录
        copy(src=local_save_dir, dst=hdfs_dir)  # 注释：拷贝到 HDFS
