# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：空行占位，保持逐行注释
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：许可证获取提示
#  # 注释：空行占位，保持逐行注释
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位，保持逐行注释
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无担保声明
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：将 GSM8K 原始数据处理为“多轮 SFT”格式的 parquet（messages 列）。  # 注释：模块用途
输入：命令行参数（本地/远端数据路径、保存目录等）与 GSM8K 原始数据集。  # 注释：输入说明
输出：train.parquet/test.parquet（本地保存，可选同步到 HDFS）。  # 注释：输出说明
关键依赖：datasets、verl.utils.hdfs_io、os、re。  # 注释：依赖说明
典型用法：  # 注释：用法说明标题
- python examples/data_preprocess/gsm8k_multiturn_sft.py --local_save_dir ~/data/gsm8k_sft  # 注释：最小命令
调用路径概览：  # 注释：调用路径标题
- 手动执行脚本 -> load_dataset -> map -> to_parquet -> (可选) copy 到 HDFS。  # 注释：调用链路
"""  # 注释：模块 docstring 结束

import argparse  # 注释：标准库，用于解析命令行参数
import os  # 注释：标准库，用于路径与环境处理
import re  # 注释：标准库，用于正则抽取答案

import datasets  # 注释：第三方库，用于加载 HuggingFace 数据集

from verl.utils.hdfs_io import copy, makedirs  # 注释：项目内工具，用于 HDFS 目录创建与拷贝


def extract_solution(solution_str):  # 注释：提取 GSM8K 解答中的最终数值
    """
    功能：从答案字符串中解析出以 "####" 标注的最终数值。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - solution_str (str)：原始答案文本（包含 "#### 数值"）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：最终数值字符串（去掉逗号）。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若未匹配到 "####"，assert 触发异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入 "abc #### 1,234" -> 输出 "1234"。  # 注释：示例说明
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：examples/data_preprocess/gsm8k_multiturn_sft.py::extract_solution。  # 注释：函数位置
    - 典型调用路径：本脚本内调用（如需提取最终答案）。  # 注释：典型调用链
    - 被谁调用：当前文件内（未在仓库其他处引用）。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：re.search。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)  # 注释：匹配 "#### 数值" 模式
    assert solution is not None  # 注释：确保匹配成功
    final_solution = solution.group(0)  # 注释：取出匹配片段
    final_solution = final_solution.split("#### ")[1].replace(",", "")  # 注释：去掉标记与逗号
    return final_solution  # 注释：返回最终数值字符串


if __name__ == "__main__":  # 注释：脚本入口保护
    parser = argparse.ArgumentParser()  # 注释：创建命令行参数解析器
    parser.add_argument("--local_dir", default=None)  # 注释：兼容旧参数（已弃用）
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")  # 注释：本地原始数据路径
    parser.add_argument(  # 注释：添加保存目录参数
        "--local_save_dir", default="~/data/gsm8k_sft", help="The save directory for the preprocessed dataset."  # 注释：默认保存目录
    )  # 注释：参数定义结束
    parser.add_argument("--hdfs_dir", default=None)  # 注释：可选 HDFS 保存目录

    args = parser.parse_args()  # 注释：解析命令行参数
    local_dataset_path = args.local_dataset_path  # 注释：读取本地数据路径

    data_source = "openai/gsm8k"  # 注释：默认数据集来源

    if local_dataset_path is not None:  # 注释：优先使用本地数据
        dataset = datasets.load_dataset(local_dataset_path, "main")  # 注释：加载本地数据集
    else:  # 注释：未提供本地路径时
        dataset = datasets.load_dataset(data_source, "main")  # 注释：从 HF Hub 加载

    train_dataset = dataset["train"]  # 注释：训练集切分
    test_dataset = dataset["test"]  # 注释：测试集切分

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'  # 注释：追加到问题后的指令

    # add a row to each data item that represents a unique id  # 注释：原注释保留（说明 map 的目的）
    def make_map_fn(split):  # 注释：构造 map 函数（按 split 名称）
        """
        功能：构造对单条样本进行格式转换的处理函数。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - split (str)：数据切分名称（train/test）。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - callable：用于 datasets.map 的处理函数。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：输入样本缺少 question/answer 会抛 KeyError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - fn = make_map_fn("train"); fn({"question":"q","answer":"a"},0)。  # 注释：示例说明
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：examples/data_preprocess/gsm8k_multiturn_sft.py::make_map_fn。  # 注释：函数位置
        - 典型调用路径：__main__ -> dataset.map(function=make_map_fn(...))。  # 注释：调用链
        - 被谁调用：当前文件内。  # 注释：调用方
        - 调用了谁（项目内）：无。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：datasets.Dataset.map。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        def process_fn(example, idx):  # 注释：实际处理单条样本的函数
            """
            功能：将 GSM8K 样本转为多轮 messages 结构。  # 注释：函数用途
            参数：  # 注释：参数说明标题
            - example (dict)：原始样本字典。  # 注释：参数含义
            - idx (int)：样本索引（来自 with_indices）。  # 注释：参数含义
            返回：  # 注释：返回值说明标题
            - dict：包含 messages 的新样本。  # 注释：返回值语义
            副作用：会从 example 中 pop 掉 question/answer。  # 注释：副作用说明
            异常/边界条件：缺少字段会抛 KeyError。  # 注释：异常说明
            最小示例：  # 注释：最小示例标题
            - 输入 {"question":"q","answer":"a"} -> 输出 messages 列表。  # 注释：示例说明
            调用路径依赖：  # 注释：调用路径说明标题
            - 所在位置：examples/data_preprocess/gsm8k_multiturn_sft.py::process_fn。  # 注释：函数位置
            - 典型调用路径：make_map_fn -> datasets.map -> process_fn。  # 注释：调用链
            - 被谁调用：datasets.map 内部。  # 注释：调用方说明
            - 调用了谁（项目内）：无。  # 注释：项目内依赖
            - 调用了谁（关键外部依赖）：无（仅 dict 操作）。  # 注释：外部依赖
            """  # 注释：函数 docstring 结束
            question_raw = example.pop("question")  # 注释：取出原始问题

            question = question_raw + " " + instruction_following  # 注释：拼接指令

            answer_raw = example.pop("answer")  # 注释：取出原始答案
            data = {  # 注释：构造多轮 messages 结构
                "messages": [  # 注释：消息列表
                    {  # 注释：user 消息
                        "role": "user",  # 注释：角色标记
                        "content": question,  # 注释：用户问题内容
                    },  # 注释：user 消息结束
                    {  # 注释：assistant 消息
                        "role": "assistant",  # 注释：角色标记
                        "content": answer_raw,  # 注释：助手回答
                    },  # 注释：assistant 消息结束
                ],  # 注释：messages 列表结束
            }  # 注释：data 字典结束
            return data  # 注释：返回新样本

        return process_fn  # 注释：返回处理函数

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)  # 注释：转换训练集
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)  # 注释：转换测试集

    hdfs_dir = args.hdfs_dir  # 注释：读取 HDFS 目录

    local_save_dir = args.local_dir  # 注释：兼容旧参数 local_dir
    if local_save_dir is not None:  # 注释：检测旧参数
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")  # 注释：打印弃用提示
    else:  # 注释：未提供旧参数
        local_save_dir = args.local_save_dir  # 注释：使用新参数

    local_save_dir = os.path.expanduser(local_save_dir)  # 注释：展开 ~ 到绝对路径

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))  # 注释：保存训练 parquet
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))  # 注释：保存测试 parquet

    if hdfs_dir is not None:  # 注释：若提供 HDFS 目录
        makedirs(hdfs_dir)  # 注释：创建 HDFS 目录

        copy(src=local_save_dir, dst=hdfs_dir)  # 注释：拷贝到 HDFS
