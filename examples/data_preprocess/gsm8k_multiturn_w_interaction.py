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
模块用途：将 GSM8K 转为“多轮 + 交互回合”格式的 parquet（interaction_kwargs）。  # 注释：模块用途
输入：GSM8K 原始数据（本地或 HF Hub）与命令行参数。  # 注释：输入说明
输出：train/test parquet（含 prompt、reward_model、interaction_kwargs）。  # 注释：输出说明
关键依赖：datasets、verl.utils.hdfs_io、os、re。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- python examples/data_preprocess/gsm8k_multiturn_w_interaction.py --local_save_dir ~/data/gsm8k_interaction  # 注释：最小命令
调用路径概览：  # 注释：调用链标题
- 手动运行脚本 -> load_dataset -> map -> to_parquet -> (可选) copy 到 HDFS。  # 注释：调用链
"""  # 注释：模块 docstring 结束

import argparse  # 注释：标准库，解析命令行参数
import os  # 注释：标准库，路径处理
import re  # 注释：标准库，正则提取答案

import datasets  # 注释：第三方库，加载 HF 数据集

from verl.utils.hdfs_io import copy, makedirs  # 注释：项目内工具，HDFS 目录创建与拷贝


def extract_solution(solution_str):  # 注释：提取最终数值答案
    """
    功能：解析答案中的 "#### 数值" 并返回数值字符串。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - solution_str (str)：原始答案文本。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：去逗号后的数值字符串。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：未匹配到 "####" 时断言失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入 "#### 1,000" -> 输出 "1000"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：examples/data_preprocess/gsm8k_multiturn_w_interaction.py::extract_solution。  # 注释：函数位置
    - 典型调用路径：process_fn -> extract_solution。  # 注释：调用链
    - 被谁调用：当前文件内 process_fn。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：re.search。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)  # 注释：正则匹配最终答案
    assert solution is not None  # 注释：确保匹配成功
    final_solution = solution.group(0)  # 注释：取得匹配片段
    final_solution = final_solution.split("#### ")[1].replace(",", "")  # 注释：去掉前缀与逗号
    return final_solution  # 注释：返回最终数值


if __name__ == "__main__":  # 注释：脚本入口
    parser = argparse.ArgumentParser()  # 注释：创建参数解析器
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")  # 注释：旧参数（保存目录）
    parser.add_argument("--hdfs_dir", default=None)  # 注释：可选 HDFS 目录
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")  # 注释：本地原始数据路径
    parser.add_argument(  # 注释：添加保存目录参数
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."  # 注释：默认保存目录
    )  # 注释：参数定义结束

    args = parser.parse_args()  # 注释：解析参数
    local_dataset_path = args.local_dataset_path  # 注释：读取本地数据路径

    data_source = "openai/gsm8k"  # 注释：默认数据集来源

    if local_dataset_path is not None:  # 注释：若提供本地路径
        dataset = datasets.load_dataset(local_dataset_path, "main")  # 注释：加载本地数据集
    else:  # 注释：否则使用 HF Hub
        dataset = datasets.load_dataset(data_source, "main")  # 注释：加载远端数据集

    train_dataset = dataset["train"]  # 注释：训练集切分
    test_dataset = dataset["test"]  # 注释：测试集切分

    instruction_following = "Let's think step by step and output the final answer after `####`."  # 注释：追加指令

    # add a row to each data item that represents a unique id  # 注释：原注释保留（map 目的）
    def make_map_fn(split):  # 注释：构造 map 处理函数
        """
        功能：构造用于 datasets.map 的样本转换函数（含 interaction_kwargs）。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - split (str)：数据切分名称。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - callable：处理单条样本的函数。  # 注释：返回值语义
        副作用：无。  # 注释：副作用说明
        异常/边界条件：样本缺字段将抛 KeyError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - fn = make_map_fn("train"); fn({"question":"q","answer":"a"},0)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：examples/data_preprocess/gsm8k_multiturn_w_interaction.py::make_map_fn。  # 注释：函数位置
        - 典型调用路径：__main__ -> dataset.map(function=make_map_fn(...))。  # 注释：调用链
        - 被谁调用：当前文件内。  # 注释：调用方说明
        - 调用了谁（项目内）：extract_solution。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：datasets.Dataset.map。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        def process_fn(example, idx):  # 注释：处理单条样本
            """
            功能：将 GSM8K 样本转为含交互字段的训练样本。  # 注释：函数用途
            参数：  # 注释：参数说明标题
            - example (dict)：原始样本字典。  # 注释：参数含义
            - idx (int)：样本索引。  # 注释：参数含义
            返回：  # 注释：返回值说明标题
            - dict：包含 prompt 与 interaction_kwargs 的样本。  # 注释：返回值语义
            副作用：会 pop question/answer 字段。  # 注释：副作用说明
            异常/边界条件：字段缺失抛 KeyError。  # 注释：异常说明
            最小示例：  # 注释：最小示例标题
            - 输入 {"question":"q","answer":"a"} -> 输出包含 interaction_kwargs 的 dict。  # 注释：示例
            调用路径依赖：  # 注释：调用路径说明标题
            - 所在位置：examples/data_preprocess/gsm8k_multiturn_w_interaction.py::process_fn。  # 注释：函数位置
            - 典型调用路径：make_map_fn -> datasets.map -> process_fn。  # 注释：调用链
            - 被谁调用：datasets.map 内部。  # 注释：调用方说明
            - 调用了谁（项目内）：extract_solution。  # 注释：项目内依赖
            - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
            """  # 注释：函数 docstring 结束
            question_raw = example.pop("question")  # 注释：取出原始问题

            question = question_raw + " " + instruction_following  # 注释：拼接指令

            answer_raw = example.pop("answer")  # 注释：取出原始答案
            solution = extract_solution(answer_raw)  # 注释：解析最终数值
            data = {  # 注释：构造输出样本
                "data_source": data_source,  # 注释：数据来源
                "prompt": [  # 注释：prompt 列表
                    {  # 注释：system 消息
                        "role": "system",  # 注释：角色
                        "content": (  # 注释：system 提示语
                            "You are a math expert. You are given a question and you need to solve it step by step. "  # 注释：提示语片段
                            "You should rethinking carefully if user point out your answer is wrong. "  # 注释：提示语片段
                            "Put your final answer in the format of `#### <answer>`."  # 注释：提示语片段
                        ),  # 注释：system 文本结束
                    },  # 注释：system 消息结束
                    {  # 注释：user 消息
                        "role": "user",  # 注释：角色
                        "content": question,  # 注释：用户问题
                    },  # 注释：user 消息结束
                ],  # 注释：prompt 列表结束
                "ability": "math",  # 注释：能力标签
                "reward_model": {"style": "rule", "ground_truth": solution},  # 注释：规则奖励配置
                "extra_info": {  # 注释：额外信息
                    "split": split,  # 注释：切分名
                    "index": idx,  # 注释：样本索引
                    "answer": answer_raw,  # 注释：原始答案
                    "question": question_raw,  # 注释：原始问题
                    "interaction_kwargs": {  # 注释：交互参数
                        "name": "gsm8k",  # 注释：交互任务名
                        "query": question,  # 注释：问题文本
                        "ground_truth": solution,  # 注释：答案数值
                    },  # 注释：interaction_kwargs 结束
                },  # 注释：extra_info 结束
            }  # 注释：data 结束
            return data  # 注释：返回处理后样本

        return process_fn  # 注释：返回处理函数

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)  # 注释：处理训练集
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)  # 注释：处理测试集

    hdfs_dir = args.hdfs_dir  # 注释：读取 HDFS 目录
    local_save_dir = args.local_dir  # 注释：兼容旧参数 local_dir
    if local_save_dir is not None:  # 注释：旧参数存在时
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")  # 注释：打印弃用提示
    else:  # 注释：旧参数未设置
        local_save_dir = args.local_save_dir  # 注释：使用新参数

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))  # 注释：保存训练 parquet
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))  # 注释：保存测试 parquet

    if hdfs_dir is not None:  # 注释：若需要写入 HDFS
        makedirs(hdfs_dir)  # 注释：创建 HDFS 目录
        copy(src=local_save_dir, dst=hdfs_dir)  # 注释：拷贝到 HDFS
