# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明，表明该文件所属与年份
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行可读
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明使用 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：说明使用本文件需遵守许可证
# You may obtain a copy of the License at  # 注释：提示可在下方链接获取许可证全文
#  # 注释：保留注释符号，保证此行也有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证链接
#  # 注释：保留注释符号，保证此行也有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明的开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供任何形式担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款说明
# limitations under the License.  # 注释：许可限制说明
"""模块用途：将 GSM8K 原始数据集预处理为单轮 SFT/RL 共用的 parquet 格式。  # 注释：模块级用途说明
输入：  # 注释：模块级输入说明标题
- --local_dataset_path：本地原始数据集路径（datasets 可识别的数据源）。  # 注释：输入参数说明
- --local_save_dir/--local_dir：本地保存 parquet 的目录。  # 注释：输入参数说明
- --hdfs_dir：可选，保存/同步到 HDFS 的目标目录。  # 注释：输入参数说明
输出：  # 注释：模块级输出说明标题
- {local_save_dir}/train.parquet 与 {local_save_dir}/test.parquet。  # 注释：输出文件说明
- 若提供 hdfs_dir，则将本地目录整体拷贝到 HDFS。  # 注释：输出到 HDFS 的说明
依赖：datasets（读取 GSM8K）、re（正则提取答案）、verl.utils.hdfs_io（拷贝/建目录）。  # 注释：关键依赖说明
典型用法：  # 注释：最小运行示例标题
- python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k --hdfs_dir hdfs://path/to/gsm8k  # 注释：示例命令
调用路径概览：  # 注释：调用路径总览标题
- 入口：本文件 __main__。  # 注释：脚本入口说明
- 典型链路：__main__ -> make_map_fn(split) -> process_fn(example, idx) -> extract_solution(answer)。  # 注释：关键调用链说明
"""  # 注释：模块 docstring 结束
# （分隔说明：模块说明结束，下面开始导入依赖）  # 注释：替代空行，保持逐行注释
import argparse  # 注释：命令行参数解析
import os  # 注释：路径拼接与文件保存
import re  # 注释：正则表达式用于提取最终答案
# （分隔说明：第三方依赖导入）  # 注释：分隔本地与第三方库
import datasets  # 注释：Hugging Face datasets，用于加载 GSM8K 数据
# （分隔说明：项目内工具导入）  # 注释：分隔第三方与项目内依赖
from verl.utils.hdfs_io import copy, makedirs  # 注释：HDFS/本地通用拷贝与建目录接口
# （分隔说明：依赖导入结束）  # 注释：替代空行，保持逐行注释
def extract_solution(solution_str):  # 注释：定义从答案字符串中提取数值解的函数
    """函数用途：从包含 "####" 的 GSM8K 答案字符串中提取最终数值答案。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - solution_str (str)：原始答案文本，需包含 "#### " 前缀。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：去掉逗号后的最终数值字符串。  # 注释：返回值语义
    副作用：无。  # 注释：函数副作用说明
    异常/边界条件：  # 注释：异常与边界说明标题
    - 若未匹配到 "####" 形式的答案，将触发 AssertionError。  # 注释：潜在异常说明
    最小示例：  # 注释：最小示例标题
    - 输入："... #### 1,234" -> 输出："1234"。  # 注释：示例输入输出
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：examples/data_preprocess/gsm8k.py::extract_solution。  # 注释：函数位置
    - 典型调用路径：__main__ -> make_map_fn -> process_fn -> extract_solution。  # 注释：典型调用链
    - 被谁调用：仅在本文件内的 process_fn 调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无（仅使用标准库）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：re.search、str.split、str.replace。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)  # 注释：用正则匹配 "#### " 后的数字片段
    assert solution is not None  # 注释：若匹配失败则直接断言，防止后续处理错误
    final_solution = solution.group(0)  # 注释：取出完整匹配字符串（含 "#### " 前缀）
    final_solution = final_solution.split("#### ")[1].replace(",", "")  # 注释：去掉前缀并移除千分位逗号
    return final_solution  # 注释：返回最终数值字符串
# （分隔说明：核心工具函数定义结束）  # 注释：替代空行，保持逐行注释
if __name__ == "__main__":  # 注释：脚本入口，仅直接执行该文件时运行
    parser = argparse.ArgumentParser()  # 注释：创建命令行参数解析器
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")  # 注释：旧参数（已弃用）
    parser.add_argument("--hdfs_dir", default=None)  # 注释：HDFS 目标目录参数
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")  # 注释：本地原始数据集路径
    parser.add_argument(  # 注释：开始定义 local_save_dir 参数（新推荐）
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."  # 注释：本地保存目录
    )  # 注释：结束 local_save_dir 参数定义
    # （分隔说明：解析命令行参数）  # 注释：替代空行，保持逐行注释
    args = parser.parse_args()  # 注释：解析命令行参数
    local_dataset_path = args.local_dataset_path  # 注释：获取本地数据集路径参数
    # （分隔说明：设置数据源名称）  # 注释：替代空行，保持逐行注释
    data_source = "openai/gsm8k"  # 注释：HF datasets 上的 GSM8K 数据集名
    # （分隔说明：加载数据集）  # 注释：替代空行，保持逐行注释
    if local_dataset_path is not None:  # 注释：若提供本地路径，则优先使用本地数据集
        dataset = datasets.load_dataset(local_dataset_path, "main")  # 注释：从本地路径加载数据集
    else:  # 注释：否则从线上数据源加载
        dataset = datasets.load_dataset(data_source, "main")  # 注释：从 HF Hub 加载 GSM8K
    # （分隔说明：取出训练与测试拆分）  # 注释：替代空行，保持逐行注释
    train_dataset = dataset["train"]  # 注释：训练集
    test_dataset = dataset["test"]  # 注释：测试集
    # （分隔说明：定义统一的指令提示）  # 注释：替代空行，保持逐行注释
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'  # 注释：与 GSM8K 评测格式一致的提示
    # （分隔说明：定义 map 函数工厂，用于构造样本转换函数）  # 注释：替代空行，保持逐行注释
    def make_map_fn(split):  # 注释：根据 split 构造样本处理函数（闭包）
        """函数用途：生成一个将原始 GSM8K 样本转换为训练格式的处理函数。  # 注释：函数用途说明
        参数：  # 注释：参数说明标题
        - split (str)：数据集拆分名，如 "train" 或 "test"。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - Callable[[dict, int], dict]：供 datasets.map 调用的处理函数。  # 注释：返回值语义
        副作用：无（但返回的 process_fn 会修改其输入 example）。  # 注释：副作用说明
        异常/边界条件：无显式异常；内部若答案格式异常会触发 extract_solution 的断言。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - 输入：make_map_fn("train") -> 返回 process_fn；process_fn({"question": "1+1?", "answer": "#### 2"}, 0)  # 注释：示例输入
        - 输出：包含 prompt/ability/reward_model 的标准化 dict。  # 注释：示例输出
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：examples/data_preprocess/gsm8k.py::make_map_fn。  # 注释：函数位置
        - 典型调用路径：__main__ -> train_dataset.map(function=make_map_fn("train"), ...) -> process_fn。  # 注释：典型调用链
        - 被谁调用：仅在本文件 __main__ 中调用。  # 注释：调用方说明
        - 调用了谁（项目内）：返回的 process_fn 会调用本文件 extract_solution。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：datasets.Dataset.map（调用返回函数）。  # 注释：外部依赖说明
        """  # 注释：函数 docstring 结束
        def process_fn(example, idx):  # 注释：单条样本处理函数（datasets.map 调用）
            """函数用途：将单条 GSM8K 样本转为训练/评估统一格式。  # 注释：函数用途说明
            参数：  # 注释：参数说明标题
            - example (dict)：原始样本，包含 "question" 与 "answer" 字段。  # 注释：参数含义
            - idx (int)：样本在拆分中的索引。  # 注释：参数含义
            返回：  # 注释：返回值说明标题
            - dict：包含 prompt、reward_model、extra_info 等字段的标准化样本。  # 注释：返回值语义
            副作用：  # 注释：副作用说明标题
            - 会从 example 中 pop 出 "question" 与 "answer"（就地修改）。  # 注释：副作用说明
            异常/边界条件：  # 注释：异常说明标题
            - 若答案不含 "####"，extract_solution 会触发 AssertionError。  # 注释：异常说明
            最小示例：  # 注释：最小示例标题
            - 输入：example={"question":"1+1?","answer":"Let\'s think...#### 2"}, idx=0。  # 注释：示例输入
            - 输出：{"prompt":[{"role":"user","content":"1+1? ..."}],"reward_model":{...}}。  # 注释：示例输出
            调用路径依赖：  # 注释：调用路径说明标题
            - 所在位置：examples/data_preprocess/gsm8k.py::process_fn。  # 注释：函数位置
            - 典型调用路径：datasets.Dataset.map -> process_fn -> extract_solution。  # 注释：典型调用链
            - 被谁调用：datasets.Dataset.map（由 train_dataset/test_dataset 调用）。  # 注释：调用方说明
            - 调用了谁（项目内）：extract_solution。  # 注释：项目内依赖说明
            - 调用了谁（关键外部依赖）：无（仅标准库字符串操作）。  # 注释：外部依赖说明
            """  # 注释：函数 docstring 结束
            question_raw = example.pop("question")  # 注释：取出原始问题文本并从样本中移除
            # （分隔说明：构造带提示的用户问题）  # 注释：替代空行，保持逐行注释
            question = question_raw + " " + instruction_following  # 注释：拼接推理提示，形成最终 prompt 内容
            answer_raw = example.pop("answer")  # 注释：取出原始答案文本并从样本中移除
            solution = extract_solution(answer_raw)  # 注释：从答案中提取最终数值解
            # （分隔说明：组装统一的数据结构）  # 注释：替代空行，保持逐行注释
            data = {  # 注释：构造标准化样本字典
                "data_source": data_source,  # 注释：记录数据来源
                "prompt": [  # 注释：模型输入 prompt 的对话列表
                    {  # 注释：单轮对话项
                        "role": "user",  # 注释：角色为用户
                        "content": question,  # 注释：用户问题文本
                    }  # 注释：结束单轮对话项
                ],  # 注释：结束 prompt 列表
                "ability": "math",  # 注释：能力标签（数学推理）
                "reward_model": {"style": "rule", "ground_truth": solution},  # 注释：规则奖励与标准答案
                "extra_info": {  # 注释：额外元信息
                    "split": split,  # 注释：数据拆分信息
                    "index": idx,  # 注释：样本索引
                    "answer": answer_raw,  # 注释：保留原始答案
                    "question": question_raw,  # 注释：保留原始问题
                },  # 注释：结束 extra_info
            }  # 注释：结束标准化样本字典
            return data  # 注释：返回处理后的样本
        return process_fn  # 注释：返回闭包函数，供 datasets.map 使用
    # （分隔说明：对训练与测试集执行 map 转换）  # 注释：替代空行，保持逐行注释
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)  # 注释：转换训练集
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)  # 注释：转换测试集
    # （分隔说明：处理输出目录参数）  # 注释：替代空行，保持逐行注释
    hdfs_dir = args.hdfs_dir  # 注释：读取 HDFS 目录参数
    local_save_dir = args.local_dir  # 注释：读取旧参数 local_dir（已弃用）
    if local_save_dir is not None:  # 注释：若使用旧参数则提示弃用
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")  # 注释：打印弃用提示
    else:  # 注释：否则使用新参数
        local_save_dir = args.local_save_dir  # 注释：设置本地保存目录
    # （分隔说明：写出 parquet 文件）  # 注释：替代空行，保持逐行注释
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))  # 注释：保存训练集 parquet
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))  # 注释：保存测试集 parquet
    # （分隔说明：可选同步到 HDFS）  # 注释：替代空行，保持逐行注释
    if hdfs_dir is not None:  # 注释：若指定 HDFS 目录则进行同步
        makedirs(hdfs_dir)  # 注释：在 HDFS 上创建目标目录
        copy(src=local_save_dir, dst=hdfs_dir)  # 注释：将本地保存目录拷贝到 HDFS
