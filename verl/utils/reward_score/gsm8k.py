# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
模块用途：提供 GSM8K 数据集的规则评分与答案抽取逻辑。（注释：模块功能概述）
输入：模型输出 solution_str 与 ground_truth。（注释：输入形态说明）
输出：格式分或正确分（float）。（注释：输出形态说明）
关键依赖：re（正则提取答案）。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - score = compute_score("...#### 5", "5")
调用路径概览：（注释：全局调用关系）
  - verl/utils/reward_score/__init__.py::default_compute_score -> gsm8k.compute_score
"""

import re  # 注释：用于正则匹配最终答案

_SOLUTION_CLIP_CHARS = 300  # 注释：仅截取末尾 300 字符以加速正则匹配


def extract_solution(solution_str, method="strict"):  # 注释：从模型输出中抽取最终答案字符串
    """
    功能：根据 strict/flexible 策略从解答文本中抽取数值答案。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - solution_str (str): 模型输出文本。（注释：待解析文本）
      - method (str): "strict" 或 "flexible"。（注释：解析策略）
    返回：（注释：返回值说明）
      - str|None：抽取到的最终答案；若无法抽取则返回 None。（注释：返回形态）
    副作用：（注释：副作用说明）
      - 无（纯解析）。
    异常/边界条件：（注释：异常与边界）
      - method 不是 strict/flexible 会触发断言。（注释：参数校验）
      - solution_str 过长时仅保留末尾 300 字符。（注释：截断策略）
    最小示例：（注释：最小可理解示例）
      - 输入：solution_str="...#### 1,234"，method="strict"
      - 输出："1234"
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/reward_score/gsm8k.py::extract_solution`
      - 典型调用路径：`gsm8k.compute_score` -> `extract_solution`
      - 被谁调用：`gsm8k.compute_score`
      - 调用了谁（项目内）：无
      - 调用了谁（外部依赖）：`re.findall`
    """
    assert method in ["strict", "flexible"]  # 注释：仅允许两种解析策略

    # Optimization: Regular expression matching on very long strings can be slow.  # 注释：长文本正则匹配较慢
    # For math problems, the final answer is usually at the end.  # 注释：最终答案通常出现在末尾
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.  # 注释：截断策略说明
    if len(solution_str) > _SOLUTION_CLIP_CHARS:  # 注释：超长文本时截断
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]  # 注释：保留末尾 300 字符

    if method == "strict":  # 注释：严格模式需要 "#### 数字" 格式
        # this also tests the formatting of the model  # 注释：同时检查格式是否符合规范
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)  # 注释：匹配 #### 后的数值
        if len(solutions) == 0:  # 注释：未匹配到答案
            final_answer = None  # 注释：返回 None
        else:  # 注释：匹配到答案
            # take the last solution  # 注释：取最后一个匹配结果
            final_answer = solutions[-1].replace(",", "").replace("$", "")  # 注释：去除逗号/美元符号
    elif method == "flexible":  # 注释：宽松模式直接抓取任意数值
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)  # 注释：匹配所有数值片段
        final_answer = None  # 注释：默认无答案
        if len(answer) == 0:  # 注释：未匹配到数值
            # no reward is there is no answer  # 注释：无答案则不给分
            pass  # 注释：保持 final_answer=None
        else:  # 注释：存在数值候选
            invalid_str = ["", "."]  # 注释：无效候选列表
            # find the last number that is not '.'  # 注释：从后往前找合法数字
            for final_answer in reversed(answer):  # 注释：反向遍历候选
                if final_answer not in invalid_str:  # 注释：过滤无效候选
                    break  # 注释：找到合法答案即停止
    return final_answer  # 注释：返回抽取结果


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):  # 注释：GSM8K 评分
    """
    功能：对 GSM8K 解答进行规则评分。（注释：函数目标说明）
    参考：Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." ACL 2024。（注释：引用来源）
    参数：（注释：函数参数说明）
      - solution_str (str): 模型输出文本。（注释：待评估答案）
      - ground_truth (str): 标准答案。（注释：正确答案）
      - method (str): 解析策略，"strict"/"flexible"。（注释：解析策略）
      - format_score (float): 仅格式正确但答案错误时的得分。（注释：格式分）
      - score (float): 答案正确时的得分。（注释：正确分）
    返回：（注释：返回值说明）
      - float：0/format_score/score。（注释：分数）
    副作用：（注释：副作用说明）
      - 无（纯函数）。
    异常/边界条件：（注释：异常与边界）
      - 若 extract_solution 返回 None，则返回 0。（注释：无答案处理）
    最小示例：（注释：最小可理解示例）
      - 输入：solution_str="...#### 5"，ground_truth="5"
      - 输出：1.0
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/reward_score/gsm8k.py::compute_score`
      - 典型调用路径：`default_compute_score` -> `gsm8k.compute_score`
      - 被谁调用：`verl/utils/reward_score/__init__.py::default_compute_score`
      - 调用了谁（项目内）：`extract_solution`
      - 调用了谁（外部依赖）：无
    """
    answer = extract_solution(solution_str=solution_str, method=method)  # 注释：抽取最终答案
    if answer is None:  # 注释：无答案则不给分
        return 0  # 注释：返回 0 分
    else:  # 注释：有答案则判断正确性
        if answer == ground_truth:  # 注释：答案正确
            return score  # 注释：返回满分
        else:  # 注释：答案错误但可能格式正确
            return format_score  # 注释：返回格式分
