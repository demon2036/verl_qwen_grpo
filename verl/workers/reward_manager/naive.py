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
模块用途：提供默认的 NaiveRewardManager，用规则评分函数对单轮文本答案打分。（注释：模块功能概述）
输入：DataProto（含 prompts/responses/ground_truth/data_source）、tokenizer、compute_score。（注释：输入形态说明）
输出：与 responses 对齐的 reward 张量，及可选的 reward_extra_info。（注释：输出形态说明）
关键依赖：torch、verl.utils.reward_score.default_compute_score。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - rm = NaiveRewardManager(tokenizer, num_examine=1)
  - reward = rm(data, return_dict=True)
调用路径概览：（注释：全局调用关系）
  - verl/trainer/ppo/reward.py::compute_reward -> NaiveRewardManager.__call__ -> default_compute_score -> gsm8k.compute_score
"""

from collections import defaultdict  # 注释：用于构造 reward_extra_info 的默认列表容器
from typing import Any  # 注释：类型提示

import torch  # 注释：张量运算与奖励张量构造

from verl import DataProto  # 注释：训练 batch 数据容器
from verl.utils.reward_score import default_compute_score  # 注释：默认规则评分函数
from verl.workers.reward_manager import register  # 注释：reward manager 注册装饰器
from verl.workers.reward_manager.abstract import AbstractRewardManager  # 注释：reward manager 抽象基类


@register("naive")  # 注释：将该类注册为 "naive" reward manager
class NaiveRewardManager(AbstractRewardManager):  # 注释：默认 reward manager 实现
    """
    功能：对单轮文本输出使用规则函数打分，并将分数写入 response 最后一个有效 token。（注释：类的职责）
    适用场景：GSM8K 等规则评分数据集。（注释：场景说明）
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:  # 注释：初始化
        """
        功能：保存 tokenizer、打印样本数量与评分函数。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - tokenizer (Any): 用于将 token ids 解码成文本。（注释：文本还原）
          - num_examine (int): 每个数据源打印的样本数。（注释：调试控制）
          - compute_score (Callable|None): 评分函数，默认使用 default_compute_score。（注释：评分逻辑）
          - reward_fn_key (str): non_tensor_batch 中数据源字段名。（注释：数据源定位）
        返回：（注释：返回值说明）
          - None（初始化过程）。
        副作用：（注释：副作用说明）
          - 保存成员变量用于后续计算。（注释：状态持有）
        异常/边界条件：（注释：异常与边界）
          - 无显式异常（依赖调用方传入正确参数）。（注释：参数校验由外部完成）
        最小示例：（注释：最小可理解示例）
          - 输入：tokenizer=..., num_examine=1
          - 输出：完成 NaiveRewardManager 初始化
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/reward_manager/naive.py::NaiveRewardManager.__init__`
          - 典型调用路径：`verl/trainer/ppo/reward.py::load_reward_manager` -> NaiveRewardManager(...)
          - 被谁调用：`verl/trainer/ppo/reward.py::load_reward_manager`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：无
        """
        self.tokenizer = tokenizer  # 注释：保存 tokenizer 以便解码 token ids
        self.num_examine = num_examine  # 注释：保存每个数据源的调试打印次数
        self.compute_score = compute_score or default_compute_score  # 注释：设置评分函数（默认规则评分）
        self.reward_fn_key = reward_fn_key  # 注释：保存 data_source 字段名

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:  # 注释：计算奖励
        """
        功能：对 batch 中每条样本解码回答并计算奖励，写入 response 最后一个有效 token。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 包含 prompts/responses/metadata 的 batch。（注释：输入批次）
          - return_dict (bool): 是否返回包含 reward_extra_info 的字典。（注释：返回控制）
        返回：（注释：返回值说明）
          - torch.Tensor 或 dict[str, Any]。（注释：返回类型）
        副作用：（注释：副作用说明）
          - 可能向控制台打印样本的 prompt/response/score。（注释：调试输出）
        异常/边界条件：（注释：异常与边界）
          - 若 data 缺失必要字段（如 reward_model/ground_truth），可能抛 KeyError。（注释：字段依赖）
          - 如果 response 全为 padding，valid_response_length 可能为 0。（注释：边界情形）
        最小示例：（注释：最小可理解示例）
          - 输入：单条样本，response 解码为 "The answer is 5 #### 5"，ground_truth="5"
          - 输出：reward_tensor 在最后一个有效 token 位置为 1
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/workers/reward_manager/naive.py::NaiveRewardManager.__call__`
          - 典型调用路径：`verl/trainer/ppo/reward.py::compute_reward` -> NaiveRewardManager.__call__
          - 被谁调用：`verl/trainer/ppo/reward.py::compute_reward`
          - 调用了谁（项目内）：`AbstractRewardManager._extract_reward_from_rm_scores`、`default_compute_score`
          - 调用了谁（外部依赖）：`tokenizer.decode`、`torch.zeros_like`
        """

        # 若已有 rm_scores 则直接返回，否则走规则评分  # 注释：兼容 reward loop 场景
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)  # 注释：尝试直接提取 rm_scores
        if reward_from_rm_scores is not None:  # 注释：若已有 rm_scores 则直接返回
            return reward_from_rm_scores  # 注释：避免重复计算

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # 注释：初始化 reward 张量
        reward_extra_info = defaultdict(list)  # 注释：收集额外信息（按 key 追加）

        already_print_data_sources = {}  # 注释：记录每个 data_source 已打印次数

        for i in range(len(data)):  # 注释：逐条样本计算 reward
            data_item = data[i]  # 注释：取出第 i 条 DataProtoItem

            prompt_ids = data_item.batch["prompts"]  # 注释：获取 prompt token ids

            prompt_length = prompt_ids.shape[-1]  # 注释：prompt 长度（含 padding）

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()  # 注释：有效 prompt 长度
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]  # 注释：截取有效 prompt ids

            response_ids = data_item.batch["responses"]  # 注释：获取 response token ids
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()  # 注释：有效 response 长度
            valid_response_ids = response_ids[:valid_response_length]  # 注释：截取有效 response ids

            # 解码 token ids 为文本  # 注释：将模型输出还原成字符串
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)  # 注释：解码 prompt
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)  # 注释：解码 response

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]  # 注释：读取标准答案
            data_source = data_item.non_tensor_batch[self.reward_fn_key]  # 注释：读取数据源名称
            extra_info = data_item.non_tensor_batch.get("extra_info", {})  # 注释：读取可选额外信息
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)  # 注释：读取多轮轮数（若有）
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})  # 注释：读取 rollout 阶段 reward
            extra_info["num_turns"] = num_turns  # 注释：补充 num_turns 到 extra_info
            extra_info["rollout_reward_scores"] = rollout_reward_scores  # 注释：补充 rollout reward 信息

            score = self.compute_score(  # 注释：调用评分函数
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):  # 注释：若评分函数返回字典（含额外信息）
                reward = score["score"]  # 注释：取出主 reward 分数
                # 保存包含原始 reward 的所有字段  # 注释：用于后续统计/分析
                for key, value in score.items():  # 注释：遍历额外信息
                    reward_extra_info[key].append(value)  # 注释：按 key 追加值列表
            else:  # 注释：若评分函数返回数值
                reward = score  # 注释：直接作为 reward

            reward_tensor[i, valid_response_length - 1] = reward  # 注释：将 reward 写入最后一个有效 token

            if data_source not in already_print_data_sources:  # 注释：初始化该 data_source 的打印计数
                already_print_data_sources[data_source] = 0  # 注释：初始计数为 0

            if already_print_data_sources[data_source] < self.num_examine:  # 注释：控制打印次数
                already_print_data_sources[data_source] += 1  # 注释：递增计数
                print("[prompt]", prompt_str)  # 注释：打印 prompt
                print("[response]", response_str)  # 注释：打印 response
                print("[ground_truth]", ground_truth)  # 注释：打印 ground truth
                if isinstance(score, dict):  # 注释：打印字典形式的额外信息
                    for key, value in score.items():  # 注释：遍历字典字段
                        print(f"[{key}]", value)  # 注释：打印字段名与值
                else:  # 注释：打印数值形式的 score
                    print("[score]", score)  # 注释：打印分数

        if return_dict:  # 注释：按需返回字典格式
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }  # 注释：返回包含 reward 与额外信息的字典
        else:  # 注释：仅返回 reward 张量
            return reward_tensor  # 注释：返回 reward 张量
