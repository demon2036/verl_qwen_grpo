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
模块用途：RewardLoop 版本的 NaiveRewardManager，使用规则评分函数计算奖励。（注释：模块功能概述）
输入：DataProto（含 responses/ground_truth/data_source 等）、tokenizer。（注释：输入形态说明）
输出：dict，包含 reward_score 与 reward_extra_info。（注释：输出形态说明）
关键依赖：default_compute_score、RewardManagerBase。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - rm = NaiveRewardManager(config, tokenizer)
  - await rm.run_single(data_item)
调用路径概览：（注释：全局调用关系）
  - RewardLoopWorker.compute_score -> reward_loop.run_single -> default_compute_score
"""

import inspect  # 注释：判断评分函数是否为协程

from verl import DataProto  # 注释：数据容器
from verl.experimental.reward_loop.reward_manager import register  # 注释：注册装饰器
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase  # 注释：基类
from verl.utils.reward_score import default_compute_score  # 注释：默认规则评分函数


@register("naive")  # 注释：注册名称为 naive
class NaiveRewardManager(RewardManagerBase):  # 注释：Naive RewardManager 实现
    """
    功能：对单条样本调用规则评分函数，返回 reward_score 与额外信息。（注释：类职责）
    适用场景：无需 RM 服务器或使用规则评分的数据集。（注释：使用场景）
    """

    def __init__(  # 注释：初始化 NaiveRewardManager
        self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None
    ):
        """
        功能：保存评分函数与可选的 reward_router/tokenizer。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config: RewardLoop 配置。（注释：配置对象）
          - tokenizer: 用于解码 responses 的 tokenizer。（注释：分词器）
          - compute_score (Callable|None): 规则评分函数，默认 default_compute_score。（注释：评分函数）
          - reward_router_address (str|None): RM 路由地址（可选）。（注释：路由地址）
          - reward_model_tokenizer (AutoTokenizer|None): RM tokenizer（可选）。（注释：RM 分词器）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 保存成员变量，判断评分函数是否为协程。（注释：状态持有）
        异常/边界条件：（注释：异常与边界）
          - 无显式异常。（注释：依赖外部输入合法）
        最小示例：（注释：最小可理解示例）
          - 输入：NaiveRewardManager(config, tokenizer)
          - 输出：对象初始化完成
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/naive.py::NaiveRewardManager.__init__`
          - 典型调用路径：`RewardLoopWorker._init_reward_fn` -> `NaiveRewardManager(...)`
          - 被谁调用：`RewardLoopWorker`
          - 调用了谁（项目内）：`RewardManagerBase.__init__`、`default_compute_score`
          - 调用了谁（外部依赖）：`inspect.iscoroutinefunction`
        """
        super().__init__(config, tokenizer)  # 注释：调用基类初始化
        self.compute_score = compute_score or default_compute_score  # 注释：设置评分函数
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)  # 注释：判断是否异步
        self.reward_router_address = reward_router_address  # 注释：保存路由地址
        self.reward_model_tokenizer = reward_model_tokenizer  # 注释：保存 RM tokenizer

    async def run_single(self, data: DataProto) -> dict:  # 注释：处理单条样本
        """
        功能：解码 responses，调用评分函数并返回 reward_score。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 单条样本数据（len==1）。（注释：输入样本）
        返回：（注释：返回值说明）
          - dict：{"reward_score": float, "reward_extra_info": dict}。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 可能调用外部 reward_router 服务。（注释：外部副作用）
        异常/边界条件：（注释：异常与边界）
          - data 非单条样本会触发断言。（注释：输入约束）
          - 缺失字段（如 ground_truth）会抛 KeyError。（注释：字段依赖）
        最小示例：（注释：最小可理解示例）
          - 输入：data[0].non_tensor_batch["reward_model"]["ground_truth"]="5"
          - 输出：{"reward_score": 1.0, ...}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/naive.py::NaiveRewardManager.run_single`
          - 典型调用路径：`RewardLoopWorker.compute_score` -> `run_single`
          - 被谁调用：`RewardLoopWorker`
          - 调用了谁（项目内）：`default_compute_score`
          - 调用了谁（外部依赖）：`tokenizer.decode`、`loop.run_in_executor`
        """
        assert len(data) == 1, "Only support single data item"  # 注释：仅支持单条样本
        data_item = data[0]  # 注释：取出样本
        response_ids = data_item.batch["responses"]  # 注释：response token ids
        response_length = response_ids.shape[-1]  # 注释：response 长度
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()  # 注释：有效长度
        valid_response_ids = response_ids[:valid_response_length]  # 注释：截取有效 response

        data_source = data_item.non_tensor_batch["data_source"]  # 注释：数据源名称
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]  # 注释：标准答案
        extra_info = data_item.non_tensor_batch.get("extra_info", {})  # 注释：额外信息
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)  # 注释：工具字段（可选）
        if tool_extra_fields is not None:  # 注释：存在工具字段时合并
            extra_info.update(tool_extra_fields.items())  # 注释：合并到 extra_info

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)  # 注释：多轮轮数（可选）
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})  # 注释：rollout reward（可选）
        extra_info["num_turns"] = num_turns  # 注释：写入 extra_info
        extra_info["rollout_reward_scores"] = rollout_reward_scores  # 注释：写入 extra_info

        response_str = await self.loop.run_in_executor(  # 注释：异步解码 response 文本
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (  # 注释：可选的 RM 额外参数
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:  # 注释：评分函数为协程
            result = await self.compute_score(  # 注释：直接 await 评分函数
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:  # 注释：评分函数为同步函数
            result = await self.loop.run_in_executor(  # 注释：在线程池中执行同步评分
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}  # 注释：收集额外信息

        score: float  # 注释：评分结果类型
        if isinstance(result, dict):  # 注释：字典形式返回
            score = result["score"]  # 注释：主分数
            for key, value in result.items():  # 注释：收集所有字段
                reward_extra_info[key] = value  # 注释：写入额外信息
        else:  # 注释：数值形式返回
            score = result  # 注释：直接作为分数
            reward_extra_info["acc"] = score  # 注释：记录准确率字段

        reward = score  # 注释：reward 即 score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}  # 注释：返回结果
