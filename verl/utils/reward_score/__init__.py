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
模块用途：根据 data_source 路由到对应的规则评分函数（GSM8K、MATH、代码题等）。（注释：模块功能概述）
输入：data_source、solution_str、ground_truth，以及可选的 sandbox 参数。（注释：输入形态说明）
输出：float 或包含附加信息的 dict。（注释：输出形态说明）
关键依赖：各子模块的 compute_score（gsm8k、math_reward、prime_code 等）。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - score = default_compute_score("openai/gsm8k", "...#### 5", "5")
调用路径概览：（注释：全局调用关系）
  - verl/workers/reward_manager/naive.py -> default_compute_score -> gsm8k.compute_score
  - verl/trainer/ppo/reward.py::load_reward_manager 在未指定 custom_reward_fn 时使用 default_compute_score
"""

# （可选）示例导入：gsm8k/math/prime_* 等评分模块（当前按需懒加载）  # 注释：避免启动时全量导入

from verl.utils.import_utils import deprecated  # 注释：提供 @deprecated 装饰器


def default_compute_score(  # 注释：根据 data_source 路由到具体评分实现
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """
    功能：根据数据源选择对应的评分规则并返回分数。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - data_source (str): 数据集或任务标识（决定评分逻辑）。（注释：路由键）
      - solution_str (str): 模型输出文本。（注释：待评估答案）
      - ground_truth (str): 标准答案或测试用例。（注释：对照答案）
      - extra_info (dict|None): 评分所需的额外信息（可选）。（注释：扩展信息）
      - sandbox_fusion_url (str|None): 代码题 sandbox 服务地址。（注释：代码执行服务）
      - concurrent_semaphore (Semaphore|None): sandbox 并发控制。（注释：并发限制）
      - memory_limit_mb (int|None): sandbox 内存限制（MB）。（注释：资源限制）
      - **kwargs: 兼容扩展参数。（注释：留作扩展）
    返回：（注释：返回值说明）
      - float 或 dict：若子评分函数返回 dict，则原样返回；否则转为 float。（注释：返回类型）
    副作用：（注释：副作用说明）
      - 无（纯函数式计算；内部可能调用外部 sandbox）。
    异常/边界条件：（注释：异常与边界）
      - 未识别的数据源会抛出 NotImplementedError。（注释：未知数据源）
      - 代码题在缺少 sandbox_fusion_url 时将回退到 prime_code。（注释：回退逻辑）
    最小示例：（注释：最小可理解示例）
      - 输入：data_source="openai/gsm8k", solution_str="...#### 5", ground_truth="5"
      - 输出：1.0
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/reward_score/__init__.py::default_compute_score`
      - 典型调用路径：`verl/workers/reward_manager/naive.py` -> `default_compute_score`
      - 被谁调用：`verl/workers/reward_manager/naive.py`、`verl/workers/reward_manager/dapo.py`、
        `verl/workers/reward_manager/prime.py`、`verl/experimental/reward_loop/reward_manager/limited.py`、
        `verl/trainer/ppo/reward.py::load_reward_manager`
      - 调用了谁（项目内）：`gsm8k.compute_score`、`math_reward.compute_score`、`prime_code.compute_score` 等
      - 调用了谁（外部依赖）：可能调用 sandbox 服务（通过 sandbox_fusion）
    """
    if data_source == "openai/gsm8k":  # 注释：GSM8K 数据集评分
        from . import gsm8k  # 注释：按需导入 gsm8k 评分模块

        res = gsm8k.compute_score(solution_str, ground_truth)  # 注释：计算 GSM8K 分数
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:  # 注释：标准 MATH 数据集
        from . import math_reward  # 注释：按需导入 MATH 评分模块

        res = math_reward.compute_score(solution_str, ground_truth)  # 注释：计算 MATH 分数
        # 可选：使用 Math-Verify 提升准确性（https://github.com/huggingface/Math-Verify）  # 注释：参考库
        # 安装提示：pip install math-verify  # 注释：需要手动安装
        # 用法：可将 compute_score 替换为 math_verify.compute_score  # 注释：替换方式说明

        # from . import math_verify  # 注释：Math-Verify 评分模块（可选）
        # res = math_verify.compute_score(solution_str, ground_truth)  # 注释：改用 Math-Verify 评分
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):  # 注释：DAPO 数学与 AIME
        from . import math_dapo  # 注释：按需导入 DAPO 数学评分模块

        res = math_dapo.compute_score(solution_str, ground_truth)  # 注释：计算 DAPO/AIME 分数
    elif data_source in [  # 注释：Prime Math 系列数据源
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math  # 注释：按需导入 prime_math 评分模块

        res = prime_math.compute_score(solution_str, ground_truth)  # 注释：计算 Prime Math 分数
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:  # 注释：代码题数据源
        # 若提供 sandbox_fusion_url 则优先使用 sandbox 执行  # 注释：优先走可执行评测
        if sandbox_fusion_url:  # 注释：若提供 sandbox 服务地址
            from . import sandbox_fusion  # 注释：按需导入 sandbox_fusion 模块

            # 直接传入 URL，ground_truth 通常为测试用例集合  # 注释：代码题用例说明
            res = sandbox_fusion.compute_score(  # 注释：通过 sandbox 执行评分
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:  # 注释：无 sandbox 服务时回退到 prime_code
            # 未提供 sandbox URL 时回退到 prime_code（或上层决定报错）  # 注释：回退策略说明
            from . import prime_code  # 注释：按需导入 prime_code 模块

            # 假设 prime_code 不需要 URL 参数  # 注释：接口差异说明
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)  # 注释：计算代码题分数
    elif data_source in ["hiyouga/geometry3k"]:  # 注释：几何题数据源
        from . import geo3k  # 注释：按需导入 geo3k 评分模块

        res = geo3k.compute_score(solution_str, ground_truth)  # 注释：计算 geometry3k 分数
    elif data_source in [  # 注释：SearchR1 QA 系列数据源
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em  # 注释：按需导入 SearchR1 QA 评分模块

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)  # 注释：计算 QA EM 分数

    else:  # 注释：未知数据源
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")  # 注释：显式报错

    if isinstance(res, dict):  # 注释：若评分函数返回字典（含额外字段）
        return res  # 注释：原样返回字典
    elif isinstance(res, int | float | bool):  # 注释：数值/布尔类型转为 float
        return float(res)  # 注释：统一返回 float
    else:  # 注释：支持返回序列类型时取第一个元素
        return float(res[0])  # 注释：转换为 float


@deprecated("verl.utils.reward_score.default_compute_score")  # 注释：标记旧 API 已弃用
def _default_compute_score(  # 注释：旧版 API 入口（兼容过渡）
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    功能：旧版 default_compute_score 的兼容接口。（注释：函数目标说明）
    参数：（注释：函数参数说明）
      - 与 default_compute_score 相同。（注释：参数复用）
    返回：（注释：返回值说明）
      - 与 default_compute_score 相同。（注释：返回复用）
    副作用：（注释：副作用说明）
      - 触发 @deprecated 警告。（注释：弃用提示）
    异常/边界条件：（注释：异常与边界）
      - 同 default_compute_score。（注释：一致性）
    最小示例：（注释：最小可理解示例）
      - 输入：_default_compute_score("openai/gsm8k", "...#### 5", "5")
      - 输出：1.0
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/utils/reward_score/__init__.py::_default_compute_score`
      - 典型调用路径：`recipe/entropy/reward.py` -> `_default_compute_score`
      - 被谁调用：`recipe/entropy/reward.py`
      - 调用了谁（项目内）：`default_compute_score`
      - 调用了谁（外部依赖）：无
    """
    return default_compute_score(  # 注释：直接转调新 API
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]  # 注释：对外导出默认评分函数
