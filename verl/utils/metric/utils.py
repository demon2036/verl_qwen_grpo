# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行说明
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
"""（模块说明：指标聚合工具函数，用于将训练过程中收集的指标列表聚合为单个统计值）

模块用途：
    本模块提供指标聚合（metric reduction）功能，用于分布式训练或多步训练中的指标汇总。
    主要功能是根据指标名称自动选择合适的聚合方式：
    - 包含 "max" 的指标使用最大值聚合
    - 包含 "min" 的指标使用最小值聚合
    - 其他指标使用均值聚合

输入/输出：
    - 输入：字典，键为指标名称（str），值为指标值列表（list[Any]）
           例如：{"loss": [1.0, 2.0, 3.0], "max_reward": [5.0, 8.0, 6.0]}
    - 输出：字典，键为指标名称（str），值为聚合后的标量（float 或其他数值类型）
           例如：{"loss": 2.0, "max_reward": 8.0}

关键依赖：
    - numpy：用于高效的数值统计计算（np.mean, np.max, np.min）
    - typing.Any：用于类型标注，支持任意可数值化类型

典型用法：
    >>> from verl.utils.metric import reduce_metrics
    >>> # 场景1：聚合多个训练步骤的loss
    >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
    >>> reduced = reduce_metrics(metrics)
    >>> print(reduced)
    {"loss": 2.0, "accuracy": 0.8}

    >>> # 场景2：聚合多个worker的奖励指标
    >>> metrics = {"max_reward": [5.0, 8.0, 6.0], "min_error": [0.1, 0.05, 0.2]}
    >>> reduced = reduce_metrics(metrics)
    >>> print(reduced)
    {"max_reward": 8.0, "min_error": 0.05}

调用路径概览：
    入口脚本（如 verl/trainer/main_ppo.py）
    -> verl.trainer.ppo.ray_trainer.collect_metrics()
    -> verl.utils.metric.reduce_metrics()
    -> 返回聚合后的指标字典，用于日志记录或 WandB/MLflow 上报

所在位置：
    - 路径：verl/utils/metric/utils.py
    - 模块：verl.utils.metric.utils

被谁调用：
    - verl/trainer/ppo/metric_utils.py（指标汇总逻辑）
    - verl/trainer/ppo/ray_trainer.py（Ray 训练循环中收集 worker 指标）
    - verl/trainer/fsdp_sft_trainer.py（可能在 SFT 训练中聚合多步指标）

调用了谁（项目内）：
    - 无（本模块为底层工具，不调用其他项目内模块）

调用了谁（外部依赖）：
    - numpy.mean：计算均值
    - numpy.max：计算最大值
    - numpy.min：计算最小值

注意事项：
    1. 聚合方式完全由指标名称的字符串内容决定（"max" / "min" / 其他）
    2. 如果指标值列表为空，numpy 会抛出异常，调用者需确保列表非空
    3. 本函数会直接修改输入字典（in-place 修改），不创建新字典
"""  # 注释：模块级 docstring 结束，下面是类型导入与依赖
# （分隔说明：导入类型标注）  # 注释：替代空行，保持逐行注释
from typing import Any  # 注释：Any 用于 reduce_metrics 函数的类型签名
# （分隔说明：导入 numpy）  # 注释：替代空行，保持逐行注释
import numpy as np  # 注释：numpy 提供向量化的统计函数
# （分隔说明：指标聚合实现）  # 注释：替代空行，保持逐行注释

def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:  # 注释：聚合指标字典
    """（函数说明：聚合指标列表，根据指标名称自动选择聚合策略）

    本函数用于将训练过程中收集的指标列表（例如多个训练步骤或多个 worker 的指标）聚合为单个统计值。
    聚合策略由指标名称（key）自动决定：
    - 如果指标名称包含 "max"（如 "max_reward"），使用 np.max 取最大值
    - 如果指标名称包含 "min"（如 "min_error"），使用 np.min 取最小值
    - 否则（如 "loss"、"accuracy"），使用 np.mean 取均值

    参数：
        metrics (dict[str, list[Any]]): 指标字典，格式为 {指标名称: 指标值列表}
            - 键（str）：指标名称，如 "loss"、"max_reward"、"min_error"
            - 值（list[Any]）：该指标在多个步骤/worker 上的值列表
              例如：[1.0, 2.0, 3.0] 或 [5.0, 8.0, 6.0]
            - 列表元素类型：可以是 int、float 或 numpy 标量等任意可数值化类型
            - 列表长度：必须 >= 1（空列表会导致 numpy 抛出异常）

    返回：
        dict[str, Any]: 聚合后的指标字典，格式为 {指标名称: 聚合后的标量值}
            - 键（str）：与输入相同的指标名称
            - 值（float 或其他数值类型）：根据策略聚合后的单个标量值
              例如：2.0（均值）、8.0（最大值）、0.05（最小值）
            - 注意：返回值与输入是同一个字典对象（in-place 修改）

    副作用：
        - 直接修改输入字典（in-place），将值列表替换为聚合后的标量
        - 不创建新字典，节省内存，但调用者需注意输入字典已被修改

    异常/边界条件：
        - 如果指标值列表为空（[]），numpy.mean/max/min 会抛出 ValueError 或 RuntimeWarning
        - 如果列表元素不可数值化（如字符串），numpy 会抛出 TypeError
        - 调用者应确保输入列表非空且元素可数值化

    最小示例（手算验证）：
        >>> # 输入：3个指标，每个有3个值
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],       # 均值指标（无 max/min）
        ...     "max_reward": [5.0, 8.0, 6.0], # 最大值指标（含 "max"）
        ...     "min_error": [0.1, 0.05, 0.2]  # 最小值指标（含 "min"）
        ... }
        >>> # 执行聚合
        >>> result = reduce_metrics(metrics)
        >>> # 预期输出（手算）：
        >>> # - "loss" -> mean([1.0, 2.0, 3.0]) = 2.0
        >>> # - "max_reward" -> max([5.0, 8.0, 6.0]) = 8.0
        >>> # - "min_error" -> min([0.1, 0.05, 0.2]) = 0.05
        >>> assert result == {"loss": 2.0, "max_reward": 8.0, "min_error": 0.05}

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/metric/utils.py
        - 函数：reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]

    典型调用路径（示例）：
        verl/trainer/main_ppo.py (入口脚本)
        -> verl/trainer/ppo/ray_trainer.py::PPOTrainer.fit()
        -> verl/trainer/ppo/metric_utils.py::collect_worker_metrics()
        -> verl/utils/metric/utils.py::reduce_metrics()  # 当前函数

    被谁调用：
        - verl/trainer/ppo/metric_utils.py（聚合多 worker 的指标列表）
        - verl/trainer/ppo/ray_trainer.py（Ray 训练循环中）
        - 任何需要聚合多步骤或多 worker 指标的地方

    调用了谁（项目内）：
        - 无（本函数为底层工具，不依赖其他项目内函数）

    调用了谁（外部依赖）：
        - numpy.mean：计算列表均值
        - numpy.max：计算列表最大值
        - numpy.min：计算列表最小值

    记忆提示：
        - 策略选择：根据键名（"max"/"min"/其他）自动选择聚合函数
        - 口诀：max取最大，min取最小，其他取均值
        - 副作用：in-place 修改，节省内存但改变输入
    """  # 注释：函数 docstring 结束
    # 注释：遍历输入字典的每个键值对（指标名称 key 与指标值列表 val）
    for key, val in metrics.items():  # 注释：metrics.items() 返回 (key, value) 迭代器
        # 注释：判断指标名称是否包含 "max" 子串
        if "max" in key:  # 注释：包含 max 时使用最大值
            metrics[key] = np.max(val)  # 注释：计算最大值并覆盖原列表
        # 注释：判断指标名称是否包含 "min" 子串
        elif "min" in key:  # 注释：包含 min 时使用最小值
            metrics[key] = np.min(val)  # 注释：计算最小值并覆盖原列表
        # 注释：默认情况（指标名称不含 max/min）
        else:  # 注释：普通指标使用均值
            metrics[key] = np.mean(val)  # 注释：计算均值并覆盖原列表
    return metrics  # 注释：返回聚合后的指标字典
