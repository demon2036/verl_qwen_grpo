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
模块用途：提供带限流能力的 RewardManager（RateLimitedRewardManager）与异步令牌桶实现。（注释：模块功能概述）
输入：DataProto、RewardModel 配置（max_rpm/max_tpm/timeout 等）。（注释：输入形态说明）
输出：dict 或奖励张量（兼容旧接口）。（注释：输出形态说明）
关键依赖：asyncio、default_compute_score、RewardManagerBase。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - rm = RateLimitedRewardManager(config, tokenizer)
  - await rm.run_single(data_item)
调用路径概览：（注释：全局调用关系）
  - reward_loop.py::_init_reward_fn -> RateLimitedRewardManager
"""

import asyncio  # 注释：异步与协程工具
import inspect  # 注释：判断评分函数是否为协程
import logging  # 注释：日志记录

from omegaconf import DictConfig  # 注释：配置类型
from transformers import AutoTokenizer  # 注释：分词器类型

from verl import DataProto  # 注释：数据容器
from verl.experimental.reward_loop.reward_manager import register as register_manager  # 注释：reward_loop 注册装饰器
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase  # 注释：基类
from verl.utils.ray_utils import get_event_loop  # 注释：获取事件循环
from verl.utils.reward_score import default_compute_score  # 注释：默认规则评分函数
from verl.workers.reward_manager import register as register_manager_legacy  # 注释：兼容旧注册表

logger = logging.getLogger(__file__)  # 注释：模块级日志器


class AsyncTokenBucket:  # 注释：异步令牌桶限流器
    """
    功能：以令牌桶算法实现异步限流，支持可变 token 消耗。（注释：类职责）
    参数：（注释：构造参数说明）
      - rate_limit (float): 每秒补充的 token 数。（注释：速率）
      - max_tokens (float|None): 桶容量上限（默认=rate_limit）。（注释：容量）
    返回：（注释：返回值说明）
      - AsyncTokenBucket 实例。（注释：实例对象）
    副作用：（注释：副作用说明）
      - 创建 asyncio.Lock，用于并发互斥。（注释：并发副作用）
    异常/边界条件：（注释：异常与边界）
      - rate_limit<=0 会导致除零或无意义限流。（注释：参数边界）
    最小示例：（注释：最小可理解示例）
      - 输入：bucket = AsyncTokenBucket(rate_limit=1.0, max_tokens=1.0)
      - 输出：await bucket.acquire(1.0)
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::AsyncTokenBucket`
      - 典型调用路径：`RateLimitedRewardManager.init_class` -> `AsyncTokenBucket(...)`
      - 被谁调用：`RateLimitedRewardManager`
      - 调用了谁（项目内）：`get_event_loop`（在 acquire 中）
      - 调用了谁（外部依赖）：`asyncio.Lock`、`asyncio.sleep`
    """

    def __init__(self, rate_limit: float, max_tokens: float = None):  # 注释：初始化令牌桶
        """
        功能：设置速率、容量与初始 token 数。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - rate_limit (float): 每秒补充 token 数。（注释：速率）
          - max_tokens (float|None): 最大 token 容量。（注释：容量）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 初始化 asyncio.Lock。（注释：并发互斥）
        异常/边界条件：（注释：异常与边界）
          - rate_limit<=0 时 acquire 可能异常。（注释：参数边界）
        最小示例：（注释：最小可理解示例）
          - 输入：AsyncTokenBucket(1.0, 2.0)
          - 输出：tokens=2.0
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::AsyncTokenBucket.__init__`
          - 典型调用路径：`RateLimitedRewardManager.init_class` -> `AsyncTokenBucket`
          - 被谁调用：`RateLimitedRewardManager`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`asyncio.Lock`
        """
        self.rate_limit = rate_limit  # 注释：保存补充速率
        self.max_tokens = max_tokens or rate_limit  # 注释：容量默认等于速率
        self.tokens = self.max_tokens  # 注释：初始 token 充满
        self.last_update = None  # 注释：上次更新时间（事件循环时间）
        self.lock = asyncio.Lock()  # 注释：并发互斥锁

    async def acquire(self, num_tokens: float = 1.0) -> None:  # 注释：获取 token（必要时等待）
        """
        功能：根据令牌桶规则等待并消耗指定 token 数。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - num_tokens (float): 需要消耗的 token 数。（注释：消耗量）
        返回：（注释：返回值说明）
          - None。（注释：成功获取后返回）
        副作用：（注释：副作用说明）
          - 可能调用 asyncio.sleep 等待。（注释：等待副作用）
        异常/边界条件：（注释：异常与边界）
          - rate_limit 过小可能导致长时间等待。（注释：性能边界）
        最小示例：（注释：最小可理解示例）
          - 输入：await bucket.acquire(5.0)
          - 输出：None（等待足够 token 后继续）
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::AsyncTokenBucket.acquire`
          - 典型调用路径：`RateLimitedRewardManager.run_single` -> `acquire`
          - 被谁调用：`RateLimitedRewardManager`
          - 调用了谁（项目内）：`get_event_loop`
          - 调用了谁（外部依赖）：`asyncio.sleep`、`asyncio.Lock`
        """
        # Handle requests larger than max_tokens separately  # 注释：单次请求超过桶容量的特殊处理
        if num_tokens > self.max_tokens:  # 注释：超容量请求
            wait_time = 0.0  # 注释：需要等待的时间
            async with self.lock:  # 注释：加锁计算 token
                loop = get_event_loop()  # 注释：获取事件循环
                now = loop.time()  # 注释：当前时间戳
                if self.last_update is None:  # 注释：首次更新
                    self.last_update = now  # 注释：记录初始时间

                elapsed = now - self.last_update  # 注释：距离上次更新的时间
                new_tokens = elapsed * self.rate_limit  # 注释：补充的 token 数
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)  # 注释：更新 token 数

                tokens_needed = num_tokens - self.tokens  # 注释：还需要的 token 数
                if tokens_needed > 0:  # 注释：不足时计算等待
                    wait_time = tokens_needed / self.rate_limit  # 注释：需要等待的秒数

                self.tokens -= num_tokens  # 注释：先消耗本次请求 token（可为负）
                self.last_update = now  # 注释：更新最后时间

            if wait_time > 0:  # 注释：需要等待时 sleep
                await asyncio.sleep(wait_time)  # 注释：等待补充 token
            return  # 注释：超容量请求完成

        # Standard case: request <= max_tokens  # 注释：常规情况
        while True:  # 注释：循环直到获取足够 token
            wait_time = 0.0  # 注释：初始化等待时间
            async with self.lock:  # 注释：加锁更新 token
                loop = get_event_loop()  # 注释：获取事件循环
                now = loop.time()  # 注释：当前时间戳
                if self.last_update is None:  # 注释：首次更新
                    self.last_update = now  # 注释：记录初始时间

                elapsed = now - self.last_update  # 注释：距离上次更新的时间
                new_tokens = elapsed * self.rate_limit  # 注释：补充的 token 数
                self.tokens = min(self.max_tokens, self.tokens + new_tokens)  # 注释：更新 token 数
                self.last_update = now  # 注释：更新最后时间

                if self.tokens >= num_tokens:  # 注释：token 足够
                    self.tokens -= num_tokens  # 注释：消耗 token
                    return  # 注释：获取成功

                tokens_needed = num_tokens - self.tokens  # 注释：还需要的 token 数
                wait_time = tokens_needed / self.rate_limit  # 注释：等待时间

            if wait_time > 0:  # 注释：释放锁后等待
                await asyncio.sleep(wait_time)  # 注释：等待补充 token


@register_manager("rate_limited")  # 注释：注册 reward_loop 名称
@register_manager_legacy("rate_limited")  # 注释：兼容旧 reward_manager 注册
class RateLimitedRewardManager(RewardManagerBase):  # 注释：带限流的 RewardManager
    """
    功能：对奖励函数进行并发/RPM/TPM 三层限流，适配 LLM-as-judge 调用。（注释：类职责）
    参数：（注释：构造参数说明，详见 __init__）
      - config.reward_model.max_concurrent/max_rpm/max_tpm/timeout 等。（注释：关键配置）
    返回：（注释：返回值说明）
      - RateLimitedRewardManager 实例。（注释：实例对象）
    副作用：（注释：副作用说明）
      - 使用类级限流器（全局共享）。（注释：类级状态）
    异常/边界条件：（注释：异常与边界）
      - 配置不合法会触发断言或导致等待过长。（注释：配置边界）
    最小示例：（注释：最小可理解示例）
      - 输入：RateLimitedRewardManager(config, tokenizer)
      - 输出：对象初始化完成，可调用 run_single
    调用路径依赖：（注释：调用关系说明）
      - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::RateLimitedRewardManager`
      - 典型调用路径：`RewardLoopWorker._init_reward_fn` -> `RateLimitedRewardManager`
      - 被谁调用：`RewardLoopWorker`
      - 调用了谁（项目内）：`AsyncTokenBucket`、`default_compute_score`
      - 调用了谁（外部依赖）：`asyncio.Semaphore`
    """

    # Class-level state for global rate limiting  # 注释：全局共享限流器
    _semaphore = None  # 注释：并发信号量
    _max_concurrent = None  # 注释：最大并发数
    _rpm_limiter = None  # 注释：请求速率限制器
    _max_rpm = None  # 注释：每分钟请求上限
    _tpm_limiter = None  # 注释：token 速率限制器
    _max_tpm = None  # 注释：每分钟 token 上限
    _estimated_tokens_per_request = None  # 注释：估计每次请求 token 数
    _class_initialized = False  # 注释：类级初始化标记

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):  # 注释：初始化类级限流器
        """
        功能：初始化全局限流器（并发/RPM/TPM），所有实例共享。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config (DictConfig): reward_model 配置。（注释：配置对象）
          - tokenizer (AutoTokenizer): 分词器（未直接使用，保持接口一致）。（注释：占位参数）
        返回：（注释：返回值说明）
          - None。（注释：无显式返回）
        副作用：（注释：副作用说明）
          - 初始化类级信号量与令牌桶。（注释：类级状态变更）
          - 记录日志。（注释：日志副作用）
        异常/边界条件：（注释：异常与边界）
          - 已初始化时直接返回。（注释：幂等逻辑）
        最小示例：（注释：最小可理解示例）
          - 输入：RateLimitedRewardManager.init_class(config, tokenizer)
          - 输出：类级限流器已初始化
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::RateLimitedRewardManager.init_class`
          - 典型调用路径：`RateLimitedRewardManager.__init__` -> `init_class`
          - 被谁调用：`RewardManagerBase.__init__`
          - 调用了谁（项目内）：`AsyncTokenBucket`
          - 调用了谁（外部依赖）：`asyncio.Semaphore`
        """
        # Check if already initialized before calling parent  # 注释：已初始化则直接返回
        if cls._class_initialized:  # 注释：幂等保护
            return  # 注释：避免重复初始化

        super().init_class(config, tokenizer)  # 注释：调用基类类级初始化

        # Concurrency limiter  # 注释：并发限流
        cls._max_concurrent = config.reward_model.get("max_concurrent", 1)  # 注释：读取最大并发
        cls._semaphore = asyncio.Semaphore(cls._max_concurrent)  # 注释：创建信号量

        # Request rate limiter (RPM)  # 注释：请求速率限流
        cls._max_rpm = config.reward_model.get("max_rpm", None)  # 注释：每分钟请求上限
        if cls._max_rpm is not None:  # 注释：启用 RPM 限流
            requests_per_second = cls._max_rpm / 60.0  # 注释：换算为每秒
            cls._rpm_limiter = AsyncTokenBucket(  # 注释：创建 RPM 令牌桶
                rate_limit=requests_per_second, max_tokens=requests_per_second
            )
        else:  # 注释：未配置 RPM
            cls._rpm_limiter = None  # 注释：不启用 RPM 限流

        # Token rate limiter (TPM)  # 注释：token 速率限流
        cls._max_tpm = config.reward_model.get("max_tpm", None)  # 注释：每分钟 token 上限
        cls._estimated_tokens_per_request = config.reward_model.get("estimated_tokens_per_request", 2000)  # 注释：估计每次 token
        if cls._max_tpm is not None:  # 注释：启用 TPM 限流
            tokens_per_second = cls._max_tpm / 60.0  # 注释：换算为每秒
            cls._tpm_limiter = AsyncTokenBucket(  # 注释：创建 TPM 令牌桶
                rate_limit=tokens_per_second, max_tokens=tokens_per_second
            )
        else:  # 注释：未配置 TPM
            cls._tpm_limiter = None  # 注释：不启用 TPM 限流

        log_msg = "Rate limiting configuration:\n"  # 注释：准备日志字符串
        log_msg += f"  - Concurrency limit: {cls._max_concurrent}\n"  # 注释：并发限制
        if cls._max_rpm is not None:  # 注释：记录 RPM 配置
            log_msg += f"  - Request rate limit: {cls._max_rpm} RPM ({cls._max_rpm / 60.0:.2f} RPS)\n"  # 注释：RPM 信息
        else:  # 注释：无限制
            log_msg += "  - Request rate limit: unlimited\n"  # 注释：RPM 无限
        if cls._max_tpm is not None:  # 注释：记录 TPM 配置
            log_msg += f"  - Token rate limit: {cls._max_tpm} TPM ({cls._max_tpm / 60.0:.2f} TPS)\n"  # 注释：TPM 信息
            log_msg += f"  - Estimated tokens per request: {cls._estimated_tokens_per_request}\n"  # 注释：估计 token
        else:  # 注释：无限制
            log_msg += "  - Token rate limit: unlimited\n"  # 注释：TPM 无限
        log_msg += "All limiters are shared globally across all workers."  # 注释：说明全局共享
        logger.info(log_msg)  # 注释：打印配置日志

        cls._class_initialized = True  # 注释：标记已初始化

    def __init__(  # 注释：初始化限流 RewardManager
        self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None
    ):
        """
        功能：保存评分函数与路由信息，并读取超时配置。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config: RewardLoop 配置。（注释：配置对象）
          - tokenizer: 用于解码 responses 的 tokenizer。（注释：分词器）
          - compute_score (Callable|None): 评分函数，默认 default_compute_score。（注释：评分函数）
          - reward_router_address (str|None): RM 路由地址。（注释：路由地址）
          - reward_model_tokenizer (AutoTokenizer|None): RM tokenizer。（注释：RM 分词器）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 保存成员变量。（注释：状态持有）
        异常/边界条件：（注释：异常与边界）
          - 配置缺失 timeout 时使用默认 300 秒。（注释：默认值）
        最小示例：（注释：最小可理解示例）
          - 输入：RateLimitedRewardManager(config, tokenizer)
          - 输出：对象初始化完成
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::RateLimitedRewardManager.__init__`
          - 典型调用路径：`RewardLoopWorker._init_reward_fn` -> `RateLimitedRewardManager(...)`
          - 被谁调用：`RewardLoopWorker`
          - 调用了谁（项目内）：`RewardManagerBase.__init__`
          - 调用了谁（外部依赖）：`inspect.iscoroutinefunction`
        """
        super().__init__(config, tokenizer)  # 注释：调用基类初始化
        self.compute_score = compute_score or default_compute_score  # 注释：设置评分函数
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)  # 注释：判断是否异步评分
        self.reward_router_address = reward_router_address  # 注释：保存路由地址
        self.reward_model_tokenizer = reward_model_tokenizer  # 注释：保存 RM tokenizer
        self.timeout = config.reward_model.get("timeout", 300.0)  # 注释：奖励计算超时

    async def _compute_reward(  # 注释：执行评分函数（支持同步/异步）
        self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict
    ) -> dict | float:
        """
        功能：调用评分函数计算 reward，支持同步/异步评分函数。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data_source (str): 数据源名称。（注释：数据源）
          - solution_str (str): 模型输出文本。（注释：答案文本）
          - ground_truth (str): 标准答案。（注释：正确答案）
          - extra_info (dict): 评分所需的额外信息。（注释：附加信息）
        返回：（注释：返回值说明）
          - dict|float：评分结果（可能含 extra 字段）。（注释：返回类型）
        副作用：（注释：副作用说明）
          - 可能调用外部 reward_router。（注释：外部 I/O）
        异常/边界条件：（注释：异常与边界）
          - compute_score 抛出的异常会向上抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：_compute_reward("openai/gsm8k", "...", "5", {})
          - 输出：{"score": 1.0} 或 1.0
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::RateLimitedRewardManager._compute_reward`
          - 典型调用路径：`run_single` -> `_compute_reward`
          - 被谁调用：`RateLimitedRewardManager.run_single`
          - 调用了谁（项目内）：`default_compute_score`
          - 调用了谁（外部依赖）：`loop.run_in_executor`
        """
        extra_reward_kwargs = (  # 注释：可选的 RM 额外参数
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:  # 注释：评分函数为协程
            return await self.compute_score(  # 注释：直接 await 评分函数
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:  # 注释：评分函数为同步
            return await self.loop.run_in_executor(  # 注释：在线程池中执行同步评分
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=solution_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

    async def run_single(self, data: DataProto) -> dict:  # 注释：处理单条样本并应用限流
        """
        功能：对单条样本进行评分，应用 RPM/TPM/并发限流与超时控制。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 单条样本数据（len==1）。（注释：输入样本）
        返回：（注释：返回值说明）
          - dict：{"reward_score": float, "reward_extra_info": dict}。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 可能调用外部评分服务或等待限流。（注释：I/O 与等待）
        异常/边界条件：（注释：异常与边界）
          - 超时返回 0 分并记录 timeout。（注释：超时处理）
          - 评分异常返回 0 分并记录 error。（注释：异常处理）
        最小示例：（注释：最小可理解示例）
          - 输入：data[0] 的 response/ground_truth
          - 输出：reward_score 与 reward_extra_info
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::RateLimitedRewardManager.run_single`
          - 典型调用路径：`RewardLoopWorker.compute_score` -> `run_single`
          - 被谁调用：`RewardLoopWorker`、`RateLimitedRewardManager.__call__`（批量包装）
          - 调用了谁（项目内）：`_compute_reward`
          - 调用了谁（外部依赖）：`asyncio.wait_for`、`tokenizer.decode`
        """
        assert len(data) == 1, "Only support single data item"  # 注释：仅支持单条样本
        data_item = data[0]  # 注释：取出样本

        response_ids = data_item.batch["responses"]  # 注释：response token ids
        response_length = response_ids.shape[-1]  # 注释：response 长度
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()  # 注释：有效长度
        valid_response_ids = response_ids[:valid_response_length]  # 注释：截取有效 response

        data_source = data_item.non_tensor_batch["data_source"]  # 注释：数据源
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]  # 注释：标准答案
        extra_info = data_item.non_tensor_batch.get("extra_info", {})  # 注释：额外信息
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)  # 注释：工具字段（可选）
        if tool_extra_fields is not None:  # 注释：存在工具字段时合并
            extra_info.update(tool_extra_fields.items())  # 注释：合并到 extra_info

        response_str = await self.loop.run_in_executor(  # 注释：异步解码 response 文本
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        reward_extra_info = {}  # 注释：收集额外信息

        # Apply rate limiting layers  # 注释：应用三层限流
        if self._rpm_limiter is not None:  # 注释：RPM 限流
            await self._rpm_limiter.acquire(1.0)  # 注释：每次请求消耗 1 token

        if self._tpm_limiter is not None:  # 注释：TPM 限流
            estimated_tokens = self._estimated_tokens_per_request  # 注释：估计 token 数
            await self._tpm_limiter.acquire(estimated_tokens)  # 注释：消耗 token

        async with self._semaphore:  # 注释：并发限制
            try:  # 注释：评分计算（带超时）
                result = await asyncio.wait_for(  # 注释：超时控制
                    self._compute_reward(  # 注释：调用评分函数
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    ),
                    timeout=self.timeout,  # 注释：超时秒数
                )

                score: float  # 注释：评分结果
                if isinstance(result, dict):  # 注释：字典形式返回
                    score = result["score"]  # 注释：主分数
                    for key, value in result.items():  # 注释：收集所有字段
                        reward_extra_info[key] = value  # 注释：写入额外信息
                else:  # 注释：数值形式返回
                    score = result  # 注释：直接作为分数
                    reward_extra_info["acc"] = score  # 注释：记录准确率字段

                reward = score  # 注释：reward 即 score

            except asyncio.TimeoutError:  # 注释：超时处理
                logger.warning(  # 注释：记录超时日志
                    f"Reward computation timed out after {self.timeout}s for data_source={data_source}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward = 0.0  # 注释：超时奖励置 0
                reward_extra_info["timeout"] = True  # 注释：记录超时标记
                reward_extra_info["acc"] = 0.0  # 注释：记录准确率 0

            except Exception as e:  # 注释：其他异常处理
                logger.error(  # 注释：记录异常日志
                    f"Reward computation failed for data_source={data_source}: {e}. "
                    f"Response preview: {response_str[:100]}..."
                )
                reward = 0.0  # 注释：异常奖励置 0
                reward_extra_info["error"] = str(e)  # 注释：记录错误信息
                reward_extra_info["acc"] = 0.0  # 注释：记录准确率 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}  # 注释：返回结果

    def __call__(self, data: DataProto, return_dict: bool = False):  # 注释：兼容旧 RewardManager 接口
        """
        功能：将异步 run_single 批量化并同步返回奖励张量。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 批量数据。（注释：输入 batch）
          - return_dict (bool): 是否返回包含 extra_info 的字典。（注释：返回控制）
        返回：（注释：返回值说明）
          - torch.Tensor 或 dict：reward_tensor 或包含 reward_extra_info。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 使用事件循环执行异步任务。（注释：事件循环副作用）
        异常/边界条件：（注释：异常与边界）
          - 若数据已包含 rm_scores 则直接返回。（注释：快路径）
        最小示例：（注释：最小可理解示例）
          - 输入：data=batch, return_dict=True
          - 输出：{"reward_tensor": ..., "reward_extra_info": ...}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_manager/limited.py::RateLimitedRewardManager.__call__`
          - 典型调用路径：旧版 reward_manager 调用 -> `__call__`
          - 被谁调用：`verl/trainer/ppo/reward.py::compute_reward`（当使用该 manager）
          - 调用了谁（项目内）：`run_single`
          - 调用了谁（外部依赖）：`asyncio.gather`、`torch.zeros_like`
        """
        from collections import defaultdict  # 注释：默认列表容器

        import torch  # 注释：张量构造

        # If there are pre-computed rm_scores, return them directly  # 注释：兼容 reward loop 预计算
        if "rm_scores" in data.batch.keys():  # 注释：已有 rm_scores 直接返回
            if return_dict:  # 注释：需要返回字典
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])  # 注释：读取额外字段名
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}  # 注释：提取额外信息
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}  # 注释：返回字典
            else:  # 注释：仅返回张量
                return data.batch["rm_scores"]  # 注释：返回 rm_scores

        # Initialize reward tensor  # 注释：初始化 reward 张量
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # 注释：与 responses 同形
        reward_extra_info = defaultdict(list)  # 注释：收集额外信息

        # Process each data item through the async event loop  # 注释：异步批处理
        async def process_batch():  # 注释：批量运行 run_single
            """
            功能：为 batch 内每条样本创建任务并并发执行。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - 无（使用外层 data）。（注释：闭包变量）
            返回：（注释：返回值说明）
              - list[dict]：每条样本的 run_single 结果。（注释：返回列表）
            副作用：（注释：副作用说明）
              - 创建并调度协程任务。（注释：异步调度）
            异常/边界条件：（注释：异常与边界）
              - run_single 异常会向上抛出。（注释：异常传播）
            最小示例：（注释：最小可理解示例）
              - 输入：data 长度 N
              - 输出：长度 N 的结果列表
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`RateLimitedRewardManager.__call__.process_batch`
              - 典型调用路径：`__call__` -> `process_batch`
              - 被谁调用：`__call__`
              - 调用了谁（项目内）：`run_single`
              - 调用了谁（外部依赖）：`asyncio.gather`
            """
            tasks = []  # 注释：任务列表
            for i in range(len(data)):  # 注释：遍历样本
                data_item = data[i : i + 1]  # 注释：取单条样本切片
                tasks.append(self.run_single(data_item))  # 注释：追加协程任务

            results = await asyncio.gather(*tasks)  # 注释：并发执行任务
            return results  # 注释：返回结果列表

        # Run the async processing using self.loop property which lazily gets/creates event loop  # 注释：使用同一事件循环
        # This ensures rate limiters and semaphores work correctly by using the same loop  # 注释：保证限流器一致性
        results = self.loop.run_until_complete(process_batch())  # 注释：同步等待结果

        # Aggregate results into reward tensor and extra info  # 注释：聚合结果
        for i, result in enumerate(results):  # 注释：遍历结果
            data_item = data[i]  # 注释：对应样本
            response_ids = data_item.batch["responses"]  # 注释：response token ids
            response_length = response_ids.shape[-1]  # 注释：response 长度
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()  # 注释：有效长度

            reward = result["reward_score"]  # 注释：取 reward 分数
            reward_tensor[i, valid_response_length - 1] = reward  # 注释：写入最后有效 token

            # Collect extra info  # 注释：收集额外信息
            if "reward_extra_info" in result:  # 注释：存在额外信息
                for key, value in result["reward_extra_info"].items():  # 注释：遍历字段
                    reward_extra_info[key].append(value)  # 注释：追加到列表

        if return_dict:  # 注释：需要返回字典
            return {  # 注释：返回包含额外信息的字典
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:  # 注释：仅返回张量
            return reward_tensor  # 注释：返回 reward 张量
