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
模块用途：实现 RewardLoopWorker 与 RewardLoopManager，用于异步/远程计算奖励。（注释：模块功能概述）
输入：DataProto（含 prompt/response/metadata）、DictConfig 配置、reward router 地址。（注释：输入形态说明）
输出：奖励得分与可选的 reward_extra_info，或包含 rm_scores 的 DataProto。（注释：输出形态说明）
关键依赖：ray、aiohttp、RewardModelManager、RewardManagerBase。（注释：关键依赖）
典型用法：（注释：最小使用示例）
  - rm_manager = RewardLoopManager(config)
  - data_with_rm_scores = rm_manager.compute_rm_score(data)
调用路径概览：（注释：全局调用关系）
  - verl/trainer/ppo/reward.py::load_reward_manager -> RewardLoopManager / RewardLoopWorker
"""

import asyncio  # 注释：异步调度与等待
import logging  # 注释：日志记录
import os  # 注释：环境变量

import aiohttp  # 注释：异步 HTTP 客户端
import numpy as np  # 注释：非张量数组处理
import ray  # 注释：分布式任务框架
import torch  # 注释：张量操作
from omegaconf import DictConfig  # 注释：配置类型
from tensordict import TensorDict  # 注释：TensorDict 容器

from verl.protocol import DataProto  # 注释：数据容器
from verl.single_controller.ray.base import RayResourcePool  # 注释：Ray 资源池
from verl.trainer.ppo.reward import get_custom_reward_fn  # 注释：加载自定义奖励函数
from verl.utils import hf_tokenizer  # 注释：加载 HuggingFace tokenizer
from verl.utils.fs import copy_to_local  # 注释：将权重/模型拷贝到本地

from .reward_manager import get_reward_manager_cls  # 注释：reward_loop 注册表查询
from .reward_model import RewardModelManager  # 注释：RewardModel 管理器

logger = logging.getLogger(__file__)  # 注释：模块级日志器
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：日志级别


@ray.remote  # 注释：将 RewardLoopWorker 作为 Ray Actor 运行
class RewardLoopWorker:  # 注释：奖励计算工作进程
    """
    功能：在 Ray Actor 中执行奖励计算，支持规则/模型/自定义函数三种路径。（注释：类职责）
    输入：DictConfig 配置、reward_router_address。（注释：输入形态说明）
    输出：reward_score 与 reward_extra_info（dict）。（注释：输出形态说明）
    关键依赖：RewardManagerBase 子类、RewardModelManager、aiohttp。（注释：关键依赖）
    典型用法：（注释：最小使用示例）
      - worker = RewardLoopWorker.remote(config, router_addr)
      - await worker.compute_score.remote(data)
    调用路径概览：（注释：全局调用关系）
      - RewardLoopManager._init_reward_loop_workers -> RewardLoopWorker
    """

    def __init__(self, config: DictConfig, reward_router_address: str = None):  # 注释：初始化 worker
        """
        功能：保存配置与路由地址，并初始化 reward manager。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config (DictConfig): RewardLoop 配置。（注释：配置对象）
          - reward_router_address (str|None): reward router 地址。（注释：路由地址）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 加载 tokenizer、reward manager。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - 配置缺失路径时会触发断言。（注释：配置校验）
        最小示例：（注释：最小可理解示例）
          - 输入：RewardLoopWorker(config, "127.0.0.1:8000")
          - 输出：worker 初始化完成
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker.__init__`
          - 典型调用路径：`RewardLoopManager._init_reward_loop_workers` -> `RewardLoopWorker`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：`_init_reward_fn`
          - 调用了谁（外部依赖）：无
        """
        self.config = config  # 注释：保存配置
        self.reward_router_address = reward_router_address  # 注释：保存路由地址
        self._init_reward_fn()  # 注释：初始化 reward 相关对象

    def _init_reward_fn(self):  # 注释：初始化 reward manager 与 tokenizer
        """
        功能：加载 tokenizer、reward manager 类，并实例化 reward_loop。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无（使用 self.config）。（注释：配置来源）
        返回：（注释：返回值说明）
          - None。（注释：无显式返回）
        副作用：（注释：副作用说明）
          - 读取模型路径并加载 tokenizer。（注释：I/O 副作用）
          - 动态导入 reward manager 类。（注释：动态导入）
        异常/边界条件：（注释：异常与边界）
          - reward_loop_source 非法会抛 ValueError。（注释：配置校验）
        最小示例：（注释：最小可理解示例）
          - 输入：config.reward_model.reward_manager="naive"
          - 输出：self.reward_loop 初始化完成
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker._init_reward_fn`
          - 典型调用路径：`RewardLoopWorker.__init__` -> `_init_reward_fn`
          - 被谁调用：`RewardLoopWorker`
          - 调用了谁（项目内）：`get_custom_reward_fn`、`get_reward_manager_cls`
          - 调用了谁（外部依赖）：`hf_tokenizer`、`copy_to_local`
        """
        input_tokenizer_local_path = copy_to_local(  # 注释：将 actor tokenizer 拷贝到本地
            self.config.actor_rollout_ref.model.path
        )
        self.input_tokenizer = hf_tokenizer(  # 注释：加载输入 tokenizer
            input_tokenizer_local_path, trust_remote_code=True
        )
        self.reward_model_tokenizer = None  # 注释：默认无 reward_model tokenizer
        if self.config.reward_model.enable:  # 注释：启用 reward_model 时加载其 tokenizer
            reward_model_tokenizer_local_path = copy_to_local(  # 注释：拷贝 RM tokenizer
                self.config.reward_model.model.path
            )
            self.reward_model_tokenizer = hf_tokenizer(  # 注释：加载 RM tokenizer
                reward_model_tokenizer_local_path, trust_remote_code=True
            )
        self.reward_fn = get_custom_reward_fn(self.config)  # 注释：加载自定义奖励函数（若有）

        # Load reward loop manager class  # 注释：选择 reward manager 类
        # Support both registry and importlib loading methods  # 注释：支持注册表/动态导入
        reward_loop_source = self.config.reward_model.get("reward_loop_source", "register")  # 注释：加载方式

        if reward_loop_source == "register":  # 注释：从注册表加载（默认）
            # Load from registry (default behavior)  # 注释：原注释保留
            reward_manager_cls = get_reward_manager_cls(self.config.reward_model.reward_manager)  # 注释：按名称取类
        elif reward_loop_source == "importlib":  # 注释：从外部模块动态加载
            # Load from external module using importlib  # 注释：原注释保留
            from verl.utils.import_utils import load_extern_object  # 注释：动态导入工具

            reward_loop_module_path = self.config.reward_model.get("reward_loop_module_path", None)  # 注释：模块路径
            reward_loop_class_name = self.config.reward_model.get("reward_loop_class_name", None)  # 注释：类名

            assert reward_loop_module_path is not None, (  # 注释：路径必须提供
                "reward_loop_module_path must be set when reward_loop_source='importlib'"
            )
            assert reward_loop_class_name is not None, (  # 注释：类名必须提供
                "reward_loop_class_name must be set when reward_loop_source='importlib'"
            )

            reward_manager_cls = load_extern_object(  # 注释：动态加载类
                module_path=reward_loop_module_path, object_name=reward_loop_class_name
            )
        else:  # 注释：未知加载方式
            raise ValueError(  # 注释：显式报错
                f"Unknown reward_loop_source: {reward_loop_source}. Must be 'register' or 'importlib'"
            )

        self.reward_loop = reward_manager_cls(  # 注释：实例化 reward manager
            self.config,
            self.input_tokenizer,
            self.reward_fn,
            self.reward_router_address,
            self.reward_model_tokenizer,
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:  # 注释：批量计算 reward
        """
        功能：对 batch 中每条样本并发调用 compute_score。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 批量数据。（注释：输入 batch）
        返回：（注释：返回值说明）
          - list[dict]：每条样本的评分结果列表。（注释：返回列表）
        副作用：（注释：副作用说明）
          - 创建 asyncio 任务并发执行。（注释：异步调度）
        异常/边界条件：（注释：异常与边界）
          - 单条 compute_score 异常会向上抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：data 长度 N
          - 输出：长度 N 的结果列表
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker.compute_score_batch`
          - 典型调用路径：`RewardLoopManager.compute_rm_score` -> `compute_score_batch`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：`compute_score`
          - 调用了谁（外部依赖）：`asyncio.create_task`、`asyncio.gather`
        """
        tasks = []  # 注释：任务列表
        for i in range(len(data)):  # 注释：遍历 batch
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))  # 注释：为每条样本创建任务
        outputs = await asyncio.gather(*tasks)  # 注释：并发执行
        return outputs  # 注释：返回结果列表

    async def compute_score(self, data: DataProto) -> dict:  # 注释：计算单条样本 reward
        """
        功能：根据配置选择自定义函数、规则评分或 RM 评分路径。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 单条样本数据（len==1）。（注释：输入样本）
        返回：（注释：返回值说明）
          - dict：reward 结果字典。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 可能调用 reward model 服务。（注释：外部 I/O）
        异常/边界条件：（注释：异常与边界）
          - 若启用 genrm 但未配置自定义函数，可能触发异常路径。（注释：配置约束）
        最小示例：（注释：最小可理解示例）
          - 输入：config.custom_reward_function.path 非空
          - 输出：调用自定义 reward 函数结果
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker.compute_score`
          - 典型调用路径：`RewardLoopWorker.compute_score_batch` -> `compute_score`
          - 被谁调用：`RewardLoopWorker.compute_score_batch`
          - 调用了谁（项目内）：`reward_loop.run_single`、`compute_score_disrm`
          - 调用了谁（外部依赖）：无
        """
        assert len(data) == 1, "RewardLoopWorker only support single data item"  # 注释：仅支持单条样本
        if self.config.custom_reward_function.path is not None:  # 注释：有自定义奖励函数
            # directly use user-customized reward function  # 注释：直接使用自定义函数
            return await self.reward_loop.run_single(data)  # 注释：调用 reward_loop
        else:  # 注释：未配置自定义函数
            if self.config.reward_model.enable:  # 注释：启用 reward_model
                # we assume the rm is disrm  # 注释：默认走 disrm
                # genrm must set custom_reward_function  # 注释：genrm 需自定义函数
                return await self.compute_score_disrm(data)  # 注释：使用 disrm 评分
            else:  # 注释：未启用 reward_model
                return await self.reward_loop.run_single(data)  # 注释：走规则评分

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):  # 注释：发送 HTTP 请求
        """
        功能：向 reward_router 发送 POST 请求并支持重试与指数退避。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - payload (dict): 请求体数据。（注释：POST 数据）
          - endpoint (str): 目标接口路径。（注释：接口路径）
          - max_retries (int): 最大重试次数。（注释：重试上限）
        返回：（注释：返回值说明）
          - dict：响应 JSON。（注释：响应内容）
        副作用：（注释：副作用说明）
          - 产生网络请求。（注释：网络 I/O）
          - 记录日志。（注释：日志副作用）
        异常/边界条件：（注释：异常与边界）
          - 4xx 错误不重试并直接抛出。（注释：客户端错误）
          - 重试耗尽后抛最后异常。（注释：重试耗尽）
        最小示例：（注释：最小可理解示例）
          - 输入：_post_request({"model": "xx"}, "classify")
          - 输出：{"data": ...}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker._post_request`
          - 典型调用路径：`compute_score_disrm` -> `_post_request`
          - 被谁调用：`compute_score_disrm`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`aiohttp.ClientSession`
        """
        url = f"http://{self.reward_router_address}/{endpoint}"  # 注释：拼接请求 URL
        last_exception = None  # 注释：记录最后异常
        for attempt in range(max_retries):  # 注释：重试循环
            try:  # 注释：发送请求
                # It's safer to have a timeout instead of None, which can hang indefinitely.  # 注释：原注释保留
                timeout = aiohttp.ClientTimeout(total=None)  # 注释：设置超时对象
                async with aiohttp.ClientSession(timeout=timeout) as session:  # 注释：创建会话
                    async with session.post(url, json=payload) as resp:  # 注释：发送 POST
                        resp.raise_for_status()  # 注释：非 2xx 抛异常
                        return await resp.json()  # 注释：返回 JSON 响应
            except aiohttp.ClientResponseError as e:  # 注释：HTTP 状态异常
                # Do not retry on 4xx client errors, but retry on 5xx server errors.  # 注释：原注释保留
                if 400 <= e.status < 500:  # 注释：客户端错误不重试
                    logger.error(  # 注释：记录错误
                        f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying."
                    )
                    raise  # 注释：直接抛出
                last_exception = e  # 注释：记录异常
                logger.warning(  # 注释：记录重试信息
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:  # 注释：超时或连接错误
                last_exception = e  # 注释：记录异常
                logger.warning(  # 注释：记录重试信息
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying..."
                )
            except Exception as e:  # 注释：其他异常
                last_exception = e  # 注释：记录异常
                logger.warning(  # 注释：记录重试信息
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:  # 注释：未到最后一次则退避
                # Using exponential backoff is generally better than a fixed sleep.  # 注释：原注释保留
                backoff_seconds = 2**attempt  # 注释：指数退避秒数
                await asyncio.sleep(min(backoff_seconds, 30))  # 注释：等待（上限 30s）

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")  # 注释：记录重试耗尽
        if last_exception:  # 注释：抛出最后异常
            raise last_exception  # 注释：异常向上抛

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:  # 注释：构造 RM 输入 prompt
        """
        功能：将 raw_prompt 与模型 response 拼接为 reward_model 输入文本。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 单条样本数据。（注释：输入样本）
        返回：（注释：返回值说明）
          - str：用于 RM 推理的 prompt 文本。（注释：返回文本）
        副作用：（注释：副作用说明）
          - 无（纯文本处理）。（注释：无副作用）
        异常/边界条件：（注释：异常与边界）
          - raw_prompt 缺失会触发断言。（注释：字段依赖）
        最小示例：（注释：最小可理解示例）
          - 输入：raw_prompt=[{"role":"user","content":"1+1"}], response="2"
          - 输出：RM 格式化 prompt
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker._preprocess_reward_inputs`
          - 典型调用路径：`compute_score_disrm` -> `_preprocess_reward_inputs`
          - 被谁调用：`compute_score_disrm`
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`tokenizer.decode`、`apply_chat_template`
        """
        assert len(data) == 1, "RewardLoopWorker only support single data item"  # 注释：仅支持单条样本
        data_item = data[0]  # 注释：取出样本
        assert "raw_prompt" in data_item.non_tensor_batch  # 注释：必须提供 raw_prompt

        # extract raw prompt  # 注释：提取原始对话
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])  # 注释：复制为列表

        # extract response  # 注释：提取 response ids
        response_ids = data_item.batch["responses"]  # 注释：response token ids
        response_length = response_ids.shape[-1]  # 注释：response 长度
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()  # 注释：有效长度
        valid_response_ids = response_ids[:valid_response_length]  # 注释：截取有效 response

        # decode  # 注释：解码 response 文本
        rollout_response = self.input_tokenizer.decode(valid_response_ids)  # 注释：解码为字符串
        # remove bos and eos  # 注释：移除 eos 等特殊符号
        rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")  # 注释：去除 eos

        chat.append({"role": "assistant", "content": rollout_response})  # 注释：将模型回答加入对话

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(  # 注释：应用 RM 的 chat 模板
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default  # 注释：llama 默认加 bos
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False  # 注释：版本提示
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(  # 注释：若有 bos 且开头匹配
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]  # 注释：移除 bos 前缀

        return rm_prompt  # 注释：返回 RM prompt

    async def compute_score_disrm(self, data: DataProto) -> dict:  # 注释：使用 disrm 计算奖励
        """
        功能：调用 disrm（判别式 RM）服务获取奖励分数。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 单条样本数据。（注释：输入样本）
        返回：（注释：返回值说明）
          - dict：{"reward_score": float}。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 发送 HTTP 请求到 reward_router。（注释：网络 I/O）
        异常/边界条件：（注释：异常与边界）
          - 不支持的 engine_name 会抛 NotImplementedError。（注释：引擎限制）
        最小示例：（注释：最小可理解示例）
          - 输入：engine_name="vllm"
          - 输出：{"reward_score": 0.87}
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopWorker.compute_score_disrm`
          - 典型调用路径：`compute_score` -> `compute_score_disrm`
          - 被谁调用：`compute_score`
          - 调用了谁（项目内）：`_preprocess_reward_inputs`、`_post_request`
          - 调用了谁（外部依赖）：无
        """
        disrm_prompt = await self._preprocess_reward_inputs(data)  # 注释：构造 RM prompt
        engine_name = self.config.reward_model.rollout.name  # 注释：推理引擎名称
        model_name = self.config.reward_model.model.path  # 注释：模型路径
        if engine_name == "vllm":  # 注释：vLLM 引擎
            # TODO (dyy): the "activation" has been changed to "use_activation" in vllm 0.11.2  # 注释：版本提示
            payloads = {  # 注释：构造 vLLM 请求体
                "model": model_name,
                "input": disrm_prompt,
                "activation": False,
                # "add_special_tokens": False,  # vllm >= 0.11.2  # 注释：新版本参数
            }
            output = await self._post_request(payloads, "classify")  # 注释：调用 classify 接口
            rm_score = output["data"][-1]["probs"][-1]  # 注释：取最后一个概率作为分数
        elif engine_name == "sglang":  # 注释：SGLang 引擎
            payloads = {  # 注释：构造 SGLang 请求体
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")  # 注释：调用 embeddings 接口
            rm_score = output["data"][-1]["embedding"][-1]  # 注释：取最后维度作为分数
        else:  # 注释：未知引擎
            raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")  # 注释：显式报错

        return {"reward_score": rm_score}  # 注释：返回分数


class RewardLoopManager:  # 注释：RewardLoop 管理器
    """
    功能：在单控制器中创建并管理 RewardLoopWorker；可选启动 RewardModelManager。（注释：类职责）
    输入：DictConfig、RayResourcePool（可选）。（注释：输入形态说明）
    输出：提供 compute_rm_score 返回包含 rm_scores 的 DataProto。（注释：输出形态说明）
    关键依赖：RewardLoopWorker、RewardModelManager、ray。（注释：关键依赖）
    典型用法：（注释：最小使用示例）
      - mgr = RewardLoopManager(config)
      - data = mgr.compute_rm_score(batch)
    调用路径概览：（注释：全局调用关系）
      - trainer/ppo/ray_trainer.py 或 reward_loop 流程调用 RewardLoopManager
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):  # 注释：初始化管理器
        """
        功能：根据配置初始化 RewardModelManager，并启动 RewardLoopWorker。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - config (DictConfig): RewardLoop 配置。（注释：配置对象）
          - rm_resource_pool (RayResourcePool|None): RM 资源池（可选）。（注释：资源池）
        返回：（注释：返回值说明）
          - None。（注释：初始化无返回）
        副作用：（注释：副作用说明）
          - 启动 Ray Actor（RewardLoopWorker）。（注释：资源副作用）
          - 可能启动 RewardModelManager。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - Ray 未初始化会抛异常。（注释：依赖边界）
        最小示例：（注释：最小可理解示例）
          - 输入：RewardLoopManager(config)
          - 输出：reward_loop_workers 已创建
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopManager.__init__`
          - 典型调用路径：上层 trainer -> RewardLoopManager
          - 被谁调用：`reward.py` 或 recipe 脚本
          - 调用了谁（项目内）：`_init_reward_loop_workers`、`RewardModelManager`
          - 调用了谁（外部依赖）：`ray`
        """
        self.config = config  # 注释：保存配置
        if self.config.reward_model.enable:  # 注释：启用 reward_model 时创建 RM 管理器
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)  # 注释：创建 RM 管理器
            self.reward_router_address = self.reward_model_manager.get_router_address()  # 注释：获取 router 地址
        else:  # 注释：未启用 reward_model
            self.reward_model_manager = None  # 注释：不创建 RM 管理器
            self.reward_router_address = None  # 注释：无 router 地址

        self._init_reward_loop_workers()  # 注释：创建 RewardLoopWorker

    def _init_reward_loop_workers(self):  # 注释：创建 reward_loop workers
        """
        功能：按配置创建 RewardLoopWorker，并进行节点亲和调度。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - 无（使用 self.config）。（注释：配置来源）
        返回：（注释：返回值说明）
          - None。（注释：无显式返回）
        副作用：（注释：副作用说明）
          - 创建 Ray Actor。（注释：资源副作用）
        异常/边界条件：（注释：异常与边界）
          - 节点列表为空会导致索引错误。（注释：资源边界）
        最小示例：（注释：最小可理解示例）
          - 输入：num_workers=2, node_ids=[n0,n1]
          - 输出：创建两个 RewardLoopWorker
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopManager._init_reward_loop_workers`
          - 典型调用路径：`RewardLoopManager.__init__` -> `_init_reward_loop_workers`
          - 被谁调用：`RewardLoopManager`
          - 调用了谁（项目内）：`RewardLoopWorker.options`
          - 调用了谁（外部依赖）：`ray.nodes`
        """
        self.reward_loop_workers = []  # 注释：保存 worker 列表
        num_workers = self.config.reward_model.num_workers  # 注释：worker 数量
        node_ids = [  # 注释：获取可用节点列表
            node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0
        ]

        for i in range(num_workers):  # 注释：为每个 worker 分配节点
            # Round-robin scheduling over the all nodes  # 注释：轮询调度
            node_id = node_ids[i % len(node_ids)]  # 注释：选择节点
            self.reward_loop_workers.append(  # 注释：创建并记录 worker
                RewardLoopWorker.options(  # 注释：设置 Ray Actor 选项
                    name=f"reward_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(  # 注释：节点亲和
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)  # 注释：启动 Actor
            )

    # this func is used to replace the legacy fsdp/megatron RewardModelWorker.compute_rm_score  # 注释：兼容旧接口
    def compute_rm_score(self, data: DataProto) -> DataProto:  # 注释：批量计算 rm_scores
        """
        功能：将 batch 切分给多个 RewardLoopWorker 并汇总 rm_scores。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - data (DataProto): 批量数据。（注释：输入 batch）
        返回：（注释：返回值说明）
          - DataProto：包含 rm_scores 与 reward_extra_info。（注释：返回结构）
        副作用：（注释：副作用说明）
          - 可能唤醒/休眠 RewardModelManager。（注释：资源副作用）
          - 发起 Ray 远程调用。（注释：分布式调用）
        异常/边界条件：（注释：异常与边界）
          - reward_loop_workers 为空会导致除零或索引错误。（注释：资源边界）
        最小示例：（注释：最小可理解示例）
          - 输入：data=batch size=4
          - 输出：DataProto.batch["rm_scores"] 形状与 responses 对齐
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopManager.compute_rm_score`
          - 典型调用路径：trainer -> RewardLoopManager.compute_rm_score
          - 被谁调用：旧 RewardModelWorker 调用路径或新 reward_loop 流程
          - 调用了谁（项目内）：`RewardLoopWorker.compute_score_batch`
          - 调用了谁（外部依赖）：`ray.get`
        """
        if self.reward_model_manager is not None:  # 注释：若有 RM 管理器则唤醒
            self.reward_model_manager.wake_up()  # 注释：唤醒 RM 服务

        chunks = data.chunk(len(self.reward_loop_workers))  # 注释：按 worker 数切分 batch
        outputs = ray.get(  # 注释：并行获取各 worker 结果
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers, chunks, strict=True)
            ]
        )
        outputs_flat = [item for sublist in outputs for item in sublist]  # 注释：展平成列表

        # compute rm score  # 注释：将分数写回到 rm_scores 张量
        scores = [item["reward_score"] for item in outputs_flat]  # 注释：提取分数列表
        prompt_length = data.batch["prompts"].size(1)  # 注释：prompt 长度
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)  # 注释：有效 response 长度
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)  # 注释：初始化 rm_scores
        rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = torch.tensor(  # 注释：写入最后 token
            scores, dtype=torch.float32
        )
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))  # 注释：构造 TensorDict

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]  # 注释：提取额外信息
        reward_extra_keys = list(reward_extra_infos[0].keys())  # 注释：额外字段名
        non_tensor_batch = {}  # 注释：非张量字段容器
        for key in reward_extra_keys:  # 注释：汇总额外信息为数组
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])  # 注释：堆叠为 numpy

        if self.reward_model_manager is not None:  # 注释：若有 RM 管理器则休眠
            self.reward_model_manager.sleep()  # 注释：释放 RM 资源

        return DataProto(  # 注释：返回包含 rm_scores 的 DataProto
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):  # 注释：同步运行异步任务列表
        """
        功能：在新事件循环中运行一组协程任务。（注释：函数目标说明）
        参数：（注释：函数参数说明）
          - tasks (list[asyncio.Task]): 待执行任务列表。（注释：任务列表）
        返回：（注释：返回值说明）
          - list：任务结果列表。（注释：返回结果）
        副作用：（注释：副作用说明）
          - 创建并运行事件循环。（注释：事件循环副作用）
        异常/边界条件：（注释：异常与边界）
          - 任务内部异常会向上抛出。（注释：异常传播）
        最小示例：（注释：最小可理解示例）
          - 输入：_run_all([task1, task2])
          - 输出：[res1, res2]
        调用路径依赖：（注释：调用关系说明）
          - 所在位置：`verl/experimental/reward_loop/reward_loop.py::RewardLoopManager._run_all`
          - 典型调用路径：内部辅助函数
          - 被谁调用：当前类内部
          - 调用了谁（项目内）：无
          - 调用了谁（外部依赖）：`asyncio.gather`、`asyncio.run`
        """
        async def run_all():  # 注释：封装 gather
            """
            功能：并发执行任务并返回结果。（注释：内部函数说明）
            参数：（注释：函数参数说明）
              - 无（使用外层 tasks）。（注释：闭包变量）
            返回：（注释：返回值说明）
              - list：任务结果列表。（注释：返回结果）
            副作用：（注释：副作用说明）
              - 调度异步任务。（注释：异步调度）
            异常/边界条件：（注释：异常与边界）
              - 任务异常会向上抛出。（注释：异常传播）
            最小示例：（注释：最小可理解示例）
              - 输入：tasks=[task1, task2]
              - 输出：[res1, res2]
            调用路径依赖：（注释：调用关系说明）
              - 所在位置：`RewardLoopManager._run_all.run_all`
              - 典型调用路径：`_run_all` -> `run_all`
              - 被谁调用：`RewardLoopManager._run_all`
              - 调用了谁（项目内）：无
              - 调用了谁（外部依赖）：`asyncio.gather`
            """
            return await asyncio.gather(*tasks)  # 注释：并发执行

        return asyncio.run(run_all())  # 注释：运行事件循环并返回结果
