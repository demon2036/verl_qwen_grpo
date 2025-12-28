# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
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
模块用途：实现单轮（single turn）Agent Loop，用于最简单的提示 -> 生成流程。  # 注释：模块用途
输入：raw_prompt、多模态数据、采样参数等。  # 注释：输入说明
输出：AgentLoopOutput（含 prompt_ids/response_ids/metrics）。  # 注释：输出说明
关键依赖：AgentLoopBase、tokenizer/processor、server_manager。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- 在配置中设置 agent_name=single_turn_agent，通过 AgentLoopManager 创建并调用 run。  # 注释：示例
调用路径概览：  # 注释：调用路径标题
- AgentLoopManager.generate_sequences -> SingleTurnAgentLoop.run。  # 注释：调用链
"""  # 注释：模块 docstring 结束

import copy  # 注释：标准库，深拷贝多模态数据
import logging  # 注释：标准库，日志
import os  # 注释：标准库，环境变量
from typing import Any  # 注释：类型注解
from uuid import uuid4  # 注释：生成请求 ID

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register  # 注释：AgentLoop 基类与注册器
from verl.utils.profiler import simple_timer  # 注释：性能计时器

logger = logging.getLogger(__file__)  # 注释：获取模块 logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：设置日志级别


@register("single_turn_agent")  # 注释：注册 AgentLoop 名称
class SingleTurnAgentLoop(AgentLoopBase):  # 注释：单轮 AgentLoop 实现
    """
    功能：提供单轮对话的生成逻辑（无工具调用/多轮状态）。  # 注释：类用途
    参数：继承 AgentLoopBase 的初始化参数。  # 注释：参数说明
    返回：实例化后的 SingleTurnAgentLoop。  # 注释：返回值说明
    副作用：初始化时读取配置与 tokenizer/processor。  # 注释：副作用说明
    异常/边界条件：配置缺失必要字段时可能抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - loop = SingleTurnAgentLoop(config, server_manager, tokenizer, processor=None)  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/single_turn_agent_loop.py::SingleTurnAgentLoop。  # 注释：类位置
    - 典型调用路径：AgentLoopManager -> SingleTurnAgentLoop.run。  # 注释：调用链
    - 被谁调用：verl/experimental/agent_loop/agent_loop.py。  # 注释：调用方说明
    - 调用了谁（项目内）：AgentLoopBase、simple_timer。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：uuid4、tokenizer/processor。  # 注释：外部依赖
    """  # 注释：类 docstring 结束

    def __init__(self, *args, **kwargs):  # 注释：初始化单轮 AgentLoop
        """
        功能：初始化单轮 AgentLoop 的配置与关键字段。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - *args/**kwargs：透传给 AgentLoopBase。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - None（原地初始化）。  # 注释：返回值语义
        副作用：设置 prompt_length/response_length 等成员变量。  # 注释：副作用说明
        异常/边界条件：配置缺失字段时可能抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - SingleTurnAgentLoop(config, server_manager, tokenizer)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/single_turn_agent_loop.py::SingleTurnAgentLoop.__init__。  # 注释：函数位置
        - 典型调用路径：AgentLoopManager._init_agent_loop_workers -> SingleTurnAgentLoop(...)。  # 注释：调用链
        - 被谁调用：verl/experimental/agent_loop/agent_loop.py。  # 注释：调用方说明
        - 调用了谁（项目内）：AgentLoopBase.__init__。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        super().__init__(*args, **kwargs)  # 注释：调用父类初始化
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length  # 注释：读取 prompt 长度
        self.response_length = self.config.actor_rollout_ref.rollout.response_length  # 注释：读取 response 长度
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})  # 注释：读取模板参数

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:  # 注释：执行单轮生成
        """
        功能：将 raw_prompt 转为 token，并调用 server_manager 生成响应。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - sampling_params (dict)：采样参数（temperature/top_p 等）。  # 注释：参数含义
        - **kwargs：包含 raw_prompt、multi_modal_data 等。  # 注释：参数含义
        返回：  # 注释：返回值说明标题
        - AgentLoopOutput：包含 prompt/response/metrics 的输出。  # 注释：返回值语义
        副作用：可能调用远端生成服务；记录计时指标。  # 注释：副作用说明
        异常/边界条件：kwargs 缺少 raw_prompt 将抛 KeyError。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - await loop.run({"temperature":1.0}, raw_prompt=[...])。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：verl/experimental/agent_loop/single_turn_agent_loop.py::SingleTurnAgentLoop.run。  # 注释：函数位置
        - 典型调用路径：AgentLoopManager.generate_sequences -> SingleTurnAgentLoop.run。  # 注释：调用链
        - 被谁调用：verl/experimental/agent_loop/agent_loop.py。  # 注释：调用方说明
        - 调用了谁（项目内）：simple_timer、server_manager.generate。  # 注释：项目内依赖
        - 调用了谁（关键外部依赖）：uuid4、tokenizer/processor。  # 注释：外部依赖
        """  # 注释：函数 docstring 结束
        messages = list(kwargs["raw_prompt"])  # 注释：复制原始 prompt 消息
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))  # 注释：拷贝图像数据

        metrics = {}  # 注释：初始化指标字典
        request_id = uuid4().hex  # 注释：生成请求 ID

        # Use processor if available for multimodal support  # 注释：原注释保留（多模态处理）
        if self.processor is not None:  # 注释：存在 processor（多模态）
            raw_prompt = await self.loop.run_in_executor(  # 注释：在线程池中构建文本 prompt
                None,  # 注释：使用默认 executor
                lambda: self.processor.apply_chat_template(  # 注释：调用 processor 模板
                    messages,  # 注释：消息列表
                    add_generation_prompt=True,  # 注释：追加生成提示
                    tokenize=False,  # 注释：不分词（后续统一处理）
                    **self.apply_chat_template_kwargs,  # 注释：额外模板参数
                ),  # 注释：apply_chat_template 结束
            )  # 注释：run_in_executor 结束
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")  # 注释：构造多模态输入
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()  # 注释：取出 prompt token id
        else:  # 注释：无 processor（纯文本）
            prompt_ids = await self.loop.run_in_executor(  # 注释：在线程池中分词
                None,  # 注释：默认 executor
                lambda: self.tokenizer.apply_chat_template(  # 注释：调用 tokenizer 模板
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs  # 注释：模板参数
                ),  # 注释：apply_chat_template 结束
            )  # 注释：run_in_executor 结束

        with simple_timer("generate_sequences", metrics):  # 注释：计时生成耗时
            output = await self.server_manager.generate(  # 注释：调用生成服务
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data  # 注释：生成参数
            )  # 注释：generate 结束
        response_mask = [1] * len(output.token_ids)  # 注释：构造 response mask（全 1）

        output = AgentLoopOutput(  # 注释：封装输出对象
            prompt_ids=prompt_ids,  # 注释：prompt token id
            response_ids=output.token_ids[: self.response_length],  # 注释：截断 response
            response_mask=response_mask[: self.response_length],  # 注释：截断 mask
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,  # 注释：截断 logprobs
            routed_experts=(  # 注释：可选专家路由信息
                output.routed_experts[: len(prompt_ids) + self.response_length]  # 注释：截断 routed_experts
                if output.routed_experts is not None  # 注释：存在 routed_experts 时
                else None  # 注释：否则为空
            ),  # 注释：routed_experts 字段结束
            multi_modal_data={"image": image_data} if image_data is not None else {},  # 注释：回填多模态数据
            num_turns=2,  # 注释：单轮对话包含 user+assistant 两条
            metrics=metrics,  # 注释：指标字典
        )  # 注释：AgentLoopOutput 构造结束
        return output  # 注释：返回结果
