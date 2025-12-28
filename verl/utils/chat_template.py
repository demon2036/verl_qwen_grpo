# Copyright 2025 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
"""
模块用途：提供 chat template 系统提示词的解析与初始化工具。  # 注释：模块用途
输入/输出：输入 tokenizer 与可选参数，输出 system/generation prompt 的 token 列表。  # 注释：模块输入输出概览
关键依赖：jinja2.TemplateError、tokenizer.apply_chat_template。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- system_ids = initialize_system_prompt(tokenizer)  # 注释：获取系统提示词 token
- system_ids, gen_ids = extract_system_prompt_and_generation(tokenizer)  # 注释：解析系统与生成提示词
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/utils/dataset/multiturn_sft_dataset.py、verl/experimental/agent_loop/tool_agent_loop.py。  # 注释：上层入口举例
- 典型链路：上层模块 -> chat_template 工具 -> tokenizer.apply_chat_template。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import logging  # 注释：日志输出
import os  # 注释：读取环境变量
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
from jinja2 import TemplateError  # 注释：捕获 chat template 渲染错误
# （分隔说明：模块级日志器）  # 注释：替代空行，保持逐行注释
logger = logging.getLogger(__name__)  # 注释：以模块名作为 logger 名称
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：读取环境变量设置日志级别
# （分隔说明：系统 prompt 初始化）  # 注释：替代空行，保持逐行注释
def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:  # 注释：初始化系统提示词 token
    """
    功能：为支持 system prompt 的 chat template 初始化系统提示词 token。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - tokenizer：支持 apply_chat_template 的 tokenizer。  # 注释：参数含义
    - **apply_chat_template_kwargs：传给 apply_chat_template 的附加参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - list[int]：系统提示词 token 列表；若不支持返回空列表。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 若模板不支持 system，会记录 warning。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - TemplateError 表示模板不支持或渲染失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - initialize_system_prompt(tokenizer) -> [<tok1>, <tok2>, ...] 或 []。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/chat_template.py::initialize_system_prompt。  # 注释：函数位置
    - 典型调用路径：tool_agent_loop.ToolAgentLoop -> initialize_system_prompt。  # 注释：典型调用链
    - 被谁调用：verl/experimental/agent_loop/tool_agent_loop.py。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：tokenizer.apply_chat_template、jinja2.TemplateError。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    try:  # 注释：尝试构造系统提示词
        return tokenizer.apply_chat_template([{}], tokenize=True, **apply_chat_template_kwargs)  # 注释：空消息生成 system prompt
    except TemplateError as e:  # 注释：模板不支持 system prompt
        logger.warning(f"Chat template does not support system prompt: {e}")  # 注释：记录告警
        return []  # 注释：返回空列表
# （分隔说明：解析 system/generation prompt）  # 注释：替代空行，保持逐行注释
def extract_system_prompt_and_generation(tokenizer):  # 注释：解析系统提示词与生成提示词 token
    """
    功能：通过对比不同长度的空对话模板输出，推断 system prompt 与 generation prompt。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - tokenizer：支持 apply_chat_template 的 tokenizer。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - tuple(list[int], list[int])：系统提示词 token 与生成提示词 token。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若模板不符合假设，切片可能为空或异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - system_prompt, gen_prompt = extract_system_prompt_and_generation(tokenizer)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/chat_template.py::extract_system_prompt_and_generation。  # 注释：函数位置
    - 典型调用路径：MultiTurnSFTDataset.__init__ -> extract_system_prompt_and_generation。  # 注释：典型调用链
    - 被谁调用：verl/utils/dataset/multiturn_sft_dataset.py。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：tokenizer.apply_chat_template。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    token1 = tokenizer.apply_chat_template(  # 注释：单轮空用户消息的模板输出
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True  # 注释：不添加生成提示词
    )  # 注释：token1 构造结束
    token2 = tokenizer.apply_chat_template(  # 注释：两轮空用户消息模板输出
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True  # 注释：重复两次对话
    )  # 注释：token2 构造结束
    # get system prompt tokens  # 注释：原注释保留并补充说明
    system_prompt = token1[: -(len(token2) - len(token1))]  # 注释：前缀差异即 system prompt
    # get generate prompt tokens  # 注释：原注释保留并补充说明
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)  # 注释：包含生成提示词的模板输出
    generate_prompt = token3[len(token1) :]  # 注释：尾部差异即 generation prompt

    return system_prompt, generate_prompt  # 注释：返回 system 与 generation prompt token
