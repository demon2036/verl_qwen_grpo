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
模块用途：提供 Agent Loop 配置路径解析与 GPT-OSS 工具响应手工格式化工具。  # 注释：模块用途
输入：配置文件路径、工具调用消息等字符串/列表。  # 注释：输入说明
输出：绝对路径字符串或格式化后的消息文本。  # 注释：输出说明
关键依赖：os、typing、（可选）verl 安装路径。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- resolve_config_path("configs/agent.yaml")  # 注释：路径解析示例
- build_gpt_oss_tool_response_text(messages, tool_call_names)  # 注释：工具响应格式化示例
调用路径概览：  # 注释：调用路径标题
- AgentLoopManager 初始化 -> resolve_config_path。  # 注释：调用链 1
- ToolAgentLoop 生成 -> build_gpt_oss_tool_response_text。  # 注释：调用链 2
"""  # 注释：模块 docstring 结束

import os  # 注释：标准库，路径与文件检查
from typing import Any  # 注释：类型注解


def resolve_config_path(config_path: str) -> str:  # 注释：解析 agent loop 配置文件路径
    """
    功能：在多机 Ray 环境中解析相对/绝对配置路径，确保可定位到文件。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - config_path (str)：相对或绝对配置路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：解析后的绝对路径。  # 注释：返回值语义
    副作用：无（仅路径判断）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若无法定位文件，抛 FileNotFoundError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - resolve_config_path("configs/agent.yaml") -> "/abs/path/configs/agent.yaml"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/utils.py::resolve_config_path。  # 注释：函数位置
    - 典型调用路径：AgentLoopManager.__init__ -> resolve_config_path。  # 注释：调用链
    - 被谁调用：verl/experimental/agent_loop/agent_loop.py。  # 注释：调用方说明
    - 调用了谁（项目内）：无（直接使用 os/verl）。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：os.path, verl（可选）。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    # Return absolute paths unchanged  # 注释：原注释保留（绝对路径直接返回）
    if os.path.isabs(config_path):  # 注释：若已是绝对路径
        return config_path  # 注释：直接返回

    # Try current working directory first  # 注释：优先使用当前工作目录
    cwd = os.path.abspath(os.getcwd())  # 注释：获取当前绝对路径
    cwd_path = os.path.abspath(os.path.join(cwd, config_path))  # 注释：拼接相对路径
    if (cwd_path == cwd or cwd_path.startswith(cwd + os.sep)) and os.path.exists(cwd_path):  # 注释：校验路径有效
        return cwd_path  # 注释：返回当前目录下路径

    # Try relative to verl project root (where verl package is installed)  # 注释：尝试基于 verl 包路径
    try:  # 注释：捕获 verl 导入异常
        import verl  # 注释：导入 verl 获取安装路径

        verl_package_dir = os.path.abspath(os.path.dirname(verl.__file__))  # 注释：获取 verl 包目录

        # Strategy 1: For development/editable installs.  # 注释：策略 1（开发/可编辑安装）
        project_root = os.path.dirname(verl_package_dir)  # 注释：推断项目根目录
        dev_path = os.path.abspath(os.path.join(project_root, config_path))  # 注释：拼接开发路径
        if (dev_path == project_root or dev_path.startswith(project_root + os.sep)) and os.path.exists(dev_path):  # 注释：路径存在则返回
            return dev_path  # 注释：返回开发路径

        # Strategy 2: For standard package installations.  # 注释：策略 2（标准安装）
        install_path = os.path.abspath(os.path.join(verl_package_dir, config_path))  # 注释：拼接包内路径
        if (install_path == verl_package_dir or install_path.startswith(verl_package_dir + os.sep)) and os.path.exists(  # 注释：路径存在检查
            install_path  # 注释：被检查的路径
        ):  # 注释：if 结束
            return install_path  # 注释：返回包内路径
    except (ImportError, AttributeError):  # 注释：捕获 verl 不可用或 __file__ 缺失
        pass  # 注释：忽略异常，继续报错

    # File not found - raise clear error  # 注释：文件未找到时抛出错误
    raise FileNotFoundError(  # 注释：抛出异常
        f"Agent loop configuration file not found: {config_path}. Tried current directory and verl project root."  # 注释：错误信息
    )  # 注释：异常结束


# tokenizer.apply_chat_template is not working properly for gpt-oss model.  # 注释：原注释保留（gpt-oss 模型说明）
# Because the chat template requires tool call messages to parse tool response messages  # 注释：原注释保留（原因说明）
# so we need to format the tool response manually.  # 注释：原注释保留（手工格式化）
def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:  # 注释：手动格式化工具响应
    """
    功能：按 gpt-oss 的函数调用格式拼接工具响应文本。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - tool_response (str)：工具响应内容。  # 注释：参数含义
    - tool_call_name (str)：工具名称。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：符合 gpt-oss 模板的工具响应消息。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - 输入 ("ok","calc") -> 输出带 <|start|>functions.calc ... 的文本。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/utils.py::format_gpt_oss_tool_response_manually。  # 注释：函数位置
    - 典型调用路径：build_gpt_oss_tool_response_text -> format_gpt_oss_tool_response_manually。  # 注释：调用链
    - 被谁调用：本文件内 build_gpt_oss_tool_response_text。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"  # 注释：拼接并返回格式化文本


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:  # 注释：为 gpt-oss 添加生成提示
    """
    功能：在消息末尾追加 assistant 开始标记，触发生成。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - message_content (str)：消息内容。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：追加生成提示后的文本。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - add_generation_prompt_for_gpt_oss("hi") -> "hi<|start|>assistant"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/utils.py::add_generation_prompt_for_gpt_oss。  # 注释：函数位置
    - 典型调用路径：build_gpt_oss_tool_response_text -> add_generation_prompt_for_gpt_oss。  # 注释：调用链
    - 被谁调用：本文件内 build_gpt_oss_tool_response_text。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    return message_content + "<|start|>assistant"  # 注释：拼接并返回


def build_gpt_oss_tool_response_text(messages: list[dict[str, Any]], tool_call_names: list[str]) -> str:  # 注释：构建 gpt-oss 工具响应文本
    """
    功能：对工具响应逐条格式化并追加生成提示。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - messages (list[dict])：工具消息列表。  # 注释：参数含义
    - tool_call_names (list[str])：工具名称列表。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：拼接后的工具响应文本。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：messages 与 tool_call_names 长度不一致可能抛异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - build_gpt_oss_tool_response_text([{...}], ["calc"]) -> "<|start|>functions.calc ..."。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/utils.py::build_gpt_oss_tool_response_text。  # 注释：函数位置
    - 典型调用路径：ToolAgentLoop._format_tool_response -> build_gpt_oss_tool_response_text。  # 注释：调用链
    - 被谁调用：verl/experimental/agent_loop/tool_agent_loop.py。  # 注释：调用方说明
    - 调用了谁（项目内）：format_gpt_oss_tool_response_manually、add_generation_prompt_for_gpt_oss。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    tool_response_texts: list[str] = []  # 注释：初始化文本列表
    for i, tool_msg in enumerate(messages):  # 注释：遍历工具消息
        actual_tool_name = tool_call_names[i]  # 注释：取对应工具名称
        formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)  # 注释：格式化工具响应
        tool_response_texts.append(formatted)  # 注释：追加格式化结果
    return add_generation_prompt_for_gpt_oss("".join(tool_response_texts))  # 注释：拼接并添加生成提示
