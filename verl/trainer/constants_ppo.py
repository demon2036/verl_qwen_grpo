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
模块用途：定义 PPO/GRPO 训练在 Ray 运行时需要的环境变量，并提供过滤函数生成 runtime_env。（注释：模块级用途概述，说明本文件职责）
输入/输出：（注释：模块级 I/O 概览）
  - 输入：系统环境变量、Ray 作业配置（RAY_JOB_CONFIG_JSON_ENV_VAR）。（注释：说明输入来源）
  - 输出：供 ray.init 使用的 runtime_env 字典。（注释：说明输出形态）
关键依赖：（注释：列出关键依赖，便于环境准备）
  - ray._private.runtime_env.constants.RAY_JOB_CONFIG_JSON_ENV_VAR。（注释：Ray 内部常量）
  - os/json。（注释：标准库依赖）
典型用法（最小示例）：（注释：给出最小可用片段）
  >>> from verl.trainer.constants_ppo import get_ppo_ray_runtime_env  # 入口调用（示例）
  >>> runtime_env = get_ppo_ray_runtime_env()  # 生成 Ray 运行时环境（示例）
调用路径概览：（注释：说明从入口到本模块的调用链）
  - examples/grpo_trainer/run_qwen2-7b.sh -> verl.trainer.main_ppo.run_ppo -> get_ppo_ray_runtime_env。（注释：GRPO 入口链路）
"""  # 注释：模块 docstring 结束

# 标准库导入（注释：json 用于解析 Ray 作业配置）
import json
# 标准库导入（注释：os 用于读取系统环境变量）
import os

# Ray 内部常量（注释：读取 Ray 作业 JSON 配置的环境变量键名）
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

# PPO/GRPO 在 Ray runtime_env 中默认注入的环境变量（注释：集中配置）
PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",  # 注释：开启 tokenizer 并行，加速分词
        "NCCL_DEBUG": "WARN",  # 注释：降低 NCCL 日志噪音
        "VLLM_LOGGING_LEVEL": "WARN",  # 注释：降低 vLLM 日志噪音
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",  # 注释：允许运行时更新 LoRA
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",  # 注释：禁用对称内存 allreduce，规避 SPMD 异常
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",  # 注释：控制 CUDA 连接数，降低通信拥塞
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",  # 注释：禁用 NCCL cuMem，避免权重同步卡死
    },
}  # 注释：默认 runtime_env 定义结束


def get_ppo_ray_runtime_env():
    """
    生成 PPO/GRPO 所需的 Ray runtime_env，并过滤已在外部设置的环境变量。（注释：函数用途概述）

    参数：（注释：本函数无显式参数）
      - 无。（注释：输入来自环境变量与 Ray 作业配置）
    返回：（注释：返回值说明）
      - runtime_env (dict): 可直接传给 ray.init 的运行时环境配置。（注释：输出形态）
    副作用：（注释：副作用说明）
      - 无（仅读取环境变量，不写入）。（注释：说明无副作用）
    异常/边界条件：（注释：异常说明）
      - 若 RAY_JOB_CONFIG_JSON_ENV_VAR 内容非 JSON，json.loads 会抛异常。（注释：潜在异常）
    最小示例：（注释：最小可运行示例）
      >>> env = get_ppo_ray_runtime_env()  # 输出包含 env_vars 的字典（示例）
    调用路径依赖：（注释：调用关系说明）
      所在位置
      --------
      - 路径：`verl/trainer/constants_ppo.py`（注释：文件路径）
      - 函数：`get_ppo_ray_runtime_env()`（注释：函数名）
      典型调用路径
      ------------
      - `examples/grpo_trainer/run_qwen2-7b.sh` -> `verl.trainer.main_ppo.run_ppo` -> `get_ppo_ray_runtime_env`。（注释：入口链路）
      被谁调用
      --------
      - `verl/trainer/main_ppo.py::run_ppo`（注释：Ray 初始化前调用）
      调用了谁（项目内）
      ----------------
      - 无。（注释：不调用仓库内其他函数）
      调用了谁（关键外部依赖）
      ----------------------
      - `json.loads` / `os.environ.get` / Ray 常量。（注释：外部依赖）
    """
    # 从 Ray 作业配置中读取 working_dir（注释：避免重复设置 working_dir）
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    # 复制默认 env_vars，并在没有 working_dir 时显式禁用（注释：避免 Ray 重复注入）
    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),  # 注释：复制默认环境变量
        **({"working_dir": None} if working_dir is None else {}),  # 注释：仅在无 working_dir 时置空
    }
    # 过滤：若外部已设置，则不再覆盖（注释：尊重外部环境）
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)  # 注释：删除已存在的环境变量
    return runtime_env  # 注释：返回最终 runtime_env 配置
