# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
模块用途：
  - 定义 HuggingFace 模型相关的配置（HFModelConfig），用于 Actor/Ref/Reward Model 等。
  - 包含模型路径、tokenizer、generation config、LoRA/patch 等运行期细节。

输入：
  - 通常由 YAML/OmegaConf 解析后构造 dataclass。

输出：
  - HFModelConfig 实例（包含已加载的 tokenizer/processor/hf_config 等）。

关键依赖：
  - `transformers.AutoConfig`：加载模型结构配置。
  - `verl.utils.hf_tokenizer/hf_processor`：加载 tokenizer/processor。
  - `verl.utils.model`：generation config 与配置更新工具。

典型用法（最小示例）：
  - `cfg = HFModelConfig(path="Qwen/Qwen2.5-0.5B-Instruct")`
  - `tok = cfg.get_processor()`  # 获取 tokenizer/processor。

调用路径概览：
  - `verl/trainer/config/*.yaml`
    -> `verl/trainer/config/config.py`
    -> `HFModelConfig`（本模块）
    -> `verl/workers/*` / `verl/models/*` 初始化模型与 tokenizer。
"""

from dataclasses import dataclass, field  # dataclass 定义
from typing import Any, Optional  # 类型提示

from omegaconf import MISSING  # OmegaConf 必填占位符
from transformers import AutoConfig  # HF 模型配置加载

from verl.base_config import BaseConfig  # 配置基类
from verl.utils import hf_processor, hf_tokenizer  # tokenizer/processor 构建器
from verl.utils.fs import copy_to_local  # 远端/共享存储转本地
from verl.utils.import_utils import import_external_libs  # 动态导入外部库
from verl.utils.model import get_generation_config, update_model_config  # 生成配置与覆盖工具

__all__ = ["HFModelConfig"]  # 对外导出符号


@dataclass
class HFModelConfig(BaseConfig):
    """
    功能：
      - 描述 HF 模型加载所需的路径与运行参数，并在 __post_init__ 中完成加载与配置修正。

    参数（关键字段）：
      - path (str): 模型主路径（本地/HF hub）。
      - tokenizer_path/hf_config_path: 可选独立路径。
      - load_tokenizer (bool): 是否加载 tokenizer/processor。
      - override_config (dict): 覆盖到 hf_config 的参数。
      - lora_rank/lora_alpha/target_modules: LoRA 相关参数。

    返回：
      - HFModelConfig 实例（包含 hf_config/tokenizer/processor 等）。

    副作用：
      - __post_init__ 中会触发文件复制、本地缓存、加载 tokenizer/hf_config。

    异常/边界条件：
      - path 为空或不可达会在 copy_to_local/AutoConfig 中报错。
      - architectures 为空或不唯一会触发断言。

    最小示例（伪输入输出）：
      - 输入：HFModelConfig(path="Qwen/Qwen2.5-0.5B-Instruct")
      - 输出：cfg.tokenizer/cfg.hf_config 已加载，architectures=单一架构名。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/config/model.py`
        - 类：`HFModelConfig`
      典型调用路径：
        - `verl/trainer/config/model/*.yaml`
          -> `verl/trainer/config/config.py`
          -> `HFModelConfig`
      被谁调用：
        - `verl/workers/fsdp_workers.py` / `verl/workers/*` 初始化模型
      调用了谁（项目内）：
        - `copy_to_local` / `hf_tokenizer` / `hf_processor`
        - `get_generation_config` / `update_model_config`
      调用了谁（关键外部依赖）：
        - `transformers.AutoConfig`
    """

    # note: 分离 model_path / config_path / tokenizer_path 以支持路径不同的场景
    _mutable_fields = {
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
    }  # 允许 OmegaConf 动态修改的字段

    path: str = MISSING  # 模型主路径（必填）
    local_path: Optional[str] = None  # 本地缓存路径
    hf_config_path: Optional[str] = None  # HF config 路径
    local_hf_config_path: Optional[str] = None  # 本地 config 路径
    tokenizer_path: Optional[str] = None  # tokenizer 路径
    local_tokenizer_path: Optional[str] = None  # 本地 tokenizer 路径

    # 是否加载 tokenizer（仅加载 config 时可关闭）
    load_tokenizer: bool = True

    hf_config: Any = None  # HF 模型配置对象
    generation_config: Any = None  # 生成配置
    tokenizer: Any = None  # tokenizer
    processor: Any = None  # processor（多模态模型可能使用）

    # 是否使用共享内存（/dev/shm）进行本地缓存
    use_shm: bool = False
    trust_remote_code: bool = False  # 是否允许远端自定义代码

    # 自定义 chat template
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None  # 动态导入的外部库名

    override_config: dict = field(default_factory=dict)  # 覆盖到 hf_config 的键值对

    enable_gradient_checkpointing: bool = True  # 是否启用梯度检查点
    enable_activation_offload: bool = False  # 是否启用激活值卸载

    use_remove_padding: bool = True  # 是否在模型前向使用 remove padding

    # TODO: unify fsdp and megatron lora config
    # FSDP LoRA 相关参数
    lora_rank: int = 0  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    target_modules: Optional[str] = "all-linear"  # 默认对所有线性层打 LoRA

    exclude_modules: Optional[str] = None  # 排除 LoRA 的模块名

    # Megatron LoRA 配置（注意：此字段在下方会被同名字段覆盖）
    lora: dict[str, Any] = field(default_factory=dict)  # 类型注释版本（被后面覆盖）

    # 预训练 LoRA adapter 路径
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False  # 是否使用 Liger kernel（若支持）
    lora: dict = field(default_factory=dict)  # 同名字段，实际生效的 lora 配置

    use_fused_kernels: bool = False  # 是否启用 fused kernel
    fused_kernel_options: dict = field(default_factory=dict)  # fused kernel 细项

    architectures: Optional[list[str]] = None  # HF config 中的模型架构列表

    def __post_init__(self):
        """
        功能：
          - 动态导入外部库，准备本地缓存路径与 tokenizer/processor。
          - 读取 HF config 与 generation config，并应用 override_config。

        参数：
          - self: HFModelConfig 实例。

        返回：
          - None。

        副作用：
          - 可能触发网络/文件系统访问（copy_to_local/AutoConfig）。
          - 会修改 self 内部字段（local_path/tokenizer/hf_config 等）。

        异常/边界条件：
          - HF 模型路径无效时会抛出异常。
          - architectures 数量不为 1 时触发断言。

        最小示例（伪输入输出）：
          - 输入：HFModelConfig(path="Qwen/Qwen2.5-0.5B-Instruct")
          - 输出：hf_config/tokenizer/generation_config 已加载。

        调用路径依赖：
          所在位置：
            - 路径：`verl/workers/config/model.py`
            - 方法：`HFModelConfig.__post_init__(self)`
          典型调用路径：
            - `HFModelConfig(...)` 构造时自动触发
          被谁调用：
            - dataclass 构造流程
          调用了谁（项目内）：
            - `import_external_libs` / `copy_to_local` / `hf_tokenizer`
            - `get_generation_config` / `update_model_config`
          调用了谁（关键外部依赖）：
            - `transformers.AutoConfig.from_pretrained`
        """
        # --- 先加载可能的外部依赖 ---
        import_external_libs(self.external_lib)

        # --- 兜底 config/tokenizer 路径 ---
        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        # --- 将模型路径拷贝到本地（可使用 shm） ---
        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # --- 构建 tokenizer/processor ---
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        # --- 应用自定义 chat template ---
        if self.custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = self.custom_chat_template
            else:
                self.tokenizer.chat_template = self.custom_chat_template

        # --- 加载 generation config ---
        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # --- 构建 hf_config ---
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )

        override_config_kwargs = {}  # 收集需要覆盖到 hf_config 的字段

        if self.tokenizer is not None:
            # 将 tokenizer 的 token_id 同步到模型配置
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). Megatron 与 FSDP 的 override_config 结构不同
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)  # 更新覆盖参数
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)  # 应用覆盖

        # --- 是否共享 embedding 与输出权重 ---
        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # --- 获取模型架构（期待只有一个） ---
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )

        # --- 特定模型 patch ---
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"  # Kimi-VL 特殊处理

    def get_processor(self):
        """
        功能：
          - 返回 processor（若存在），否则返回 tokenizer。

        参数：
          - self: HFModelConfig 实例。

        返回：
          - processor 或 tokenizer 对象。

        副作用：
          - 无。

        异常/边界条件：
          - 若未加载 tokenizer（load_tokenizer=False），可能返回 None。

        最小示例（伪输入输出）：
          - 输入：processor=None, tokenizer=Tok
          - 输出：Tok

        调用路径依赖：
          所在位置：
            - 路径：`verl/workers/config/model.py`
            - 方法：`HFModelConfig.get_processor(self)`
          典型调用路径：
            - `verl/workers/*` 初始化模型或 rollout 时获取 tokenizer
          被谁调用：
            - `verl/workers/fsdp_workers.py` 等模型构建流程
          调用了谁（项目内）：
            - 无
          调用了谁（关键外部依赖）：
            - 无
        """
        return self.processor if self.processor is not None else self.tokenizer  # 优先返回 processor
