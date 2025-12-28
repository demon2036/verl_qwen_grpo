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
  - 在 GRPO/PPO 训练中，将 batch 的 left-right padding 表示与无 padding（NestedTensor）表示互相转换。
  - 解决 “按真实长度计算 loss/values” 与 “统一长度写回指标/张量” 的格式对齐问题。

输入：
  - TensorDict：包含 input_ids/attention_mask/response_mask/position_ids 等张量或嵌套张量。

输出：
  - TensorDict 或 Tensor：将左/右 padding 形式转换为 NestedTensor 或反向恢复到定长张量。

关键依赖：
  - torch / TensorDict：张量与嵌套张量承载。
  - `verl.utils.attention_utils.pad_input` / `unpad_input`：pad/unpad 核心实现。
  - `verl.utils.tensordict_utils`：非张量元数据写入/读取。

典型用法（最小示例）：
  - `batch_td = left_right_2_no_padding(batch_td)`  # 训练前把 batch 变成 NestedTensor。
  - `values = no_padding_2_padding(values, batch_td)`  # 将模型输出恢复到右 padding 形式。

调用路径概览：
  - `verl/trainer/main_ppo.py`
    -> `verl/trainer/ppo/ray_trainer.py`
    -> `left_right_2_no_padding` / `no_padding_2_padding`
    -> 下游 loss/metric 计算与日志汇总。
"""

import torch  # PyTorch 张量与 nested tensor 支持
from tensordict import TensorDict  # 统一承载 batch 结构的张量字典

from verl.utils import tensordict_utils as tu  # TensorDict 的非张量元数据读写
from verl.utils.attention_utils import pad_input, unpad_input  # pad/unpad 的底层实现


def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    """
    功能：
      - 将 left-right padding 形式的 TensorDict 转为无 padding（NestedTensor）格式。
      - 同时在 TensorDict 中记录恢复所需的索引与最大长度元信息。

    参数：
      - data (TensorDict): 必须包含 "input_ids"/"attention_mask"/"response_mask"/"position_ids"。

    返回：
      - TensorDict:
        - 张量字段被替换为 NestedTensor（如 input_ids/position_ids/loss_mask）。
        - 非张量元数据新增 max_seq_len/max_response_len/indices。

    副作用：
      - 会从 `data` 中弹出（pop）原始 `input_ids`，并原地写入新的字段。
      - 会写入非张量元数据，后续 `no_padding_2_padding` 依赖这些元信息。

    异常/边界条件：
      - 缺少必要键会触发断言。
      - attention_mask 与 input_ids 维度不匹配时，`unpad_input` 可能报错。

    最小示例（伪输入输出）：
      - 输入：input_ids 形状 [2, 5]，attention_mask=[[1,1,1,0,0],[1,1,0,0,0]]；
      - 关键中间量：unpad 后得到 jagged 序列长度 [3,2]；
      - 输出：input_ids 变为 NestedTensor（长度 3 和 2），并写入 indices/max_seq_len。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/utils/padding.py`
        - 函数：`left_right_2_no_padding(data)`
      典型调用路径：
        - `verl/trainer/main_ppo.py`
          -> `verl/trainer/ppo/ray_trainer.py::train(...)`
          -> `left_right_2_no_padding(...)`
      被谁调用：
        - `verl/trainer/ppo/ray_trainer.py`（rollout/训练前转换 batch）
      调用了谁（项目内）：
        - `verl.utils.attention_utils.unpad_input`
        - `verl.utils.tensordict_utils.assign_non_tensor_data`
      调用了谁（关键外部依赖）：
        - `torch.nested.*`（构建 NestedTensor）
    """
    # --- 校验必备字段（保证后续 unpad 有足够信息） ---
    assert "input_ids" in data, "input_ids is required in left-right padding data"  # 输入 token 必需
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"  # attention 掩码必需
    assert "response_mask" in data, "response_mask is required in left-right padding data"  # response 掩码必需
    assert "position_ids" in data, "position_ids is required in left-right padding data"  # 位置编码必需

    # --- 取出/读取字段（input_ids 会被替换成 NestedTensor） ---
    input_ids = data.pop("input_ids")  # 移除原始定长 input_ids，避免后续歧义
    attention_mask = data["attention_mask"]  # 注意力 mask 仍保留
    response_mask = data["response_mask"]  # response 掩码用于 loss/长度统计

    # --- 记录最大长度信息（用于恢复 padding） ---
    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]  # 取定长维度
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)  # 写入序列最大长度
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)  # 写入响应最大长度

    # --- 去 padding，得到 jagged 表示与索引 ---
    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # 核心 unpad
    tu.assign_non_tensor_data(data, "indices", indices)  # 保存恢复所需索引

    # --- 将 unpad 的扁平结果转成 NestedTensor ---
    input_ids_nested = torch.nested.nested_tensor_from_jagged(
        input_ids_rmpad.squeeze(-1), offsets=cu_seqlens
    )  # jagged -> NestedTensor

    # --- 构建对应的 position_ids（按真实长度展开） ---
    seq_lens = cu_seqlens.diff().tolist()  # 每条样本真实长度
    response_lens = response_mask.sum(dim=1).tolist()  # 每条样本 response 长度（用于理解）

    position_ids_list = []  # 收集每条样本的 position_ids
    for seq_len, response_len in zip(seq_lens, response_lens, strict=False):
        # 注：response_len 仅作语义提示，position_ids 按完整 seq_len 构造
        position_ids_list.append(torch.arange(seq_len, device=input_ids.device))  # [0,1,...,seq_len-1]

    position_ids_nested = torch.nested.as_nested_tensor(
        position_ids_list, layout=torch.jagged
    )  # list -> NestedTensor

    # --- 写回 TensorDict 字段（使用 NestedTensor 表示） ---
    data["input_ids"] = input_ids_nested  # 替换为 NestedTensor
    data["position_ids"] = position_ids_nested  # 与 input_ids 对齐的 position_ids
    data["loss_mask"] = data["response_mask"]  # loss_mask 直接复用 response_mask

    return data  # 返回就地更新后的 TensorDict


def no_padding_2_padding(nested_tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    """
    功能：
      - 将无 padding 的 NestedTensor 恢复为右 padding 的定长张量。
      - 主要用于将模型输出（values/log_probs 等）恢复成固定长度以便统计/写回。

    参数：
      - nested_tensor (torch.Tensor): NestedTensor 或 jagged 表示的模型输出。
      - data (TensorDict): 必须包含 `indices`/`max_seq_len`/`max_response_len` 的元数据。

    返回：
      - torch.Tensor: 形状 [bsz, max_response_len] 的右 padding 张量。

    副作用：
      - 无（仅读取 data 中的元信息）。

    异常/边界条件：
      - 缺失 indices 或 max_seq_len/max_response_len 会触发断言。
      - 若 nested_tensor 的 batch 维与 data 记录不一致，会导致 pad_input 维度错误。

    最小示例（伪输入输出）：
      - 输入：nested_tensor 长度 [3,2]，max_seq_len=5，max_response_len=2；
      - 中间：pad_input 先恢复为 [bsz, max_seq_len]；
      - 输出：裁剪得到 [bsz, max_response_len]。

    调用路径依赖：
      所在位置：
        - 路径：`verl/workers/utils/padding.py`
        - 函数：`no_padding_2_padding(nested_tensor, data)`
      典型调用路径：
        - `verl/trainer/ppo/ray_trainer.py`
          -> `no_padding_2_padding(...)`
      被谁调用：
        - `verl/trainer/ppo/ray_trainer.py`（将模型输出恢复为定长）
      调用了谁（项目内）：
        - `verl.utils.attention_utils.pad_input`
        - `verl.utils.tensordict_utils.get_non_tensor_data`
      调用了谁（关键外部依赖）：
        - `torch.nested` 张量 API
    """
    # --- 校验必须的元信息 ---
    assert "indices" in data, "indices is required in left-right padding data"  # 恢复索引必需
    assert "max_seq_len" in data, "max_seq_len is required in left-right padding data"  # 最大长度必需
    assert "max_response_len" in data, "max_response_len is required in left-right padding data"  # 最大响应长度必需

    # --- 读取元信息 ---
    indices = tu.get_non_tensor_data(data=data, key="indices", default=None)  # unpad 时记录的索引
    max_seq_len = tu.get_non_tensor_data(data=data, key="max_seq_len", default=2048)  # 默认兜底值
    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=1024)  # 默认兜底值
    batch_size = nested_tensor.size(0)  # batch 维度

    # --- 执行 pad 恢复 ---
    values = nested_tensor.values()  # jagged -> 扁平 values
    full_values = pad_input(
        hidden_states=values.unsqueeze(-1),  # pad_input 期望带最后一维
        indices=indices,  # unpad 产生的索引
        batch=batch_size,  # batch 大小
        seqlen=max_seq_len,  # 恢复到最大序列长度
    )
    # 取最后 max_response_len 段作为 response（排除最后一个 token 的对齐）
    values = full_values.squeeze(-1)[:, -max_response_len - 1 : -1]  # (bsz, response_length)

    return values  # 返回右 padding 形式
