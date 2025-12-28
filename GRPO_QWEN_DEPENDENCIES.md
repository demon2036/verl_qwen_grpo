# Qwen 单轮 GRPO RL + SFT 训练文件依赖清单（非 Agent / 非多轮）

说明：
- 目标：梳理“**Qwen 模型 + 单轮 SFT + 单轮 GRPO RL**”训练链路中**必须经过/直接依赖**的仓库文件。
- 范围：仅单轮任务（非 agent、非多轮交互），以 GSM8K parquet 为例。
- 路径：全部为仓库根目录 `/home/john/github/verl_annotated` 的**相对路径**。
- 参考入口脚本：
  - SFT：`examples/sft/gsm8k/run_qwen_05_peft.sh`
  - GRPO RL：`examples/grpo_trainer/run_qwen2-7b.sh`

---

## 0. 数据准备（SFT/RL 共用）

### 0.1 数据预处理脚本（生成 parquet）
- `examples/data_preprocess/gsm8k.py`

### 0.2 数据拷贝/缓存（本地或 HDFS）
- `verl/utils/hdfs_io.py`
- `verl/utils/fs.py`

---

## 1. SFT 阶段（单轮）

### 1.1 入口脚本（启动命令）
- `examples/sft/gsm8k/run_qwen_05_peft.sh`

> 说明：该脚本调用 `torchrun -m verl.trainer.fsdp_sft_trainer`，使用 Qwen2.5-0.5B-Instruct 做单轮 SFT。

### 1.2 SFT 训练入口与主流程
- `verl/trainer/fsdp_sft_trainer.py`  （SFT 训练主入口，FSDP 后端）
- `verl/trainer/sft_trainer.py`  （引擎无关版 SFT Trainer，若切换到新引擎流程可用）

### 1.3 SFT 配置链（Hydra）
- `verl/trainer/config/sft_trainer.yaml`
- `verl/trainer/config/optim/fsdp.yaml`
- `verl/trainer/config/sft_trainer_engine.yaml`  （仅在 engine-agnostic SFT 时启用）

### 1.4 SFT 数据集与分词/模板
- `verl/utils/dataset/sft_dataset.py`
- `verl/utils/dataset/dataset_utils.py`
- `verl/utils/dataset/__init__.py`
- `verl/utils/tokenizer.py`
- `verl/utils/chat_template.py`
- `verl/utils/model.py`  （位置编码/position id 相关工具）
- `verl/utils/attention_utils.py`
- `verl/utils/py_functional.py`
- `verl/utils/torch_dtypes.py`
- `verl/utils/torch_functional.py`

> 备注：`verl/trainer/fsdp_sft_trainer.py` 会导入 `multiturn_sft_dataset.py`，但在 `sft_trainer.yaml` 中 `multiturn.enable=false` 时不会走多轮逻辑。
- `verl/utils/dataset/multiturn_sft_dataset.py`

### 1.5 Qwen 模型与补丁（单轮 SFT）
- `verl/models/transformers/monkey_patch.py`
- `verl/models/transformers/qwen2.py`
- `verl/models/transformers/qwen2_vl.py`  （仅在使用 VL 版本时）
- `verl/models/transformers/npu_patch.py`  （仅在 Ascend NPU 场景）

### 1.6 FSDP / Ulysses 并行与分片
- `verl/utils/fsdp_utils.py`
- `verl/workers/sharding_manager/fsdp_ulysses.py`
- `verl/utils/ulysses.py`
- `verl/utils/distributed.py`
- `verl/workers/config/optimizer.py`

### 1.7 SFT Checkpoint / 日志
- `verl/utils/checkpoint/checkpoint_manager.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/utils/logger.py`
- `verl/utils/tracking.py`
- `verl/utils/profiler/__init__.py`

### 1.8 SFT 运行时工具（fsdp_sft_trainer 直接导入）
- `verl/utils/device.py`
- `verl/utils/import_utils.py`

---

## 2. GRPO RL 阶段（单轮）

### 2.1 入口脚本（启动命令）
- `examples/grpo_trainer/run_qwen2-7b.sh`

> 说明：该脚本调用 `python3 -m verl.trainer.main_ppo` 并设置 `algorithm.adv_estimator=grpo`，使用 Qwen2-7B-Instruct 做单轮 GRPO。

### 2.2 GRPO 训练入口与调度
- `verl/trainer/main_ppo.py`  （PPO/GRPO 主入口）
- `verl/trainer/constants_ppo.py`  （Ray 运行时环境配置）
- `verl/trainer/ppo/ray_trainer.py`  （Ray 调度与训练循环）
- `verl/trainer/ppo/utils.py`  （是否需要 critic / reference policy 的判断逻辑）
- `verl/trainer/ppo/metric_utils.py`  （指标汇总与统计）
- `verl/trainer/ppo/rollout_corr_helper.py`  （rollout 修正）
- `verl/trainer/ppo/core_algos.py`  （GRPO/GAE 等优势估计核心实现）

### 2.3 GRPO 配置链（Hydra defaults）
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/trainer/config/actor/actor.yaml`
- `verl/trainer/config/actor/dp_actor.yaml`
- `verl/trainer/config/engine/fsdp.yaml`
- `verl/trainer/config/optim/fsdp.yaml`
- `verl/trainer/config/data/legacy_data.yaml`
- `verl/trainer/config/reward_manager.yaml`
- `verl/trainer/config/ref/dp_ref.yaml`
- `verl/trainer/config/ref/ref.yaml`
- `verl/trainer/config/rollout/rollout.yaml`
- `verl/trainer/config/model/hf_model.yaml`
- `verl/trainer/config/critic/dp_critic.yaml`  （GRPO 不训练 critic，但仍在配置链中）
- `verl/trainer/config/critic/critic.yaml`
- `verl/trainer/config/reward_model/dp_reward_loop.yaml`
- `verl/trainer/config/reward_model/dp_reward_model.yaml`
- `verl/trainer/config/reward_model/reward_model.yaml`
- `verl/trainer/config/algorithm/rollout_correction.yaml`
- `verl/trainer/config/config.py`
- `verl/trainer/config/algorithm.py`

### 2.4 RL 数据集与采样
- `verl/trainer/main_ppo.py`  （`create_rl_dataset` / `create_rl_sampler`）
- `verl/utils/dataset/rl_dataset.py`
- `verl/utils/dataset/dataset_utils.py`
- `verl/utils/dataset/__init__.py`
- `verl/utils/tokenizer.py`
- `verl/utils/chat_template.py`
- `verl/utils/model.py`
- `verl/protocol.py`
- `verl/experimental/dataset/sampler/__init__.py`
- `verl/experimental/dataset/sampler/base.py`

> 可选（仅在多模态 / 工具 schema / 动态生成开启时）：
- `verl/utils/dataset/vision_utils.py`
- `verl/models/transformers/qwen2_vl.py`
- `verl/models/transformers/qwen3_vl.py`
- `verl/models/transformers/glm4v.py`
- `verl/tools/utils/tool_registry.py`
- `verl/utils/dataset/dynamicgen_dataset.py`

### 2.5 Worker / 引擎层（FSDP + Ray）
- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/critic/dp_critic.py`  （GRPO 不用 critic，但 worker 仍可被实例化）
- `verl/workers/reward_model/dp_reward_loop.py`
- `verl/workers/config/actor.py`
- `verl/workers/config/critic.py`
- `verl/workers/config/engine.py`
- `verl/workers/config/model.py`
- `verl/workers/config/optimizer.py`
- `verl/workers/config/reward_model.py`
- `verl/workers/config/rollout.py`
- `verl/single_controller/ray/base.py`
- `verl/single_controller/ray/__init__.py`
- `verl/single_controller/base/__init__.py`
- `verl/single_controller/base/worker_group.py`
- `verl/single_controller/base/decorator.py`
- `verl/workers/utils/__init__.py`
- `verl/workers/utils/losses.py`
- `verl/workers/utils/padding.py`

> Qwen 模型补丁（RL Actor/Ref 也会用到，复用 SFT 同名文件）：
- `verl/models/transformers/monkey_patch.py`
- `verl/models/transformers/qwen2.py`

> 若 `trainer.use_legacy_worker_impl=disable`，会走新引擎 worker：
- `verl/workers/engine_workers.py`

### 2.6 Rollout（vLLM 后端）
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/rollout/vllm_rollout/utils.py`
- `verl/workers/rollout/base.py`
- `verl/workers/rollout/schemas.py`
- `verl/workers/rollout/tokenizer.py`
- `verl/workers/rollout/utils.py`
- `verl/workers/rollout/replica.py`
- `verl/utils/vllm/__init__.py`
- `verl/utils/vllm/vllm_fp8_utils.py`
- `verl/utils/vllm/patch.py`
- `verl/third_party/vllm/__init__.py`

### 2.7 Reward / 评分（GSM8K 规则奖励）
- `verl/trainer/ppo/reward.py`
- `verl/workers/reward_manager/registry.py`
- `verl/workers/reward_manager/abstract.py`
- `verl/workers/reward_manager/naive.py`  （默认 reward manager）
- `verl/utils/reward_score/__init__.py`
- `verl/utils/reward_score/gsm8k.py`

> 可选（仅在启用 transfer queue 或 reward loop 时）：
- `verl/utils/transferqueue_utils.py`
- `verl/experimental/reward_loop/`

### 2.8 Checkpoint / 日志 / 设备与配置工具
- `verl/utils/checkpoint/checkpoint_manager.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/utils/logger.py`
- `verl/utils/tracking.py`
- `verl/utils/profiler/__init__.py`
- `verl/utils/profiler/config.py`
- `verl/utils/profiler/performance.py`
- `verl/utils/device.py`
- `verl/utils/config.py`
- `verl/utils/import_utils.py`
- `verl/utils/fs.py`

### 2.9 GRPO 训练运行时工具（ray_trainer/fsdp_workers/dp_actor 直接导入）
- `verl/utils/metric.py`
- `verl/utils/rollout_skip.py`
- `verl/utils/seqlen_balancing.py`
- `verl/utils/py_functional.py`
- `verl/utils/torch_functional.py`
- `verl/utils/torch_dtypes.py`
- `verl/utils/attention_utils.py`
- `verl/utils/ulysses.py`
- `verl/utils/distributed.py`
- `verl/utils/ray_utils.py`
- `verl/utils/memory_utils.py`
- `verl/utils/activation_offload.py`
- `verl/utils/flops_counter.py`
- `verl/utils/debug/__init__.py`
- `verl/utils/debug/metrics.py`
- `verl/utils/debug/trajectory_tracker.py`

---

## 3. 与“非 Agent / 非多轮”有关的排除项（本清单不包含）

- 多轮 / 工具交互：`verl/workers/rollout/sglang_rollout/`、`verl/experimental/agent_loop/`
- 多轮 SFT：`examples/sft/multiturn/`、`verl/utils/dataset/multiturn_sft_dataset.py`（仅被导入，不启用）
- 工具调用数据准备：`examples/data_preprocess/*_multiturn*.py`

---

## 4. 快速核对（单轮 Qwen GRPO + SFT 最小链路）

**SFT 最小链路**：
- `examples/sft/gsm8k/run_qwen_05_peft.sh`
- `verl/trainer/fsdp_sft_trainer.py`
- `verl/trainer/config/sft_trainer.yaml`
- `verl/utils/dataset/sft_dataset.py`
- `verl/models/transformers/monkey_patch.py`

**GRPO 最小链路**：
- `examples/grpo_trainer/run_qwen2-7b.sh`
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/config/ppo_trainer.yaml`
- `verl/utils/dataset/rl_dataset.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
- `verl/workers/reward_manager/naive.py`
- `verl/utils/reward_score/gsm8k.py`
