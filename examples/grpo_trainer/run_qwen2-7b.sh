#!/usr/bin/env bash  # 注释：指定脚本解释器，便于直接执行
# 用途：启动 Qwen2-7B-Instruct 的单轮 GRPO 训练（GSM8K 数据集示例）。  # 注释：脚本作用
# 依赖：CUDA + PyTorch + Ray + vLLM；已安装 verl 环境；可访问模型权重与数据路径。  # 注释：运行依赖
# 输出：默认由配置写入训练日志与 checkpoint（见 trainer.default_local_dir 等配置）。  # 注释：输出位置说明
# 单卡简化：将 trainer.n_gpus_per_node=1、actor_rollout_ref.rollout.tensor_model_parallel_size=1，  # 注释：单卡提示
#           并适当降低 data.train_batch_size / ppo_micro_batch_size_per_gpu / rollout.n 以避免 OOM。  # 注释：单卡提示

set -x  # 注释：打印执行命令，便于排错

# 说明：使用数组保存 Hydra 覆盖参数，便于逐条中文注释且不影响参数传递。  # 注释：为何用数组
ARGS=(  # 注释：Hydra 覆盖参数列表开始
  "algorithm.adv_estimator=grpo"  # 注释：使用 GRPO 优势估计器（单轮 RL）
  "data.train_files=$HOME/data/gsm8k/train.parquet"  # 注释：训练集 parquet 路径
  "data.val_files=$HOME/data/gsm8k/test.parquet"  # 注释：验证集 parquet 路径
  "data.train_batch_size=1024"  # 注释：训练批大小（全局）
  "data.max_prompt_length=512"  # 注释：最大 prompt token 长度
  "data.max_response_length=1024"  # 注释：最大 response token 长度
  "data.filter_overlong_prompts=True"  # 注释：过滤超长 prompt
  "data.truncation=error"  # 注释：截断策略（超长直接报错）
  "actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct"  # 注释：Actor/Ref 模型权重路径
  "actor_rollout_ref.actor.optim.lr=1e-6"  # 注释：Actor 学习率
  "actor_rollout_ref.model.use_remove_padding=True"  # 注释：启用 padding 移除以节省显存
  "actor_rollout_ref.actor.ppo_mini_batch_size=256"  # 注释：PPO mini-batch 大小
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=40"  # 注释：每 GPU micro-batch
  "actor_rollout_ref.actor.use_kl_loss=True"  # 注释：启用 KL loss 约束
  "actor_rollout_ref.actor.kl_loss_coef=0.001"  # 注释：KL loss 系数
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"  # 注释：KL loss 计算方式
  "actor_rollout_ref.actor.entropy_coeff=0"  # 注释：熵正则系数（此处为 0）
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"  # 注释：开启梯度检查点节省显存
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"  # 注释：关闭参数 offload
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"  # 注释：关闭优化器 offload
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40"  # 注释：rollout log_prob 微批大小
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"  # 注释：rollout 的张量并行度
  "actor_rollout_ref.rollout.name=vllm"  # 注释：rollout 后端使用 vLLM
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"  # 注释：vLLM 显存占用比例
  "actor_rollout_ref.rollout.n=5"  # 注释：每 prompt 采样 5 个 response
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40"  # 注释：ref policy log_prob 微批大小
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"  # 注释：ref policy 参数 offload
  "algorithm.use_kl_in_reward=False"  # 注释：不在 reward 中加入 KL 惩罚
  "trainer.critic_warmup=0"  # 注释：critic 预热步数（0 表示不预热）
  "trainer.logger=[\"console\",\"wandb\"]"  # 注释：日志后端（控制台 + wandb）
  "trainer.project_name=verl_grpo_example_gsm8k"  # 注释：实验项目名
  "trainer.experiment_name=qwen2_7b_function_rm"  # 注释：实验名称
  "trainer.n_gpus_per_node=8"  # 注释：每节点 GPU 数
  "trainer.nnodes=1"  # 注释：节点数
  "trainer.save_freq=20"  # 注释：保存 checkpoint 频率（step）
  "trainer.test_freq=5"  # 注释：验证频率（step）
  "trainer.total_epochs=15"  # 注释：训练总 epoch
)  # 注释：Hydra 覆盖参数列表结束

python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@"  # 注释：启动 PPO/GRPO 主入口（保留额外 CLI 参数）
