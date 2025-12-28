# VERL GRPO Qwen 代码阅读顺序指南

本文档提供一个**推荐的代码阅读顺序**，帮助你理解 Qwen GRPO 训练的完整调用流程。按照这个顺序阅读代码，你可以从入口一步步深入到核心算法实现。

---

## 目录

1. [第一阶段：Shell 脚本入口](#第一阶段shell-脚本入口)
2. [第二阶段：Python 主入口与配置](#第二阶段python-主入口与配置)
3. [第三阶段：分布式训练调度器](#第三阶段分布式训练调度器)
4. [第四阶段：核心算法实现](#第四阶段核心算法实现)
5. [第五阶段：奖励计算](#第五阶段奖励计算)
6. [第六阶段：Worker 实现（可选深入）](#第六阶段worker-实现可选深入)
7. [完整调用链图示](#完整调用链图示)

---

## 第一阶段：Shell 脚本入口

### 1.1 阅读文件：`examples/grpo_trainer/run_qwen2-7b.sh`

**阅读目的**：理解训练是如何启动的，有哪些关键配置参数

**关键内容（行号）**：

| 行号 | 内容 | 说明 |
|------|------|------|
| 1-7 | shebang 与注释 | 脚本用途说明 |
| 12 | `algorithm.adv_estimator=grpo` | **关键**：指定使用 GRPO 优势估计器 |
| 13-14 | `data.train_files` / `data.val_files` | 数据集路径 |
| 15-17 | `train_batch_size` / `max_prompt_length` / `max_response_length` | 数据配置 |
| 20 | `actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct` | 模型路径 |
| 21-28 | Actor 配置（学习率、KL loss、熵正则等） | PPO/GRPO 超参 |
| 33-36 | Rollout 配置（vLLM、采样数 n=5） | 推理配置 |
| 44-48 | Trainer 配置（GPU 数、epoch、保存频率） | 训练控制 |
| 51 | `python3 -m verl.trainer.main_ppo "${ARGS[@]}"` | **入口命令** |

**阅读时间**：5 分钟

**阅读后你应该理解**：
- GRPO 训练通过 `verl.trainer.main_ppo` 模块启动
- 每个 prompt 采样 5 个 response（`rollout.n=5`）
- 使用 vLLM 进行高效推理
- 开启了 KL loss 约束（`use_kl_loss=True`）

---

## 第二阶段：Python 主入口与配置

### 2.1 阅读文件：`verl/trainer/main_ppo.py`

**阅读目的**：理解 Python 层面的启动流程和 Ray 分布式初始化

**推荐阅读顺序（按函数）**：

| 顺序 | 函数/类 | 行号 | 阅读时间 | 说明 |
|------|---------|------|----------|------|
| 1 | 模块 docstring | 14-28 | 2 分钟 | 模块用途概述 |
| 2 | `main(config)` | 60-98 | 3 分钟 | Hydra 入口，调用 `run_ppo` |
| 3 | `run_ppo(config)` | 102-182 | 5 分钟 | Ray 初始化，创建 TaskRunner |
| 4 | `TaskRunner.__init__` | 218-221 | 1 分钟 | 初始化角色映射 |
| 5 | `TaskRunner.add_actor_rollout_worker` | 223-314 | 5 分钟 | **关键**：选择 Worker 实现 |
| 6 | `TaskRunner.run` | 535-659 | 10 分钟 | **核心**：主训练流程编排 |
| 7 | `create_rl_dataset` | 662-731 | 3 分钟 | 数据集创建 |
| 8 | `create_rl_sampler` | 734-800 | 3 分钟 | 采样器创建 |

**关键调用链（在 `TaskRunner.run` 中）**：

```
TaskRunner.run (行 535-659)
    ├── add_actor_rollout_worker (行 579)     # 注册 actor/rollout worker
    ├── add_critic_worker (行 581)            # 注册 critic worker
    ├── add_reward_model_worker (行 584)      # 注册 reward model worker
    ├── add_ref_policy_worker (行 587)        # 注册 ref policy worker
    ├── validate_config (行 590)              # 校验配置
    ├── copy_to_local (行 597)                # 下载模型权重
    ├── hf_tokenizer (行 605)                 # 加载 tokenizer
    ├── load_reward_manager (行 609-614)      # 加载奖励函数 ★
    ├── init_resource_pool_mgr (行 617)       # 初始化资源池
    ├── create_rl_dataset (行 622-637)        # 创建数据集
    ├── RayPPOTrainer.__init__ (行 641-654)   # 创建 Trainer
    ├── trainer.init_workers (行 656)         # 初始化分布式 worker
    └── trainer.fit (行 659)                  # 启动训练循环 ★★★
```

**阅读时间**：30 分钟

**阅读后你应该理解**：
- Hydra 配置如何传递给 Python
- Ray 集群如何初始化
- 不同角色（Actor/Critic/Ref/Reward）如何注册
- 奖励函数如何加载
- `RayPPOTrainer` 是训练的核心类

---

## 第三阶段：分布式训练调度器

### 3.1 阅读文件：`verl/trainer/ppo/ray_trainer.py`

**阅读目的**：理解训练循环的完整流程，包括 rollout、奖励计算、优势估计、参数更新

**推荐阅读顺序（按函数）**：

| 顺序 | 函数/类 | 行号 | 阅读时间 | 说明 |
|------|---------|------|----------|------|
| 1 | 模块 docstring | 16-31 | 2 分钟 | 模块用途概述 |
| 2 | `ResourcePoolManager` | 84-189 | 5 分钟 | Ray 资源池管理 |
| 3 | `compute_response_mask` | 257-292 | 2 分钟 | 计算 response 区域的 mask |
| 4 | `compute_advantage` | 295-391 | 8 分钟 | **关键**：优势估计调度 |
| 5 | `RayPPOTrainer.__init__` | 414-505 | 5 分钟 | Trainer 初始化 |
| 6 | `RayPPOTrainer._create_dataloader` | 507-612 | 3 分钟 | 数据加载器创建 |
| 7 | `RayPPOTrainer.init_workers` | 989-1183 | 10 分钟 | 分布式 worker 初始化 |
| 8 | `RayPPOTrainer.fit` | 1629-2022 | **20 分钟** | **核心**：训练主循环 |

### 3.2 `RayPPOTrainer.fit` 详细调用链（行 1629-2022）

这是**最核心的函数**，建议仔细阅读。以下是主循环内的关键步骤：

```python
# fit() 主循环结构（伪代码 + 行号）

def fit(self):                                          # 行 1629
    logger = Tracking(...)                              # 行 1650-1655
    self._load_checkpoint()                             # 行 1660

    # 训练前验证
    if val_before_train:
        val_metrics = self._validate()                  # 行 1667

    for epoch in range(total_epochs):                   # 行 1694
        for batch_dict in self.train_dataloader:        # 行 1695

            # === 1. Rollout 生成 ===
            gen_batch = self._get_gen_batch(batch)      # 行 1715
            gen_batch_output = actor_rollout_wg.generate_sequences(gen_batch_output)  # 行 1728

            # === 2. 奖励计算 ===
            reward_tensor, extra = self._compute_or_extract_reward(batch, reward_fn)  # 行 1802
            batch.batch["token_level_scores"] = reward_tensor                          # 行 1864

            # === 3. 计算 old_log_prob ===
            old_log_prob, mfu = self._compute_old_log_prob(batch)                      # 行 1822

            # === 4. 计算 ref_log_prob（如启用 KL） ===
            if self.use_reference_policy:
                ref_log_prob = self._compute_ref_log_prob(batch)                       # 行 1850

            # === 5. 计算 values（如启用 critic） ===
            if self.use_critic:
                values = self._compute_values(batch)                                   # 行 1856

            # === 6. 优势估计 ★★★ ===
            batch = compute_advantage(                                                 # 行 1898-1906
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,  # "grpo"
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,  # 5
            )

            # === 7. 更新 Critic（如启用） ===
            if self.use_critic:
                critic_output = self._update_critic(batch)                             # 行 1911

            # === 8. 更新 Actor ★★★ ===
            actor_output = self._update_actor(batch)                                   # 行 1919

            # === 9. 验证 ===
            if should_validate:
                val_metrics = self._validate()                                         # 行 1935

            # === 10. 保存 Checkpoint ===
            if should_save:
                self._save_checkpoint()                                                # 行 1958

            # === 11. 日志记录 ===
            logger.log(data=metrics, step=self.global_steps)                           # 行 1997
```

**阅读时间**：50 分钟

**阅读后你应该理解**：
- 训练循环的完整流程
- 各个组件（rollout、reward、advantage、update）的调用顺序
- GRPO 与 GAE 在优势计算上的分支（行 345-373）

---

## 第四阶段：核心算法实现

### 4.1 阅读文件：`verl/trainer/ppo/core_algos.py`

**阅读目的**：理解 GRPO 优势估计的数学实现和 PPO loss 计算

**推荐阅读顺序（按函数）**：

| 顺序 | 函数/类 | 行号 | 阅读时间 | 说明 |
|------|---------|------|----------|------|
| 1 | 模块 docstring | 15-28 | 2 分钟 | 模块用途概述 |
| 2 | `AdvantageEstimator` (Enum) | 109-127 | 2 分钟 | 所有支持的优势估计器 |
| 3 | `register_adv_est` | 133-156 | 2 分钟 | 优势估计器注册装饰器 |
| 4 | `get_adv_estimator_fn` | 159-173 | 1 分钟 | 按名称获取优势函数 |
| 5 | `AdaptiveKLController` / `FixedKLController` | 176-222 | 3 分钟 | KL 系数控制器 |
| 6 | `compute_gae_advantage_return` | 225-270 | 5 分钟 | GAE 优势估计（对比用） |
| 7 | **`compute_grpo_outcome_advantage`** | 274-328 | **10 分钟** | **GRPO 核心实现 ★★★** |
| 8 | `compute_grpo_vectorized_outcome_advantage` | 331-354 | 5 分钟 | GRPO 向量化版本 |
| 9 | `compute_rewards` | 744-757 | 2 分钟 | 带 KL 惩罚的奖励计算 |
| 10 | `agg_loss` | 760-814 | 5 分钟 | 损失聚合（token-mean/seq-mean） |
| 11 | **`compute_policy_loss_vanilla`** | 893-984 | **10 分钟** | **PPO Clip Loss ★★★** |

### 4.2 GRPO 核心算法详解（行 274-328）

```python
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,   # (B, T) token 奖励
    response_mask: torch.Tensor,         # (B, T) 有效 mask
    index: np.ndarray,                   # 分组索引（同一 prompt 的样本共享 id）
    epsilon: float = 1e-6,               # 防止除零
    norm_adv_by_std_in_grpo: bool = True,# 是否按 std 归一化
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 优势计算：

    公式：A_i = (R_i - mean(R_group)) / (std(R_group) + eps)

    其中：
    - R_i: 第 i 个 response 的累积奖励
    - R_group: 同一 prompt 下所有 response 的奖励集合
    """

    # 步骤 1: 计算序列级奖励（token 求和）
    scores = token_level_rewards.sum(dim=-1)           # 行 301

    # 步骤 2: 按 prompt 分组
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])           # 行 310

    # 步骤 3: 计算每组的均值和标准差
    for idx in id2score:
        scores_tensor = torch.stack(id2score[idx])
        id2mean[idx] = torch.mean(scores_tensor)       # 行 317
        id2std[idx] = torch.std(scores_tensor)         # 行 318

    # 步骤 4: 标准化优势
    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)  # 行 323

    # 步骤 5: 扩展回 token 维度
    scores = scores.unsqueeze(-1) * response_mask      # 行 326

    return scores, scores  # GRPO 的 returns = advantages
```

### 4.3 PPO Clip Loss 详解（行 893-984）

```python
@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,      # 旧策略 log_prob
    log_prob: torch.Tensor,          # 当前策略 log_prob
    advantages: torch.Tensor,        # 优势值
    response_mask: torch.Tensor,     # 有效 mask
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    PPO Clip Loss:

    L = -min(r * A, clip(r, 1-ε, 1+ε) * A)

    其中：
    - r = exp(log_π_new - log_π_old)  # 策略比率
    - A = advantages                   # 优势值
    - ε = clip_ratio                   # 裁剪范围（默认 0.2）
    """

    # 步骤 1: 计算策略比率
    negative_approx_kl = log_prob - old_log_prob       # 行 944
    ratio = torch.exp(negative_approx_kl)              # 行 947

    # 步骤 2: 计算未裁剪损失
    pg_losses1 = -advantages * ratio                   # 行 950

    # 步骤 3: 计算裁剪损失
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )                                                  # 行 955-957

    # 步骤 4: 取最大值（即最小化）
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # 行 958-960

    # 步骤 5: 聚合损失
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)  # 行 975-977

    return pg_loss, pg_metrics
```

**阅读时间**：45 分钟

**阅读后你应该理解**：
- GRPO 的核心思想：组内标准化奖励作为优势
- GRPO 与 GAE 的区别：GRPO 不需要 Value 函数
- PPO Clip Loss 的实现细节

---

## 第五阶段：奖励计算

### 5.1 阅读文件：`verl/trainer/ppo/reward.py`

**阅读目的**：理解奖励函数如何加载和调用

**推荐阅读顺序（按函数）**：

| 顺序 | 函数 | 行号 | 阅读时间 | 说明 |
|------|------|------|----------|------|
| 1 | 模块 docstring | 14-26 | 2 分钟 | 模块用途概述 |
| 2 | `get_custom_reward_fn` | 113-152 | 5 分钟 | 加载自定义奖励函数 |
| 3 | **`load_reward_manager`** | 155-246 | **10 分钟** | **奖励管理器加载 ★** |
| 4 | `compute_reward` | 249-283 | 5 分钟 | 同步计算奖励 |
| 5 | `compute_reward_async` | 286-322 | 3 分钟 | 异步计算奖励（Ray） |

### 5.2 `load_reward_manager` 调用链（行 155-246）

```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    加载奖励管理器的流程：

    1. 尝试加载自定义奖励函数
    2. 若无自定义，使用 default_compute_score
    3. 按 reward_manager 配置选择管理器类
    4. 实例化并返回
    """

    # 步骤 1: 尝试加载自定义奖励函数
    compute_score = get_custom_reward_fn(config)       # 行 188

    # 步骤 2: 若无自定义，使用默认评分函数
    if compute_score is None:
        final_compute_score = default_compute_score    # 行 225

    # 步骤 3: 选择 RewardManager 类
    if reward_manager_cfg.source == "register":
        reward_manager_cls = get_reward_manager_cls(reward_manager_cfg.name)  # 行 196

    # 步骤 4: 实例化
    return reward_manager_cls(
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        ...
    )                                                  # 行 240-246
```

### 5.3 阅读文件：`verl/utils/reward_score/gsm8k.py`

**阅读目的**：理解 GSM8K 数据集的规则评分实现

**推荐阅读顺序（按函数）**：

| 顺序 | 函数 | 行号 | 阅读时间 | 说明 |
|------|------|------|----------|------|
| 1 | 模块 docstring | 14-23 | 1 分钟 | 模块用途概述 |
| 2 | `extract_solution` | 30-81 | 5 分钟 | 从模型输出抽取答案 |
| 3 | `compute_score` | 84-117 | 5 分钟 | **GSM8K 评分逻辑 ★** |

### 5.4 GSM8K 评分逻辑详解（行 84-117）

```python
def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    GSM8K 评分规则：

    1. 严格模式：只认可 "#### 数字" 格式
    2. 若答案正确，得分 = 1.0
    3. 若答案错误但格式正确，得分 = format_score（默认 0）
    4. 若无法抽取答案，得分 = 0
    """

    # 步骤 1: 抽取最终答案
    answer = extract_solution(solution_str, method=method)  # 行 110

    # 步骤 2: 判断并返回分数
    if answer is None:
        return 0                                       # 行 112
    elif answer == ground_truth:
        return score                                   # 行 115
    else:
        return format_score                            # 行 117
```

**阅读时间**：25 分钟

**阅读后你应该理解**：
- 奖励函数如何从配置加载
- GSM8K 的评分逻辑（正则匹配 `#### 数字`）
- 自定义奖励函数的扩展方式

---

## 第六阶段：Worker 实现（可选深入）

如果你想深入了解分布式 Worker 的实现，可以继续阅读以下文件：

### 6.1 Actor/Rollout Worker

| 文件 | 说明 |
|------|------|
| `verl/workers/fsdp_workers.py` | FSDP 并行实现 |
| `verl/workers/engine_workers.py` | 新引擎实现 |
| `verl/workers/actor/dp_actor.py` | 数据并行 Actor |

### 6.2 Rollout 实现

| 文件 | 说明 |
|------|------|
| `verl/workers/rollout/base.py` | Rollout 抽象基类 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | vLLM 推理实现 |

### 6.3 奖励管理器

| 文件 | 说明 |
|------|------|
| `verl/workers/reward_manager/naive.py` | 默认奖励管理器 |
| `verl/utils/reward_score/__init__.py` | 评分函数路由 |

---

## 完整调用链图示

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              GRPO 训练完整调用链                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

examples/grpo_trainer/run_qwen2-7b.sh
    │
    │ python3 -m verl.trainer.main_ppo "${ARGS[@]}"
    ▼
verl/trainer/main_ppo.py
    │
    ├── main(config)                                    # 行 61，Hydra 入口
    │       │
    │       └── run_ppo(config)                         # 行 98，Ray 初始化
    │               │
    │               └── TaskRunner.run(config)          # 行 176，远程执行
    │                       │
    │                       ├── add_actor_rollout_worker()     # 行 579
    │                       ├── add_critic_worker()            # 行 581
    │                       ├── load_reward_manager()          # 行 609-614
    │                       │       │
    │                       │       └── verl/trainer/ppo/reward.py
    │                       │               │
    │                       │               └── default_compute_score
    │                       │                       │
    │                       │                       └── verl/utils/reward_score/gsm8k.py
    │                       │                               │
    │                       │                               └── compute_score()    # 行 84
    │                       │
    │                       ├── RayPPOTrainer.__init__()       # 行 641
    │                       ├── trainer.init_workers()         # 行 656
    │                       │
    │                       └── trainer.fit()                  # 行 659 ★★★
    │
    ▼
verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit()           # 行 1629
    │
    │ for epoch in range(total_epochs):                        # 行 1694
    │     for batch in train_dataloader:                       # 行 1695
    │
    ├── [1] Rollout 生成
    │       │
    │       └── actor_rollout_wg.generate_sequences()          # 行 1728
    │               │
    │               └── verl/workers/rollout/vllm_rollout/vllm_rollout.py
    │
    ├── [2] 奖励计算
    │       │
    │       └── _compute_or_extract_reward(batch, reward_fn)   # 行 1802
    │               │
    │               └── verl/trainer/ppo/reward.py::compute_reward()
    │                       │
    │                       └── verl/utils/reward_score/gsm8k.py::compute_score()
    │
    ├── [3] 计算 old_log_prob
    │       │
    │       └── _compute_old_log_prob(batch)                   # 行 1822
    │
    ├── [4] 计算 ref_log_prob（如启用 KL）
    │       │
    │       └── _compute_ref_log_prob(batch)                   # 行 1850
    │
    ├── [5] 计算 values（如启用 critic）
    │       │
    │       └── _compute_values(batch)                         # 行 1856
    │
    ├── [6] 优势估计 ★★★
    │       │
    │       └── compute_advantage(batch, adv_estimator="grpo") # 行 1898
    │               │
    │               └── verl/trainer/ppo/core_algos.py
    │                       │
    │                       └── compute_grpo_outcome_advantage() # 行 274
    │                               │
    │                               ├── scores = rewards.sum(dim=-1)        # 序列级奖励
    │                               ├── id2mean[idx] = mean(group_scores)   # 组内均值
    │                               ├── id2std[idx] = std(group_scores)     # 组内标准差
    │                               └── A = (R - mean) / (std + eps)        # 标准化优势
    │
    ├── [7] 更新 Critic（如启用）
    │       │
    │       └── _update_critic(batch)                          # 行 1911
    │
    ├── [8] 更新 Actor ★★★
    │       │
    │       └── _update_actor(batch)                           # 行 1919
    │               │
    │               └── actor_rollout_wg.update_actor(batch)
    │                       │
    │                       └── verl/trainer/ppo/core_algos.py
    │                               │
    │                               └── compute_policy_loss_vanilla()  # 行 893
    │                                       │
    │                                       ├── ratio = exp(log_π_new - log_π_old)
    │                                       ├── pg_loss1 = -A * ratio
    │                                       ├── pg_loss2 = -A * clip(ratio, 1-ε, 1+ε)
    │                                       └── loss = max(pg_loss1, pg_loss2)
    │
    ├── [9] 验证
    │       │
    │       └── _validate()                                    # 行 1935
    │
    ├── [10] 保存 Checkpoint
    │       │
    │       └── _save_checkpoint()                             # 行 1958
    │
    └── [11] 日志记录
            │
            └── logger.log(metrics, step)                      # 行 1997
```

---

## 推荐阅读时长

| 阶段 | 文件 | 建议时长 | 重要程度 |
|------|------|----------|----------|
| 第一阶段 | `run_qwen2-7b.sh` | 5 分钟 | ★★☆ |
| 第二阶段 | `main_ppo.py` | 30 分钟 | ★★★ |
| 第三阶段 | `ray_trainer.py` | 50 分钟 | ★★★ |
| 第四阶段 | `core_algos.py` | 45 分钟 | ★★★ |
| 第五阶段 | `reward.py` + `gsm8k.py` | 25 分钟 | ★★☆ |
| 第六阶段 | Worker 实现 | 可选 | ★☆☆ |

**总计**：约 2.5-3 小时（不含第六阶段）

---

## 关键概念速查

| 概念 | 文件位置 | 行号 | 说明 |
|------|----------|------|------|
| GRPO 优势估计 | `core_algos.py` | 274-328 | `A = (R - mean) / std` |
| PPO Clip Loss | `core_algos.py` | 893-984 | `max(r*A, clip(r)*A)` |
| 训练主循环 | `ray_trainer.py` | 1629-2022 | `fit()` 方法 |
| 奖励加载 | `reward.py` | 155-246 | `load_reward_manager()` |
| GSM8K 评分 | `gsm8k.py` | 84-117 | `compute_score()` |
| 配置入口 | `main_ppo.py` | 60-98 | `main()` |
| Worker 初始化 | `ray_trainer.py` | 989-1183 | `init_workers()` |

---

## 常见问题

### Q1: GRPO 和 PPO 有什么区别？

**A**: GRPO 是单轮 RL 的优势估计方法：
- **PPO (GAE)**: 需要 Value 函数，多步 TD 误差累积
- **GRPO**: 不需要 Value 函数，直接用组内标准化奖励作为优势

### Q2: 为什么每个 prompt 要采样 5 个 response？

**A**: GRPO 需要组内对比来计算优势。采样多个 response 可以：
1. 估计同一 prompt 下的奖励分布
2. 计算更稳定的均值和标准差
3. 增强样本效率

### Q3: KL loss 和 KL in reward 有什么区别？

**A**:
- **KL loss** (`actor.use_kl_loss`): 在 policy loss 中加入 KL 惩罚项
- **KL in reward** (`algorithm.use_kl_in_reward`): 在奖励计算时减去 KL 惩罚

---

*文档生成时间: 2025-12-28*
*基于 VERL GRPO Qwen 代码库*
