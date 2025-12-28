# VERL（超详细中文注释版）注释计划（注释：文档主标题，明确这是对 VERL 仓库进行超详细中文注释的总体计划。）

目标：对本仓库所有关键脚本与配置进行“超详细中文注释”维护，**不改变原始行为**的前提下补齐：模块级说明、函数级说明、关键逻辑逐段解释，并为复杂逻辑提供**最小可运行示例或伪输入输出**。（注释：说明本计划的总体目标、范围与不改变行为的约束。）

> 硬性要求（本计划的核心约束）（注释：引用块标题，强调以下内容为强制执行的核心约束。）
> 1) 注释必须使用**中文**，且足够详细，能让首次接触者独立理解流程。（注释：强制要求条目，必须逐条满足以保证注释质量。）
> 2) 每个复杂逻辑必须给**示例**（最小输入 + 关键中间结果 + 预期输出）。（注释：强制要求条目，必须逐条满足以保证注释质量。）
> 3) 只新增注释/文档，不修改代码行为、参数含义或数据格式。（注释：强制要求条目，必须逐条满足以保证注释质量。）
> 4) 需注释覆盖的文件类型：`*.py`、`*.sh`、`*.yaml`、`*.yml`、`*.json`、`*.slurm`。（注释：强制要求条目，必须逐条满足以保证注释质量。）

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

## 0. 当前状态（注释：本节说明当前注释工作的阶段与已有成果。）

- 已创建 `verl_annotated/` 用于放置注释版代码（注释：说明当前已完成的准备动作，例如创建注释版目录。）
- 目前已复制代码文件（`*.py` / `*.sh` / `*.yaml` / `*.yml` / `*.json` / `*.slurm`），其余文件仍保留在原仓库结构中（注释：说明已有文件复制进展，便于确认现阶段的覆盖范围。）

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

## 1. 注释规范（统一约定，必须遵守）（注释：本节统一说明所有文件类型的注释标准与写法约束。）

### 1.1 Python（`.py`）（注释：小节标题，说明 Python 文件的详细注释要求。）

**模块顶部必须包含**：（注释：补充解释该行在计划中的作用与含义。）
- 模块用途（做什么）（注释：强调模块顶部必须说明用途，帮助读者快速定位功能。）
- 输入/输出（文件、张量、数据结构）（注释：提醒必须写清输入输出类型与形态，避免理解歧义。）
- 关键依赖（外部库、环境变量、模型/权重路径）（注释：要求列出外部依赖，便于环境准备与复现。）
- 典型用法（最小运行示例）（注释：要求提供最小用法示例，帮助读者快速上手。）
- **调用路径概览**（入口脚本 -> 本模块 -> 下游关键函数）（注释：强调必须描述调用路径，帮助理解模块在全局中的位置。）

**函数/类 docstring 必须包含**：（注释：补充解释该行在计划中的作用与含义。）
- 参数含义（类型、形状、单位、取值范围）（注释：要求说明参数类型/形状/范围，保证可理解与可验证。）
- 返回值（类型、形状、语义）（注释：要求说明返回值语义，避免调用者误用。）
- 副作用（写文件、修改全局状态）（注释：提示需写明副作用，便于评估影响范围。）
- 异常/边界条件（空输入、缺字段、超参不合法）（注释：要求标出异常与边界，减少运行时意外。）
- 最小示例（伪输入输出即可）（注释：要求给出最小示例，便于记忆与验证理解。）
- **调用路径依赖**（必须清楚）：该函数/类内部**调用了哪些函数**，以及在全局**被哪些文件路径调用**（注释：强调必须写清被谁调用与调用了谁的双向关系。）

**调用路径依赖的写法要求（必须遵守）**：（注释：补充解释该行在计划中的作用与含义。）
- 必须写出“调用 → 被调用”的双向信息（注释：说明调用关系的表达方式与必要字段。）
- 形式建议（可按需细化为小标题）：（注释：要点条目，补充说明当前小节的具体要求或状态。）
  - 所在位置：`路径/文件名.py` + 函数名（注释：规定调用路径说明应包含具体位置与函数名。）
  - 典型调用路径：入口脚本 -> ... -> 当前函数（注释：要求给出从入口到当前函数的典型调用链。）
  - 被谁调用：列出仓库内引用该函数/类的**具体文件路径**（注释：要求列出调用方文件路径，方便反向追踪。）
  - 调用了谁（项目内）：列出同仓库内被调用的函数/类（注释：要求列出被调用函数或外部依赖，建立依赖图。）
  - 调用了谁（外部依赖）：列出关键库调用（如 torch/pandas/transformers）（注释：要求列出被调用函数或外部依赖，建立依赖图。）
- 若仅在本文件内使用，明确写“仅在本文件内调用/无外部引用”（注释：强调要明确内部使用范围，避免误判影响面。）

**调用路径依赖示例（格式示意）**：（注释：补充解释该行在计划中的作用与含义。）
```
所在位置  # 注释：小节标题，说明下面是位置与函数信息的展示格式
--------  # 注释：分隔线，帮助阅读时快速识别小节边界
- 路径：`verl/trainer/ppo/core_algos.py`  # 注释：给出文件路径，便于读者直接定位到源码
- 函数：`compute_gae(rewards, values, gamma, lam)`  # 注释：给出函数名和关键参数，明确关注对象
# （空行说明：此处用注释行代替空行，强调不同块的分隔语义）  # 注释：保证每一行都有中文说明
典型调用路径  # 注释：说明从入口脚本到当前函数的完整调用链
------------  # 注释：分隔线，突出“调用路径”这一小节
- `verl/trainer/main_ppo.py` -> `PPOTrainer.fit(...)` -> `rollout_and_train(...)` -> `compute_gae(...)`  # 注释：示例链路，展示真实调用顺序
# （空行说明：此处为段落分隔，避免信息挤在一起）  # 注释：保持结构清晰、可读
被谁调用  # 注释：列出外部调用当前函数的入口或模块
--------  # 注释：分隔线，明确“被谁调用”区域
- `verl/trainer/ppo/ray_trainer.py`（训练阶段计算优势）  # 注释：给出调用方与调用场景，便于理解用途
# （空行说明：分隔被调用与调用关系两部分）  # 注释：强调结构层次
调用了谁（项目内）  # 注释：列出当前函数内部依赖的项目内函数/模块
----------------  # 注释：分隔线，突出“项目内依赖”
- `verl/trainer/ppo/utils.py::discount_cumsum(...)`  # 注释：指出内部调用的具体函数，方便追踪
# （空行说明：分隔项目内依赖与外部依赖）  # 注释：让依赖类别一目了然
调用了谁（关键外部依赖）  # 注释：列出核心第三方库调用，帮助准备环境
----------------------  # 注释：分隔线，突出“外部依赖”
- `torch` 张量运算/广播  # 注释：说明依赖库及其作用场景
```

**逐段注释原则**：（注释：补充解释该行在计划中的作用与含义。）
- 本次注释版要求“每一行或每个完整逻辑段都必须有中文注释”，且每个逻辑块必须说明意图与数据流（注释：强调本任务需要逐行/逐段注释，同时解释数据流与意图。）
- 对复杂逻辑（如：异步 rollout、奖励模型路由、分布式并行、张量分片、checkpoint 合并等）必须做到**逐句级**解释（注释：强调复杂流程必须逐句说明，不能略过关键步骤。）

**补充示例：复杂逻辑如何写成可记忆的最小示例**：（注释：新增说明，示范如何把抽象算法变成可手算的例子。）
- 示例主题：GAE 优势计算（注释：选择常见算法作演示，便于迁移到其他复杂逻辑。）
- 最小输入：rewards=[1,1]，values=[0.5,0.2]，gamma=0.9，lam=0.95，V_2=0（注释：给出可手算的最小输入，含终止值设为 0 的假设。）
- 关键中间量：delta=[1+0.9*0.2-0.5=0.68，1+0-0.2=0.8]（注释：明确每一步计算，展示中间量如何得到。）
- 预期输出：adv=[0.68+0.9*0.95*0.8=1.364，0.8]（注释：展示最终优势值，形成“输入→中间量→输出”的闭环。）
- 记忆提示：先算 delta，再做折扣累积（注释：用口诀式提示增强记忆点。）
- 示例主题：张量分片与合并（注释：再给一个常见工程场景，避免只懂一个算法例子。）
- 最小输入：矩阵 W 形状 [4,4]，按列切成 2 片（注释：给出简单维度，便于想象切分方式。）
- 关键中间量：片0=前2列（形状[4,2]），片1=后2列（形状[4,2]）（注释：展示分片后的实际形状变化。）
- 预期输出：将片0与片1按列拼接，恢复为原始 W（注释：说明合并操作应回到原始形状与数值顺序。）
- 记忆提示：分片=切块，合并=拼回（注释：给出简短记忆口诀，降低理解成本。）

**Python 注释示例（格式示意）**：（注释：补充解释该行在计划中的作用与含义。）
```python
"""（注释：docstring 开始，下面展示模块级说明的写法）
模块用途：PPO 训练入口，负责加载配置并启动训练流程。（说明：一句话概括模块功能）
输入：（说明：列出主要输入项）
  - config_path: 配置文件路径（yaml）。（说明：标明输入类型与含义）
输出：（说明：列出主要输出项）
  - 在 output_dir 下保存 checkpoint 与日志。（说明：指出输出目录与产物）
依赖：ray, torch, hydra。（说明：列出关键外部依赖）
示例：（说明：提供最小可运行命令）
  python main_ppo.py --config_path ./verl/trainer/config/ppo_trainer.yaml。（说明：示例命令便于复现）
"""  # 注释：docstring 结束，回到正常代码结构
# （分隔说明：模块级说明结束，下面进入函数定义）  # 注释：用注释行代替空行，保证逐行可读
def build_trainer(cfg):  # 注释：定义构建训练器的函数
    """（注释：函数 docstring 开始，说明函数级细节）
    根据配置构建 Trainer。（说明：一句话说明函数目标）
    （空行说明：这里用文字代替空行，强调段落分隔）（说明：保持每行都有注释）
    参数：（说明：列出参数说明）
      cfg (DictConfig): Hydra 配置对象。（说明：参数类型与语义）
    返回：（说明：列出返回值说明）
      trainer (PPOTrainer): 训练器实例。（说明：返回对象及用途）
    调用：（说明：列出内部调用）
      - PPOTrainer(...)。（说明：示例内部调用的关键函数）
    被调用：（说明：列出外部调用方）
      - main_ppo.py::main(...)。（说明：指出入口位置）
    示例：（说明：给出最小输入输出）
      输入 cfg["trainer"]["name"]="ppo"。（说明：示例输入）
      输出 PPOTrainer 实例。（说明：示例输出）
    """  # 注释：函数 docstring 结束
    ...  # 注释：省略函数体，仅示意结构
```

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

### 1.2 Shell / Slurm（`.sh` / `.slurm`）（注释：小节标题，说明 Shell/Slurm 脚本的注释要求。）

- 文件头部必须说明：用途、依赖（CUDA/conda/环境变量/集群设置）、输出目录（注释：要点条目，补充说明当前小节的具体要求或状态。）
- 每条关键命令都要行内中文注释：做什么、输入/输出、依赖什么（注释：提醒必须写清输入输出类型与形态，避免理解歧义。）
- 如脚本依赖多机/多卡，必须写明**单机/单卡简化运行**方式（仅注释说明）（注释：要点条目，补充说明当前小节的具体要求或状态。）

**Shell 注释示例**：（注释：补充解释该行在计划中的作用与含义。）
```bash
#!/usr/bin/env bash  # 注释：指定脚本解释器，保证脚本可直接执行
# 用途：启动 PPO 训练（单机多卡）  # 注释：一句话概括脚本作用
# 依赖：CUDA>=11.8, conda 环境 verl  # 注释：列出运行所需的关键环境
# 输出：./outputs/ 下保存 checkpoint 与日志  # 注释：说明输出目录与产物
python verl/trainer/main_ppo.py --config_path ./verl/trainer/config/ppo_trainer.yaml --output_dir ./outputs/ppo  # 注释：最小可运行命令，一行写清关键参数
# 提示：如需多行书写，可在行尾加 \\ 并把注释放到上一行  # 注释：避免行内注释破坏反斜杠换行
# 单卡简化：将分布式参数设为 1，并调小 batch_size  # 注释：说明单卡/小规模运行的简化策略
```

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

### 1.3 配置文件（`.yaml` / `.yml` / `.json`）（注释：小节标题，说明 YAML/JSON 等配置文件的注释要求。）

- **YAML**：允许行内注释，需写清每个参数含义、取值范围与影响（注释：要求说明参数类型/形状/范围，保证可理解与可验证。）
- **JSON**：原文件不改结构；建立同名 sidecar（旁注文档）文档 `*.json.md`（注释：要点条目，补充说明当前小节的具体要求或状态。）
  - 说明字段 schema、典型样例、由哪个脚本生成/消费（注释：要点条目，补充说明当前小节的具体要求或状态。）
  - 给出最小样例（1~3 行）+ 解释（注释：要点条目，补充说明当前小节的具体要求或状态。）

**sidecar（旁注文档）示例**：（注释：补充解释该行在计划中的作用与含义，同时翻译 sidecar 的含义。）
```
mcp_server.json.md  # 注释：sidecar 文档文件名，用于解释同名 JSON 的字段含义
- 字段：{"name": "server", "cmd": "..."}  # 注释：说明字段结构与键名示意
- 样例：{"name": "search", "cmd": "python server.py"}  # 注释：给出最小样例，便于快速理解
- 使用脚本：verl/tools/utils/mcp_clients/*.py  # 注释：指出被哪些脚本读取/消费
```

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

## 2. 注释执行顺序（建议）（注释：本节给出推荐的执行优先级，便于按重要性推进。）

1) 训练/评测入口：`verl/trainer/main_*.py`、`scripts/*.sh`、`examples/**/run_*.sh`（注释：补充解释该行在计划中的作用与含义。）
2) 核心训练逻辑：`verl/trainer/`、`verl/workers/`、`verl/utils/`（注释：补充解释该行在计划中的作用与含义。）
3) 模型与并行实现：`verl/models/`、`verl/workers/engine/`、`verl/model_merger/`（注释：补充解释该行在计划中的作用与含义。）
4) 工具与交互：`verl/tools/`、`verl/interactions/`、`verl/experimental/`（注释：补充解释该行在计划中的作用与含义。）
5) 配置文件：`verl/trainer/config/`、`examples/**/config/`、`recipe/**/config/`（注释：补充解释该行在计划中的作用与含义。）
6) 实验配方与脚本：`recipe/`、`examples/`（注释：补充解释该行在计划中的作用与含义。）
7) 测试：`tests/`（注释：补充解释该行在计划中的作用与含义。）

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

## 3. 注释覆盖清单（勾选=已完成）（注释：本节列出需要覆盖注释的全部文件清单与状态。）

> 状态说明：（注释：引用块说明，用于突出重要规则或补充说明。）
> - ⬜ 未注释：尚未开始（注释：引用块说明，用于突出重要规则或补充说明。）
> - 🟨 进行中：已加部分说明但未达到“逐段清楚/含示例”（注释：引用块说明，用于突出重要规则或补充说明。）
> - ✅ 已完成：达到本计划的超详细注释标准（注释：引用块说明，用于突出重要规则或补充说明。）

### 根目录（注释：小节标题，用于展开更细粒度的说明或清单。）
- ⬜ `.pre-commit-config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `.readthedocs.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `setup.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

### examples（注释：小节标题，用于展开更细粒度的说明或清单。）
#### examples/cispo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/cispo_trainer/run_cispo_qwen2_5_0_5b_gsm8k.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/data_preprocess（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/data_preprocess/aime2024_multiturn_w_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/dapo_multiturn_w_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/full_hh_rlhf.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/geo3k.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/geo3k_multiturn_w_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/gsm8k.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/gsm8k_multiturn_sft.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/gsm8k_multiturn_w_interaction.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/gsm8k_multiturn_w_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/gsm8k_tool_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/hellaswag.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/math_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/multiturn.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/pokemon.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/data_preprocess/preprocess_search_r1_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/generation（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/generation/run_deepseek7b_mutli_node.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/generation/run_deepseek_v2_lite_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/gmpo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/gmpo_trainer/run_qwen2_5-7b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/gmpo_trainer/test_dapo_7b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/gmpo_trainer/test_dapo_qwen3_30b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/gpg_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/gpg_trainer/run_qwen2-7b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/gpg_trainer/run_qwen2-7b_math_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/grpo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/grpo_trainer/run_deepseek671b_math_megatron_80gb.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_deepseek671b_math_megatron_96gb.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_deepseek7b_llm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_deepseek7b_llm_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_deepseek7b_llm_math_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_deepseek7b_llm_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_glm41v_9b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_gptoss_20b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_minicpmo2_6.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_mistral13b_skyworkrm_hhrlhf.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_moonlight16b_math_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b_math_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b_math_megatron_lora.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b_seq_balance_math_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2-7b_sgl_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora_from_adapter.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5-7b_math_megatron_diff_tp.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_32b_grpo_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_7b_grpo_discrete_prof_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_7b_grpo_e2e_prof_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_7b_grpo_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl-7b-megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl-7b-sglang.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl-7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl-7b_freeze_vision.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl-7b_lora.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl-7b_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl_32b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl_3b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen2_5_vl_7b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3-235b_megatron_96gb.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3-32b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3-8b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3-8b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3_4b_grpo_vllm_1k_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3_8b_grpo_sglang_1k_spmd_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3_8b_grpo_sglang_32k_spmd_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3_vl-235b-megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3_vl-30b-megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3_vl-8b-megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3moe-30b_megatron_96gb.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_qwen3moe-30b_megatron_lora.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/grpo_trainer/run_seed_oss_36b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/gspo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/gspo_trainer/run_qwen30b_gspo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/gspo_trainer/test_gspo_3b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/gspo_trainer/test_gspo_3b_math_slurm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/gspo_trainer/test_gspo_qwen30b_a3b_ep.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/ppo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/ppo_trainer/run_deepseek7b_llm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek7b_llm_modelscope.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek7b_llm_pfppo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek7b_llm_sandbox_fusion.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek7b_llm_sp2.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek_full_hh_rlhf.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek_math_gsm8k_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_deepseek_math_gsm8k_megatron_nsys.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_gemma.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_moonlight16b_a3b_gsm8k_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen1.5_moe_a2.7b-gsm8k_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_math_gsm8k_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_rm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_rm_legacy.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_rm_reward_loop_colocate.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_rm_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_rm_seq_balance_fused_kernels.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_rm_seq_balance_nsys.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2-7b_sglang_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2.5-32b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2.5-3b_rm_legacy.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen2.5-3b_rm_reward_loop_colocate.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/ppo_trainer/run_qwen3-8b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/reinforce_plus_plus_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/reinforce_plus_plus_trainer/run_qwen2-7b_math_rf.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/reinforce_plus_plus_trainer/run_qwen2-7b_math_rf_baseline.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/remax_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/remax_trainer/run_qwen2.5-3b_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/remax_trainer/run_qwen2.5-7b_seq_balance.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/rloo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/rloo_trainer/run_qwen2-7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/rollout_correction（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/rollout_correction/run_with_rollout_corr.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/router_replay（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/router_replay/run_qwen30_a3b_megatron_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/sapo_trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/sapo_trainer/run_qwen30b_sapo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/sft（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/sft/gsm8k/run_deepseek_6b7.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_gemma_2b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_gemma_7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_qwen3_8b_sft_peft_sp2_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_qwen_05_peft.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_qwen_05_sp2.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_qwen_05_sp2_liger.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/gsm8k/run_seed_oss_36b_sft.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/multiturn/run_qwen_05_sp2.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sft/vlm/run_qwen3_vl_2b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/sglang_multiturn（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/sglang_multiturn/config/geo3k_multiturn_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/geo3k_multiturn_megatron_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/gsm8k_multiturn_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/gsm8k_multiturn_grpo_server.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/gsm8k_multiturn_grpo_w_interaction.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/gsm8k_multiturn_megatron_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/retool_multiturn_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/search_multiturn_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/search_multiturn_grpo_one_step_off.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/tool_config/mcp_server.json`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/tool_config/mcp_tool_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/config/tool_config/search_tool_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/geo3k/run_qwen2.5-3b_geo3k_multiturn_4xgpu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/geo3k/run_qwen2.5-3b_megatron_geo3k_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen0.5b_gsm8k_multiturn_curriculum.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-0.5b_gsm8k_multiturn_w_interaction.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_4xgpu_server.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_server.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_multiturn_vllm_fsdp.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen2.5-3b_megatron_gsm8k_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen3-4b_gsm8k_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/run_qwen3_4b_dapo_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/skypilot（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/skypilot/verl-grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/skypilot/verl-multiturn-tools.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/skypilot/verl-ppo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/slurm（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/slurm/ray_on_slurm.slurm`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/split_placement（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/split_placement/config/ppo_trainer_split.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/split_placement/main_ppo_split.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/split_placement/run_deepseek7b_llm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/split_placement/split_monkey_patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/tuning（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/tuning/0.5b/qwen2-0.5b_grpo-lora_1_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/14b/qwen2-14b_grpo-lora_2_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/14b/qwen2_14b_grpo_4_h800_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/32b/qwen2-32b_grpo-lora_4_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/32b/qwen2_32B_grpo_8_h20_megatron_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/3b/qwen2-3b_grpo-lora_1_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/70b/qwen2-70b_grpo_32_h20_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/70b/qwen2-70b_grpo_32_h800_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/70b/qwen2-72b_grpo-lora_8_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/7b/qwen2-7b_grpo-lora_1_h100_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `examples/tuning/7b/qwen2-7b_grpo_2_h800_fsdp_vllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### examples/tutorial（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `examples/tutorial/agent_loop_get_started/sandbox.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

### recipe（注释：小节标题，用于展开更细粒度的说明或清单。）
#### recipe/collabllm（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/collabllm/collabllm_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/collabllm_interation.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/config/agent.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/config/collabllm_interaction_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/metrics/accuracy.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/metrics/bleu_score.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/metrics/interactivity.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/metrics/pass_rate.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/metrics/token_amount.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/process_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/reward_function.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/train_rl_collabllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/train_sft_collabllm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/collabllm/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/dapo（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/dapo/config/dapo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/config/dapo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/dapo_ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/main_dapo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/prepare_dapo_data.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_early_qwen2.5_32b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen2.5_32b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen2.5_32b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen2.5_32b_rollout_corr.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen2.5_7b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen3_14b_base_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen3_8b_base_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen3_moe_30b_base_fsdp_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen3_moe_30b_megatron_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_qwen3_moe_30b_vllm_fp8_rollout.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/run_dapo_wo_ds_qwen2.5_32b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/runtime_env.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_7b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_7b_math_lora.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_7b_math_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_8b_megatron_fp16.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_8b_megatron_fp8train.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_dspk_671b_megatron_96gb.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_glm_air_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_gptoss_20b_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_qwen3_30b_math.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_qwen3_30b_math_single_node.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_qwen3_moe_30b_megatron_fp16.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/dapo/test_dapo_qwen3next_80b_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/deepeyes（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/deepeyes/configs/deepeyes_multiturn_grpo.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/deepeyes/configs/image_zoom_in_tool_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/deepeyes/deepeyes.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/deepeyes/run_deepeyes_grpo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/entropy（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/entropy/32b_clip_cov.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/32b_kl_cov.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/32b_kl_cov_mininbsz.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/7b_clip_cov.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/7b_kl_cov.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/config/entropy_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/entropy_ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/main_entropy.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/reward.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/reward_score/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/reward_score/entropy_math/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/reward_score/entropy_math/grader.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/entropy/reward_score/entropy_math/math_normalize.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/fapo（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/fapo/config/rm_config.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/prepare_fapo_data.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/reward_fn_genrm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/reward_fn_reasoning.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/reward_fn_reasoning_remote.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_baseline_32b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_baseline_7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_fapo_32b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_fapo_32b_remote.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_fapo_7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_fapo_7b_remote.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/run_fapo_genrm_train.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fapo/runtime_env.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/flowrl（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/flowrl/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/config/flowrl_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/flowrl_actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/flowrl_fsdp_worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/flowrl_ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/main_flowrl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/prepare/prepare_data.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/prepare/prepare_model.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/flowrl/run_flowrl_qwen2.5_7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/fully_async_policy（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/fully_async_policy/agent_loop/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/agent_loop/agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/agent_loop/partial_single_turn_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/agent_loop/partial_tool_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/checkpoint_engine.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/config/fully_async_ppo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/config/fully_async_ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/detach_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/fsdp2_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/fully_async_main.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/fully_async_rollouter.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/fully_async_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/megatron_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/megatron_worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/message_queue.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/param_sync.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_async_retool.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_16_16.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_32_32.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_4_12.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_64_64.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_64_64_mis.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/dapo_7b_math_fsdp2_8_8.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_96_32.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/grpo_30b_a3b_base_math_megatron_96_32_mis.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/shell/runtime_env.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/unittest/simple_streaming_demo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/vllm_rollout/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/fully_async_policy/vllm_rollout/vllm_async_server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/genrm_remote（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/genrm_remote/reward_function.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/genrm_remote/run_genrm_remote.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/gkd（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/gkd/config/on_policy_distill_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/config/runtime_env.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/main_gkd.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/megatron_kl_loss.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/megatron_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/megatron_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/run_moonlight_dsv3_training.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/client.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/join_server.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/proxy.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/start_server.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/vllm_engine.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher/worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/teacher_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/test_qwen.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/test_qwen_sglang.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/gkd/test_teacher_server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/infigui-g1（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/infigui-g1/reward_fn.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/infigui-g1/run_3b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/infigui-g1/run_7b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/minicpmo（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/minicpmo/rl_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/one_step_off_policy（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/one_step_off_policy/agent_loop/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/agent_loop/agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/config/one_step_off_ppo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/config/one_step_off_ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/distributed_util.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/main_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/megatron_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_fsdp2_4_12.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_fsdp2_64_64.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_fsdp2_64_64_ris.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_fsdp2_colocate.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_fsdp2_sglang_4_12.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_fsdp2_sglang_colocate.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_megatron_4_12.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/dapo_7b_math_megatron_colocate.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_2_6.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/grpo_0.6b_gsm8k_fsdp2_sglang_2_6.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/grpo_3b_gsm8k_fsdp2_2_6.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/shell/grpo_qwen3_8b_gsm8k_fsdp2_8_8_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/one_step_off_policy/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/open_math_reasoning（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/open_math_reasoning/compute_score.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/open_math_reasoning/prepare_eval_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/open_math_reasoning/prepare_nvidia-OpenMathReasoning_sft.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/open_math_reasoning/run_eval.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/open_math_reasoning/run_generation.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/open_math_reasoning/run_sft_qwen3_8b.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/prime（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/prime/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/config/prime_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/main_prime.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/prime_core_algos.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/prime_dp_rm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/prime_fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/prime_ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/run_prime_qwen.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/prime/run_prime_qwen_code.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/r1（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/r1/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/config/evaluation.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/data_process.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/main_eval.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/reward_score.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/run_r1_distill_qwen.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/tasks/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/tasks/gpqa.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/tasks/livecodebench.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1/tasks/math_reward.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/r1_ascend（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/r1_ascend/deepscaler.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/engine_core.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/json_to_parquet.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/main_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/megatron_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/ray_start_grpo_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/run_deepseekv3_671b_grpo_megatron_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/vllm_parallel_state.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/r1_ascend/vllm_rollout_spmd.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/specRL（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/specRL/cache_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/config/specRL_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/main_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/vllm_plugin/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/vllm_plugin/patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/vllm_plugin/patch_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/vllm_plugin/v0_10_0/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/specRL/vllm_plugin/v0_10_0/patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/spin（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/spin/config/spin_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/core_algos.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/dp_actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/main_spin.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/run_spin.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/spin_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/spin/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/sppo（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/sppo/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/config.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/config/sppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/dp_actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/main_sppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/run_qwen2.5-7b_rm.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/sppo_ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/sppo/sppo_worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/transfer_queue（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/transfer_queue/agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/transfer_queue/config/transfer_queue_ppo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/transfer_queue/config/transfer_queue_ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/transfer_queue/main_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/transfer_queue/ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/transfer_queue/run_qwen3-8b_transferqueue.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### recipe/vla（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `recipe/vla/config/rob_ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/dp_rob.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/env_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/action_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/isaac_env/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/isaac_env/isaac_env.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/libero_env/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/libero_env/libero_env.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/libero_env/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/envs/libero_env/venv.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/main_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/models/openvla_oft/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/models/openvla_oft/configuration_prismatic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/models/openvla_oft/constants.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/models/openvla_oft/modeling_prismatic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/models/openvla_oft/processing_prismatic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/models/openvla_oft/train_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/naive_rollout_rob.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/prepare_libero_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/rob_ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/run_simpleVLA_isaac_disagg.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/run_simpleVLA_libero_grpo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/workers/env/env_loop_wg_test.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/workers/env/env_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `recipe/vla/workers/env/env_worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

### verl（注释：小节标题，用于展开更细粒度的说明或清单。）
#### verl/(root)（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/base_config.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/protocol.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/experimental（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/experimental/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/prometheus_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/single_turn_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/tool_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/tool_parser.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/agent_loop/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/dataset/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/dataset/sampler.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/dynamic_dataset/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/dynamic_dataset/dynamicgen_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_manager/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_manager/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_manager/dapo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_manager/limited.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_manager/naive.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_manager/registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/reward_model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/router/inner_sglang_router.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/experimental/reward_loop/router/naive_router.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/interactions（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/interactions/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/interactions/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/interactions/gsm8k_interaction.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/interactions/utils/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/interactions/utils/interaction_registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/interactions/weather_interaction.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/model_merger（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/model_merger/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/model_merger/__main__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/model_merger/base_model_merger.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/model_merger/fsdp_model_merger.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/model_merger/megatron_model_merger.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/models（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/models/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/checkpoint_utils/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/checkpoint_utils/llama_loader.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/checkpoint_utils/llama_loader_depracated.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/checkpoint_utils/llama_saver.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/layers/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/layers/parallel_attention.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/layers/parallel_decoder.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/layers/parallel_linear.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/layers/parallel_mlp.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/layers/parallel_rmsnorm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/llama/megatron/modeling_llama_megatron.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/bridge.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/config_converter.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/loader.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/mbridge.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/model_forward.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/model_forward_1f1b_overlap.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/model_forward_fused.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/model_initializer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/patch_v012.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/attention.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/rope_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/vision_config.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/vision_model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/qwen2_5_vl/vision_transformer_block.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/saver.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/util.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/mcore/weight_converter.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/checkpoint_utils/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/checkpoint_utils/qwen2_loader_depracated.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/checkpoint_utils/qwen2_saver.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/layers/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/layers/parallel_attention.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/layers/parallel_decoder.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/layers/parallel_linear.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/layers/parallel_mlp.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/layers/parallel_rmsnorm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/qwen2/megatron/modeling_qwen2_megatron.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/apertus.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/dense_common.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/glm4v.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/kimi_vl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/llama.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/monkey_patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/npu_patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/qwen2.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/qwen2_vl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/transformers/qwen3_vl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/models/weight_loader_registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/single_controller（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/single_controller/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/single_controller/base/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/single_controller/base/decorator.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/single_controller/base/worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/single_controller/base/worker_group.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/single_controller/ray/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/single_controller/ray/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/third_party（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/third_party/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/third_party/torch/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/third_party/torch/distributed/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/third_party/torch/distributed/_state_dict_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/third_party/torch/distributed/checkpoint/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/third_party/torch/distributed/checkpoint/state_dict.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/third_party/vllm/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/tools（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/tools/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/base_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/geo3k_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/gsm8k_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/image_zoom_in_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/mcp_base_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/mcp_search_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/sandbox_fusion_tools.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/schemas.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/search_tool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/utils/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/utils/mcp_clients/McpClientManager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/utils/mcp_clients/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/utils/search_r1_like_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/tools/utils/tool_registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/trainer/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/_generated_ppo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/_generated_ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/actor/actor.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/actor/dp_actor.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/actor/megatron_actor.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/algorithm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/algorithm/rollout_correction.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/config.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/critic/critic.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/critic/dp_critic.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/critic/megatron_critic.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/data/legacy_data.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/engine/fsdp.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/engine/megatron.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/engine/veomni.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/evaluation.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/generation.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/model/hf_model.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/npu_profile/npu_profile.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/optim/fsdp.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/optim/megatron.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/optim/veomni.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/ppo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/ref/dp_ref.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/ref/megatron_ref.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/ref/ref.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/reward_manager.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/reward_model/dp_reward_loop.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/reward_model/dp_reward_model.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/reward_model/megatron_reward_loop.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/reward_model/megatron_reward_model.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/reward_model/reward_model.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/rollout/rollout.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/sft_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/config/sft_trainer_engine.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/constants_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/fsdp_sft_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/main_eval.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/main_generation.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/main_generation_server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/main_ppo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/core_algos.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/metric_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/ray_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/reward.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/rollout_corr_helper.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/ppo/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/runtime_env.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/sft_trainer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/trainer/sft_trainer_ray.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/utils（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/utils/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/activation_offload.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/attention_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/chat_template.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/checkpoint/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/checkpoint/checkpoint_handler.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/checkpoint/checkpoint_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/checkpoint/fsdp_checkpoint_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/checkpoint/megatron_checkpoint_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/config.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/dataset_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/multiturn_sft_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/rl_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/rm_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/sft_dataset.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/dataset/vision_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/debug/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/debug/metrics.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/debug/performance.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/debug/trajectory_tracker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/device.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/distributed.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/experimental/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/experimental/torch_functional.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/flops_counter.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/fs.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/fsdp_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/groupwise.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/hdfs_io.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/import_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/kernel/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/kernel/kernels.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/kernel/linear_cross_entropy.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/logger/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/logger/aggregate_logger.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/logging_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/dist_checkpointing.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/memory.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/optimizer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/pipeline_parallel.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/router_replay_patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/router_replay_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/sequence_parallel.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron/tensor_parallel.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron_peft_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/megatron_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/memory_buffer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/memory_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/metric/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/metric/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/net_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/npu_flash_attn_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/config.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/empty_annotations.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/mstx_profile.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/nvtx_profile.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/performance.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/profiler/profile.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/py_functional.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/ray_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/rendezvous/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/rendezvous/ray_backend.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/geo3k.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/gsm8k.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/math_batch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/math_dapo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/math_reward.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/math_verify.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/prime_code/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/prime_code/testing_util.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/prime_code/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/prime_math/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/prime_math/grader.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/prime_math/math_normalize.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/sandbox_fusion/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/sandbox_fusion/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/reward_score/search_r1_like_qa_em.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/rollout_skip.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/rollout_trace.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/seqlen_balancing.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/sglang/sglang_fp8_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/tensordict_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/tokenizer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/torch_dtypes.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/torch_functional.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/tracking.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/transferqueue_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/transformers_compat.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/ulysses.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/vllm/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/vllm/patch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/vllm/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/utils/vllm/vllm_fp8_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### verl/workers（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `verl/workers/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/actor/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/actor/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/actor/dp_actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/actor/megatron_actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/critic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/engine.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/megatron_peft.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/optimizer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/reward_model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/config/rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/critic/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/critic/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/critic/dp_critic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/critic/megatron_critic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/fsdp/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/fsdp/transformer_impl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/fsdp/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/megatron/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/megatron/transformer_impl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/megatron/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/mindspeed/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/mindspeed/transformer_impl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/veomni/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/veomni/transformer_impl.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine/veomni/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/engine_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/megatron_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/abstract.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/batch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/dapo.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/naive.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/prime.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_manager/registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_model/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_model/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_model/megatron/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/reward_model/megatron/reward_model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/hf_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/naive/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/naive/naive_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/replica.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/schemas.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/sglang_rollout/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/sglang_rollout/async_sglang_server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/sglang_rollout/http_server_engine.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/sglang_rollout/sglang_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/sglang_rollout/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/tokenizer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/vllm_rollout/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/vllm_rollout/utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/vllm_rollout/vllm_async_server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/rollout/vllm_rollout/vllm_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/sharding_manager/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/sharding_manager/base.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/sharding_manager/fsdp_ulysses.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/utils/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/utils/losses.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `verl/workers/utils/padding.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

### scripts（注释：小节标题，用于展开更细粒度的说明或清单。）
#### scripts/(root)（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `scripts/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/converter_hf_to_mcore.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/diagnose.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/generate_trainer_config.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/init_random_model.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/install_vllm_sglang_mcore.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/legacy_model_merger.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/print_cfg.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `scripts/rollout_viewer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

### tests（注释：小节标题，用于展开更细粒度的说明或清单。）
#### tests/(root)（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/kill_github_tests.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/test_base_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/test_protocol_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/test_protocol_v2_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/experimental（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/experimental/agent_loop/agent_utils.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/agent_loop/test_basic_agent_loop.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/agent_loop/test_gpt_oss_tool_parser.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/agent_loop/test_multi_modal.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/agent_loop/test_standalone_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/reward_fn.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/test_agent_loop_reward_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/test_agent_reward_loop_colocate.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/test_async_token_bucket_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/test_rate_limited_reward_manager_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/test_reward_model_disrm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/reward_loop/test_reward_model_genrm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/experimental/vla/test_sim_envs.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/interactions（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/interactions/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/interactions/test_gsm8k_interaction.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/interactions/test_interaction_registry.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/models（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/models/test_engine.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/models/test_transformer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/models/test_transformers_ulysses.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/single_controller（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/single_controller/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/base/test_decorator.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/check_worker_alive/main.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/detached_worker/client.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/detached_worker/run.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/detached_worker/server.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_auto_padding_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_colocated_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_colocated_workers_fused.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_data_transfer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_decorator_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_device_mesh_register.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_driverfunc_to_worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_fused_workers_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_get_set_dispatch_collect_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_high_level_scheduling_api.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_nested_worker.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_ray_collectives.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_ray_local_envs_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_ray_utils_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_rvdz.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_split_resource_pool.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_worker_group_basics.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/single_controller/test_worker_group_torch.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/special_distributed（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/special_distributed/run_all.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_distributed/test_fsdp_ckpt.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_distributed/test_mcore_config_converter.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_distributed/test_tensor_dict.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_distributed/test_torch_functional.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/special_e2e（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/special_e2e/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/check_custom_rwd_fn.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/check_results.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/envs/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/envs/digit_completion/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/envs/digit_completion/task.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/envs/digit_completion/tokenizer.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/generation/run_gen_qwen05.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/generation/run_gen_qwen05_server.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/ppo_trainer/expert_parallel/qwen2moe_minimal.json`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/ppo_trainer/expert_parallel/qwen3moe_minimal.json`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/ppo_trainer/run_function_reward.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/ppo_trainer/run_model_reward.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/ppo_trainer/run_single_gpu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/ppo_trainer/run_single_gpu_with_engine.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_dapo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_fully_async_policy.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_genrm_remote.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_geo3k_fsdp_sgl_multiturn_w_tool.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_grpo_lora_with_merge.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_gsm8k_fsdp_sgl_multiturn_sf_tool.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_gsm8k_fsdp_sgl_multiturn_w_tool.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_one_step_off_policy.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_ppo_trainer_megatron.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_prime.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_r1_distill_qwen_aime24_eval.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_spin.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_sppo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_test.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/run_transferqueue.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/sft/compare_sft_engine_results.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/sft/run_sft.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/sft/run_sft_engine.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/sft/test_sft_engine_all.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_e2e/sft/test_sp_loss_match.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/special_npu（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/special_npu/run_one_step_off_policy.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen2_5_05b_dapo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen2_5_05b_grpo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen2_5_05b_grpo_mindspeed.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen2_5_05b_sft_peft_sp2.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen2_5_vl_3b_npu.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen3_06b_ppo.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_npu/run_qwen3_30b_dapo_mindspeed.sh`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/special_sanity（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/special_sanity/check_api_docs.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_dataproto_usage.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_device_api_usage.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_docs_time_info.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_docstrings.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_license.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_pr_description.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/check_pr_title.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/test_config_docs.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/test_import.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/type_coverage_check.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/validate_imported_docs.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/special_sanity/validate_structure.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/special_standalone（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/special_standalone/test_memory_buffers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/trainer（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/trainer/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/config/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/config/legacy_ppo_megatron_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/config/legacy_ppo_trainer.yaml`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/config/test_algo_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/config/test_legacy_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/ppo/__init__.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/ppo/test_core_algos_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/ppo/test_metric_utils_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/ppo/test_rollout_corr.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/trainer/ppo/test_rollout_corr_integration.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/utils（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/utils/_test_module.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/ckpt/test_esi_save_ckpt_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/dataset/test_create_rl_sampler_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/dataset/test_multiturn_sft_dataset_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/dataset/test_rl_collate_fn_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/dataset/test_rl_dataset_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/dataset/test_sft_dataset_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/debug/test_metrics.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/megatron/test_pipeline_parallel.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/reward_score/reward_score/test_sandbox_fusion_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/reward_score/test_sandbox_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_activation_offload.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_flops_counter.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_fs_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_groupwise.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_import_utils_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_linear_cross_entropy.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_mlflow_key_sanitization.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_model_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_nvtx_profile.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_rollout_skip_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_rollout_trace_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_seqlen_balancing.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_special_linear_cross_entropy_tp.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_special_mstx_profile.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_temp_env_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_timeout_decorator_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/utils/test_torch_functional.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

#### tests/workers（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `tests/workers/actor/test_special_dp_actor.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/config/test_actor_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/config/test_critic_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/config/test_engine_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/config/test_optim_config_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/critic/test_special_dp_critic.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/reward_manager/test_registry_on_cpu.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/perf/vllm_async_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/resource/tool_configs/mcp_server.json`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/rollout_sglang/test_http_server_engine.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/rollout_vllm/run_fsdp_vllm.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/rollout_vllm/test_vllm_abort.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/test_hf_rollout.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/test_sglang_async_rollout_multimodal_delta.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/rollout/test_sglang_rollout_sharding_manager.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/test_fsdp_attn_implementation.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）
- ⬜ `tests/workers/test_fsdp_workers.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

### docs（注释：小节标题，用于展开更细粒度的说明或清单。）
#### docs/(root)（注释：子小节标题，通常对应某个目录或模块的文件清单。）
- ⬜ `docs/conf.py`（注释：清单条目，状态为“未注释”，表示该文件后续需要按规范补齐中文注释与示例。）

---
（注释：分隔线，用于将不同章节或主题清晰分开，便于阅读。）

## 4. 验收标准（注释：本节给出完成注释后的质量检查与验收要点。）

- 注释版脚本在不改参数的情况下，行为与原脚本完全一致（注释：要点条目，补充说明当前小节的具体要求或状态。）
- 每个复杂模块至少提供 1 个**最小示例**（或伪输入输出）（注释：要求给出最小示例，便于记忆与验证理解。）
- 配置项说明清晰：参数含义、取值范围、影响写全（注释：要求说明参数类型/形状/范围，保证可理解与可验证。）
- 所有新增注释为中文，且足够详细，能独立指导读者理解与复现流程（注释：要点条目，补充说明当前小节的具体要求或状态。）
