# 已在 2 卡 / 4 卡环境测试通过（注释：说明脚本的基本验证范围）
# 用途：调用 torchrun 启动 FSDP SFT Trainer，对 Qwen2.5-0.5B-Instruct 做单轮 SFT（注释：脚本用途）
# 依赖：PyTorch 分布式、Hydra 配置、VERL 运行环境、Qwen 权重、GSM8K parquet 数据（注释：关键依赖）
# 输出：trainer.default_local_dir 指定目录保存 checkpoint 与日志（注释：输出位置）
# 单卡简化：将 nproc_per_node=1，适当减小 data.micro_batch_size_per_gpu（注释：单机单卡简化建议）

set -x  # 打印执行命令，便于调试与复现（注释：shell 调试开关）

if [ "$#" -lt 2 ]; then  # 参数不足时给出用法提示（注释：参数数量检查）
    echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"  # 打印用法（注释：用户提示）
    exit 1  # 退出并返回失败码（注释：提前终止）
fi

nproc_per_node=$1  # 读取每节点进程数（通常等于 GPU 数）（注释：位置参数 1）
save_path=$2  # 读取保存目录（注释：位置参数 2）

# Shift the arguments so $@ refers to the rest（注释：下面把额外参数留给 Hydra 配置覆盖）
shift 2  # 移除前两个位置参数，剩余参数透传给 Hydra（注释：参数处理）

# 下面是 torchrun 命令的关键参数说明（注释：逐段解释多行命令）
# -m verl.trainer.fsdp_sft_trainer：以模块方式启动 SFT 训练入口（注释：入口模块）
# data.train_files / data.val_files：训练/验证 parquet 数据路径（注释：数据输入）
# data.prompt_key / data.response_key：单轮字段名（注释：字段对齐）
# data.prompt_dict_keys / data.response_dict_keys：从 dict 中抽取字段（注释：字段抽取）
# model.partial_pretrain：Qwen2.5-0.5B-Instruct 权重（注释：模型权重）
# trainer.default_local_dir：保存 checkpoint 的目录（注释：输出目录）
# trainer.*：实验命名、日志与训练轮数（注释：训练元信息）
# model.lora_* / model.target_modules：LoRA 适配配置（注释：PEFT 相关参数）
# $@：透传用户追加的 Hydra 覆盖项（注释：可扩展配置）

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=console \
    trainer.total_epochs=1 $@ \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear

# Or you can do this:（注释：下面给出可选替代写法）
# model.target_modules=[q_proj,v_proj] \
