#!/bin/bash  # 注释：指定脚本解释器
# 用途：调用 torchrun 启动 FSDP SFT Trainer，执行多轮（multiturn）SFT 并开启 Ulysses SP=2。  # 注释：脚本用途
# 依赖：PyTorch 分布式/torchrun、Hydra 配置、VERL 环境、Qwen 权重、多轮 parquet 数据。  # 注释：关键依赖
# 输出：trainer.default_local_dir 指定目录保存 checkpoint 与日志。  # 注释：输出位置
# 单卡简化：nproc_per_node=1，ulysses_sequence_parallel_size=1，并适当减小 data.micro_batch_size。  # 注释：单机单卡建议

set -x  # 注释：打印执行命令，便于调试与复现

if [ "$#" -lt 2 ]; then  # 注释：参数数量检查（至少需要进程数与保存目录）
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"  # 注释：打印用法提示
    exit 1  # 注释：参数不足，退出并返回失败码
fi

nproc_per_node=$1  # 注释：读取每节点进程数（通常等于 GPU 数）
save_path=$2  # 注释：读取保存目录

# Shift the arguments so $@ refers to the rest  # 注释：说明后续参数透传给 Hydra
shift 2  # 注释：移除前两个位置参数

# 下面是 torchrun 命令的关键参数说明（多轮 SFT + SP2）。  # 注释：多行命令说明
# -m verl.trainer.fsdp_sft_trainer：以模块方式启动 SFT 训练入口。  # 注释：入口模块
# data.train_files / data.val_files：多轮 parquet 训练/验证路径。  # 注释：数据输入
# data.multiturn.enable=true：开启多轮消息拼接逻辑。  # 注释：多轮开关
# data.multiturn.messages_key=messages：多轮消息字段名。  # 注释：字段对齐
# ulysses_sequence_parallel_size=2：Ulysses 序列并行大小（SP2）。  # 注释：并行配置
# use_remove_padding=true：启用变长去 padding 优化。  # 注释：性能优化
# $@：透传用户追加 Hydra 覆盖项。  # 注释：可扩展配置

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/multiturn/train.parquet \
    data.val_files=$HOME/data/multiturn/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=multiturn-sft-qwen-2.5-0.5b-instruct-sp2 \
    trainer.logger=console \
    trainer.total_training_steps=1 $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true  # 注释：开启去 padding 与 SP2
