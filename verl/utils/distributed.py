# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行，保持逐行说明
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：提示许可证链接
#  # 注释：保留注释符号，保证该行有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证地址
#  # 注释：保留注释符号，保证该行有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：分布式训练/推理的进程组初始化与 NUMA 亲和设置工具。  # 注释：模块用途
输入/输出：输入主要来自环境变量与 Ray 上下文；输出为初始化后的进程组与 rank 信息。  # 注释：模块输入输出概览
关键依赖：torch.distributed、ray、pynvml（可选）、verl.utils.device。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- local_rank, rank, world_size = initialize_global_process_group()  # 注释：单机/多机初始化
- initialize_global_process_group_ray(timeout_second=None)  # 注释：Ray worker 初始化
调用路径概览：  # 注释：调用路径说明标题
- trainer/worker 入口 -> initialize_global_process_group -> torch.distributed。  # 注释：标准分布式链路
- Ray worker -> initialize_global_process_group_ray -> torch.distributed。  # 注释：Ray 环境链路
"""  # 注释：模块 docstring 结束
# （分隔说明：标准库导入）  # 注释：替代空行，保持逐行注释
import ctypes  # 注释：动态加载 libnuma/pynvml 等系统库
import os  # 注释：读取环境变量
from datetime import timedelta  # 注释：构建超时对象
# （分隔说明：第三方依赖导入）  # 注释：替代空行，保持逐行注释
import ray  # 注释：Ray 运行时上下文用于获取设备绑定信息
import torch.distributed  # 注释：torch 分布式通信核心接口
# （分隔说明：项目内工具导入）  # 注释：替代空行，保持逐行注释
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device, is_npu_available  # 注释：设备与后端选择
# （分隔说明：NUMA 亲和设置）  # 注释：替代空行，保持逐行注释

def set_numa_affinity():  # 注释：设置 NUMA 亲和性
    """
    功能：在可用时设置 NUMA 亲和，提高 GPU 访问本地 CPU 内存的效率。  # 注释：函数用途
    参数：无。  # 注释：参数说明标题
    返回：无。  # 注释：返回值说明标题
    副作用：可能调用 pynvml 设置 GPU 与 CPU 亲和。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - NPU 环境或 libnuma/pynvml 不可用时直接跳过。  # 注释：跳过条件
    最小示例：  # 注释：最小示例标题
    - set_numa_affinity()  # 注释：示例调用
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/distributed.py::set_numa_affinity。  # 注释：函数位置
    - 典型调用路径：megatron_workers -> set_numa_affinity。  # 注释：典型调用链
    - 被谁调用：verl/workers/megatron_workers.py、verl/model_merger/megatron_model_merger.py、recipe/gkd/megatron_workers.py。  # 注释：调用方示例
    - 调用了谁（项目内）：verl.utils.device.is_npu_available。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：ray.get_runtime_context、pynvml.*。  # 注释：外部依赖说明
    """
    if is_npu_available:  # 注释：NPU 环境暂不支持 NUMA 绑定（当前实现按变量使用）
        # TODO (FightingZhen) libnuma.so is not available in e2e_ascend CI image, remove this code after image update.  # 注释：原 TODO 保留
        return  # 注释：直接退出，避免调用 libnuma/pynvml

    initialized = False  # 注释：记录 pynvml 是否成功初始化
    try:  # 注释：捕获初始化/设置过程异常
        libnuma = ctypes.CDLL("libnuma.so")  # 注释：动态加载 NUMA 库
        if libnuma.numa_available() < 0:  # 注释：系统不支持 NUMA
            return  # 注释：直接退出

        import pynvml  # 注释：延迟导入以避免无依赖时报错

        pynvml.nvmlInit()  # 注释：初始化 NVML
        initialized = True  # 注释：标记已初始化，便于 finally 中释放
        device_name = "NPU" if is_npu_available else "GPU"  # 注释：Ray 设备类型名称
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])  # 注释：获取本地设备索引
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)  # 注释：获取设备句柄
        pynvml.nvmlDeviceSetCpuAffinity(handle)  # 注释：设置 NUMA 亲和
    except ImportError:  # 注释：缺少 pynvml 依赖
        print("Warning: pynvml not available, skipping NUMA affinity setup")  # 注释：缺少 pynvml 时降级
    except Exception as e:  # 注释：捕获其他异常
        print(f"Warning: Failed to set NUMA affinity: {e}")  # 注释：异常提示
    finally:  # 注释：清理资源
        if initialized:  # 注释：仅在成功初始化时关闭 NVML
            pynvml.nvmlShutdown()  # 注释：释放 NVML 资源
# （分隔说明：初始化进程组）  # 注释：替代空行，保持逐行注释

def initialize_global_process_group(timeout_second=36000):  # 注释：初始化全局进程组
    """
    功能：初始化默认 torch.distributed 进程组，并返回 rank 信息。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - timeout_second (int)：初始化超时时间（秒）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - tuple(local_rank, rank, world_size)：本地进程号、全局 rank、总进程数。  # 注释：返回值语义
    副作用：创建默认进程组；设置当前设备为 local_rank。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 依赖环境变量 LOCAL_RANK/RANK/WORLD_SIZE/DIST_INIT_METHOD。  # 注释：环境变量要求
    最小示例：  # 注释：最小示例标题
    - LOCAL_RANK=0, RANK=0, WORLD_SIZE=8 -> (0, 0, 8)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/distributed.py::initialize_global_process_group。  # 注释：函数位置
    - 典型调用路径：fsdp_sft_trainer/main_ppo -> initialize_global_process_group。  # 注释：典型调用链
    - 被谁调用：verl/trainer/fsdp_sft_trainer.py、verl/trainer/sft_trainer.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：verl.utils.device.get_nccl_backend、get_torch_device。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.distributed.init_process_group。  # 注释：外部依赖说明
    """
    torch.distributed.init_process_group(  # 注释：创建默认进程组
        get_nccl_backend(),  # 注释：根据设备类型选择 nccl/hccl
        timeout=timedelta(seconds=timeout_second),  # 注释：设置初始化超时
        init_method=os.environ.get("DIST_INIT_METHOD", None),  # 注释：使用指定 init_method（可选）
    )  # 注释：进程组初始化结束
    local_rank = int(os.environ["LOCAL_RANK"])  # 注释：读取本地 rank
    rank = int(os.environ["RANK"])  # 注释：读取全局 rank
    world_size = int(os.environ["WORLD_SIZE"])  # 注释：读取世界大小

    if torch.distributed.is_initialized():  # 注释：确认进程组已建立
        get_torch_device().set_device(local_rank)  # 注释：绑定当前进程到本地设备
    return local_rank, rank, world_size  # 注释：返回 rank 信息
# （分隔说明：销毁进程组）  # 注释：替代空行，保持逐行注释

def destroy_global_process_group():  # 注释：销毁默认进程组
    """
    功能：销毁全局进程组，释放通信资源。  # 注释：函数用途
    参数：无。  # 注释：参数说明标题
    返回：无。  # 注释：返回值说明标题
    副作用：销毁默认进程组。  # 注释：副作用说明
    异常/边界条件：若未初始化则跳过。  # 注释：边界条件
    最小示例：  # 注释：最小示例标题
    - destroy_global_process_group()  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/distributed.py::destroy_global_process_group。  # 注释：函数位置
    - 典型调用路径：fsdp_sft_trainer -> destroy_global_process_group。  # 注释：典型调用链
    - 被谁调用：verl/trainer/fsdp_sft_trainer.py、tests/special_distributed/test_mcore_config_converter.py。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.distributed.destroy_process_group。  # 注释：外部依赖说明
    """
    if torch.distributed.is_initialized():  # 注释：仅在初始化后销毁
        torch.distributed.destroy_process_group()  # 注释：销毁默认进程组
# （分隔说明：Ray 环境初始化）  # 注释：替代空行，保持逐行注释

def initialize_global_process_group_ray(timeout_second=None):  # 注释：Ray worker 初始化分布式进程组
    """
    功能：在 Ray worker 内初始化分布式进程组。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - timeout_second (Optional[int])：超时时间（秒），None 使用默认。  # 注释：参数含义
    返回：无。  # 注释：返回值说明标题
    副作用：在 Ray 环境下初始化默认进程组（CPU + 设备后端）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若已初始化则不重复初始化。  # 注释：幂等性
    - 依赖 RANK/WORLD_SIZE/DIST_INIT_METHOD 环境变量。  # 注释：环境变量要求
    最小示例：  # 注释：最小示例标题
    - RANK=0, WORLD_SIZE=2 -> init_process_group 成功。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/distributed.py::initialize_global_process_group_ray。  # 注释：函数位置
    - 典型调用路径：engine_workers/vllm_rollout -> initialize_global_process_group_ray。  # 注释：典型调用链
    - 被谁调用：verl/workers/engine_workers.py、verl/workers/rollout/vllm_rollout/vllm_rollout.py、recipe/vla/workers/env/env_worker.py。  # 注释：调用方示例
    - 调用了谁（项目内）：verl.utils.device.get_device_name、get_nccl_backend。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.distributed.init_process_group。  # 注释：外部依赖说明
    """
    # in current ray environment, LOCAL_RANK is always zero.  # 注释：Ray 环境下 LOCAL_RANK 固定为 0

    import torch.distributed  # 注释：局部导入，避免顶层导入冲突

    timeout = timedelta(seconds=timeout_second) if timeout_second is not None else None  # 注释：构造超时对象

    if not torch.distributed.is_initialized():  # 注释：仅在未初始化时创建进程组
        rank = int(os.environ.get("RANK", 0))  # 注释：读取 rank
        world_size = int(os.environ.get("WORLD_SIZE", 1))  # 注释：读取 world_size
        torch.distributed.init_process_group(  # 注释：初始化进程组
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",  # 注释：CPU+设备后端组合
            rank=rank,  # 注释：当前 rank
            world_size=world_size,  # 注释：总进程数
            timeout=timeout,  # 注释：初始化超时
            init_method=os.environ.get("DIST_INIT_METHOD", None),  # 注释：init_method（可选）
        )  # 注释：进程组初始化完成
