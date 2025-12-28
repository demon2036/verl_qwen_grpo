# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释:版权声明
#  # 注释:保留注释符号,保证该行有中文说明
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释:声明 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释:使用需遵守许可证
# You may obtain a copy of the License at  # 注释:提示许可证链接
#  # 注释:保留注释符号,保证该行有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释: Apache 2.0 许可证地址
#  # 注释:保留注释符号,保证该行有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释:免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释:软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释:不提供担保
# See the License for the specific language governing permissions and  # 注释:更多许可条款
# limitations under the License.  # 注释:许可限制说明
"""
模块用途:基于新引擎架构的 Worker 实现,支持训练(TrainingWorker)与混合(ActorRolloutRefWorker)。  # 注释:模块用途
输入/输出:  # 注释:模块输入输出概览标题
- 输入:TensorDict 格式的批次数据、Hydra 配置对象。  # 注释:输入说明
- 输出:训练指标(metrics)、模型checkpoint、推理结果(TensorDict)。  # 注释:输出说明
关键依赖:  # 注释:关键依赖说明标题
- verl.workers.engine:新引擎抽象层(BaseEngine、EngineRegistry)。  # 注释:核心引擎依赖
- verl.single_controller.base:Worker 基类与装饰器。  # 注释:控制器依赖
- tensordict.TensorDict:数据容器。  # 注释:数据结构依赖
- ray:分布式调度框架。  # 注释:分布式依赖
典型用法:  # 注释:最小用法示例标题
- TrainingWorker:用于独立训练场景(SFT、actor训练)。  # 注释:训练worker用法
- ActorRolloutRefWorker:RL场景下混合worker(actor+rollout+可选ref)。  # 注释:混合worker用法
调用路径概览:  # 注释:调用路径说明标题
- 入口示例:verl/trainer/ppo/ray_trainer.py -> create_workers -> ActorRolloutRefWorker。  # 注释:上层入口举例
- 典型链路:ray_trainer -> ActorRolloutRefWorker.init_model -> TrainingWorker -> BaseEngine。  # 注释:关键调用链
"""  # 注释:模块 docstring 结束
# (分隔说明:导入标准库)  # 注释:替代空行,保持逐行注释
import logging  # 注释:日志记录
import os  # 注释:环境变量与路径
from functools import partial  # 注释:偏函数工具
from itertools import chain  # 注释:迭代器链接
from typing import Any, Optional  # 注释:类型注解
# (分隔说明:导入第三方库)  # 注释:替代空行,保持逐行注释
import torch  # 注释:张量操作
from codetiming import Timer  # 注释:性能计时
from omegaconf import DictConfig, open_dict  # 注释:Hydra 配置对象
from tensordict import NonTensorData, TensorDict  # 注释:TensorDict 数据容器
from torch.distributed.device_mesh import init_device_mesh  # 注释:设备网格初始化
# (分隔说明:导入项目内依赖)  # 注释:替代空行,保持逐行注释
from verl.single_controller.base import Worker  # 注释:Worker 基类
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register  # 注释:装饰器与dispatch模式
from verl.utils import tensordict_utils as tu  # 注释:TensorDict 工具函数
from verl.utils.config import omega_conf_to_dataclass  # 注释:配置转dataclass
from verl.utils.device import (  # 注释:设备管理工具
    get_device_name,  # 注释:获取设备名称
    get_torch_device,  # 注释:获取torch设备对象
    set_expandable_segments,  # 注释:设置可扩展段
)  # 注释:导入结束
from verl.utils.distributed import initialize_global_process_group_ray  # 注释:Ray分布式进程组初始化
from verl.utils.flops_counter import FlopsCounter  # 注释:FLOPS计数器
from verl.utils.memory_utils import aggressive_empty_cache  # 注释:激进清空缓存
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage  # 注释:分布式profiler与内存日志
from verl.utils.py_functional import append_to_dict  # 注释:字典追加工具
from verl.utils.torch_functional import allgather_dict_into_dict  # 注释:分布式all_gather字典
from verl.workers.config import ActorConfig, HFModelConfig, RolloutConfig, TrainingWorkerConfig  # 注释:worker配置类
from verl.workers.rollout.base import BaseRollout, get_rollout_class  # 注释:rollout基类与工厂函数
from verl.workers.utils.losses import ppo_loss  # 注释:PPO loss函数
# (分隔说明:初始化 logger)  # 注释:替代空行,保持逐行注释
logger = logging.getLogger(__file__)  # 注释:创建模块级logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释:从环境变量读取日志级别
# (分隔说明:TrainingWorker 类定义)  # 注释:替代空行,保持逐行注释
class TrainingWorker(Worker):  # 注释:训练worker类,基于新引擎架构
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.  # 注释:保留英文说明

    功能:基于新引擎架构的训练worker,提供train_batch/infer_batch等接口。  # 注释:类用途
    参数:  # 注释:参数说明标题
    - config (TrainingWorkerConfig):训练worker配置,包含model、engine、optimizer、checkpoint配置。  # 注释:参数含义
    返回:  # 注释:返回值说明标题
    - TrainingWorker实例。  # 注释:返回值语义
    副作用:  # 注释:副作用说明标题
    - 初始化分布式进程组、创建engine实例、初始化FLOPS计数器。  # 注释:副作用说明
    异常/边界条件:  # 注释:异常说明标题
    - 配置缺失或格式错误时抛异常。  # 注释:异常说明
    最小示例:  # 注释:最小示例标题
    - config = TrainingWorkerConfig(...); worker = TrainingWorker(config)。  # 注释:示例
    调用路径依赖:  # 注释:调用路径说明标题
    - 所在位置:verl/workers/engine_workers.py::TrainingWorker。  # 注释:类位置
    - 典型调用路径:ActorRolloutRefWorker.init_model -> TrainingWorker.__init__。  # 注释:典型调用链
    - 被谁调用:ActorRolloutRefWorker、verl/trainer/sft_trainer_ray.py。  # 注释:调用方说明
    - 调用了谁(项目内):EngineRegistry.new、FlopsCounter。  # 注释:项目内依赖说明
    - 调用了谁(关键外部依赖):initialize_global_process_group_ray。  # 注释:外部依赖说明
    """  # 注释:类 docstring 结束

    def __init__(self, config: TrainingWorkerConfig):  # 注释:初始化方法
        """
        功能:初始化TrainingWorker,创建engine、配置dispatch、初始化FLOPS计数器。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - config (TrainingWorkerConfig):训练worker配置。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None (就地初始化)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 创建self.engine、self.flops_counter、self.loss_fn等实例变量。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 配置缺失时抛异常。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker = TrainingWorker(config)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.__init__。  # 注释:方法位置
        - 典型调用路径:ActorRolloutRefWorker.init_model -> TrainingWorker(config)。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.init_model。  # 注释:调用方说明
        - 调用了谁(项目内):EngineRegistry.new、_register_dispatch_collect_info、FlopsCounter。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):initialize_global_process_group_ray。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        Worker.__init__(self)  # 注释:调用基类初始化
        # (分隔说明:导入engine模块)  # 注释:替代空行,保持逐行注释
        from verl.workers.engine import BaseEngine, EngineRegistry  # 注释:导入engine基类与注册表
        # (分隔说明:初始化分布式进程组)  # 注释:替代空行,保持逐行注释
        initialize_global_process_group_ray(timeout_second=None)  # 注释:初始化Ray全局进程组
        # (分隔说明:保存配置)  # 注释:替代空行,保持逐行注释
        self.config = config  # 注释:保存完整配置
        self.model_config = self.config.model_config  # 注释:模型配置
        self.engine_config = self.config.engine_config  # 注释:引擎配置
        self.optimizer_config = self.config.optimizer_config  # 注释:优化器配置
        self.checkpoint_config = self.config.checkpoint_config  # 注释:checkpoint配置
        self.device_name = get_device_name()  # 注释:获取当前设备名称(cuda/cpu/npu)
        # (分隔说明:同步use_remove_padding配置)  # 注释:替代空行,保持逐行注释
        # we use the one defined in model  # 注释:原注释保留
        self.engine_config.use_remove_padding = self.model_config.use_remove_padding  # 注释:从model_config同步到engine_config
        # (分隔说明:TODO profiler)  # 注释:替代空行,保持逐行注释
        # TODO: add DistProfilerExtension  # 注释:原TODO
        # self.profiler_config = self.config.profiler_config  # 注释:原注释代码
        # tool_config = self.profiler_config.tool_config  # 注释:原注释代码
        # DistProfilerExtension.__init__(  # 注释:原注释代码
        #     self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=tool_config)  # 注释:原注释代码
        # )  # 注释:原注释代码
        # (分隔说明:创建engine实例)  # 注释:替代空行,保持逐行注释
        self.engine: BaseEngine = EngineRegistry.new(  # 注释:通过注册表创建engine实例
            model_type=self.config.model_type,  # 注释:模型类型(language_model等)
            backend=self.engine_config.strategy,  # 注释:后端策略(fsdp/megatron等)
            model_config=self.model_config,  # 注释:模型配置
            engine_config=self.engine_config,  # 注释:引擎配置
            optimizer_config=self.optimizer_config,  # 注释:优化器配置
            checkpoint_config=self.checkpoint_config,  # 注释:checkpoint配置
        )  # 注释:engine创建结束
        # (分隔说明:注册dispatch信息)  # 注释:替代空行,保持逐行注释
        # build dispatch info  # 注释:原注释保留
        self._register_dispatch_collect_info(  # 注释:注册dispatch/collect配置
            mesh_name="train",  # 注释:mesh名称
            dp_rank=self.engine.get_data_parallel_rank(),  # 注释:数据并行rank
            is_collect=self.engine.is_mp_src_rank_with_outputs(),  # 注释:是否收集输出(仅MP source rank收集)
        )  # 注释:注册结束
        # (分隔说明:初始化FLOPS计数器)  # 注释:替代空行,保持逐行注释
        self.flops_counter = FlopsCounter(self.model_config.hf_config)  # 注释:创建FLOPS计数器用于计算MFU
        # (分隔说明:初始化loss_fn为None)  # 注释:替代空行,保持逐行注释
        self.loss_fn = None  # 注释:损失函数初始化为None,后续通过set_loss_fn设置

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,dispatch模式为ONE_TO_ALL(广播到所有worker)
    def to(self, device, model=True, optimizer=True, grad=True):  # 注释:手动控制模型/优化器/梯度的加载/卸载
        """Manual control of load/offload  # 注释:保留英文说明

        功能:手动控制模型/优化器/梯度在设备(cuda/cpu)间迁移。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - device (str):目标设备('cpu'或'device'),'device'会自动映射到cuda/npu。  # 注释:参数含义
        - model (bool):是否迁移模型参数。  # 注释:参数含义
        - optimizer (bool):是否迁移优化器状态。  # 注释:参数含义
        - grad (bool):是否迁移梯度。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用self.engine.to()迁移数据。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - device不是'cpu'或'device'时assert失败。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.to('cpu', model=True, optimizer=False)  # 注释:示例(仅将模型迁移到CPU)
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.to。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.to('cpu')。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.to、外部trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.engine.to。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        assert device in ["cpu", "device"]  # 注释:校验device参数
        # (分隔说明:device映射)  # 注释:替代空行,保持逐行注释
        if device == "device":  # 注释:若指定'device'
            device = get_device_name()  # 注释:自动映射到实际设备名称(cuda/npu)
        # (分隔说明:调用engine.to)  # 注释:替代空行,保持逐行注释
        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)  # 注释:迁移数据

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,dispatch模式为ONE_TO_ALL
    def set_loss_fn(self, loss_fn):  # 注释:设置损失函数
        """
        功能:设置损失函数。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - loss_fn (Callable):损失函数,签名为fn(data, model_output) -> loss。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 修改self.loss_fn。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.set_loss_fn(ppo_loss)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.set_loss_fn。  # 注释:方法位置
        - 典型调用路径:ActorRolloutRefWorker.init_model -> actor.set_loss_fn。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.set_loss_fn、ActorRolloutRefWorker.init_model。  # 注释:调用方说明
        - 调用了谁(项目内):无。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        self.loss_fn = loss_fn  # 注释:保存损失函数

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,dispatch模式为ONE_TO_ALL
    def reset(self):  # 注释:重置engine到初始状态
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states  # 注释:保留英文说明

        功能:重置engine到初始状态,若未初始化则初始化,否则重新加载checkpoint。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - 无。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用self.engine.initialize()初始化或重置engine。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.reset()。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.reset。  # 注释:方法位置
        - 典型调用路径:ActorRolloutRefWorker.init_model -> worker.reset()。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.init_model。  # 注释:调用方说明
        - 调用了谁(项目内):self.engine.initialize。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        self.engine.initialize()  # 注释:初始化或重置engine

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only):  # 注释:后处理输出,计算MFU等指标
        """
        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        功能:后处理engine输出,计算loss、all_gather metrics、计算MFU。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - output (dict):engine输出,包含loss、model_output、metrics。  # 注释:参数含义
        - global_token_num (list):全局token数列表,用于MFU计算。  # 注释:参数含义
        - delta_time (float):训练耗时(秒)。  # 注释:参数含义
        - forward_only (bool):是否仅forward(True时MFU除以3)。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:包含metrics与model_output的TensorDict。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 修改output字典(pop loss、metrics、model_output)。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - final_output = worker._postprocess_output(output, global_token_num=[100,200], delta_time=1.5, forward_only=False)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker._postprocess_output。  # 注释:方法位置
        - 典型调用路径:train_batch/infer_batch -> _postprocess_output。  # 注释:典型调用链
        - 被谁调用:train_batch、infer_batch。  # 注释:调用方说明
        - 调用了谁(项目内):allgather_dict_into_dict、FlopsCounter.estimate_flops、tu.get_tensordict。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):torch.distributed.all_reduce。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        # TODO: whether to log memory  # 注释:原TODO
        # metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024 ** 3)  # 注释:原注释代码
        # metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024 ** 3)  # 注释:原注释代码
        # metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)  # 注释:原注释代码
        # (分隔说明:提取metrics与loss)  # 注释:替代空行,保持逐行注释
        metrics: dict = output.pop("metrics")  # 注释:提取metrics字典
        # perform all gather in dp group to ensure that it's correct.  # 注释:原注释保留
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton  # 注释:原注释保留
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token  # 注释:原注释保留
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))  # 注释:将loss列表求和(可能包含多个micro-batch)
        torch.distributed.all_reduce(  # 注释:在DP组内all_reduce loss
            loss, op=torch.distributed.ReduceOp.AVG, group=self.engine.get_data_parallel_group()  # 注释:平均loss
        )  # 注释:all_reduce结束
        loss = loss.item()  # 注释:转为Python标量
        # (分隔说明:处理grad_norm与lr)  # 注释:替代空行,保持逐行注释
        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad  # 注释:原注释保留
        grad_norm = metrics.pop("grad_norm", None)  # 注释:提取grad_norm(已经all_reduce过)
        lr = metrics.pop("lr", None)  # 注释:提取学习率
        # (分隔说明:all_gather其他metrics)  # 注释:替代空行,保持逐行注释
        # For other metrics, we perform all gather in dp group  # 注释:原注释保留
        final_metrics = allgather_dict_into_dict(data=metrics, group=self.engine.get_data_parallel_group())  # 注释:在DP组内all_gather其他metrics
        final_metrics["loss"] = loss  # 注释:添加loss到final_metrics
        if grad_norm is not None:  # 注释:若有grad_norm
            final_metrics["grad_norm"] = grad_norm  # 注释:添加grad_norm
        if lr is not None:  # 注释:若有lr
            final_metrics["lr"] = lr  # 注释:添加lr
        # (分隔说明:计算MFU)  # 注释:替代空行,保持逐行注释
        # compute mfu  # 注释:原注释保留
        if global_token_num is not None:  # 注释:若提供global_token_num
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_token_num, delta_time)  # 注释:估算实际FLOPS与理论FLOPS
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()  # 注释:计算MFU(Model FLOPs Utilization)
            if forward_only:  # 注释:若仅forward
                final_metrics["mfu"] /= 3.0  # 注释:MFU除以3(因为backward占2/3)
        # (分隔说明:提取model_output)  # 注释:替代空行,保持逐行注释
        # model outputs  # 注释:原注释保留
        model_output = output.pop("model_output", {})  # 注释:提取model_output字典
        # We only return final_metrics  # 注释:原注释保留
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})  # 注释:构造TensorDict,包含model_output与metrics
        return final_output  # 注释:返回final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)  # 注释:注册方法,使用nd_compute_dataproto dispatch,非阻塞
    def train_mini_batch(self, data: TensorDict) -> TensorDict:  # 注释:训练多个mini-batch(支持多epoch)
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        功能:将批次切分为N个mini-batch,训练多个epoch。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - data (TensorDict):完整批次数据,包含mini_batch_size/num_mini_batch/epochs等参数。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:聚合后的metrics(仅MP source rank返回,其他返回None)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用train_batch训练多次,更新模型参数。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - mini_batch_size与num_mini_batch必须提供其一。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - data["mini_batch_size"]=32; output = worker.train_mini_batch(data)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.train_mini_batch。  # 注释:方法位置
        - 典型调用路径:ActorRolloutRefWorker.update_actor -> actor.train_mini_batch。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.update_actor、外部trainer。  # 注释:调用方说明
        - 调用了谁(项目内):train_batch、tu.make_iterator、append_to_dict。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):Timer、torch.distributed.all_gather_object。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        # (分隔说明:解析参数)  # 注释:替代空行,保持逐行注释
        batch_size_per_dp = data.shape[0]  # 注释:每个DP rank的batch size
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)  # 注释:是否禁用自动offload
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)  # 注释:mini_batch大小(全局)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)  # 注释:mini_batch数量
        epochs = tu.pop(data, key="epochs", default=1)  # 注释:训练epoch数
        seed = tu.pop(data, key="seed", default=42)  # 注释:随机种子
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})  # 注释:dataloader额外参数
        # (分隔说明:校验参数)  # 注释:替代空行,保持逐行注释
        assert mini_batch_size is not None or num_mini_batch is not None  # 注释:两者至少提供其一
        # (分隔说明:计算mini_batch_size_per_gpu)  # 注释:替代空行,保持逐行注释
        if mini_batch_size is None:  # 注释:若未提供mini_batch_size
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"  # 注释:校验可整除
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch  # 注释:计算每GPU的mini_batch size
        else:  # 注释:若提供mini_batch_size
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (  # 注释:校验可整除
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"  # 注释:错误信息
            )  # 注释:assert结束
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()  # 注释:计算每GPU的mini_batch size
        # (分隔说明:创建dataloader)  # 注释:替代空行,保持逐行注释
        # make iterator  # 注释:原注释保留
        dataloader = tu.make_iterator(  # 注释:创建TensorDict迭代器
            data,  # 注释:完整数据
            mini_batch_size=mini_batch_size_per_gpu,  # 注释:每GPU的mini_batch size
            epochs=epochs,  # 注释:epoch数
            seed=seed + self.engine.get_data_parallel_rank(),  # 注释:seed加DP rank以保证各rank随机性不同
            dataloader_kwargs=dataloader_kwargs,  # 注释:额外参数
        )  # 注释:迭代器创建结束
        # (分隔说明:训练循环)  # 注释:替代空行,保持逐行注释
        with (  # 注释:上下文管理器
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),  # 注释:进入train模式
            Timer(name="train_batch", logger=None),  # 注释:计时器(不打印)
        ):  # 注释:with结束
            # update  # 注释:原注释保留
            output_lst = []  # 注释:收集每个mini-batch的输出
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs  # 注释:总迭代次数
            # (分隔说明:遍历mini-batch)  # 注释:替代空行,保持逐行注释
            for batch_idx, mini_batch_td in enumerate(dataloader):  # 注释:遍历dataloader
                # add global token num  # 注释:原注释保留
                global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)  # 注释:计算每个序列的token数
                # allgather from dp rank  # 注释:原注释保留
                global_token_num_output = [None] * self.engine.get_data_parallel_size()  # 注释:预分配列表
                torch.distributed.all_gather_object(  # 注释:all_gather token数(用于MFU计算)
                    global_token_num_output, global_token_num, self.engine.get_data_parallel_group()  # 注释:在DP组内all_gather
                )  # 注释:all_gather结束
                global_token_num = [x for xs in global_token_num_output for x in xs]  # 注释:展平列表
                tu.assign_non_tensor(  # 注释:向TensorDict添加非tensor字段
                    mini_batch_td,  # 注释:目标TensorDict
                    global_token_num=NonTensorData(global_token_num),  # 注释:全局token数
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,  # 注释:最后一个mini-batch才更新lr scheduler
                    disable_auto_offload=True,  # 注释:禁用自动offload
                )  # 注释:assign结束
                actor_output = self.train_batch(mini_batch_td)  # 注释:训练单个mini-batch
                output_lst.append(actor_output)  # 注释:收集输出
            # (分隔说明:聚合metrics)  # 注释:替代空行,保持逐行注释
            if self.engine.is_mp_src_rank_with_outputs():  # 注释:仅MP source rank聚合metrics
                actor_output = [tu.get(output, "metrics") for output in output_lst]  # 注释:提取所有metrics
                metrics = {}  # 注释:聚合后的metrics
                for output in actor_output:  # 注释:遍历每个输出
                    for key, val in output.items():  # 注释:遍历每个metric
                        # flattn dp and micro batch  # 注释:原注释保留
                        if isinstance(val, list):  # 注释:若为列表
                            output[key] = list(chain.from_iterable(val))  # 注释:展平列表(DP与micro-batch维度)
                    append_to_dict(metrics, output)  # 注释:追加到metrics字典
                # (分隔说明:构造输出)  # 注释:替代空行,保持逐行注释
                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()  # 注释:构造TensorDict并移到CPU
            else:  # 注释:非MP source rank
                output = None  # 注释:返回None
        return output  # 注释:返回聚合后的输出

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)  # 注释:注册方法,非阻塞
    def train_batch(self, data: TensorDict) -> TensorDict:  # 注释:训练单个batch
        """
        功能:训练单个batch,计算loss并更新参数。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - data (TensorDict):单个batch数据,包含input_ids、labels等。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:包含metrics的TensorDict(仅MP source rank返回,其他返回None)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 更新模型参数、可能更新lr scheduler。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - self.loss_fn为None时assert失败。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - output = worker.train_batch(data)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.train_batch。  # 注释:方法位置
        - 典型调用路径:train_mini_batch -> train_batch。  # 注释:典型调用链
        - 被谁调用:train_mini_batch、外部trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.engine.train_batch、_postprocess_output、self.engine.lr_scheduler_step。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):Timer。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"  # 注释:校验loss_fn已设置
        # global_token_num should be a list of number of tokens of each seq in this batch  # 注释:原注释保留
        global_token_num = tu.get(data, key="global_token_num")  # 注释:提取global_token_num
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)  # 注释:提取disable_auto_offload
        # (分隔说明:注入默认参数)  # 注释:替代空行,保持逐行注释
        # inject engineering parameters if not specified  # 注释:原注释保留
        default_keys = dict(  # 注释:默认参数字典
            use_remove_padding=self.model_config.use_remove_padding,  # 注释:是否移除padding
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,  # 注释:是否动态batch size
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,  # 注释:每GPU最大token数
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,  # 注释:每GPU micro batch size
            use_fused_kernels=self.engine_config.use_fused_kernels,  # 注释:是否使用fused kernels
        )  # 注释:字典结束
        # (分隔说明:注入未指定的参数)  # 注释:替代空行,保持逐行注释
        for key, val in default_keys.items():  # 注释:遍历默认参数
            if key not in data.keys():  # 注释:若data中未指定
                tu.assign_non_tensor(data, **{key: val})  # 注释:注入默认值
        # (分隔说明:训练)  # 注释:替代空行,保持逐行注释
        with (  # 注释:上下文管理器
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),  # 注释:进入train模式
            Timer(name="train_batch", logger=None) as timer,  # 注释:计时器
        ):  # 注释:with结束
            output = self.engine.train_batch(data, loss_function=self.loss_fn)  # 注释:调用engine训练
            # containing loss, model_output and metrics  # 注释:原注释保留
            # for training, we only care about loss and metrics  # 注释:原注释保留
        delta_time = timer.last  # 注释:训练耗时
        # (分隔说明:更新lr scheduler)  # 注释:替代空行,保持逐行注释
        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)  # 注释:是否更新lr scheduler
        # update lr scheduler  # 注释:原注释保留
        if update_lr_scheduler:  # 注释:若需要更新
            lr = self.engine.lr_scheduler_step()  # 注释:更新lr scheduler并返回新lr
        else:  # 注释:若不更新
            lr = None  # 注释:lr为None
        # (分隔说明:后处理输出)  # 注释:替代空行,保持逐行注释
        if self.engine.is_mp_src_rank_with_outputs():  # 注释:仅MP source rank后处理
            # we don't need model_output in training. Maybe we change out mind later  # 注释:原注释保留
            output.pop("model_output")  # 注释:移除model_output(训练时不需要)
            if lr is not None:  # 注释:若有lr
                output["metrics"]["lr"] = lr  # 注释:添加lr到metrics
            final_output = self._postprocess_output(  # 注释:后处理输出
                output, global_token_num=global_token_num, delta_time=delta_time, forward_only=False  # 注释:非forward_only
            ).cpu()  # 注释:移到CPU
        else:  # 注释:非MP source rank
            final_output = None  # 注释:返回None
        return final_output  # 注释:返回final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)  # 注释:注册方法,非阻塞
    def infer_batch(self, data: TensorDict) -> TensorDict:  # 注释:推理单个batch
        """
        功能:推理单个batch,可选计算loss。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - data (TensorDict):单个batch数据,包含input_ids等。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:包含metrics与model_output的TensorDict(仅MP source rank返回,其他返回None)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 无(推理不更新参数)。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - data["compute_loss"]=False; output = worker.infer_batch(data)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.infer_batch。  # 注释:方法位置
        - 典型调用路径:ActorRolloutRefWorker.compute_log_prob -> actor.infer_batch。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.compute_log_prob/compute_ref_log_prob。  # 注释:调用方说明
        - 调用了谁(项目内):self.engine.infer_batch、_postprocess_output。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):Timer。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        # add mfu calculator  # 注释:原注释保留
        global_token_num = tu.get(data, key="global_token_num")  # 注释:提取global_token_num
        compute_loss = tu.get(data, key="compute_loss", default=True)  # 注释:是否计算loss
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)  # 注释:是否禁用自动offload
        # (分隔说明:注入默认参数)  # 注释:替代空行,保持逐行注释
        default_keys = dict(  # 注释:默认参数字典(注意使用infer参数)
            use_remove_padding=self.model_config.use_remove_padding,  # 注释:是否移除padding
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,  # 注释:是否动态batch size
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,  # 注释:推理时每GPU最大token数
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,  # 注释:推理时每GPU micro batch size
            use_fused_kernels=self.engine_config.use_fused_kernels,  # 注释:是否使用fused kernels
        )  # 注释:字典结束
        # (分隔说明:注入未指定的参数)  # 注释:替代空行,保持逐行注释
        for key, val in default_keys.items():  # 注释:遍历默认参数
            if key not in data.keys():  # 注释:若data中未指定
                tu.assign_non_tensor(data, **{key: val})  # 注释:注入默认值
        # (分隔说明:设置loss_function)  # 注释:替代空行,保持逐行注释
        # for sft training, we need to compute loss in eval  # 注释:原注释保留
        loss_function = self.loss_fn if compute_loss else None  # 注释:若compute_loss为True则使用loss_fn,否则None
        # (分隔说明:推理)  # 注释:替代空行,保持逐行注释
        with (  # 注释:上下文管理器
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),  # 注释:进入eval模式
            Timer(name="eval_batch", logger=None) as timer,  # 注释:计时器
        ):  # 注释:with结束
            output = self.engine.infer_batch(data, loss_function=loss_function)  # 注释:调用engine推理
        delta_time = timer.last  # 注释:推理耗时
        # (分隔说明:后处理输出)  # 注释:替代空行,保持逐行注释
        if self.engine.is_mp_src_rank_with_outputs():  # 注释:仅MP source rank后处理
            final_output = self._postprocess_output(  # 注释:后处理输出
                output, global_token_num=global_token_num, delta_time=delta_time, forward_only=True  # 注释:forward_only为True
            ).cpu()  # 注释:移到CPU
        else:  # 注释:非MP source rank
            final_output = None  # 注释:返回None
        return final_output  # 注释:返回final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):  # 注释:保存checkpoint
        """
        功能:保存checkpoint到本地与可选的HDFS。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - local_path (str):本地保存路径。  # 注释:参数含义
        - hdfs_path (str, optional):HDFS保存路径。  # 注释:参数含义
        - global_step (int):全局步数。  # 注释:参数含义
        - max_ckpt_to_keep (int, optional):最大保留checkpoint数量。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - engine.save_checkpoint的返回值。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 写文件到local_path与hdfs_path。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 路径不存在时可能抛异常。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.save_checkpoint('./ckpt', global_step=1000)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.save_checkpoint。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.save_checkpoint。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.save_checkpoint、外部trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.engine.save_checkpoint。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)  # 注释:调用engine保存

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):  # 注释:加载checkpoint
        """
        功能:加载checkpoint从本地或HDFS。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - local_path (str):本地加载路径。  # 注释:参数含义
        - hdfs_path (str, optional):HDFS加载路径。  # 注释:参数含义
        - del_local_after_load (bool):加载后是否删除本地文件。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - engine.load_checkpoint的返回值。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 修改模型参数、优化器状态,可能删除本地文件。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 路径不存在时抛异常。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.load_checkpoint('./ckpt')。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::TrainingWorker.load_checkpoint。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.load_checkpoint。  # 注释:典型调用链
        - 被谁调用:ActorRolloutRefWorker.load_checkpoint、外部trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.engine.load_checkpoint。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)  # 注释:调用engine加载
# (分隔说明:ActorRolloutRefWorker 类定义)  # 注释:替代空行,保持逐行注释

class ActorRolloutRefWorker(Worker, DistProfilerExtension):  # 注释:混合worker类,包含actor/rollout/可选ref
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.  # 注释:保留英文说明

    功能:RL场景下的混合worker,根据role参数组合actor/rollout/ref功能。  # 注释:类用途
    参数:  # 注释:参数说明标题
    - config (DictConfig):Hydra配置对象,包含actor、rollout、ref、model配置。  # 注释:参数含义
    - role (str):角色名,可选'actor'/'rollout'/'ref'/'actor_rollout'/'actor_rollout_ref'。  # 注释:参数含义
    返回:  # 注释:返回值说明标题
    - ActorRolloutRefWorker实例。  # 注释:返回值语义
    副作用:  # 注释:副作用说明标题
    - 根据role初始化不同组件(actor/rollout/ref)。  # 注释:副作用说明
    异常/边界条件:  # 注释:异常说明标题
    - role不在合法值中时assert失败。  # 注释:异常说明
    最小示例:  # 注释:最小示例标题
    - worker = ActorRolloutRefWorker(config, role='actor_rollout')。  # 注释:示例
    调用路径依赖:  # 注释:调用路径说明标题
    - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker。  # 注释:类位置
    - 典型调用路径:ray_trainer -> create_workers -> ActorRolloutRefWorker。  # 注释:典型调用链
    - 被谁调用:verl/trainer/ppo/ray_trainer.py。  # 注释:调用方说明
    - 调用了谁(项目内):TrainingWorker、BaseRollout、DistProfiler。  # 注释:项目内依赖说明
    - 调用了谁(关键外部依赖):omega_conf_to_dataclass、init_device_mesh。  # 注释:外部依赖说明
    """  # 注释:类 docstring 结束

    def __init__(self, config: DictConfig, role: str, **kwargs):  # 注释:初始化方法
        """
        功能:初始化ActorRolloutRefWorker,根据role设置组件标志位。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - config (DictConfig):配置对象。  # 注释:参数含义
        - role (str):角色名。  # 注释:参数含义
        - **kwargs:额外参数(未使用)。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None(就地初始化)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 设置self._is_actor、self._is_rollout、self._is_ref标志位,初始化profiler。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - role不在合法值中时assert失败。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker = ActorRolloutRefWorker(config, role='actor_rollout')。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.__init__。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> create_workers -> ActorRolloutRefWorker(config, role)。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):omega_conf_to_dataclass、DistProfiler。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        Worker.__init__(self)  # 注释:调用Worker基类初始化
        self.config = config  # 注释:保存配置
        self.role = role  # 注释:保存角色名
        self.actor: TrainingWorker = None  # 注释:actor worker(若role包含actor则初始化)
        self.ref: TrainingWorker = None  # 注释:ref worker(若role包含ref则初始化)
        self.rollout: BaseRollout = None  # 注释:rollout实例(若role包含rollout则初始化)
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]  # 注释:校验role合法性
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]  # 注释:是否包含actor
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]  # 注释:是否包含rollout
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]  # 注释:是否包含ref
        # (分隔说明:获取profiler配置)  # 注释:替代空行,保持逐行注释
        if self._is_actor:  # 注释:若包含actor
            omega_profiler_config = config.actor.get("profiler", {})  # 注释:从actor配置获取profiler
        elif self._is_rollout:  # 注释:若包含rollout
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)  # 注释:原注释保留
            # This is for extendability in AsyncRL cases  # 注释:原注释保留
            omega_profiler_config = config.rollout.get("profiler", {})  # 注释:从rollout配置获取profiler
        else:  # 注释:若仅包含ref
            omega_profiler_config = config.ref.get("profiler", {})  # 注释:从ref配置获取profiler
        # (分隔说明:转换profiler配置)  # 注释:替代空行,保持逐行注释
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)  # 注释:转为dataclass
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:  # 注释:若指定tool
            tool_config = omega_conf_to_dataclass(  # 注释:转换tool_config
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))  # 注释:获取对应tool的配置
            )  # 注释:转换结束
        else:  # 注释:若未指定tool
            tool_config = None  # 注释:tool_config为None
        # (分隔说明:初始化profiler)  # 注释:替代空行,保持逐行注释
        DistProfilerExtension.__init__(  # 注释:调用DistProfilerExtension基类初始化
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)  # 注释:创建DistProfiler实例
        )  # 注释:初始化结束

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def set_loss_fn(self, loss_fn):  # 注释:设置actor的loss函数
        """
        功能:设置actor的loss函数。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - loss_fn (Callable):损失函数。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用self.actor.set_loss_fn。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.set_loss_fn(ppo_loss)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.set_loss_fn。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.set_loss_fn。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.actor.set_loss_fn。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        self.actor.set_loss_fn(loss_fn=loss_fn)  # 注释:调用actor的set_loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def to(self, device, model=True, optimizer=True, grad=True):  # 注释:控制actor的加载/卸载
        """Manual control of load/offload  # 注释:保留英文说明

        功能:手动控制actor模型/优化器/梯度的加载/卸载。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - device (str):目标设备。  # 注释:参数含义
        - model (bool):是否迁移模型。  # 注释:参数含义
        - optimizer (bool):是否迁移优化器。  # 注释:参数含义
        - grad (bool):是否迁移梯度。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用self.actor.to。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.to('cpu', model=True)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.to。  # 注释:方法位置
        - 典型调用路径:wake_up -> actor.to('cpu')。  # 注释:典型调用链
        - 被谁调用:wake_up、外部trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.actor.to。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)  # 注释:调用actor的to

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def init_model(self):  # 注释:初始化ref/actor/rollout组件
        """
        功能:根据role初始化ref/actor/rollout组件。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - 无。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 创建self.ref、self.actor、self.rollout实例,设置dispatch_collect信息。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 配置缺失或格式错误时抛异常。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.init_model()。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.init_model。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.init_model()。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):TrainingWorker、get_rollout_class、omega_conf_to_dataclass、ppo_loss。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):init_device_mesh、open_dict。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)  # 注释:转换model配置为dataclass
        # (分隔说明:1. 初始化ref model)  # 注释:替代空行,保持逐行注释
        # 1. build reference model  # 注释:原注释保留
        if "ref" in self.role:  # 注释:若role包含ref
            # TODO: align ref config with actor config  # 注释:原TODO
            with open_dict(self.config.ref):  # 注释:打开ref配置以修改
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size  # 注释:同步mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)  # 注释:重命名配置
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(  # 注释:重命名配置
                    "log_prob_micro_batch_size_per_gpu", None  # 注释:默认值None
                )  # 注释:pop结束
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)  # 注释:重命名配置
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)  # 注释:重命名配置
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)  # 注释:转换ref配置为dataclass
            ref_config.model_config = model_config  # 注释:设置model_config
            # (分隔说明:构造ref TrainingWorkerConfig)  # 注释:替代空行,保持逐行注释
            # construct TrainingWorkerConfig  # 注释:原注释保留
            ref_training_config = TrainingWorkerConfig(  # 注释:创建ref的TrainingWorkerConfig
                model_type="language_model",  # 注释:模型类型
                model_config=ref_config.model_config,  # 注释:模型配置
                engine_config=ref_config.engine,  # 注释:引擎配置
                optimizer_config=ref_config.optim,  # 注释:优化器配置
                checkpoint_config=ref_config.checkpoint,  # 注释:checkpoint配置
            )  # 注释:TrainingWorkerConfig创建结束
            # (分隔说明:设置ref engine配置)  # 注释:替代空行,保持逐行注释
            # assign engine configs  # 注释:原注释保留
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz  # 注释:设置use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu  # 注释:设置推理max_token_len
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (  # 注释:设置推理micro_batch_size
                self.config.ref.ppo_micro_batch_size_per_gpu  # 注释:从配置获取
            )  # 注释:赋值结束
            ref_training_config.engine_config.use_remove_padding = model_config.use_remove_padding  # 注释:设置use_remove_padding
            # (分隔说明:创建ref worker)  # 注释:替代空行,保持逐行注释
            self.ref = TrainingWorker(config=ref_training_config)  # 注释:创建ref worker
            self.ref.reset()  # 注释:初始化ref worker
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())  # 注释:设置ref的dispatch_collect信息
        # (分隔说明:2. 初始化actor model)  # 注释:替代空行,保持逐行注释
        # 2. build actor model  # 注释:原注释保留
        if "actor" in self.role:  # 注释:若role包含actor
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)  # 注释:转换actor配置为dataclass
            actor_config.model_config = model_config  # 注释:设置model_config
            # (分隔说明:构造actor TrainingWorkerConfig)  # 注释:替代空行,保持逐行注释
            actor_training_config = TrainingWorkerConfig(  # 注释:创建actor的TrainingWorkerConfig
                model_type="language_model",  # 注释:模型类型
                model_config=actor_config.model_config,  # 注释:模型配置
                engine_config=actor_config.engine,  # 注释:引擎配置
                optimizer_config=actor_config.optim,  # 注释:优化器配置
                checkpoint_config=actor_config.checkpoint,  # 注释:checkpoint配置
            )  # 注释:TrainingWorkerConfig创建结束
            # (分隔说明:校验配置一致性)  # 注释:替代空行,保持逐行注释
            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz  # 注释:actor与rollout的use_dynamic_bsz必须一致
            # (分隔说明:设置actor engine配置)  # 注释:替代空行,保持逐行注释
            # assign engine configs  # 注释:原注释保留
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz  # 注释:设置use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (  # 注释:设置推理max_token_len
                self.config.rollout.log_prob_max_token_len_per_gpu  # 注释:从rollout配置获取
            )  # 注释:赋值结束
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (  # 注释:设置推理micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu  # 注释:从rollout配置获取
            )  # 注释:赋值结束
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu  # 注释:设置训练max_token_len
            actor_training_config.engine_config.micro_batch_size_per_gpu = (  # 注释:设置训练micro_batch_size
                self.config.actor.ppo_micro_batch_size_per_gpu  # 注释:从actor配置获取
            )  # 注释:赋值结束
            actor_training_config.engine_config.use_remove_padding = model_config.use_remove_padding  # 注释:设置use_remove_padding
            # (分隔说明:校验dynamic_bsz配置)  # 注释:替代空行,保持逐行注释
            if self.config.actor.use_dynamic_bsz:  # 注释:若使用dynamic_bsz
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None  # 注释:校验推理max_token_len已设置
                assert self.config.actor.ppo_max_token_len_per_gpu is not None  # 注释:校验训练max_token_len已设置
            else:  # 注释:若不使用dynamic_bsz
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None  # 注释:校验推理micro_batch_size已设置
                assert self.config.rollout.ppo_micro_batch_size_per_gpu is not None  # 注释:校验训练micro_batch_size已设置(应为actor配置?)
            # (分隔说明:创建actor worker)  # 注释:替代空行,保持逐行注释
            self.loss_fn = partial(ppo_loss, config=actor_config)  # 注释:创建ppo_loss偏函数
            self.actor = TrainingWorker(config=actor_training_config)  # 注释:创建actor worker
            self.actor.reset()  # 注释:初始化actor worker
            self.actor.set_loss_fn(self.loss_fn)  # 注释:设置loss函数
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())  # 注释:设置actor的dispatch_collect信息
        # (分隔说明:3. 初始化rollout engine)  # 注释:替代空行,保持逐行注释
        # 3. build rollout engine  # 注释:原注释保留
        # - vllm: vLLMAsyncRollout  # 注释:原注释保留
        # - sglang: ServerAdapter  # 注释:原注释保留
        if "rollout" in self.role:  # 注释:若role包含rollout
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)  # 注释:转换rollout配置为dataclass
            # (分隔说明:3.1 构建rollout device mesh)  # 注释:替代空行,保持逐行注释
            # 3.1 build rollout device mesh (sglang need only)  # 注释:原注释保留
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size  # 注释:计算推理TP大小(TP*DP)
            infer_pp = rollout_config.pipeline_model_parallel_size  # 注释:推理PP大小
            infer_world_size = infer_tp * infer_pp  # 注释:推理world size
            dp = self.world_size // infer_world_size  # 注释:计算DP size(训练world_size / 推理world_size)
            assert self.world_size % infer_world_size == 0, (  # 注释:校验可整除
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"  # 注释:错误信息
            )  # 注释:assert结束
            rollout_device_mesh = init_device_mesh(  # 注释:初始化rollout device mesh
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]  # 注释:mesh配置
            )  # 注释:init_device_mesh结束
            # (分隔说明:3.2 初始化随机状态)  # 注释:替代空行,保持逐行注释
            # 3.2 init trainer and rollout random states  # 注释:原注释保留
            self.torch_random_states = get_torch_device().get_rng_state()  # 注释:保存训练阶段的随机状态
            gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()  # 注释:获取rollout的DP rank
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states  # 注释:设置rollout随机种子(所有TP rank相同)
            self.gen_random_states = get_torch_device().get_rng_state()  # 注释:保存rollout阶段的随机状态
            get_torch_device().set_rng_state(self.torch_random_states)  # 注释:恢复训练阶段的随机状态
            # (分隔说明:3.3 创建rollout实例)  # 注释:替代空行,保持逐行注释
            # 3.3 initialize rollout engine  # 注释:原注释保留
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)  # 注释:获取rollout类(vllm/sglang)
            self.rollout = rollout_cls(  # 注释:创建rollout实例
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh  # 注释:传入配置
            )  # 注释:rollout创建结束
            # (分隔说明:LoRA相关配置)  # 注释:替代空行,保持逐行注释
            # used for LoRA  # 注释:原注释保留
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format  # 注释:base模型是否已同步(dummy格式不同步)
            self.layered_summon = self.config.rollout.get("layered_summon", False)  # 注释:是否分层summon

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))  # 注释:注册方法,使用ref mesh
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")  # 注释:profiler注解
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:  # 注释:计算ref model的log prob
        """
        功能:使用ref model计算log prob。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - data (TensorDict):输入数据。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:包含log_prob的TensorDict(仅MP source rank返回,其他返回None)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 无(仅推理)。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - output = worker.compute_ref_log_prob(data)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.compute_ref_log_prob。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.compute_ref_log_prob。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.ref.infer_batch。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        output = self.ref.infer_batch(data=data)  # 注释:调用ref的infer_batch
        return output.cpu() if output is not None else None  # 注释:若有输出则移到CPU,否则返回None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))  # 注释:注册方法,使用actor mesh
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")  # 注释:profiler注解
    def compute_log_prob(self, data: TensorDict) -> TensorDict:  # 注释:计算actor model的log prob
        """
        功能:使用actor model计算log prob。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - data (TensorDict):输入数据。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:包含log_prob的TensorDict(仅MP source rank返回,其他返回None)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 无(仅推理)。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - output = worker.compute_log_prob(data)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.compute_log_prob。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.compute_log_prob。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.actor.infer_batch。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        output = self.actor.infer_batch(data)  # 注释:调用actor的infer_batch
        return output.cpu() if output is not None else None  # 注释:若有输出则移到CPU,否则返回None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))  # 注释:注册方法,使用actor mesh
    @DistProfiler.annotate(color="red", role="actor_update")  # 注释:profiler注解
    def update_actor(self, data: TensorDict) -> TensorDict:  # 注释:更新actor参数
        """
        功能:训练actor model,更新参数。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - data (TensorDict):训练数据。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - TensorDict:包含metrics的TensorDict(仅MP source rank返回,其他返回None)。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 更新actor参数。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - output = worker.update_actor(data)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.update_actor。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.update_actor。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.actor.train_mini_batch。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        output = self.actor.train_mini_batch(data=data)  # 注释:调用actor的train_mini_batch
        return output.cpu() if output is not None else None  # 注释:若有输出则移到CPU,否则返回None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):  # 注释:加载actor checkpoint
        """
        功能:加载actor checkpoint。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - local_path (str):本地路径。  # 注释:参数含义
        - hdfs_path (str, optional):HDFS路径。  # 注释:参数含义
        - del_local_after_load (bool):加载后是否删除本地文件。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - actor.load_checkpoint的返回值。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 修改actor参数。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - role不包含actor时assert失败。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.load_checkpoint('./ckpt')。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.load_checkpoint。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.load_checkpoint。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.actor.load_checkpoint。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        assert "actor" in self.role, "load_checkpoint only support actor role"  # 注释:校验role包含actor
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)  # 注释:调用actor的load_checkpoint

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)  # 注释:注册方法,ONE_TO_ALL模式
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):  # 注释:保存actor checkpoint
        """
        功能:保存actor checkpoint。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - local_path (str):本地路径。  # 注释:参数含义
        - hdfs_path (str, optional):HDFS路径。  # 注释:参数含义
        - global_step (int):全局步数。  # 注释:参数含义
        - max_ckpt_to_keep (int, optional):最大保留checkpoint数。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - actor.save_checkpoint的返回值。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 写文件到本地与可选HDFS。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - role不包含actor时assert失败。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - worker.save_checkpoint('./ckpt', global_step=1000)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.save_checkpoint。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.save_checkpoint。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.actor.save_checkpoint。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        assert "actor" in self.role, "save_checkpoint only support actor role"  # 注释:校验role包含actor
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)  # 注释:调用actor的save_checkpoint

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)  # 注释:注册方法,DIRECT_ROLLOUT_METHOD模式
    async def sleep(self):  # 注释:从rollout模式切换到trainer模式
        """Context switch from rollout mode to trainer mode.  # 注释:保留英文说明

        功能:从rollout模式切换到trainer模式,释放rollout资源,恢复训练随机状态。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - 无。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用rollout.release()释放资源,清空GPU缓存,切换随机状态。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - await worker.sleep()。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.sleep。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.sleep()。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.rollout.release、aggressive_empty_cache、set_expandable_segments、log_gpu_memory_usage。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):torch.get_rng_state、torch.set_rng_state。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        if self.config.rollout.free_cache_engine:  # 注释:若配置free_cache_engine
            log_gpu_memory_usage("Before rollout offload", logger=logger)  # 注释:记录内存使用(offload前)
            await self.rollout.release()  # 注释:释放rollout资源(weights+kv_cache)
            log_gpu_memory_usage("After rollout offload", logger=logger)  # 注释:记录内存使用(offload后)
        # (分隔说明:清空GPU缓存)  # 注释:替代空行,保持逐行注释
        # add empty cache after each compute  # 注释:原注释保留
        aggressive_empty_cache(force_sync=True)  # 注释:激进清空GPU缓存
        set_expandable_segments(True)  # 注释:启用可扩展段(训练模式)
        # (分隔说明:切换随机状态)  # 注释:替代空行,保持逐行注释
        # restore random states  # 注释:原注释保留
        self.gen_random_states = get_torch_device().get_rng_state()  # 注释:保存rollout随机状态
        get_torch_device().set_rng_state(self.torch_random_states)  # 注释:恢复训练随机状态

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)  # 注释:注册方法,DIRECT_ROLLOUT_METHOD模式
    async def wake_up(self):  # 注释:从trainer模式切换到rollout模式
        """Context switch trainer mode to rollout mode.  # 注释:保留英文说明

        功能:从trainer模式切换到rollout模式,同步actor权重到rollout,offload actor到CPU。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - 无。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - None。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 恢复rollout资源,同步权重,offload actor,切换随机状态。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - await worker.wake_up()。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.wake_up。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.wake_up()。  # 注释:典型调用链
        - 被谁调用:ray_trainer。  # 注释:调用方说明
        - 调用了谁(项目内):self.rollout.resume/update_weights、self.actor.engine.get_per_tensor_param/to、aggressive_empty_cache、log_gpu_memory_usage。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):torch.get_rng_state、torch.set_rng_state。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        aggressive_empty_cache(force_sync=True)  # 注释:清空GPU缓存
        set_expandable_segments(False)  # 注释:禁用可扩展段(推理模式)
        # (分隔说明:1. 获取actor参数)  # 注释:替代空行,保持逐行注释
        # 1. get per tensor generator from engine, this will load model to gpu  # 注释:原注释保留
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param()  # 注释:从actor engine获取参数(会加载模型到GPU)
        # (分隔说明:2. 恢复rollout weights并同步)  # 注释:替代空行,保持逐行注释
        # 2. resume weights and update weights  # 注释:原注释保留
        if self.config.rollout.free_cache_engine:  # 注释:若配置free_cache_engine
            await self.rollout.resume(tags=["weights"])  # 注释:恢复rollout weights到GPU
        log_gpu_memory_usage("After resume weights", logger=logger)  # 注释:记录内存使用
        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=self.base_sync_done)  # 注释:同步actor权重到rollout
        log_gpu_memory_usage("After update_weights", logger=logger)  # 注释:记录内存使用
        # (分隔说明:3. offload actor到CPU)  # 注释:替代空行,保持逐行注释
        # 3. offload model to cpu  # 注释:原注释保留
        self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)  # 注释:将actor模型offload到CPU(保留optimizer与grad在GPU)
        aggressive_empty_cache(force_sync=True)  # 注释:清空GPU缓存
        # (分隔说明:4. 恢复rollout kv_cache)  # 注释:替代空行,保持逐行注释
        # 4. resume kv_cache  # 注释:原注释保留
        if self.config.rollout.free_cache_engine:  # 注释:若配置free_cache_engine
            await self.rollout.resume(tags=["kv_cache"])  # 注释:恢复rollout kv_cache到GPU
        log_gpu_memory_usage("After resume kv_cache", logger=logger)  # 注释:记录内存使用
        # (分隔说明:标记base模型已同步)  # 注释:替代空行,保持逐行注释
        self.base_sync_done = True  # 注释:标记base模型已同步(LoRA场景)
        # important: need to manually set the random states of each tp to be identical.  # 注释:原注释保留
        self.torch_random_states = get_torch_device().get_rng_state()  # 注释:保存训练随机状态
        get_torch_device().set_rng_state(self.gen_random_states)  # 注释:恢复rollout随机状态

    # ============================ vLLM related ============================  # 注释:原注释保留
    # (分隔说明:vLLM相关方法)  # 注释:替代空行,保持逐行注释
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)  # 注释:注册方法,DIRECT_ROLLOUT_METHOD模式
    def get_zeromq_address(self):  # 注释:获取vLLM的ZeroMQ地址
        """
        功能:获取vLLM rollout的ZeroMQ地址。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - 无。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - str:ZeroMQ地址。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 无。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - addr = worker.get_zeromq_address()。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.get_zeromq_address。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.get_zeromq_address()。  # 注释:典型调用链
        - 被谁调用:ray_trainer(vLLM rollout场景)。  # 注释:调用方说明
        - 调用了谁(项目内):self.rollout.get_zeromq_address。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        return self.rollout.get_zeromq_address()  # 注释:调用rollout的get_zeromq_address

    # ============================ SGLang related ============================  # 注释:原注释保留
    # (分隔说明:SGLang相关方法)  # 注释:替代空行,保持逐行注释
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)  # 注释:注册方法,非阻塞
    async def chat_completion(self, json_request):  # 注释:调用SGLang的chat_completion接口
        """
        功能:调用SGLang rollout的chat_completion接口。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - json_request (dict):chat completion请求。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - dict:chat completion响应。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用rollout生成响应。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - ret = await worker.chat_completion(json_request)。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.chat_completion。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.chat_completion(json_request)。  # 注释:典型调用链
        - 被谁调用:ray_trainer(SGLang rollout场景)。  # 注释:调用方说明
        - 调用了谁(项目内):self.rollout.chat_completion。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        ret = await self.rollout.chat_completion(json_request)  # 注释:调用rollout的chat_completion
        return ret  # 注释:返回响应

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)  # 注释:注册方法,非阻塞
    async def generate(  # 注释:调用SGLang的generate接口
        self,  # 注释:self参数
        prompt_ids: list[int],  # 注释:prompt token ids
        sampling_params: dict[str, Any],  # 注释:采样参数
        request_id: str,  # 注释:请求ID
        image_data: Optional[list[Any]] = None,  # 注释:图像数据(多模态场景)
    ) -> list[int]:  # 注释:返回生成的token ids
        """
        功能:调用SGLang rollout的generate接口生成token。  # 注释:方法用途
        参数:  # 注释:参数说明标题
        - prompt_ids (list[int]):prompt token ids。  # 注释:参数含义
        - sampling_params (dict):采样参数(temperature、top_p等)。  # 注释:参数含义
        - request_id (str):请求ID。  # 注释:参数含义
        - image_data (list, optional):图像数据。  # 注释:参数含义
        返回:  # 注释:返回值说明标题
        - list[int]:生成的token ids。  # 注释:返回值语义
        副作用:  # 注释:副作用说明标题
        - 调用rollout生成token。  # 注释:副作用说明
        异常/边界条件:  # 注释:异常说明标题
        - 无。  # 注释:异常说明
        最小示例:  # 注释:最小示例标题
        - ret = await worker.generate([1,2,3], {'temperature':0.7}, 'req_001')。  # 注释:示例
        调用路径依赖:  # 注释:调用路径说明标题
        - 所在位置:verl/workers/engine_workers.py::ActorRolloutRefWorker.generate。  # 注释:方法位置
        - 典型调用路径:ray_trainer -> worker_group.generate(...)。  # 注释:典型调用链
        - 被谁调用:ray_trainer(SGLang rollout场景)。  # 注释:调用方说明
        - 调用了谁(项目内):self.rollout.generate。  # 注释:项目内依赖说明
        - 调用了谁(关键外部依赖):无。  # 注释:外部依赖说明
        """  # 注释:方法 docstring 结束
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)  # 注释:调用rollout的generate
        return ret  # 注释:返回生成结果
