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
模块用途：提供 Ray 相关的轻量工具函数（可见设备开关、并行 put、事件循环获取）。  # 注释：模块用途
输入/输出：输入为环境变量/数据列表/事件循环需求，输出布尔值、ObjectRef 列表或事件循环对象。  # 注释：模块输入输出概览
关键依赖：ray、asyncio、concurrent.futures、os.environ。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- ray_noset_visible_devices() -> True/False  # 注释：检测 Ray 是否禁用可见设备设置
- refs = parallel_put([obj1, obj2])  # 注释：并行写入 Ray 对象存储
- loop = get_event_loop()  # 注释：获取/创建 asyncio 事件循环
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/fsdp_workers.py、verl/single_controller/base/worker.py。  # 注释：上层入口举例
- 典型链路：worker/ray_trainer -> ray_utils.* -> ray/asyncio 标准库。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：导入标准库）  # 注释：替代空行，保持逐行注释
import asyncio  # 注释：事件循环工具
import concurrent.futures  # 注释：线程池并行
import os  # 注释：环境变量读取
from typing import Any, Optional  # 注释：类型标注
# （分隔说明：导入第三方依赖）  # 注释：替代空行，保持逐行注释
import ray  # 注释：Ray 运行时与对象存储
# （分隔说明：Ray 可见设备开关检查）  # 注释：替代空行，保持逐行注释

def ray_noset_visible_devices(env_vars=os.environ):  # 注释：检查 Ray 是否设置了不注入可见设备的开关
    """
    功能：检查 Ray 是否启用了“不自动设置可见设备环境变量”的实验开关。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - env_vars：环境变量映射（默认 os.environ）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：任意一个 NOSET_* 环境变量存在则为 True。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]="1" -> True。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/ray_utils.py::ray_noset_visible_devices。  # 注释：函数位置
    - 典型调用路径：worker 初始化 -> ray_noset_visible_devices。  # 注释：典型调用链
    - 被谁调用：verl/single_controller/base/worker.py、verl/workers/rollout/vllm_rollout/vllm_rollout.py。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.environ.get。  # 注释：外部依赖说明
    """
    # Refer to  # 注释：原注释保留，说明来源
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96  # 注释：Ray GPU 环境变量参考
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103  # 注释：Ray AMD GPU 环境变量参考
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115  # 注释：Ray AMD GPU 环境变量参考
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95  # 注释：Ray NPU 环境变量参考
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117  # 注释：Ray HPU 环境变量参考
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109  # 注释：Ray Neuron 环境变量参考
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172  # 注释：Ray TPU 环境变量参考
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98  # 注释：Ray Intel GPU 环境变量参考
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [  # 注释：Ray 用于禁用自动设置的环境变量列表
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",  # 注释：CUDA 可见设备开关
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",  # 注释：ROCR 可见设备开关
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",  # 注释：HIP 可见设备开关
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",  # 注释：Ascend 可见设备开关
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",  # 注释：Habana 可见模块开关
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",  # 注释：Neuron 可见核心开关
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",  # 注释：TPU 可见芯片开关
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",  # 注释：Intel OneAPI 设备选择开关
    ]  # 注释：列表定义结束
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)  # 注释：任意开关存在即返回 True
# （分隔说明：并行 put 到 Ray 对象存储）  # 注释：替代空行，保持逐行注释

def parallel_put(data_list: list[Any], max_workers: Optional[int] = None):  # 注释：并行写入 Ray 对象存储
    """
    功能：用线程池并行调用 ray.put，将对象列表写入对象存储并保持顺序。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data_list (list[Any])：需要 put 的对象列表（不能为空）。  # 注释：参数含义
    - max_workers (Optional[int])：线程池大小；默认 min(len(data_list), 16)。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - list[ray.ObjectRef]：与 data_list 顺序一致的 ObjectRef 列表。  # 注释：返回值语义
    副作用：将对象写入 Ray 对象存储，占用对象存储内存。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - data_list 为空触发 AssertionError。  # 注释：边界条件
    - ray 未初始化或对象过大可能抛异常。  # 注释：外部依赖异常
    最小示例：  # 注释：最小示例标题
    - refs = parallel_put([{"x": 1}, {"y": 2}])。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/ray_utils.py::parallel_put。  # 注释：函数位置
    - 典型调用路径：Ray 调度 -> worker 装饰器 -> parallel_put。  # 注释：典型调用链
    - 被谁调用：verl/single_controller/base/decorator.py、tests/single_controller/test_ray_utils_on_cpu.py。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：ray.put、concurrent.futures.ThreadPoolExecutor。  # 注释：外部依赖说明
    """
    assert len(data_list) > 0, "data_list must not be empty"  # 注释：空列表不允许，避免线程池无任务

    def put_data(index, data):  # 注释：内部函数：写入单个对象并返回索引
        return index, ray.put(data)  # 注释：返回 (索引, ObjectRef)

    if max_workers is None:  # 注释：未指定线程数时使用默认策略
        max_workers = min(len(data_list), 16)  # 注释：上限 16，避免过多线程

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 注释：创建线程池
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]  # 注释：提交任务
        res_lst = []  # 注释：收集结果列表
        for future in concurrent.futures.as_completed(data_list_f):  # 注释：按完成顺序迭代
            res_lst.append(future.result())  # 注释：收集 (index, ObjectRef)

        # reorder based on index  # 注释：原注释保留：按索引恢复顺序
        output = [None for _ in range(len(data_list))]  # 注释：初始化输出列表
        for res in res_lst:  # 注释：遍历结果
            index, data_ref = res  # 注释：解包索引与引用
            output[index] = data_ref  # 注释：按原顺序放回

    return output  # 注释：返回有序 ObjectRef 列表
# （分隔说明：事件循环获取）  # 注释：替代空行，保持逐行注释

def get_event_loop():  # 注释：获取或创建当前线程事件循环
    """
    功能：获取当前线程的 asyncio 事件循环，若不存在则创建并设置。  # 注释：函数用途
    参数：无。  # 注释：参数说明标题
    返回：  # 注释：返回值说明标题
    - asyncio.AbstractEventLoop：当前线程可用的事件循环对象。  # 注释：返回值语义
    副作用：必要时创建新事件循环并设置为当前线程默认。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - RuntimeError 表示当前线程未绑定事件循环。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - loop = get_event_loop(); loop.run_until_complete(coro)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/ray_utils.py::get_event_loop。  # 注释：函数位置
    - 典型调用路径：worker/rollout -> get_event_loop -> asyncio。  # 注释：典型调用链
    - 被谁调用：verl/workers/fsdp_workers.py、verl/workers/rollout/vllm_rollout/vllm_rollout.py、recipe/*/workers.py。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：asyncio.get_event_loop/new_event_loop/set_event_loop。  # 注释：外部依赖说明
    """
    try:  # 注释：尝试获取当前线程事件循环
        loop = asyncio.get_event_loop()  # 注释：获取事件循环
    except RuntimeError:  # 注释：当前线程没有事件循环
        loop = asyncio.new_event_loop()  # 注释：创建新的事件循环
        asyncio.set_event_loop(loop)  # 注释：注册为当前线程事件循环

    return loop  # 注释：返回事件循环
