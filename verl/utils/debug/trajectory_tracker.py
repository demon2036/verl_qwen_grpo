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
模块用途：提供轨迹（中间结果）追踪与保存工具，便于离线比对与调试。  # 注释：模块用途
输入/输出：输入为任意可序列化数据与名称，输出为保存到 HDFS 的 .pth 文件。  # 注释：模块输入输出概览
关键依赖：ray、torch、io、tempfile、verl.utils.hdfs_io。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- os.environ["VERL_ENABLE_TRACKER"]="1" 后调用 dump_data(data, name)。  # 注释：开启并写入轨迹
- tracker = get_trajectory_tracker(); ray.get(tracker.dump.remote(...))。  # 注释：手动调用
调用路径概览：  # 注释：调用路径说明标题
- 调试/测试代码 -> dump_data -> TrajectoryTracker.dump -> save_to_hdfs。  # 注释：典型链路
"""  # 注释：模块 docstring 结束
# （分隔说明：导入标准库）  # 注释：替代空行，保持逐行注释
import io  # 注释：内存缓冲区
import os  # 注释：环境变量读取
import tempfile  # 注释：临时目录创建
from collections import deque  # 注释：队列存储异步句柄
# （分隔说明：导入第三方依赖）  # 注释：替代空行，保持逐行注释
import ray  # 注释：Ray 分布式与远程函数
import torch  # 注释：张量序列化
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl.utils.hdfs_io import copy, makedirs  # 注释：HDFS 复制与创建目录
# （分隔说明：远程函数引用）  # 注释：替代空行，保持逐行注释
remote_copy = ray.remote(copy)  # 注释：预留 remote 版本（当前未使用）
# （分隔说明：保存到 HDFS 的远程函数）  # 注释：替代空行，保持逐行注释

@ray.remote  # 注释：Ray 远程函数装饰器
def save_to_hdfs(data: io.BytesIO, name, hdfs_dir, verbose):  # 注释：保存内存缓冲区到 HDFS
    """
    功能：将 BytesIO 数据写入临时文件并上传到 HDFS。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data (io.BytesIO)：待保存的二进制缓冲区。  # 注释：参数含义
    - name (str)：文件名基准（不含扩展名）。  # 注释：参数含义
    - hdfs_dir (str)：目标 HDFS 目录。  # 注释：参数含义
    - verbose (bool)：是否打印上传日志。  # 注释：参数含义
    返回：无。  # 注释：返回值说明标题
    副作用：在 HDFS 写入 .pth 文件。  # 注释：副作用说明
    异常/边界条件：copy 失败会打印异常但不抛出。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - save_to_hdfs.remote(buffer, "step1", "hdfs://path", True)。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/debug/trajectory_tracker.py::save_to_hdfs。  # 注释：函数位置
    - 典型调用路径：TrajectoryTracker.dump -> save_to_hdfs。  # 注释：典型调用链
    - 被谁调用：TrajectoryTracker.dump。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.hdfs_io.copy。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：tempfile.TemporaryDirectory、open。  # 注释：外部依赖说明
    """
    filename = name + ".pth"  # 注释：拼接保存文件名
    with tempfile.TemporaryDirectory() as tmpdirname:  # 注释：创建临时目录
        local_filepath = os.path.join(tmpdirname, filename)  # 注释：本地临时文件路径
        with open(local_filepath, "wb") as f:  # 注释：打开本地文件写入
            f.write(data.getbuffer())  # 注释：写入缓冲区内容
        # upload to hdfs  # 注释：原注释保留：上传到 HDFS

        if verbose:  # 注释：可选打印日志
            print(f"Saving {local_filepath} to {hdfs_dir}")  # 注释：打印上传路径
        try:  # 注释：捕获上传异常
            copy(local_filepath, hdfs_dir)  # 注释：执行 HDFS 复制
        except Exception as e:  # 注释：捕获异常并打印
            print(e)  # 注释：输出异常信息
# （分隔说明：轨迹追踪器 Actor）  # 注释：替代空行，保持逐行注释

@ray.remote  # 注释：Ray 远程 Actor 装饰器
class TrajectoryTracker:  # 注释：轨迹追踪器 Actor
    """
    功能：作为 Ray Actor 管理轨迹保存请求并可等待完成。  # 注释：类用途
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/debug/trajectory_tracker.py::TrajectoryTracker。  # 注释：类位置
    - 典型调用路径：dump_data -> get_trajectory_tracker -> TrajectoryTracker.dump。  # 注释：典型调用链
    - 被谁调用：dump_data、get_trajectory_tracker。  # 注释：调用方说明
    - 调用了谁（项目内）：verl.utils.hdfs_io.makedirs。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：ray Actor 机制。  # 注释：外部依赖说明
    """

    def __init__(self, hdfs_dir, verbose) -> None:  # 注释：初始化 Actor
        """
        功能：初始化 HDFS 目录并创建句柄队列。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - hdfs_dir (str)：保存目录。  # 注释：参数含义
        - verbose (bool)：是否打印日志。  # 注释：参数含义
        返回：无。  # 注释：返回值说明标题
        副作用：创建/保证 HDFS 目录存在。  # 注释：副作用说明
        异常/边界条件：makedirs 失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - TrajectoryTracker.remote("hdfs://dir", True)。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：TrajectoryTracker.__init__。  # 注释：函数位置
        - 典型调用路径：get_trajectory_tracker -> TrajectoryTracker.remote。  # 注释：典型调用链
        - 被谁调用：get_trajectory_tracker。  # 注释：调用方说明
        - 调用了谁（项目内）：verl.utils.hdfs_io.makedirs。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
        """
        self.hdfs_dir = hdfs_dir  # 注释：保存目录
        makedirs(hdfs_dir)  # 注释：确保目录存在
        self.verbose = verbose  # 注释：保存日志开关

        self.handle = deque()  # 注释：保存异步任务句柄队列

    def dump(self, data: io.BytesIO, name):  # 注释：提交保存任务
        """
        功能：将保存任务提交给 save_to_hdfs，并记录异步句柄。  # 注释：函数用途
        参数：  # 注释：参数说明标题
        - data (io.BytesIO)：待保存缓冲区。  # 注释：参数含义
        - name (str)：文件名基准。  # 注释：参数含义
        返回：无。  # 注释：返回值说明标题
        副作用：向队列追加 Ray 任务句柄。  # 注释：副作用说明
        异常/边界条件：Ray 提交失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - tracker.dump.remote(buffer, "step1")。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：TrajectoryTracker.dump。  # 注释：函数位置
        - 典型调用路径：dump_data -> tracker.dump。  # 注释：典型调用链
        - 被谁调用：dump_data。  # 注释：调用方说明
        - 调用了谁（项目内）：save_to_hdfs（同文件）。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：ray.get（由等待方调用）。  # 注释：外部依赖说明
        """
        # get a temp file and write to it  # 注释：原注释保留：提交保存任务
        self.handle.append(save_to_hdfs.remote(data, name, self.hdfs_dir, self.verbose))  # 注释：记录任务句柄

    def wait_for_hdfs(self):  # 注释：等待所有保存任务完成
        """
        功能：依次等待队列中的保存任务完成。  # 注释：函数用途
        参数：无。  # 注释：参数说明标题
        返回：无。  # 注释：返回值说明标题
        副作用：清空任务队列。  # 注释：副作用说明
        异常/边界条件：ray.get 失败会抛异常。  # 注释：异常说明
        最小示例：  # 注释：最小示例标题
        - ray.get(tracker.wait_for_hdfs.remote())。  # 注释：示例
        调用路径依赖：  # 注释：调用路径说明标题
        - 所在位置：TrajectoryTracker.wait_for_hdfs。  # 注释：函数位置
        - 典型调用路径：测试/调试 -> wait_for_hdfs。  # 注释：典型调用链
        - 被谁调用：__main__ 测试示例。  # 注释：调用方说明
        - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
        - 调用了谁（关键外部依赖）：ray.get。  # 注释：外部依赖说明
        """
        while len(self.handle) != 0:  # 注释：循环直到队列为空
            future = self.handle.popleft()  # 注释：取出最早任务
            ray.get(future)  # 注释：等待任务完成
# （分隔说明：便捷 API：直接 dump）  # 注释：替代空行，保持逐行注释

def dump_data(data, name):  # 注释：根据环境变量控制是否保存数据
    """
    功能：根据环境变量开关，将数据序列化并提交保存。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data：任意可 torch.save 的对象。  # 注释：参数含义
    - name (str)：文件名基准。  # 注释：参数含义
    返回：无。  # 注释：返回值说明标题
    副作用：可能向 HDFS 写入文件。  # 注释：副作用说明
    异常/边界条件：未设置 HDFS 目录会在 get_trajectory_tracker 处断言失败。  # 注释：边界说明
    最小示例：  # 注释：最小示例标题
    - dump_data({"x": torch.tensor(1)}, "sample")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/debug/trajectory_tracker.py::dump_data。  # 注释：函数位置
    - 典型调用路径：调试代码 -> dump_data -> tracker.dump。  # 注释：典型调用链
    - 被谁调用：外部调试/测试代码。  # 注释：调用方说明
    - 调用了谁（项目内）：get_trajectory_tracker。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：torch.save、ray.get。  # 注释：外部依赖说明
    """
    enable = os.getenv("VERL_ENABLE_TRACKER", "0") == "1"  # 注释：读取是否开启追踪
    if not enable:  # 注释：未开启则直接返回
        return  # 注释：跳过保存
    buffer = io.BytesIO()  # 注释：创建内存缓冲区
    torch.save(data, buffer)  # 注释：序列化写入缓冲区
    tracker = get_trajectory_tracker()  # 注释：获取全局追踪器
    ray.get(tracker.dump.remote(buffer, name))  # 注释：提交保存并等待完成
# （分隔说明：获取/创建全局追踪器）  # 注释：替代空行，保持逐行注释

def get_trajectory_tracker():  # 注释：获取全局 TrajectoryTracker Actor
    """
    功能：获取（或创建）全局 TrajectoryTracker Actor。  # 注释：函数用途
    参数：无。  # 注释：参数说明标题
    返回：  # 注释：返回值说明标题
    - TrajectoryTracker ActorHandle：可用于 dump 与 wait。  # 注释：返回值语义
    副作用：若不存在则创建一个 detached Actor。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - VERL_TRACKER_HDFS_DIR 未设置会触发断言。  # 注释：边界条件
    最小示例：  # 注释：最小示例标题
    - tracker = get_trajectory_tracker()  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/debug/trajectory_tracker.py::get_trajectory_tracker。  # 注释：函数位置
    - 典型调用路径：dump_data -> get_trajectory_tracker。  # 注释：典型调用链
    - 被谁调用：dump_data、测试脚本。  # 注释：调用方说明
    - 调用了谁（项目内）：TrajectoryTracker（本文件）。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：ray Actor API。  # 注释：外部依赖说明
    """
    hdfs_dir = os.getenv("VERL_TRACKER_HDFS_DIR", default=None)  # 注释：读取 HDFS 目录
    verbose = os.getenv("VERL_TRACKER_VERBOSE", default="0") == "1"  # 注释：读取 verbose 开关
    assert hdfs_dir is not None  # 注释：必须配置 HDFS 目录
    tracker = TrajectoryTracker.options(name="global_tracker", get_if_exists=True, lifetime="detached").remote(  # 注释：获取或创建全局 Actor
        hdfs_dir, verbose  # 注释：传入目录与日志开关
    )  # 注释：Actor 句柄创建结束
    return tracker  # 注释：返回 Actor 句柄
# （分隔说明：自测入口）  # 注释：替代空行，保持逐行注释

if __name__ == "__main__":  # 注释：脚本自测入口
    # testing  # 注释：原注释保留
    os.environ["VERL_ENABLE_TRACKER"] = "1"  # 注释：开启追踪
    os.environ["VERL_TRACKER_HDFS_DIR"] = "~/debug/test"  # 注释：设置 HDFS 目录（示例）

    @ray.remote  # 注释：示例远程函数
    def process(iter):  # 注释：示例：生成并保存数据
        data = {"obs": torch.randn(10, 20)}  # 注释：构造随机张量
        dump_data(data, f"process_{iter}_obs")  # 注释：保存轨迹数据

    ray.init()  # 注释：初始化 Ray 运行时

    output_lst = []  # 注释：保存任务列表

    for i in range(10):  # 注释：提交 10 个任务
        output_lst.append(process.remote(i))  # 注释：提交远程任务

    out = ray.get(output_lst)  # 注释：等待所有任务完成

    tracker = get_trajectory_tracker()  # 注释：获取追踪器
    ray.get(tracker.wait_for_hdfs.remote())  # 注释：等待 HDFS 写入完成
