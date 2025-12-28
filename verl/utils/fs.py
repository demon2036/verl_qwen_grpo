#!/usr/bin/env python  # 注释：脚本解释器声明，便于直接执行
# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
# （分隔说明：许可证段落分隔）  # 注释：用注释行替代空行
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：许可证链接提示
#  # 注释：保留注释符号，保证该行有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证地址
#  # 注释：保留注释符号，保证该行有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
# （分隔说明：编码声明）  # 注释：替代空行，保持逐行注释
# -*- coding: utf-8 -*-  # 注释：声明 UTF-8 编码，便于中文注释
"""模块用途：提供与文件系统无关的 IO 工具（本地/HDFS/共享内存缓存）。  # 注释：模块用途
输入/输出：主要处理路径字符串，返回本地路径或布尔校验结果。  # 注释：输入输出概览
关键依赖：os、shutil、tempfile、hashlib、filelock、verl.utils.hdfs_io。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- from verl.utils.fs import copy_to_local  # 注释：导入函数
- local_path = copy_to_local("hdfs://cluster/data/train.parquet", cache_dir="/tmp/cache")  # 注释：拷贝 HDFS 文件到本地缓存
调用路径概览：  # 注释：调用路径说明标题
- 数据集加载：verl/utils/dataset/sft_dataset.py -> copy_to_local -> copy_local_path_from_hdfs -> hdfs_io.copy。  # 注释：典型调用链
- 训练/worker/评测模块也会调用 copy_to_local 与 local_mkdir_safe。  # 注释：其他调用场景
"""  # 注释：模块 docstring 结束
# （分隔说明：导入依赖）  # 注释：替代空行，保持逐行注释
import hashlib  # 注释：用于生成路径哈希
import os  # 注释：本地路径与系统接口
import shutil  # 注释：本地复制工具
import tempfile  # 注释：系统临时目录
# （分隔说明：兼容直接运行与包内导入的 hdfs_io）  # 注释：替代空行，保持逐行注释
try:  # 注释：优先尝试相对 import 失败时的直接导入
    from hdfs_io import copy, exists, makedirs  # for internal use only  # 注释：直接导入（兼容脚本运行）
except ImportError:  # 注释：若直接导入失败
    from .hdfs_io import copy, exists, makedirs  # 注释：包内相对导入
# （分隔说明：对外导出符号）  # 注释：替代空行，保持逐行注释
__all__ = ["copy", "exists", "makedirs"]  # 注释：对外公开的函数列表
# （分隔说明：常量定义）  # 注释：替代空行，保持逐行注释
_HDFS_PREFIX = "hdfs://"  # 注释：HDFS 路径前缀
# （分隔说明：路径判定函数）  # 注释：替代空行，保持逐行注释
def is_non_local(path):  # 注释：判断路径是否为 HDFS 路径
    """函数用途：判断路径是否为 HDFS 前缀路径。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - path (str)：待检查路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：True 表示 HDFS 路径，否则 False。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - is_non_local("hdfs://cluster/foo") -> True。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::is_non_local。  # 注释：函数位置
    - 典型调用路径：checkpoint/dataset 模块 -> is_non_local。  # 注释：典型调用链
    - 被谁调用：verl/utils/dataset/rm_dataset.py、verl/utils/checkpoint/*、fs.py 内部逻辑。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：str.startswith。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    return path.startswith(_HDFS_PREFIX)  # 注释：前缀匹配判断
# （分隔说明：MD5 编码函数）  # 注释：替代空行，保持逐行注释
def md5_encode(path: str) -> str:  # 注释：对路径字符串做 MD5 编码
    """函数用途：生成路径字符串的 MD5 哈希，用于缓存目录或锁文件命名。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - path (str)：待编码的路径字符串。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：32 位十六进制 MD5 字符串。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - md5_encode("/tmp/a") -> "e4f..."。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::md5_encode。  # 注释：函数位置
    - 典型调用路径：get_local_temp_path/copy_local_path_from_hdfs -> md5_encode。  # 注释：典型调用链
    - 被谁调用：仅在本文件内部函数调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：hashlib.md5。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    return hashlib.md5(path.encode()).hexdigest()  # 注释：计算并返回 MD5 哈希
# （分隔说明：HDFS 临时路径生成函数）  # 注释：替代空行，保持逐行注释
def get_local_temp_path(hdfs_path: str, cache_dir: str) -> str:  # 注释：为 HDFS 资源生成本地缓存路径
    """函数用途：为 HDFS 资源生成唯一的本地缓存路径。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - hdfs_path (str)：HDFS 源路径。  # 注释：参数含义
    - cache_dir (str)：本地缓存根目录。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：形如 {cache_dir}/{md5(hdfs_path)}/{basename(hdfs_path)} 的绝对路径。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 会创建 md5 子目录（os.makedirs）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - cache_dir 不可写时会触发 OSError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - get_local_temp_path("hdfs://a/b.txt", "/tmp/cache") -> "/tmp/cache/<md5>/b.txt"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::get_local_temp_path。  # 注释：函数位置
    - 典型调用路径：copy_local_path_from_hdfs -> get_local_temp_path。  # 注释：典型调用链
    - 被谁调用：仅在本文件内部调用。  # 注释：调用方说明
    - 调用了谁（项目内）：md5_encode。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：os.path.join、os.makedirs、os.path.basename。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    # make a base64 encoding of hdfs_path to avoid directory conflict  # 注释：原注释保留（实际使用 MD5）
    encoded_hdfs_path = md5_encode(hdfs_path)  # 注释：用 MD5 避免目录冲突
    temp_dir = os.path.join(cache_dir, encoded_hdfs_path)  # 注释：拼接缓存子目录路径
    os.makedirs(temp_dir, exist_ok=True)  # 注释：创建缓存子目录
    dst = os.path.join(temp_dir, os.path.basename(hdfs_path))  # 注释：拼接最终文件路径
    return dst  # 注释：返回本地缓存路径
# （分隔说明：复制校验函数）  # 注释：替代空行，保持逐行注释
def verify_copy(src: str, dest: str) -> bool:  # 注释：比较源与目标的大小和结构，验证拷贝
    """函数用途：通过文件大小与目录结构验证拷贝结果。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - src (str)：源路径（文件或目录）。  # 注释：参数含义
    - dest (str)：目标路径（文件或目录）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：验证通过返回 True，否则 False。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 读取文件系统元数据，可能较慢。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 权限不足或路径损坏时可能抛出 OSError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - verify_copy("/tmp/a", "/tmp/b") -> True/False。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::verify_copy。  # 注释：函数位置
    - 典型调用路径：copy_to_shm -> verify_copy。  # 注释：典型调用链
    - 被谁调用：仅在本文件 copy_to_shm 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：os.path、os.walk、os.listdir。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    if not os.path.exists(src):  # 注释：源路径不存在则直接失败
        return False  # 注释：返回 False
    if not os.path.exists(dest):  # 注释：目标路径不存在则失败
        return False  # 注释：返回 False
    if os.path.isfile(src) != os.path.isfile(dest):  # 注释：类型不一致（文件/目录）则失败
        return False  # 注释：返回 False
    if os.path.isfile(src):  # 注释：文件路径分支
        src_size = os.path.getsize(src)  # 注释：获取源文件大小
        dest_size = os.path.getsize(dest)  # 注释：获取目标文件大小
        if src_size != dest_size:  # 注释：大小不一致
            return False  # 注释：返回 False
        return True  # 注释：文件大小一致则通过
    src_files = set()  # 注释：记录源目录下所有条目
    dest_files = set()  # 注释：记录目标目录下所有条目
    for root, dirs, files in os.walk(src):  # 注释：遍历源目录树
        rel_path = os.path.relpath(root, src)  # 注释：计算相对路径
        dest_root = os.path.join(dest, rel_path) if rel_path != "." else dest  # 注释：对应目标根路径
        if not os.path.exists(dest_root):  # 注释：目标对应目录不存在
            return False  # 注释：返回 False
        for entry in os.listdir(root):  # 注释：遍历源目录当前层条目
            src_entry = os.path.join(root, entry)  # 注释：源条目绝对路径
            src_files.add(os.path.relpath(src_entry, src))  # 注释：记录源相对路径
        for entry in os.listdir(dest_root):  # 注释：遍历目标目录当前层条目
            dest_entry = os.path.join(dest_root, entry)  # 注释：目标条目绝对路径
            dest_files.add(os.path.relpath(dest_entry, dest))  # 注释：记录目标相对路径
    if src_files != dest_files:  # 注释：源与目标条目集合不一致
        return False  # 注释：返回 False
    for rel_path in src_files:  # 注释：逐一校验每个条目
        src_entry = os.path.join(src, rel_path)  # 注释：源条目绝对路径
        dest_entry = os.path.join(dest, rel_path)  # 注释：目标条目绝对路径
        if os.path.isdir(src_entry) != os.path.isdir(dest_entry):  # 注释：目录/文件类型不一致
            return False  # 注释：返回 False
        if os.path.isfile(src_entry):  # 注释：若为文件则比对大小
            src_size = os.path.getsize(src_entry)  # 注释：源文件大小
            dest_size = os.path.getsize(dest_entry)  # 注释：目标文件大小
            if src_size != dest_size:  # 注释：大小不一致
                return False  # 注释：返回 False
    return True  # 注释：所有检查通过
# （分隔说明：共享内存拷贝函数）  # 注释：替代空行，保持逐行注释
def copy_to_shm(src: str):  # 注释：将模型复制到 /dev/shm 缓存
    """函数用途：将模型或文件复制到 /dev/shm 提升多次加载效率。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - src (str)：源路径（文件或目录）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：/dev/shm 下的目标路径。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 创建 /dev/shm/verl-cache 目录并复制文件。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - /dev/shm 不可写或空间不足时可能抛出异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - copy_to_shm("/models/qwen") -> "/dev/shm/verl-cache/<md5>/qwen"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::copy_to_shm。  # 注释：函数位置
    - 典型调用路径：copy_to_local(use_shm=True) -> copy_to_shm。  # 注释：典型调用链
    - 被谁调用：仅在本文件 copy_to_local 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：verify_copy。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：os.path、hashlib.md5、shutil.copytree/copy2。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    shm_model_root = "/dev/shm/verl-cache/"  # 注释：共享内存缓存根目录
    src_abs = os.path.abspath(os.path.normpath(src))  # 注释：规范化并获取源绝对路径
    dest = os.path.join(shm_model_root, hashlib.md5(src_abs.encode("utf-8")).hexdigest())  # 注释：用 MD5 生成唯一子目录
    os.makedirs(dest, exist_ok=True)  # 注释：创建缓存子目录
    dest = os.path.join(dest, os.path.basename(src_abs))  # 注释：拼接最终目标路径
    if os.path.exists(dest) and verify_copy(src, dest):  # 注释：若目标存在且验证通过
        # inform user and depends on him  # 注释：原注释保留
        print(  # 注释：提示用户目标已存在
            f"[WARNING]: The memory model path {dest} already exists. If it is not you want, please clear it and "  # 注释：警告信息
            f"restart the task."  # 注释：警告信息续行
        )  # 注释：结束打印
    else:  # 注释：目标不存在或验证失败时重新拷贝
        if os.path.isdir(src):  # 注释：源为目录
            shutil.copytree(src, dest, symlinks=False, dirs_exist_ok=True)  # 注释：拷贝目录
        else:  # 注释：源为文件
            shutil.copy2(src, dest)  # 注释：拷贝文件并保留元数据
    return dest  # 注释：返回共享内存路径
# （分隔说明：目录结构记录函数）  # 注释：替代空行，保持逐行注释
def _record_directory_structure(folder_path):  # 注释：记录目录结构到 .directory_record.txt
    """函数用途：记录目录的相对路径结构，用于后续校验。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - folder_path (str)：待记录的目录路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：记录文件路径（.directory_record.txt）。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 在目录内写入 .directory_record.txt。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 文件不可写时会抛出异常。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _record_directory_structure("/tmp/data") -> "/tmp/data/.directory_record.txt"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::_record_directory_structure。  # 注释：函数位置
    - 典型调用路径：copy_local_path_from_hdfs -> _record_directory_structure。  # 注释：典型调用链
    - 被谁调用：仅在本文件 copy_local_path_from_hdfs 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：os.walk、open/write。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    record_file = os.path.join(folder_path, ".directory_record.txt")  # 注释：记录文件路径
    with open(record_file, "w") as f:  # 注释：打开记录文件
        for root, dirs, files in os.walk(folder_path):  # 注释：遍历目录树
            for dir_name in dirs:  # 注释：遍历子目录
                relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)  # 注释：计算相对路径
                f.write(f"dir:{relative_dir}\n")  # 注释：写入目录记录
            for file_name in files:  # 注释：遍历文件
                if file_name != ".directory_record.txt":  # 注释：跳过记录文件自身
                    relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)  # 注释：计算相对路径
                    f.write(f"file:{relative_file}\n")  # 注释：写入文件记录
    return record_file  # 注释：返回记录文件路径
# （分隔说明：目录结构校验函数）  # 注释：替代空行，保持逐行注释
def _check_directory_structure(folder_path, record_file):  # 注释：校验目录结构与记录是否一致
    """函数用途：读取记录文件并与当前目录结构比对。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - folder_path (str)：待校验目录路径。  # 注释：参数含义
    - record_file (str)：记录文件路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：结构一致返回 True，否则 False。  # 注释：返回值语义
    副作用：无（仅读取文件与遍历目录）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 记录文件不存在返回 False。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _check_directory_structure("/tmp/data", "/tmp/data/.directory_record.txt")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::_check_directory_structure。  # 注释：函数位置
    - 典型调用路径：copy_local_path_from_hdfs -> _check_directory_structure。  # 注释：典型调用链
    - 被谁调用：仅在本文件 copy_local_path_from_hdfs 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：os.walk、open/read。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    if not os.path.exists(record_file):  # 注释：记录文件不存在
        return False  # 注释：返回 False
    existing_entries = set()  # 注释：当前目录结构集合
    for root, dirs, files in os.walk(folder_path):  # 注释：遍历目录树
        for dir_name in dirs:  # 注释：遍历子目录
            relative_dir = os.path.relpath(os.path.join(root, dir_name), folder_path)  # 注释：相对路径
            existing_entries.add(f"dir:{relative_dir}")  # 注释：记录目录条目
        for file_name in files:  # 注释：遍历文件
            if file_name != ".directory_record.txt":  # 注释：跳过记录文件
                relative_file = os.path.relpath(os.path.join(root, file_name), folder_path)  # 注释：相对路径
                existing_entries.add(f"file:{relative_file}")  # 注释：记录文件条目
    with open(record_file) as f:  # 注释：读取记录文件
        recorded_entries = set(f.read().splitlines())  # 注释：记录文件条目集合
    return existing_entries == recorded_entries  # 注释：比较集合是否一致
# （分隔说明：统一拷贝入口）  # 注释：替代空行，保持逐行注释
def copy_to_local(  # 注释：将 HDFS/本地/HF 资源复制到本地
    src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False, use_shm: bool = False
) -> str:  # 注释：函数返回本地路径
    """函数用途：将 HDFS/本地/模型 ID 资源复制到本地缓存，并可选加载到共享内存。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - src (str)：源路径（HDFS、本地路径或 Hugging Face 模型 ID）。  # 注释：参数含义
    - cache_dir (str, optional)：本地缓存目录；None 时使用系统临时目录。  # 注释：参数含义
    - filelock (str)：锁文件名基准（copy_local_path_from_hdfs 使用）。  # 注释：参数含义
    - verbose (bool)：是否打印复制日志。  # 注释：参数含义
    - always_recopy (bool)：是否强制重新拷贝忽略缓存。  # 注释：参数含义
    - use_shm (bool)：是否复制到 /dev/shm 共享内存。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：本地路径（或共享内存路径）。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 可能进行文件复制、创建缓存目录、触发 Hugging Face 下载。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - Hugging Face 下载失败会打印警告；导入 huggingface_hub 失败会被忽略。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - copy_to_local("hdfs://cluster/data/train.parquet", cache_dir="/tmp/cache")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::copy_to_local。  # 注释：函数位置
    - 典型调用路径：sft_dataset/rl_dataset/model loader -> copy_to_local -> copy_local_path_from_hdfs。  # 注释：典型调用链
    - 被谁调用：verl/utils/dataset/*、verl/utils/model.py、verl/trainer/*、verl/workers/*。  # 注释：调用方示例
    - 调用了谁（项目内）：copy_local_path_from_hdfs、copy_to_shm。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：huggingface_hub.snapshot_download（可选）。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    # Save to a local path for persistence.  # 注释：原注释保留
    local_path = copy_local_path_from_hdfs(src, cache_dir, filelock, verbose, always_recopy)  # 注释：先拷贝到本地缓存
    if use_shm and isinstance(local_path, str) and not os.path.exists(local_path):  # 注释：若使用共享内存且路径不存在
        try:  # 注释：尝试从 Hugging Face 下载
            from huggingface_hub import snapshot_download  # 注释：延迟导入，避免硬依赖
            resolved = snapshot_download(local_path)  # 注释：下载模型并返回本地路径
            if isinstance(resolved, str) and os.path.exists(resolved):  # 注释：若下载成功
                local_path = resolved  # 注释：更新本地路径为下载结果
        except ImportError:  # 注释：未安装 huggingface_hub
            pass  # 注释：忽略并继续
        except Exception as e:  # 注释：其他下载异常
            print(f"WARNING: Failed to download model from Hugging Face: {e}")  # 注释：打印警告
    # Load into shm to improve efficiency.  # 注释：原注释保留
    if use_shm:  # 注释：若开启共享内存
        return copy_to_shm(local_path)  # 注释：复制到 /dev/shm 并返回路径
    return local_path  # 注释：返回本地缓存路径
# （分隔说明：HDFS -> 本地拷贝实现）  # 注释：替代空行，保持逐行注释
def copy_local_path_from_hdfs(  # 注释：从 HDFS 拷贝到本地缓存（已弃用）
    src: str, cache_dir=None, filelock=".file.lock", verbose=False, always_recopy=False
) -> str:  # 注释：返回本地路径
    """函数用途：从 HDFS 拷贝文件/目录到本地缓存（已弃用，建议用 copy_to_local）。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - src (str)：源路径，若为 HDFS 则会触发拷贝。  # 注释：参数含义
    - cache_dir (str, optional)：本地缓存根目录。  # 注释：参数含义
    - filelock (str)：锁文件名称基准（实际会被 md5 覆盖）。  # 注释：参数含义
    - verbose (bool)：是否打印拷贝日志。  # 注释：参数含义
    - always_recopy (bool)：是否强制重新拷贝。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：本地路径（若非 HDFS 则直接返回 src）。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 可能创建缓存目录/锁文件，并进行拷贝或删除。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若 src 以 '/' 结尾会触发 AssertionError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - copy_local_path_from_hdfs("hdfs://cluster/data") -> "/tmp/<md5>/data"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::copy_local_path_from_hdfs。  # 注释：函数位置
    - 典型调用路径：copy_to_local -> copy_local_path_from_hdfs。  # 注释：典型调用链
    - 被谁调用：verl/utils/dataset/multiturn_sft_dataset.py 与本文件 copy_to_local。  # 注释：调用方示例
    - 调用了谁（项目内）：is_non_local、get_local_temp_path、md5_encode、_record_directory_structure、_check_directory_structure。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：filelock.FileLock、shutil、tempfile。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    """Deprecated. Please use copy_to_local instead."""  # 注释：原注释保留，说明已弃用
    from filelock import FileLock  # 注释：文件锁用于并发保护
    assert src[-1] != "/", f"Make sure the last char in src is not / because it will cause error. Got {src}"  # 注释：防止末尾斜杠导致错误
    if is_non_local(src):  # 注释：若为 HDFS 路径
        # download from hdfs to local  # 注释：原注释保留
        if cache_dir is None:  # 注释：未指定缓存目录
            # get a temp folder  # 注释：原注释保留
            cache_dir = tempfile.gettempdir()  # 注释：使用系统临时目录
        os.makedirs(cache_dir, exist_ok=True)  # 注释：确保缓存目录存在
        assert os.path.exists(cache_dir)  # 注释：确认缓存目录创建成功
        local_path = get_local_temp_path(src, cache_dir)  # 注释：生成本地缓存路径
        # get a specific lock  # 注释：原注释保留
        filelock = md5_encode(src) + ".lock"  # 注释：锁文件名使用 MD5
        lock_file = os.path.join(cache_dir, filelock)  # 注释：锁文件路径
        with FileLock(lock_file=lock_file):  # 注释：加锁防止并发冲突
            if always_recopy and os.path.exists(local_path):  # 注释：强制重拷贝且缓存已存在
                if os.path.isdir(local_path):  # 注释：若为目录
                    shutil.rmtree(local_path, ignore_errors=True)  # 注释：删除目录
                else:  # 注释：若为文件
                    os.remove(local_path)  # 注释：删除文件
            if not os.path.exists(local_path):  # 注释：缓存不存在则拷贝
                if verbose:  # 注释：需要打印日志
                    print(f"Copy from {src} to {local_path}")  # 注释：打印拷贝信息
                copy(src, local_path)  # 注释：调用 hdfs_io.copy 进行拷贝
                if os.path.isdir(local_path):  # 注释：若为目录则记录结构
                    _record_directory_structure(local_path)  # 注释：写入目录结构记录
            elif os.path.isdir(local_path):  # 注释：缓存存在且为目录
                # always_recopy=False, local path exists, and it is a folder: check whether there is anything missed  # 注释：原注释保留
                record_file = os.path.join(local_path, ".directory_record.txt")  # 注释：记录文件路径
                if not _check_directory_structure(local_path, record_file):  # 注释：结构不一致则重拷贝
                    if verbose:  # 注释：需要打印日志
                        print(f"Recopy from {src} to {local_path} due to missing files or directories.")  # 注释：打印重拷贝原因
                    shutil.rmtree(local_path, ignore_errors=True)  # 注释：删除旧目录
                    copy(src, local_path)  # 注释：重新拷贝
                    _record_directory_structure(local_path)  # 注释：重新记录结构
        return local_path  # 注释：返回本地缓存路径
    else:  # 注释：若为本地路径
        return src  # 注释：直接返回源路径
# （分隔说明：安全创建目录函数）  # 注释：替代空行，保持逐行注释
def local_mkdir_safe(path):  # 注释：线程安全的目录创建
    """函数用途：线程/进程安全地创建目录，避免并发创建冲突。  # 注释：函数用途说明
    参数：  # 注释：参数说明标题
    - path (str)：要创建的目录路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：最终创建的目录绝对路径。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 在系统临时目录创建锁文件并创建目标目录。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 获取锁失败时会打印警告，但仍会尝试创建目录。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - local_mkdir_safe("./outputs/ckpt") -> "/abs/path/outputs/ckpt"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/fs.py::local_mkdir_safe。  # 注释：函数位置
    - 典型调用路径：checkpoint 管理器 -> local_mkdir_safe。  # 注释：典型调用链
    - 被谁调用：verl/utils/checkpoint/*、verl/utils/megatron_utils.py、verl/trainer/*。  # 注释：调用方示例
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：filelock.FileLock、os.makedirs、tempfile。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束
    from filelock import FileLock  # 注释：延迟导入 FileLock
    if not os.path.isabs(path):  # 注释：若为相对路径
        working_dir = os.getcwd()  # 注释：获取当前工作目录
        path = os.path.join(working_dir, path)  # 注释：拼接为绝对路径
    # Using hash value of path as lock file name to avoid long file name  # 注释：原注释保留
    lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"  # 注释：构造锁文件名
    lock_path = os.path.join(tempfile.gettempdir(), lock_filename)  # 注释：锁文件存放在临时目录
    try:  # 注释：尝试获取锁并创建目录
        with FileLock(lock_path, timeout=60):  # Add timeout  # 注释：加锁，设置超时
            # make a new dir  # 注释：原注释保留
            os.makedirs(path, exist_ok=True)  # 注释：创建目录
    except Exception as e:  # 注释：锁获取失败或其他异常
        print(f"Warning: Failed to acquire lock for {path}: {e}")  # 注释：打印警告
        # Even if the lock is not acquired, try to create the directory  # 注释：原注释保留
        os.makedirs(path, exist_ok=True)  # 注释：继续尝试创建目录
    return path  # 注释：返回最终目录路径
