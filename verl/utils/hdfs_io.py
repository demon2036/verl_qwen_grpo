# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明，标明所属与年份
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
"""模块用途：提供支持 HDFS 前缀路径的通用文件系统操作（exists/makedirs/copy）。  # 注释：模块用途
输入/输出：各函数接受路径字符串，返回布尔值或复制结果（见函数 docstring）。  # 注释：模块输入输出概览
关键依赖：os、shutil、logging、HDFS CLI（hdfs）与环境变量 VERL_SFT_LOGGING_LEVEL。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- from verl.utils.hdfs_io import copy, makedirs  # 注释：导入函数
- makedirs("hdfs://cluster/path")  # 注释：创建 HDFS 目录
- copy("/tmp/data", "hdfs://cluster/path")  # 注释：拷贝到 HDFS
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：examples/data_preprocess/gsm8k.py / 训练与 checkpoint 逻辑。  # 注释：上层入口举例
- 典型链路：上层脚本 -> hdfs_io.copy/makedirs -> _copy/_mkdir -> _run_cmd（或 os/shutil）。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束
# （分隔说明：开始导入依赖）  # 注释：替代空行，保持逐行注释
import logging  # 注释：日志输出
import os  # 注释：本地路径与系统命令
import shutil  # 注释：本地文件拷贝与查找可执行文件
# （分隔说明：初始化模块级日志器）  # 注释：替代空行，保持逐行注释
logger = logging.getLogger(__file__)  # 注释：以文件名作为 logger 名称
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))  # 注释：从环境变量设置日志级别
# （分隔说明：HDFS 前缀与可执行路径常量）  # 注释：替代空行，保持逐行注释
_HDFS_PREFIX = "hdfs://"  # 注释：HDFS 路径前缀约定
_HDFS_BIN_PATH = shutil.which("hdfs")  # 注释：查找系统中的 hdfs 可执行文件路径
# （分隔说明：对外函数：exists）  # 注释：替代空行，保持逐行注释
def exists(path: str, **kwargs) -> bool:  # 注释：判断路径是否存在（支持 HDFS）
    r"""函数用途：判断路径是否存在，兼容本地与 HDFS。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - path (str)：待检查的路径。  # 注释：参数含义
    - kwargs：预留给 HDFS 的关键字参数（当前未使用）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：存在返回 True，否则 False。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 若为 HDFS 路径，会触发一次外部命令调用（hdfs dfs -test -e）。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 若系统未安装 hdfs 命令，_HDFS_BIN_PATH 可能为 None，命令执行会失败。  # 注释：潜在异常
    最小示例：  # 注释：最小示例标题
    - exists("/tmp/foo") -> True/False。  # 注释：本地示例
    - exists("hdfs://cluster/foo") -> True/False。  # 注释：HDFS 示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::exists。  # 注释：函数位置
    - 典型调用路径：上层脚本 -> exists -> _is_non_local -> _exists 或 os.path.exists。  # 注释：典型调用链
    - 被谁调用：verl/utils/fs.py 与多种数据预处理脚本（如 examples/data_preprocess/gsm8k.py）。  # 注释：调用方说明
    - 调用了谁（项目内）：_is_non_local、_exists。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.path.exists、hdfs dfs -test -e。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if _is_non_local(path):  # 注释：若为 HDFS 路径
        return _exists(path, **kwargs)  # 注释：走 HDFS 检查逻辑
    return os.path.exists(path)  # 注释：本地路径直接用 os.path.exists
# （分隔说明：内部函数：_exists）  # 注释：替代空行，保持逐行注释
def _exists(file_path: str):  # 注释：HDFS 兼容的 exists 实现
    """函数用途：检查 HDFS 或本地路径是否存在（内部使用）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - file_path (str)：待检查路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：存在返回 True，否则 False。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 若为 HDFS 路径，会执行外部命令。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - HDFS 命令不存在或执行失败会导致返回 False。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _exists("hdfs://cluster/foo") -> True/False。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::_exists。  # 注释：函数位置
    - 典型调用路径：exists -> _exists -> _run_cmd/_hdfs_cmd。  # 注释：典型调用链
    - 被谁调用：仅在本文件的 exists 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：_run_cmd、_hdfs_cmd。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.path.exists、hdfs dfs -test -e。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if file_path.startswith("hdfs"):  # 注释：若为 HDFS 路径
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0  # 注释：调用 hdfs dfs -test -e 判断存在
    return os.path.exists(file_path)  # 注释：本地路径走 os.path.exists
# （分隔说明：对外函数：makedirs）  # 注释：替代空行，保持逐行注释
def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:  # 注释：创建目录（支持 HDFS）
    r"""函数用途：创建目录并支持 HDFS 路径。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - name (str)：要创建的目录路径。  # 注释：参数含义
    - mode (int)：本地目录权限位（仅本地生效）。  # 注释：参数含义
    - exist_ok (bool)：本地目录已存在是否忽略错误。  # 注释：参数含义
    - kwargs：预留给 HDFS 的关键字参数（当前未使用）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 会在本地或 HDFS 创建目录，产生文件系统副作用。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - HDFS 路径下未实现 exist_ok 逻辑；错误处理依赖 hdfs 命令行为。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - makedirs("/tmp/foo", exist_ok=True)。  # 注释：本地示例
    - makedirs("hdfs://cluster/foo")。  # 注释：HDFS 示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::makedirs。  # 注释：函数位置
    - 典型调用路径：上层脚本 -> makedirs -> _is_non_local -> _mkdir 或 os.makedirs。  # 注释：典型调用链
    - 被谁调用：examples/data_preprocess/gsm8k.py、verl/utils/checkpoint/checkpoint_handler.py 等。  # 注释：调用方示例
    - 调用了谁（项目内）：_is_non_local、_mkdir。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.makedirs、hdfs dfs -mkdir。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if _is_non_local(name):  # 注释：若为 HDFS 路径
        # TODO(haibin.lin):  # 注释：原作者 TODO 保留
        # - handle OSError for hdfs(?)  # 注释：TODO：处理 HDFS 错误
        # - support exist_ok for hdfs(?)  # 注释：TODO：支持 exist_ok
        _mkdir(name, **kwargs)  # 注释：调用 HDFS mkdir
    else:  # 注释：本地路径
        os.makedirs(name, mode=mode, exist_ok=exist_ok)  # 注释：本地递归建目录
# （分隔说明：内部函数：_mkdir）  # 注释：替代空行，保持逐行注释
def _mkdir(file_path: str) -> bool:  # 注释：HDFS mkdir 实现
    """函数用途：创建 HDFS 或本地目录（内部使用）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - file_path (str)：目标目录路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：始终返回 True（表示调用完成）。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 创建目录或执行外部 hdfs 命令。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - hdfs 命令失败时不会抛异常，但可能未创建成功。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _mkdir("hdfs://cluster/foo")。  # 注释：HDFS 示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::_mkdir。  # 注释：函数位置
    - 典型调用路径：makedirs -> _mkdir。  # 注释：典型调用链
    - 被谁调用：仅在本文件的 makedirs 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：_run_cmd、_hdfs_cmd。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.makedirs、hdfs dfs -mkdir。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if file_path.startswith("hdfs"):  # 注释：若为 HDFS 路径
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))  # 注释：执行 hdfs dfs -mkdir -p
    else:  # 注释：本地路径
        os.makedirs(file_path, exist_ok=True)  # 注释：本地递归建目录
    return True  # 注释：返回 True 表示流程完成
# （分隔说明：对外函数：copy）  # 注释：替代空行，保持逐行注释
def copy(src: str, dst: str, **kwargs) -> bool:  # 注释：复制文件/目录（支持 HDFS）
    r"""函数用途：复制文件或目录，兼容本地与 HDFS。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - src (str)：源路径。  # 注释：参数含义
    - dst (str)：目标路径。  # 注释：参数含义
    - kwargs：透传给 shutil.copy/copytree 的参数（仅本地生效）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool 或路径：对 HDFS 返回 True/False，对本地返回 shutil 的返回值。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 会创建/覆盖目标文件或目录，并可能执行外部 hdfs 命令。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - HDFS 模式下未显式处理 SameFileError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - copy("/tmp/a", "/tmp/b")。  # 注释：本地示例
    - copy("/tmp/a", "hdfs://cluster/a")。  # 注释：HDFS 示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::copy。  # 注释：函数位置
    - 典型调用路径：上层脚本 -> copy -> _is_non_local -> _copy 或 shutil.copy/copytree。  # 注释：典型调用链
    - 被谁调用：examples/data_preprocess/*.py、verl/utils/checkpoint/* 等。  # 注释：调用方示例
    - 调用了谁（项目内）：_is_non_local、_copy。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：shutil.copy、shutil.copytree、hdfs dfs -cp/-put/-get。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if _is_non_local(src) or _is_non_local(dst):  # 注释：任一路径为 HDFS，则走 HDFS 拷贝
        # TODO(haibin.lin):  # 注释：原作者 TODO 保留
        # - handle SameFileError for hdfs files(?)  # 注释：TODO：处理 HDFS 同文件拷贝
        # - return file destination for hdfs files  # 注释：TODO：返回 HDFS 目标路径
        return _copy(src, dst)  # 注释：调用 HDFS 复制逻辑
    else:  # 注释：纯本地拷贝
        if os.path.isdir(src):  # 注释：源为目录
            return shutil.copytree(src, dst, **kwargs)  # 注释：递归拷贝目录
        else:  # 注释：源为文件
            return shutil.copy(src, dst, **kwargs)  # 注释：拷贝单文件
# （分隔说明：内部函数：_copy）  # 注释：替代空行，保持逐行注释
def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:  # 注释：HDFS/本地通用复制实现
    """函数用途：根据路径类型选择 hdfs dfs 或本地复制（内部使用）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - from_path (str)：源路径。  # 注释：参数含义
    - to_path (str)：目标路径。  # 注释：参数含义
    - timeout (int, optional)：命令超时（当前未直接使用）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：复制成功返回 True，否则 False。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 可能执行外部 hdfs 命令或本地文件复制。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 本地复制异常会被捕获并记录日志，函数返回 False。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _copy("hdfs://cluster/a", "/tmp/a")。  # 注释：HDFS -> 本地示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::_copy。  # 注释：函数位置
    - 典型调用路径：copy -> _copy -> _run_cmd 或 shutil.copy。  # 注释：典型调用链
    - 被谁调用：仅在本文件 copy 中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：_run_cmd、_hdfs_cmd、logger。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：shutil.copy、hdfs dfs -cp/-put/-get。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    if to_path.startswith("hdfs"):  # 注释：目标是 HDFS
        if from_path.startswith("hdfs"):  # 注释：HDFS -> HDFS
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)  # 注释：hdfs dfs -cp
        else:  # 注释：本地 -> HDFS
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)  # 注释：hdfs dfs -put
    else:  # 注释：目标是本地
        if from_path.startswith("hdfs"):  # 注释：HDFS -> 本地
            returncode = _run_cmd(  # 注释：执行 hdfs dfs -get
                _hdfs_cmd(  # 注释：构造命令字符串
                    f"-get \
                {from_path} {to_path}"  # 注释：命令参数
                ),  # 注释：结束命令构造
                timeout=timeout,  # 注释：超时参数
            )  # 注释：结束 _run_cmd 调用
        else:  # 注释：本地 -> 本地
            try:  # 注释：捕获本地复制异常
                shutil.copy(from_path, to_path)  # 注释：本地复制文件
                returncode = 0  # 注释：成功返回码
            except shutil.SameFileError:  # 注释：同文件拷贝
                returncode = 0  # 注释：视为成功
            except Exception as e:  # 注释：其他异常
                logger.warning(f"copy {from_path} {to_path} failed: {e}")  # 注释：记录警告日志
                returncode = -1  # 注释：失败返回码
    return returncode == 0  # 注释：返回是否成功
# （分隔说明：内部函数：_run_cmd）  # 注释：替代空行，保持逐行注释
def _run_cmd(cmd: str, timeout=None):  # 注释：执行系统命令
    """函数用途：执行系统命令并返回退出码（内部使用）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - cmd (str)：待执行的命令字符串。  # 注释：参数含义
    - timeout：预留参数（当前未使用）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - int：os.system 的退出码。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 触发系统命令执行。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - os.system 可能因权限/命令不存在失败。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _run_cmd("ls") -> 0。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::_run_cmd。  # 注释：函数位置
    - 典型调用路径：_exists/_mkdir/_copy -> _run_cmd。  # 注释：典型调用链
    - 被谁调用：仅在本文件内部函数中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：os.system。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    return os.system(cmd)  # 注释：直接调用系统命令
# （分隔说明：内部函数：_hdfs_cmd）  # 注释：替代空行，保持逐行注释
def _hdfs_cmd(cmd: str) -> str:  # 注释：拼接 hdfs dfs 命令
    """函数用途：构造 hdfs dfs 命令字符串（内部使用）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - cmd (str)：hdfs dfs 子命令参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - str：完整的命令字符串。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 当 _HDFS_BIN_PATH 为 None 时，命令字符串会以 "None dfs ..." 开头。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _hdfs_cmd("-ls hdfs://cluster") -> "<hdfs_path> dfs -ls hdfs://cluster"。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::_hdfs_cmd。  # 注释：函数位置
    - 典型调用路径：_exists/_mkdir/_copy -> _hdfs_cmd。  # 注释：典型调用链
    - 被谁调用：仅在本文件内部函数中调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无（仅字符串拼接）。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    return f"{_HDFS_BIN_PATH} dfs {cmd}"  # 注释：拼接可执行路径与 dfs 子命令
# （分隔说明：内部函数：_is_non_local）  # 注释：替代空行，保持逐行注释
def _is_non_local(path: str):  # 注释：判断是否为 HDFS 路径前缀
    """函数用途：判断路径是否以 HDFS 前缀开头（内部使用）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - path (str)：待检查路径。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - bool：True 表示 HDFS 路径，否则 False。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：无。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _is_non_local("hdfs://foo") -> True。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/hdfs_io.py::_is_non_local。  # 注释：函数位置
    - 典型调用路径：exists/makedirs/copy -> _is_non_local。  # 注释：典型调用链
    - 被谁调用：仅在本文件内被 exists/makedirs/copy 调用。  # 注释：调用方说明
    - 调用了谁（项目内）：无。  # 注释：项目内依赖说明
    - 调用了谁（关键外部依赖）：无。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    return path.startswith(_HDFS_PREFIX)  # 注释：检查前缀是否为 hdfs://
