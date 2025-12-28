# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：声明 Apache 2.0 许可证
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：提示许可证链接
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：Apache 2.0 许可证地址
#
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：不提供担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
Contain small python utility functions  # 注释：保留英文说明
模块用途：提供通用 Python 工具（超时控制、字典操作、动态枚举、环境变量上下文等）。  # 注释：模块用途
输入/输出：输入为 Python 对象或配置，输出为处理后的对象/上下文管理器。  # 注释：模块输入输出概览
关键依赖：multiprocessing、signal、contextmanager、omegaconf（可选）。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- @timeout_limit(3) 装饰耗时函数  # 注释：超时装饰器示例
- with temp_env_var("KEY", "VAL"): ...  # 注释：临时环境变量示例
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：训练/评测脚本调用工具函数。  # 注释：上层入口举例
- 典型链路：上层模块 -> py_functional.* -> 标准库/第三方库。  # 注释：关键调用链
"""  # 注释：模块 docstring 结束

import importlib  # 注释：动态导入模块
import multiprocessing  # 注释：多进程实现超时
import os  # 注释：环境变量操作
import queue  # Import the queue module for exception type hint  # 注释：队列异常类型
import signal  # 注释：信号超时（POSIX）
from contextlib import contextmanager  # 注释：上下文管理器
from functools import wraps  # 注释：保持函数签名
from types import SimpleNamespace  # 注释：简单命名空间
from typing import Any, Callable, Iterator, Optional  # 注释：类型注解


# --- Top-level helper for multiprocessing timeout ---  # 注释：多进程超时辅助说明
# This function MUST be defined at the top level to be pickleable  # 注释：必须在顶层以便 pickle
def _mp_target_wrapper(target_func: Callable, mp_queue: multiprocessing.Queue, args: tuple, kwargs: dict[str, Any]):  # 注释：子进程执行包装器
    """
    Internal wrapper function executed in the child process.
    Calls the original target function and puts the result or exception into the queue.

    功能：在子进程中执行目标函数，将结果/异常写入队列。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - target_func：要执行的函数。  # 注释：参数含义
    - mp_queue：多进程队列用于返回结果。  # 注释：参数含义
    - args/kwargs：传给目标函数的参数。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None（通过队列传回结果）。  # 注释：返回值语义
    副作用：向队列写入 (success, result/exception)。  # 注释：副作用说明
    异常/边界条件：异常会被捕获并尝试 pickle。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - _mp_target_wrapper(func, q, (1,), {})。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/py_functional.py::_mp_target_wrapper。  # 注释：函数位置
    - 典型调用路径：timeout_limit -> wrapper_mp -> _mp_target_wrapper。  # 注释：典型调用链
    - 被谁调用：timeout_limit 内部。  # 注释：调用方说明
    - 调用了谁（关键外部依赖）：multiprocessing.Queue。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束
    try:  # 注释：捕获执行异常
        result = target_func(*args, **kwargs)  # 注释：执行目标函数
        mp_queue.put((True, result))  # Indicate success and put result  # 注释：写入成功结果
    except Exception as e:  # 注释：捕获异常
        # Ensure the exception is pickleable for the queue  # 注释：确保异常可 pickle
        try:  # 注释：尝试 pickle 异常
            import pickle  # 注释：导入 pickle

            pickle.dumps(e)  # Test if the exception is pickleable  # 注释：测试可 pickle
            mp_queue.put((False, e))  # Indicate failure and put exception  # 注释：写入异常
        except (pickle.PicklingError, TypeError):  # 注释：不可 pickle 的异常
            # Fallback if the original exception cannot be pickled  # 注释：原注释保留
            mp_queue.put((False, RuntimeError(f"Original exception type {type(e).__name__} not pickleable: {e}")))  # 注释：包裹异常


# Renamed the function from timeout to timeout_limit  # 注释：函数重命名说明
def timeout_limit(seconds: float, use_signals: bool = False):  # 注释：超时装饰器入口
    """
    Decorator to add a timeout to a function.

    Args:
        seconds: The timeout duration in seconds.
        use_signals: (Deprecated)  This is deprecated because signals only work reliably in the main thread
                     and can cause issues in multiprocessing or multithreading contexts.
                     Defaults to False, which uses the more robust multiprocessing approach.

    Returns:
        A decorated function with timeout.

    Raises:
        TimeoutError: If the function execution exceeds the specified time.
        RuntimeError: If the child process exits with an error (multiprocessing mode).
        NotImplementedError: If the OS is not POSIX (signals are only supported on POSIX).

    功能：为函数增加超时限制（信号或多进程实现）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - seconds：超时时间（秒）。  # 注释：参数含义
    - use_signals：是否使用 signal（不推荐）。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 装饰后的函数。  # 注释：返回值语义
    副作用：可能创建子进程或注册信号处理器。  # 注释：副作用说明
    异常/边界条件：非 POSIX 系统使用信号会抛 NotImplementedError。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - @timeout_limit(1)
      def f(): time.sleep(2)  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/py_functional.py::timeout_limit。  # 注释：函数位置
    - 典型调用路径：上层函数装饰 -> wrapper_mp/wrapper_signal。  # 注释：典型调用链
    - 被谁调用：项目内需要超时控制的函数。  # 注释：调用方说明
    - 调用了谁（关键外部依赖）：multiprocessing、signal。  # 注释：外部依赖说明
    """  # 注释：函数 docstring 结束

    def decorator(func):  # 注释：装饰器内部函数
        if use_signals:  # 注释：使用信号实现
            if os.name != "posix":  # 注释：仅 POSIX 支持信号
                raise NotImplementedError(f"Unsupported OS: {os.name}")  # 注释：抛异常
            # Issue deprecation warning if use_signals is explicitly True  # 注释：原注释保留
            print(  # 注释：输出弃用警告
                "WARN: The 'use_signals=True' option in the timeout decorator is deprecated. \
                Signals are unreliable outside the main thread. \
                Please use the default multiprocessing-based timeout (use_signals=False)."  # 注释：警告文本
            )  # 注释：print 结束

            @wraps(func)  # 注释：保持原函数信息
            def wrapper_signal(*args, **kwargs):  # 注释：信号超时包装
                def handler(signum, frame):  # 注释：信号处理函数
                    # Update function name in error message if needed (optional but good practice)  # 注释：原注释保留
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (signal)!")  # 注释：抛超时异常

                old_handler = signal.getsignal(signal.SIGALRM)  # 注释：保存旧处理器
                signal.signal(signal.SIGALRM, handler)  # 注释：注册新处理器
                # Use setitimer for float seconds support, alarm only supports integers  # 注释：原注释保留
                signal.setitimer(signal.ITIMER_REAL, seconds)  # 注释：设置定时器

                try:  # 注释：执行函数
                    result = func(*args, **kwargs)  # 注释：调用原函数
                finally:  # 注释：确保恢复
                    # Reset timer and handler  # 注释：原注释保留
                    signal.setitimer(signal.ITIMER_REAL, 0)  # 注释：关闭定时器
                    signal.signal(signal.SIGALRM, old_handler)  # 注释：恢复旧处理器
                return result  # 注释：返回结果

            return wrapper_signal  # 注释：返回信号包装器
        else:  # 注释：使用多进程实现
            # --- Multiprocessing based timeout (existing logic) ---  # 注释：原注释保留
            @wraps(func)  # 注释：保持原函数信息
            def wrapper_mp(*args, **kwargs):  # 注释：多进程包装
                q = multiprocessing.Queue(maxsize=1)  # 注释：创建队列
                process = multiprocessing.Process(target=_mp_target_wrapper, args=(func, q, args, kwargs))  # 注释：创建子进程
                process.start()  # 注释：启动子进程
                process.join(timeout=seconds)  # 注释：等待指定时间

                if process.is_alive():  # 注释：超时仍在运行
                    process.terminate()  # 注释：终止进程
                    process.join(timeout=0.5)  # Give it a moment to terminate  # 注释：等待退出
                    if process.is_alive():  # 注释：仍未退出
                        print(f"Warning: Process {process.pid} did not terminate gracefully after timeout.")  # 注释：警告
                    # Update function name in error message if needed (optional but good practice)  # 注释：原注释保留
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds (multiprocessing)!")  # 注释：抛超时异常

                try:  # 注释：读取队列结果
                    success, result_or_exc = q.get(timeout=0.1)  # Small timeout for queue read  # 注释：读取队列
                    if success:  # 注释：成功
                        return result_or_exc  # 注释：返回结果
                    else:  # 注释：子进程异常
                        raise result_or_exc  # Reraise exception from child  # 注释：重新抛出
                except queue.Empty as err:  # 注释：队列为空
                    exitcode = process.exitcode  # 注释：子进程退出码
                    if exitcode is not None and exitcode != 0:  # 注释：非正常退出
                        raise RuntimeError(  # 注释：抛运行时错误
                            f"Child process exited with error (exitcode: {exitcode}) before returning result."  # 注释：错误信息
                        ) from err  # 注释：保留上下文
                    else:  # 注释：可能异常退出
                        # Should have timed out if queue is empty after join unless process died unexpectedly  # 注释：原注释保留
                        # Update function name in error message if needed (optional but good practice)  # 注释：原注释保留
                        raise TimeoutError(  # 注释：抛超时异常
                            f"Operation timed out or process finished unexpectedly without result "  # 注释：错误信息1
                            f"(exitcode: {exitcode})."  # 注释：错误信息2
                        ) from err  # 注释：保留上下文
                finally:  # 注释：清理资源
                    q.close()  # 注释：关闭队列
                    q.join_thread()  # 注释：等待队列线程结束

            return wrapper_mp  # 注释：返回多进程包装器

    return decorator  # 注释：返回装饰器


def union_two_dict(dict1: dict, dict2: dict):  # 注释：合并两个字典
    """Union two dict. Will throw an error if there is an item not the same object with the same key.

    功能：合并 dict2 到 dict1，若相同 key 值不同则断言失败。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - dict1：目标字典。  # 注释：参数含义
    - dict2：待合并字典。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict：合并后的 dict1。  # 注释：返回值语义
    副作用：就地修改 dict1。  # 注释：副作用说明
    异常/边界条件：key 冲突且值不一致会触发 assert。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - union_two_dict({"a":1}, {"b":2}) -> {"a":1,"b":2}。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/utils/py_functional.py::union_two_dict。  # 注释：函数位置
    - 典型调用路径：上层配置合并逻辑。  # 注释：典型调用链
    - 被谁调用：项目内工具函数。  # 注释：调用方说明
    """  # 注释：函数 docstring 结束
    for key, val in dict2.items():  # 注释：遍历 dict2
        if key in dict1:  # 注释：key 冲突
            assert dict2[key] == dict1[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"  # 注释：值不一致则报错
        dict1[key] = val  # 注释：写入 dict1

    return dict1  # 注释：返回合并后的字典


def rename_dict(data: dict, prefix: str = "") -> dict:  # 注释：为字典 key 添加前缀
    """Add a prefix to all the keys in the data dict if it's name is not started with prefix

    功能：为所有 key 添加前缀（若未包含前缀）。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data：原始字典。  # 注释：参数含义
    - prefix：前缀字符串。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - dict：重命名后的字典。  # 注释：返回值语义
    副作用：无（返回新 dict）。  # 注释：副作用说明
    最小示例：  # 注释：最小示例标题
    - rename_dict({"a":1}, "x_") -> {"x_a":1}。  # 注释：示例
    """  # 注释：函数 docstring 结束
    new_data = {}  # 注释：新字典
    for key, val in data.items():  # 注释：遍历原字典
        new_key = f"{prefix}{key}" if not key.startswith(prefix) else key  # 注释：生成新 key
        new_data[new_key] = val  # 注释：写入新字典
    return new_data  # 注释：返回新字典


def append_to_dict(data: dict, new_data: dict, prefix: str = ""):  # 注释：将值追加到字典列表中
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    功能：将 new_data 的值追加到 data 中对应 key 的列表。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - data：目标字典（值为 list）。  # 注释：参数含义
    - new_data：新增数据。  # 注释：参数含义
    - prefix：key 前缀。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None（就地修改 data）。  # 注释：返回值语义
    最小示例：  # 注释：最小示例标题
    - append_to_dict({}, {"a":1}) -> {"a":[1]}。  # 注释：示例
    """  # 注释：函数 docstring 结束
    for key, val in new_data.items():  # 注释：遍历 new_data
        new_key = f"{prefix}{key}" if not key.startswith(prefix) else key  # 注释：构造新 key
        if new_key not in data:  # 注释：不存在则初始化
            data[new_key] = []  # 注释：初始化列表
        if isinstance(val, list):  # 注释：若为列表
            data[new_key].extend(val)  # 注释：扩展列表
        else:  # 注释：单值
            data[new_key].append(val)  # 注释：追加单值


class NestedNamespace(SimpleNamespace):  # 注释：递归命名空间
    """A nested version of SimpleNamespace that recursively converts dictionaries to namespaces.

    This class allows for dot notation access to nested dictionary structures by recursively
    converting dictionaries to NestedNamespace objects.

    Example:
        config_dict = {"a": 1, "b": {"c": 2, "d": 3}}
        config = NestedNamespace(config_dict)
        # Access with: config.a, config.b.c, config.b.d

    Args:
        dictionary: The dictionary to convert to a nested namespace.
        **kwargs: Additional attributes to set on the namespace.

    功能：将嵌套 dict 转为可点访问的命名空间。  # 注释：类用途
    最小示例：  # 注释：最小示例标题
    - NestedNamespace({"a":1,"b":{"c":2}}).b.c == 2。  # 注释：示例
    """  # 注释：类 docstring 结束

    def __init__(self, dictionary, **kwargs):  # 注释：初始化并递归转换
        super().__init__(**kwargs)  # 注释：初始化父类
        for key, value in dictionary.items():  # 注释：遍历字典
            if isinstance(value, dict):  # 注释：嵌套 dict
                self.__setattr__(key, NestedNamespace(value))  # 注释：递归转换
            else:  # 注释：普通值
                self.__setattr__(key, value)  # 注释：直接赋值


class DynamicEnumMeta(type):  # 注释：动态枚举元类
    def __iter__(cls) -> Iterator[Any]:  # 注释：迭代注册项
        return iter(cls._registry.values())  # 注释：返回迭代器

    def __contains__(cls, item: Any) -> bool:  # 注释：判断成员是否存在
        # allow `name in EnumClass` or `member in EnumClass`  # 注释：原注释保留
        if isinstance(item, str):  # 注释：字符串 key
            return item in cls._registry  # 注释：检查注册表
        return item in cls._registry.values()  # 注释：检查实例

    def __getitem__(cls, name: str) -> Any:  # 注释：通过名字获取成员
        return cls._registry[name]  # 注释：从注册表取值

    def __reduce_ex__(cls, protocol):  # 注释：pickle 支持
        # Always load the existing module and grab the class  # 注释：原注释保留
        return getattr, (importlib.import_module(cls.__module__), cls.__name__)  # 注释：返回反序列化函数

    def names(cls):  # 注释：返回名字列表
        return list(cls._registry.keys())  # 注释：返回 keys

    def values(cls):  # 注释：返回值列表
        return list(cls._registry.values())  # 注释：返回 values


class DynamicEnum(metaclass=DynamicEnumMeta):  # 注释：动态枚举类
    _registry: dict[str, "DynamicEnum"] = {}  # 注释：注册表
    _next_value: int = 0  # 注释：下一个值

    def __init__(self, name: str, value: int):  # 注释：初始化成员
        self.name = name  # 注释：成员名
        self.value = value  # 注释：成员值

    def __repr__(self):  # 注释：repr 输出
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"  # 注释：格式化输出

    def __reduce_ex__(self, protocol):  # 注释：pickle 支持
        """
        Unpickle via: getattr(import_module(module).Dispatch, 'ONE_TO_ALL')
        so the existing class is reused instead of re-executed.
        """  # 注释：保留英文说明
        module = importlib.import_module(self.__class__.__module__)  # 注释：导入模块
        enum_cls = getattr(module, self.__class__.__name__)  # 注释：获取类
        return getattr, (enum_cls, self.name)  # 注释：返回反序列化指令

    @classmethod  # 注释：类方法
    def register(cls, name: str) -> "DynamicEnum":  # 注释：注册新成员
        key = name.upper()  # 注释：规范化 key
        if key in cls._registry:  # 注释：已存在
            raise ValueError(f"{key} already registered")  # 注释：抛异常
        member = cls(key, cls._next_value)  # 注释：创建新成员
        cls._registry[key] = member  # 注释：写入注册表
        setattr(cls, key, member)  # 注释：设置类属性
        cls._next_value += 1  # 注释：递增计数
        return member  # 注释：返回成员

    @classmethod  # 注释：类方法
    def remove(cls, name: str):  # 注释：移除成员
        key = name.upper()  # 注释：规范化 key
        member = cls._registry.pop(key)  # 注释：移除并返回成员
        delattr(cls, key)  # 注释：删除类属性
        return member  # 注释：返回成员

    @classmethod  # 注释：类方法
    def from_name(cls, name: str) -> Optional["DynamicEnum"]:  # 注释：通过名字获取成员
        return cls._registry.get(name.upper())  # 注释：查找注册表


@contextmanager  # 注释：上下文管理器装饰器
def temp_env_var(key: str, value: str):  # 注释：临时环境变量
    """Context manager for temporarily setting an environment variable.

    This context manager ensures that environment variables are properly set and restored,
    even if an exception occurs during the execution of the code block.

    Args:
        key: Environment variable name to set
        value: Value to set the environment variable to

    Yields:
        None

    Example:
        >>> with temp_env_var("MY_VAR", "test_value"):
        ...     # MY_VAR is set to "test_value"
        ...     do_something()
        ... # MY_VAR is restored to its original value or removed if it didn't exist

    功能：临时设置环境变量，并在退出时恢复。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - key：环境变量名。  # 注释：参数含义
    - value：环境变量值。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 生成器上下文。  # 注释：返回值语义
    副作用：修改 os.environ。  # 注释：副作用说明
    最小示例：  # 注释：最小示例标题
    - with temp_env_var("A","1"): ...  # 注释：示例
    """  # 注释：函数 docstring 结束
    original = os.environ.get(key)  # 注释：保存原值
    os.environ[key] = value  # 注释：设置新值
    try:  # 注释：进入上下文
        yield  # 注释：让出执行权
    finally:  # 注释：退出上下文
        if original is None:  # 注释：原本不存在
            os.environ.pop(key, None)  # 注释：删除变量
        else:  # 注释：原本存在
            os.environ[key] = original  # 注释：恢复原值


def convert_to_regular_types(obj):  # 注释：将配置对象转换为普通 Python 类型
    """Convert Hydra configs and other special types to regular Python types.

    功能：将 DictConfig/ListConfig 等转为 dict/list，递归处理。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - obj：待转换对象。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - 普通 Python 类型对象。  # 注释：返回值语义
    最小示例：  # 注释：最小示例标题
    - convert_to_regular_types(ListConfig([1,2])) -> [1,2]。  # 注释：示例
    """  # 注释：函数 docstring 结束
    from omegaconf import DictConfig, ListConfig  # 注释：局部导入

    if isinstance(obj, ListConfig | DictConfig):  # 注释：OmegaConf 类型
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)  # 注释：递归转换
    elif isinstance(obj, list | tuple):  # 注释：列表/元组
        return [convert_to_regular_types(x) for x in obj]  # 注释：递归转换
    elif isinstance(obj, dict):  # 注释：字典
        return {k: convert_to_regular_types(v) for k, v in obj.items()}  # 注释：递归转换
    return obj  # 注释：返回原对象
