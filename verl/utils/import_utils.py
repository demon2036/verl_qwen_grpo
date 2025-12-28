# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
模块用途：检测可选依赖包是否可用，并提供动态导入工具。（注释：模块职责）
输入/输出：
  - 输入：包名、模块路径、对象名。（注释：输入来源）
  - 输出：可用性布尔值或动态加载的模块/对象。（注释：输出内容）
关键依赖：importlib、warnings。（注释：依赖说明）
典型用法（最小示例）：
  - `if is_nvtx_available(): ...`。（注释：可选依赖检测）
  - `cls = load_extern_object("file://.../x.py", "MyClass")`。（注释：动态加载）
调用路径概览：
  - 训练入口/工具 -> `verl.utils.import_utils` -> 动态加载数据集/Reward/Agent 组件。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import importlib  # 注释：动态导入模块
import importlib.util  # 注释：低层导入工具
import os  # 注释：文件路径检查
import warnings  # 注释：弃用警告
from functools import cache, wraps  # 注释：缓存与装饰器
from typing import Optional  # 注释：类型标注


@cache
def is_megatron_core_available():
    """
    检测 megatron.core 是否可用。（注释：函数用途）

    返回：bool。（注释：返回说明）
    副作用：无，结果缓存。（注释：副作用）
    最小示例：`is_megatron_core_available()` -> True/False。（注释：示例）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::is_megatron_core_available`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.util.find_spec`。（注释：外部依赖）
    """
    try:  # 注释：捕获模块不存在的异常
        mcore_spec = importlib.util.find_spec("megatron.core")  # 注释：尝试查找模块
    except ModuleNotFoundError:
        mcore_spec = None  # 注释：模块不存在
    return mcore_spec is not None  # 注释：返回是否可用


@cache
def is_vllm_available():
    """
    检测 vllm 是否可用。（注释：函数用途）

    返回：bool。（注释：返回说明）
    副作用：无，结果缓存。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::is_vllm_available`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用（供外部逻辑使用）。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.util.find_spec`。（注释：外部依赖）
    """
    try:
        vllm_spec = importlib.util.find_spec("vllm")  # 注释：查找 vllm 模块
    except ModuleNotFoundError:
        vllm_spec = None  # 注释：模块不存在
    return vllm_spec is not None  # 注释：返回可用性


@cache
def is_sglang_available():
    """
    检测 sglang 是否可用。（注释：函数用途）

    返回：bool。（注释：返回说明）
    副作用：无，结果缓存。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::is_sglang_available`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.util.find_spec`。（注释：外部依赖）
    """
    try:
        sglang_spec = importlib.util.find_spec("sglang")  # 注释：查找 sglang 模块
    except ModuleNotFoundError:
        sglang_spec = None  # 注释：模块不存在
    return sglang_spec is not None  # 注释：返回可用性


@cache
def is_nvtx_available():
    """
    检测 nvtx 是否可用。（注释：函数用途）

    返回：bool。（注释：返回说明）
    副作用：无，结果缓存。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::is_nvtx_available`。（注释：定位）
      - 被谁调用：`verl/utils/profiler/__init__.py`。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.util.find_spec`。（注释：外部依赖）
    """
    try:
        nvtx_spec = importlib.util.find_spec("nvtx")  # 注释：查找 nvtx 模块
    except ModuleNotFoundError:
        nvtx_spec = None  # 注释：模块不存在
    return nvtx_spec is not None  # 注释：返回可用性


@cache
def is_trl_available():
    """
    检测 trl 是否可用。（注释：函数用途）

    返回：bool。（注释：返回说明）
    副作用：无，结果缓存。（注释：副作用）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::is_trl_available`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.util.find_spec`。（注释：外部依赖）
    """
    try:
        trl_spec = importlib.util.find_spec("trl")  # 注释：查找 trl 模块
    except ModuleNotFoundError:
        trl_spec = None  # 注释：模块不存在
    return trl_spec is not None  # 注释：返回可用性


def import_external_libs(external_libs=None):
    """
    批量导入外部库（字符串或列表）。（注释：函数用途）

    参数：
      - external_libs：库名或库名列表。（注释：输入说明）
    返回：无。（注释：仅副作用）
    副作用：触发 import 并可能抛出 ImportError。（注释：副作用）
    异常/边界条件：external_libs 为 None 时直接返回。（注释：边界条件）
    最小示例：
      - 输入：["numpy", "torch"]。（注释：示例输入）
      - 输出：模块被导入。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::import_external_libs`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.import_module`。（注释：外部依赖）
    """
    if external_libs is None:  # 注释：无输入则直接返回
        return
    if not isinstance(external_libs, list):  # 注释：统一为列表
        external_libs = [external_libs]
    import importlib  # 注释：局部导入以避免污染命名空间

    for external_lib in external_libs:  # 注释：逐个导入
        importlib.import_module(external_lib)  # 注释：动态导入库


PKG_PATH_PREFIX = "pkg://"  # 注释：包路径前缀
FILE_PATH_PREFIX = "file://"  # 注释：文件路径前缀


def load_module(module_path: str, module_name: Optional[str] = None) -> object:
    """
    从包路径或文件路径加载模块。（注释：函数用途）

    参数：
      - module_path (str)：`pkg://` 或 `file://` 前缀的路径，或直接文件路径。（注释：输入说明）
      - module_name (str, optional)：注册到 `sys.modules` 的名称。（注释：输入说明）
    返回：
      - object：加载后的模块对象。（注释：返回说明）
    副作用：
      - 若提供 module_name，会写入 `sys.modules`。（注释：副作用）
    异常/边界条件：
      - 路径不存在抛 FileNotFoundError。（注释：边界条件）
      - spec 创建失败抛 ImportError。（注释：边界条件）
    最小示例：
      - 输入：`pkg://verl.utils.dataset.rl_dataset`。（注释：示例输入）
      - 输出：对应模块对象。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::load_module`。（注释：定位）
      - 被谁调用：`load_extern_object`、`load_extern_type`。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.import_module`、`importlib.util.spec_from_file_location`。（注释：外部依赖）
    """
    if not module_path:
        return None  # 注释：空路径直接返回

    if module_path.startswith(PKG_PATH_PREFIX):  # 注释：包路径模式
        module_name = module_path[len(PKG_PATH_PREFIX) :].replace("/", ".")  # 注释：转换为 Python 模块名
        module = importlib.import_module(module_name)  # 注释：导入模块

    else:
        if module_path.startswith(FILE_PATH_PREFIX):  # 注释：文件路径前缀
            module_path = module_path[len(FILE_PATH_PREFIX) :]  # 注释：去除前缀

        if not os.path.exists(module_path):  # 注释：路径不存在直接报错
            raise FileNotFoundError(f"Custom module file not found: {module_path=}")

        # Use the provided module_name for the spec, or derive a unique name to avoid collisions.
        spec_name = module_name or f"custom_module_{hash(os.path.abspath(module_path))}"  # 注释：构造 spec 名称
        spec = importlib.util.spec_from_file_location(spec_name, module_path)  # 注释：创建模块 spec
        if spec is None or spec.loader is None:  # 注释：spec 失败
            raise ImportError(f"Could not load module from {module_path=}")

        module = importlib.util.module_from_spec(spec)  # 注释：从 spec 构建模块对象
        try:
            spec.loader.exec_module(module)  # 注释：执行模块代码
        except Exception as e:
            raise RuntimeError(f"Error loading module from {module_path=}") from e  # 注释：包装异常

        if module_name is not None:  # 注释：如需注册到 sys.modules
            import sys  # 注释：延迟导入 sys

            # Avoid overwriting an existing module with a different object.
            if module_name in sys.modules and sys.modules[module_name] is not module:  # 注释：避免覆盖不同对象
                raise RuntimeError(
                    f"Module name '{module_name}' already in `sys.modules` and points to a different module."
                )
            sys.modules[module_name] = module  # 注释：注册模块

    return module  # 注释：返回模块对象


def _get_qualified_name(func):
    """
    获取函数/类的完全限定名（module + qualname）。（注释：函数用途）

    参数：
      - func：函数或类对象。（注释：输入说明）
    返回：
      - str：完全限定名。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::_get_qualified_name`。（注释：定位）
      - 被谁调用：`deprecated` 装饰器。（注释：调用方）
      - 调用了谁（外部依赖）：`__module__`/`__qualname__`。（注释：外部依赖）
    """
    module = func.__module__  # 注释：模块名
    qualname = func.__qualname__  # 注释：限定名
    return f"{module}.{qualname}"  # 注释：拼接返回


def deprecated(replacement: str = ""):
    """
    标记函数/类为弃用的装饰器。（注释：函数用途）

    参数：
      - replacement (str)：替代建议。（注释：输入说明）
    返回：
      - decorator：用于包装函数/类。（注释：返回说明）
    副作用：调用时发出 FutureWarning。（注释：副作用）
    最小示例：
      - `@deprecated("new_func")`。（注释：示例）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::deprecated`。（注释：定位）
      - 被谁调用：本模块 `load_extern_type`。（注释：调用方）
      - 调用了谁（项目内）：`_get_qualified_name`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`warnings.warn`。（注释：外部依赖）
    """

    def decorator(obj):
        """内部装饰器：包装类或函数以发出弃用警告。（注释：内部函数用途）"""
        qualified_name = _get_qualified_name(obj)  # 注释：生成完全限定名

        if isinstance(obj, type):  # 注释：处理类
            original_init = obj.__init__  # 注释：保存原始 __init__

            @wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                """包装 __init__ 以发出弃用提示。（注释：内部函数用途）"""
                msg = f"Warning: Class '{qualified_name}' is deprecated."  # 注释：构建警告信息
                if replacement:  # 注释：追加替代提示
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)  # 注释：发出警告
                return original_init(self, *args, **kwargs)  # 注释：调用原始 __init__

            obj.__init__ = wrapped_init
            return obj

        else:

            @wraps(obj)
            def wrapped(*args, **kwargs):
                """包装函数以发出弃用提示。（注释：内部函数用途）"""
                msg = f"Warning: Function '{qualified_name}' is deprecated."  # 注释：构建警告信息
                if replacement:  # 注释：追加替代提示
                    msg += f" Please use '{replacement}' instead."
                warnings.warn(msg, category=FutureWarning, stacklevel=2)  # 注释：发出警告
                return obj(*args, **kwargs)  # 注释：调用原函数

            return wrapped  # 注释：返回包装后的函数

    return decorator  # 注释：返回装饰器


def load_extern_object(module_path: str, object_name: str) -> object:
    """
    从模块路径加载对象。（注释：函数用途）

    参数：
      - module_path (str)：参见 `load_module`。（注释：输入说明）
      - object_name (str)：对象名称（类/函数/常量）。（注释：输入说明）
    返回：
      - object：加载到的对象。（注释：返回说明）
    副作用：可能触发动态导入。（注释：副作用）
    异常/边界条件：对象不存在抛 AttributeError。（注释：边界条件）
    最小示例：
      - 输入：("file://.../m.py", "MyClass")。（注释：示例输入）
      - 输出：MyClass 对象。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::load_extern_object`。（注释：定位）
      - 典型调用路径：`verl/trainer/fsdp_sft_trainer.py` -> `load_extern_object`。（注释：典型路径）
      - 被谁调用：`verl/trainer/main_ppo.py`、`verl/trainer/fsdp_sft_trainer.py`、`verl/experimental/*` 等。（注释：调用方）
      - 调用了谁（项目内）：`load_module`。（注释：内部依赖）
      - 调用了谁（外部依赖）：`getattr`。（注释：外部依赖）
    """
    module = load_module(module_path)  # 注释：先加载模块

    if not hasattr(module, object_name):  # 注释：对象不存在直接报错
        raise AttributeError(f"Object not found in module: {object_name=}, {module_path=}.")

    return getattr(module, object_name)  # 注释：返回对象


def load_class_from_fqn(fqn: str, description: str = "class") -> type:
    """
    从全限定名加载类。（注释：函数用途）

    参数：
      - fqn (str)：形如 `pkg.module.ClassName` 的全限定名。（注释：输入说明）
      - description (str)：用于错误提示的描述。（注释：输入说明）
    返回：
      - type：加载到的类。（注释：返回说明）
    异常/边界条件：
      - 缺少点分隔抛 ValueError。（注释：边界条件）
      - 模块导入失败抛 ImportError。（注释：边界条件）
      - 类不存在抛 AttributeError。（注释：边界条件）
    最小示例：
      - 输入：`"verl.experimental.agent_loop.AgentLoopManager"`。（注释：示例输入）
      - 输出：AgentLoopManager 类。（注释：示例输出）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::load_class_from_fqn`。（注释：定位）
      - 典型调用路径：`verl/trainer/ppo/ray_trainer.py` -> `load_class_from_fqn`。（注释：典型路径）
      - 被谁调用：`verl/trainer/ppo/ray_trainer.py`。（注释：调用方）
      - 调用了谁（外部依赖）：`importlib.import_module`、`getattr`。（注释：外部依赖）
    """
    if "." not in fqn:  # 注释：格式校验
        raise ValueError(
            f"Invalid {description} '{fqn}'. Expected fully qualified class name (e.g., 'mypackage.module.ClassName')."
        )
    try:
        module_path, class_name = fqn.rsplit(".", 1)  # 注释：分离模块路径与类名
        module = importlib.import_module(module_path)  # 注释：导入模块
        return getattr(module, class_name)  # 注释：获取类对象
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}' for {description}: {e}") from e  # 注释：包装异常
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}") from e  # 注释：包装异常


@deprecated(replacement="load_module(file_path); getattr(module, type_name)")
def load_extern_type(file_path: str, type_name: str) -> type:
    """
    已弃用：请改用 `load_extern_object`。（注释：函数用途）

    参数：
      - file_path (str)：模块文件路径。（注释：输入说明）
      - type_name (str)：对象名称。（注释：输入说明）
    返回：
      - type：加载到的对象。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/import_utils.py::load_extern_type`。（注释：定位）
      - 被谁调用：未在仓库内直接检索到显式调用。（注释：调用方）
      - 调用了谁（项目内）：`load_extern_object`。（注释：内部依赖）
    """
    return load_extern_object(file_path, type_name)  # 注释：转发到新接口
