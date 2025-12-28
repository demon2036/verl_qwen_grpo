# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：分隔说明，保持逐行注释
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：获取许可证地址提示
#  # 注释：空行占位，保持逐行注释
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位，保持逐行注释
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无明示或暗示担保
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明
"""
模块用途：统一封装 vLLM 版本检测与兼容导入，兼顾 SGLang 回退。  # 注释：模块用途
输入/输出：输入包名获取版本；输出 LLM/parallel_state 与 sleep level 配置。  # 注释：输入输出概览
关键依赖：importlib.metadata、packaging.version、verl.utils.device/is_sglang_available。  # 注释：关键依赖说明
典型用法：  # 注释：最小用法示例标题
- from verl.third_party.vllm import LLM, parallel_state, VLLM_SLEEP_LEVEL  # 注释：示例用法
调用路径概览：  # 注释：调用路径说明标题
- 入口示例：verl/workers/rollout/vllm_rollout/vllm_rollout.py。  # 注释：上层入口举例
- 典型链路：rollout 初始化 -> import 本模块 -> 选择 vLLM/SGLang。  # 注释：调用链路
"""  # 注释：模块 docstring 结束
# （分隔说明：标准库依赖）  # 注释：替代空行，保持逐行注释
from importlib.metadata import PackageNotFoundError, version  # 注释：读取包版本
# （分隔说明：第三方依赖）  # 注释：替代空行，保持逐行注释
from packaging import version as vs  # 注释：版本比较工具
# （分隔说明：项目内依赖）  # 注释：替代空行，保持逐行注释
from verl.utils.device import is_npu_available  # 注释：是否 NPU 环境
from verl.utils.import_utils import is_sglang_available  # 注释：是否可用 SGLang
# （分隔说明：版本获取工具）  # 注释：替代空行，保持逐行注释
def get_version(pkg):  # 注释：获取指定包版本
    """安全获取包版本号。  # 注释：函数用途

    参数：pkg (str)：包名。  # 注释：参数说明
    返回：str 或 None。  # 注释：返回值语义
    副作用：无。  # 注释：副作用说明
    异常/边界条件：包不存在则返回 None。  # 注释：异常说明
    最小示例：get_version("vllm")。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/third_party/vllm/__init__.py::get_version。  # 注释：位置
    - 典型调用路径：模块初始化 -> get_version。  # 注释：调用链
    - 被谁调用：本文件顶部初始化逻辑。  # 注释：调用方
    - 调用了谁（项目内）：无。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：importlib.metadata.version。  # 注释：外部依赖
    """  # 注释：docstring 结束
    try:  # 注释：捕获包不存在异常
        return version(pkg)  # 注释：读取版本
    except PackageNotFoundError:  # 注释：包不存在
        return None  # 注释：返回 None
# （分隔说明：版本与兼容逻辑）  # 注释：替代空行，保持逐行注释
package_name = "vllm"  # 注释：目标包名
package_version = get_version(package_name)  # 注释：读取 vllm 版本
vllm_version = None  # 注释：记录可用版本
VLLM_SLEEP_LEVEL = 1  # 注释：默认 sleep level
# （分隔说明：兼容分支）  # 注释：替代空行，保持逐行注释
if package_version is None:  # 注释：未安装 vllm
    if not is_sglang_available():  # 注释：SGLang 也不可用
        raise ValueError(  # 注释：抛出错误
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "  # 注释：错误信息
            f"vllm versions are 0.7.0+"  # 注释：支持版本提示
        )  # 注释：错误结束
elif is_npu_available:  # 注释：NPU 场景
    # sleep_mode=2 is not supported on vllm-ascend for now, will remove this restriction when this ability is ready.  # 注释：原注释保留
    VLLM_SLEEP_LEVEL = 1  # 注释：NPU 固定为 1
    from vllm import LLM  # 注释：导入 vllm LLM
    from vllm.distributed import parallel_state  # 注释：导入并行状态
elif vs.parse(package_version) >= vs.parse("0.7.0"):  # 注释：vllm 版本满足最低要求
    vllm_version = package_version  # 注释：记录版本
    if vs.parse(package_version) >= vs.parse("0.8.5"):  # 注释：更高版本支持 sleep_level=2
        VLLM_SLEEP_LEVEL = 2  # 注释：设置更高 sleep level
    from vllm import LLM  # 注释：导入 vllm LLM
    from vllm.distributed import parallel_state  # 注释：导入并行状态
else:  # 注释：版本过低
    if vs.parse(package_version) in [vs.parse("0.5.4"), vs.parse("0.6.3")]:  # 注释：已移除支持的旧版本
        raise ValueError(  # 注释：抛出错误
            f"vLLM version {package_version} support has been removed. vLLM 0.5.4 and 0.6.3 are no longer "  # 注释：错误信息
            f"supported. Please use vLLM 0.7.0 or later."  # 注释：升级提示
        )  # 注释：错误结束
    if not is_sglang_available():  # 注释：无 SGLang 兜底
        raise ValueError(  # 注释：抛出错误
            f"vllm version {package_version} not supported and SGLang also not Found. Currently supported "  # 注释：错误信息
            f"vllm versions are 0.7.0+"  # 注释：支持版本提示
        )  # 注释：错误结束
# （分隔说明：对外导出）  # 注释：替代空行，保持逐行注释
__all__ = ["LLM", "parallel_state"]  # 注释：公开导出符号
