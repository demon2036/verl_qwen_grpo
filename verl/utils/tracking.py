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
模块用途：统一的实验跟踪/日志接口，支持多种后端（WandB/MLflow/SwanLab/TensorBoard/Console 等）。（注释：模块职责）
输入/输出：
  - 输入：实验名、配置、日志数据字典。（注释：输入说明）
  - 输出：各后端的指标、表格或文本记录。（注释：输出说明）
关键依赖：wandb/mlflow/swanlab/clearml/trackio/tensorboard（可选）。（注释：依赖说明）
典型用法（最小示例）：
  - `tracker = Tracking(project, exp, default_backend="wandb")`。（注释：实例化）
  - `tracker.log({"train/loss": 1.23}, step=10)`。（注释：记录指标）
调用路径概览：
  - 训练入口 -> `Tracking` -> 对应后端适配器。（注释：调用链路）
"""  # 注释：模块级 docstring 结束

# ===== 标准库导入 =====
import dataclasses  # 注释：dataclass 处理
import json  # 注释：JSON 序列化
import os  # 注释：路径与环境变量
from enum import Enum  # 注释：枚举类型
from functools import partial  # 注释：函数偏应用
from pathlib import Path  # 注释：路径对象
from typing import Any  # 注释：类型标注


class Tracking:
    """
    统一的实验跟踪接口，封装多种日志后端。（注释：类用途）

    属性：
      - supported_backend：支持的后端列表。（注释：属性说明）
      - logger：各后端的实例字典。（注释：属性说明）
    调用路径依赖：
      - 所在位置：`verl/utils/tracking.py::Tracking`。（注释：定位）
      - 典型调用路径：训练入口 -> `Tracking` -> `log`。（注释：典型路径）
      - 被谁调用：`verl/trainer/fsdp_sft_trainer.py`、`verl/trainer/ppo/ray_trainer.py` 等。（注释：调用方）
      - 调用了谁（项目内）：`LocalLogger`、`_MlflowLoggingAdapter`、`_TensorboardAdapter`。（注释：内部依赖）
      - 调用了谁（外部依赖）：wandb/mlflow/swanlab/clearml/tensorboard。（注释：外部依赖）
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        """
        初始化 Tracking 并按需创建后端实例。（注释：方法用途）

        参数：
          - project_name：项目名称。（注释：输入说明）
          - experiment_name：实验名称。（注释：输入说明）
          - default_backend：后端名称或列表。（注释：输入说明）
          - config：训练配置，用于传递给后端。（注释：输入说明）
        返回：无。（注释：构造函数）
        副作用：可能启动外部后端（wandb/mlflow/swanlab/clearml）。（注释：副作用）
        异常/边界条件：未知后端会触发断言。（注释：边界条件）
        最小示例：
          - 输入：default_backend="console"。（注释：示例输入）
          - 输出：仅控制台 logger。（注释：示例输出）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::Tracking.__init__`。（注释：定位）
          - 典型调用路径：训练入口初始化 -> `Tracking(...)`。（注释：典型路径）
          - 被谁调用：训练器/脚本入口。（注释：调用方）
          - 调用了谁（项目内）：`LocalLogger`、`_MlflowLoggingAdapter`、`_TensorboardAdapter`。（注释：内部依赖）
          - 调用了谁（外部依赖）：wandb/mlflow/swanlab/clearml。（注释：外部依赖）
        """
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}  # 注释：后端实例字典

        # ===== WandB / tracking 兼容 =====
        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            entity = os.environ.get("WANDB_ENTITY", None)
            wandb.init(project=project_name, name=experiment_name, entity=entity, config=config, settings=settings)
            self.logger["wandb"] = wandb  # 注释：记录 wandb 实例

        # ===== Trackio =====
        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio  # 注释：记录 trackio 实例

        # ===== MLflow =====
        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()  # 注释：包装 MLflow 适配器

        # ===== SwanLab =====
        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab  # 注释：记录 swanlab 实例

        # ===== 火山 VEMLP WandB =====
        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb  # 注释：记录 vemlp_wandb 实例

        # ===== TensorBoard =====
        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)  # 注释：记录 TB 适配器

        # ===== Console =====
        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger  # 注释：记录 console logger

        # ===== ClearML =====
        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)  # 注释：记录 ClearML 适配器

        # ===== File Logger =====
        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)  # 注释：记录文件 logger

    def log(self, data, step, backend=None):
        """
        向所有或指定后端记录日志。（注释：方法用途）

        参数：
          - data (dict)：日志数据。（注释：输入说明）
          - step (int)：当前步。（注释：输入说明）
          - backend (list|None)：限定后端列表。（注释：输入说明）
        返回：无。（注释：仅副作用）
        副作用：调用各后端的 log 方法。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::Tracking.log`。（注释：定位）
          - 典型调用路径：训练循环 -> `tracker.log(...)`。（注释：典型路径）
          - 被谁调用：训练器/评估器。（注释：调用方）
          - 调用了谁（项目内）：各 backend 的 `log` 方法。（注释：内部依赖）
          - 调用了谁（外部依赖）：wandb/mlflow/swanlab/tensorboard 等。（注释：外部依赖）
        """
        for default_backend, logger_instance in self.logger.items():  # 注释：遍历后端实例
            if backend is None or default_backend in backend:  # 注释：若未指定或在白名单
                logger_instance.log(data=data, step=step)  # 注释：记录日志

    def __del__(self):
        """
        析构时关闭各后端资源。（注释：方法用途）

        副作用：调用后端 `finish/close`。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::Tracking.__del__`。（注释：定位）
          - 被谁调用：对象生命周期结束时自动触发。（注释：调用方）
          - 调用了谁（项目内）：各 backend 的 `finish`。（注释：内部依赖）
        """
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        """
        初始化 ClearML 记录器。（注释：方法用途）

        参数：
          - project_name：项目名。（注释：输入说明）
          - experiment_name：实验名。（注释：输入说明）
          - config：配置对象。（注释：输入说明）
        返回：无。（注释：构造函数）
        副作用：创建/连接 ClearML Task。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ClearMLLogger.__init__`。（注释：定位）
          - 被谁调用：`Tracking.__init__`。（注释：调用方）
          - 调用了谁（项目内）：无。（注释：内部依赖）
          - 调用了谁（外部依赖）：`clearml.Task.init`。（注释：外部依赖）
        """
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(  # 注释：初始化 ClearML Task
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")  # 注释：上传配置

    def _get_logger(self):
        """
        获取 ClearML logger。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ClearMLLogger._get_logger`。（注释：定位）
          - 被谁调用：`ClearMLLogger.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`clearml.Task.get_logger`。（注释：外部依赖）
        """
        return self._task.get_logger()  # 注释：返回 logger

    def log(self, data, step):
        """
        将日志写入 ClearML。（注释：方法用途）

        参数：
          - data (dict)：日志数据。（注释：输入说明）
          - step (int)：步数。（注释：输入说明）
        返回：无。（注释：仅副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ClearMLLogger.log`。（注释：定位）
          - 被谁调用：`Tracking.log`、`ValidationGenerationsLogger.log_generations_to_clearml`。（注释：调用方）
          - 调用了谁（项目内）：`ClearMLLogger._get_logger`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`clearml.Task.get_logger`。（注释：外部依赖）
        """
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()  # 注释：获取 ClearML logger
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(  # 注释：记录标量
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(  # 注释：记录表格
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(  # 注释：类型不支持时给出提示
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        """
        关闭 ClearML Task。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ClearMLLogger.finish`。（注释：定位）
          - 被谁调用：`Tracking.__del__`。（注释：调用方）
          - 调用了谁（外部依赖）：`clearml.Task.close`。（注释：外部依赖）
        """
        self._task.close()  # 注释：关闭任务


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        """
        初始化文件日志记录器。（注释：方法用途）

        参数：
          - project_name：项目名。（注释：输入说明）
          - experiment_name：实验名。（注释：输入说明）
        返回：无。（注释：构造函数）
        副作用：打开日志文件句柄。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::FileLogger.__init__`。（注释：定位）
          - 被谁调用：`Tracking.__init__`。（注释：调用方）
          - 调用了谁（外部依赖）：`open`、`os.makedirs`。（注释：外部依赖）
        """
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "w")  # 注释：打开文件句柄

    def log(self, data, step):
        """
        将日志写入 jsonl 文件。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::FileLogger.log`。（注释：定位）
          - 被谁调用：`Tracking.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`json.dumps`、文件句柄写入。（注释：外部依赖）
        """
        data = {"step": step, "data": data}  # 注释：封装 step 与 data
        self.fp.write(json.dumps(data) + "\n")  # 注释：写入一行 JSON

    def finish(self):
        """
        关闭日志文件。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::FileLogger.finish`。（注释：定位）
          - 被谁调用：`Tracking.__del__`。（注释：调用方）
          - 调用了谁（外部依赖）：文件句柄 close。（注释：外部依赖）
        """
        self.fp.close()  # 注释：关闭文件句柄


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        """
        初始化 TensorBoard writer。（注释：方法用途）

        参数：
          - project_name/experiment_name：用于构造日志目录。（注释：输入说明）
        返回：无。（注释：构造函数）
        副作用：创建目录并打开 SummaryWriter。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::_TensorboardAdapter.__init__`。（注释：定位）
          - 被谁调用：`Tracking.__init__`。（注释：调用方）
          - 调用了谁（外部依赖）：`torch.utils.tensorboard.SummaryWriter`。（注释：外部依赖）
        """
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        """
        记录标量到 TensorBoard。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::_TensorboardAdapter.log`。（注释：定位）
          - 被谁调用：`Tracking.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`SummaryWriter.add_scalar`。（注释：外部依赖）
        """
        for key in data:  # 注释：遍历指标
            self.writer.add_scalar(key, data[key], step)  # 注释：写入标量

    def finish(self):
        """
        关闭 TensorBoard writer。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::_TensorboardAdapter.finish`。（注释：定位）
          - 被谁调用：`Tracking.__del__`。（注释：调用方）
          - 调用了谁（外部依赖）：`SummaryWriter.close`。（注释：外部依赖）
        """
        self.writer.close()  # 注释：关闭 writer


class _MlflowLoggingAdapter:
    def __init__(self):
        """
        初始化 MLflow 日志适配器。（注释：方法用途）

        副作用：编译 key 校验正则。（注释：副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::_MlflowLoggingAdapter.__init__`。（注释：定位）
          - 被谁调用：`Tracking.__init__`。（注释：调用方）
          - 调用了谁（外部依赖）：`re.compile`。（注释：外部依赖）
        """
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # 注释：允许 slashes/字母数字/下划线/点/横杠/冒号/空格
        self._consecutive_slashes_pattern = re.compile(r"/+")  # 注释：连续斜杠模式

    def log(self, data, step):
        """
        将指标写入 MLflow。（注释：方法用途）

        参数：
          - data (dict)：指标字典。（注释：输入说明）
          - step (int)：步数。（注释：输入说明）
        返回：无。（注释：仅副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::_MlflowLoggingAdapter.log`。（注释：定位）
          - 被谁调用：`Tracking.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`mlflow.log_metrics`。（注释：外部依赖）
        """
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")  # 注释：兼容旧键名
            # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
            sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)  # 注释：压缩连续斜杠
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)  # 注释：替换非法字符
            if sanitized != key:
                self.logger.warning(
                    "[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized
                )  # 注释：告警提示键名被改写
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}  # 注释：清洗键名
        mlflow.log_metrics(metrics=results, step=step)  # 注释：写入 MLflow


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    """
    将配置对象转换为 MLflow 可记录的扁平参数字典。（注释：函数用途）

    参数：
      - params：原始配置对象。（注释：输入说明）
    返回：
      - dict[str, Any]：扁平化后的参数字典。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/tracking.py::_compute_mlflow_params_from_objects`。（注释：定位）
      - 被谁调用：`Tracking.__init__`（MLflow 初始化时）。（注释：调用方）
      - 调用了谁（项目内）：`_transform_params_to_json_serializable`、`_flatten_dict`。（注释：内部依赖）
    """
    if params is None:  # 注释：无参数则返回空字典
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")  # 注释：先转 JSON 再扁平化


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    """
    将对象递归转换为 JSON 可序列化结构。（注释：函数用途）

    参数：
      - x：任意对象。（注释：输入说明）
      - convert_list_to_dict (bool)：是否将 list 转为 dict。（注释：输入说明）
    返回：可序列化对象。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/tracking.py::_transform_params_to_json_serializable`。（注释：定位）
      - 被谁调用：`_compute_mlflow_params_from_objects`。（注释：调用方）
      - 调用了谁（外部依赖）：`dataclasses.asdict`。（注释：外部依赖）
    """
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)  # 注释：递归函数

    if dataclasses.is_dataclass(x):  # 注释：dataclass 转 dict
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):  # 注释：字典递归转换
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):  # 注释：列表转换
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):  # 注释：Path 转字符串
        return str(x)
    if isinstance(x, Enum):  # 注释：Enum 转值
        return x.value

    return x  # 注释：默认返回原值


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    """
    使用 pandas 将嵌套字典扁平化。（注释：函数用途）

    参数：
      - raw (dict)：嵌套字典。（注释：输入说明）
      - sep (str)：层级分隔符。（注释：输入说明）
    返回：
      - dict：扁平化后的字典。（注释：返回说明）
    调用路径依赖：
      - 所在位置：`verl/utils/tracking.py::_flatten_dict`。（注释：定位）
      - 被谁调用：`_compute_mlflow_params_from_objects`。（注释：调用方）
      - 调用了谁（外部依赖）：`pandas.json_normalize`。（注释：外部依赖）
    """
    import pandas as pd  # 注释：依赖 pandas 扁平化

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]  # 注释：扁平化并取第一条
    assert isinstance(ans, dict)  # 注释：类型断言
    return ans  # 注释：返回结果


@dataclasses.dataclass
class ValidationGenerationsLogger:
    """
    用于记录验证集生成结果的日志工具。（注释：类用途）

    参数：
      - project_name/experiment_name：用于 TensorBoard 路径。（注释：参数说明）
    调用路径依赖：
      - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger`。（注释：定位）
      - 被谁调用：训练/评估流程（如 validation loop）。（注释：调用方）
      - 调用了谁（项目内）：`_log_generations_to_wandb` 等。（注释：内部依赖）
    """
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        """
        根据已启用的后端记录生成结果。（注释：方法用途）

        参数：
          - loggers (dict)：启用的 logger 字典。（注释：输入说明）
          - samples (list)：样本列表，每个样本为 [input, output, score]。（注释：输入说明）
          - step (int)：当前步。（注释：输入说明）
        返回：无。（注释：仅副作用）
        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log`。（注释：定位）
          - 被谁调用：验证/评估流程。（注释：调用方）
          - 调用了谁（项目内）：`log_generations_to_*` 系列方法。（注释：内部依赖）
        """
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        """
        使用 VEMLP WandB 记录生成样本。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log_generations_to_vemlp_wandb`。（注释：定位）
          - 被谁调用：`ValidationGenerationsLogger.log`。（注释：调用方）
          - 调用了谁（项目内）：`_log_generations_to_wandb`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`volcengine_ml_platform.wandb`。（注释：外部依赖）
        """
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        """
        使用 WandB 记录生成样本。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log_generations_to_wandb`。（注释：定位）
          - 被谁调用：`ValidationGenerationsLogger.log`。（注释：调用方）
          - 调用了谁（项目内）：`_log_generations_to_wandb`。（注释：内部依赖）
          - 调用了谁（外部依赖）：`wandb`。（注释：外部依赖）
        """
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """
        将样本以表格形式记录到 wandb。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger._log_generations_to_wandb`。（注释：定位）
          - 被谁调用：`log_generations_to_wandb`、`log_generations_to_vemlp_wandb`。（注释：调用方）
          - 调用了谁（外部依赖）：`wandb.Table`、`wandb.log`。（注释：外部依赖）
        """

        # Create column names for all samples
        columns = ["step"] + sum(  # 注释：构造表头列
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):  # 注释：首次调用初始化表
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)  # 注释：复制旧表数据

        # Add new row with all data
        row_data = []  # 注释：构造行数据
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        if wandb.run is not None:  # 注释：确保有运行上下文
            wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """
        以表格形式记录到 SwanLab。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log_generations_to_swanlab`。（注释：定位）
          - 被谁调用：`ValidationGenerationsLogger.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`swanlab.log`。（注释：外部依赖）
        """
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]  # 注释：表头

        swanlab_row_list = [[step, *sample] for sample in samples]  # 注释：构造行
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)  # 注释：记录日志

    def log_generations_to_mlflow(self, samples, step):
        """
        将验证生成结果写入 MLflow 作为 artifact。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log_generations_to_mlflow`。（注释：定位）
          - 被谁调用：`ValidationGenerationsLogger.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`mlflow.log_artifact`。（注释：外部依赖）
        """
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:  # 注释：捕获写入异常
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:  # 注释：写入临时文件
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)  # 注释：上传 artifact
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """
        将验证生成结果记录到 ClearML 表格。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log_generations_to_clearml`。（注释：定位）
          - 被谁调用：`ValidationGenerationsLogger.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`clearml.Task.get_logger`。（注释：外部依赖）
        """

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()  # 注释：获取当前任务
        if task is None:
            return

        table = [  # 注释：构造表格数据
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()  # 注释：获取 logger
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """
        将生成结果写入 TensorBoard 文本。（注释：方法用途）

        调用路径依赖：
          - 所在位置：`verl/utils/tracking.py::ValidationGenerationsLogger.log_generations_to_tensorboard`。（注释：定位）
          - 被谁调用：`ValidationGenerationsLogger.log`。（注释：调用方）
          - 调用了谁（外部依赖）：`SummaryWriter.add_text`。（注释：外部依赖）
        """
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:  # 注释：构造默认目录
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)  # 注释：创建 writer

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"  # 注释：文本头

        for i, sample in enumerate(samples):  # 注释：逐条样本拼接文本
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:  # 注释：标准样本格式
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:  # 注释：非标准格式兜底
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)  # 注释：写入文本
        # Flush to ensure data is written
        self.writer.flush()  # 注释：刷新写入
