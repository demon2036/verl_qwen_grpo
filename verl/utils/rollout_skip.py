# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""（模块说明：Rollout 缓存跳过机制，用于加速调试和实验复现）

模块用途：
    本模块实现了 Rollout 跳过（RolloutSkip）机制，用于在 RLHF/PPO 训练的 rollout 阶段缓存生成的序列数据。
    主要功能包括：
    1. 尝试从磁盘加载之前缓存的 rollout 结果（prompt + response）
    2. 如果缓存不存在，执行正常的 rollout 生成流程，并将结果保存到磁盘
    3. 下次运行时可以直接加载缓存，跳过耗时的生成过程

    适用场景：
    - 调试训练流程时，避免重复生成相同的 rollout 数据
    - 复现实验时，确保使用完全相同的 rollout 结果
    - 快速迭代奖励函数或训练超参数，而不改变 rollout 数据

输入/输出：
    - 输入：
      * config: 训练配置对象（包含 rollout.skip_dump_dir、data.experiment_name 等）
      * rollout_wg: Rollout worker group（负责执行序列生成的 worker 组）
      * batch: 输入批次数据（prompt 数据，由 RL 数据集提供）
    - 输出：
      * DataProto 对象：包含生成的 rollout 数据（prompt、response、logprobs 等）
      * 副作用：如果缓存不存在，会将生成结果保存到磁盘

关键依赖：
    - pathlib.Path：用于跨平台的路径操作
    - verl.protocol.DataProto：统一的数据协议类，支持序列化与反序列化
    - rollout_wg.generate_sequences：实际执行 rollout 生成的函数（被本模块包装）

典型用法：
    >>> from verl.utils.rollout_skip import RolloutSkip
    >>> # 初始化 RolloutSkip 对象
    >>> rollout_skip = RolloutSkip(config, rollout_wg)
    >>> # 包装 rollout_wg 的 generate_sequences 方法
    >>> rollout_skip.wrap_generate_sequences()
    >>> # 后续调用 rollout_wg.generate_sequences 会自动尝试加载缓存
    >>> outputs = rollout_wg.generate_sequences(batch)

调用路径概览：
    入口脚本（如 verl/trainer/main_ppo.py）
    -> verl.trainer.ppo.ray_trainer.PPOTrainer.fit()
    -> （可选）初始化 RolloutSkip 并包装 rollout_wg.generate_sequences
    -> 训练循环中调用 rollout_wg.generate_sequences(batch)
    -> RolloutSkip.wrap_generate_sequences() 包装后的函数
       -> 尝试加载缓存 RolloutSkip.try_load()
       -> 如果缓存不存在，调用原始 generate_sequences() 并缓存结果
    -> 返回 DataProto 对象（可能来自缓存或新生成）

所在位置：
    - 路径：verl/utils/rollout_skip.py
    - 模块：verl.utils.rollout_skip

被谁调用：
    - verl/trainer/ppo/ray_trainer.py（可选启用，用于调试或复现实验）
    - 用户自定义训练脚本（手动启用 rollout skip 功能）

调用了谁（项目内）：
    - verl.protocol.DataProto.load_from_disk()（从磁盘加载缓存数据）
    - verl.protocol.DataProto.save_to_disk()（将数据保存到磁盘）

调用了谁（外部依赖）：
    - pathlib.Path：路径操作（创建目录、检查文件是否存在等）
    - Python 内置 print：输出日志信息

注意事项：
    1. 缓存文件名格式：{exp_name}_{project_name}_GBS{gen_batch_size}__N{n}
       - 参数变化时会生成不同的缓存文件，避免混淆
    2. 不建议使用 /tmp/ray/session* 作为缓存路径（Ray 会话结束后可能被删除）
    3. 缓存加载失败时会自动回退到正常生成流程
    4. 本模块通过 monkey patch 方式包装 rollout_wg.generate_sequences 方法
"""  # 注释：模块级 docstring 结束，下面是依赖导入

# 注释：导入 Path 类，用于跨平台的路径操作（创建目录、拼接路径、检查文件是否存在等）
from pathlib import Path  # 注释：Path 提供面向对象的路径 API，比 os.path 更易用

# 注释：导入 DataProto 类，这是 VERL 项目中统一的数据协议类，用于序列化/反序列化 rollout 数据
from verl.protocol import DataProto  # 注释：DataProto 封装了 rollout 的所有数据（prompt、response、logprobs 等），支持保存到磁盘和从磁盘加载


class RolloutSkip:  # 注释：类定义开始，RolloutSkip 用于管理 rollout 结果的缓存与加载
    """（类说明：Rollout 缓存跳过类，通过缓存生成结果加速调试和实验复现）

    本类提供 rollout 结果的缓存机制，允许在调试或实验复现时跳过耗时的序列生成过程。
    工作原理：
    1. 根据配置参数（experiment_name、gen_batch_size、n 等）生成唯一的缓存文件名
    2. 首次运行时，执行正常的 rollout 生成，并将结果保存到磁盘
    3. 后续运行时，尝试从磁盘加载缓存数据，如果加载成功则直接返回，否则重新生成

    参数：
        config: 训练配置对象（通常来自 Hydra），包含以下关键字段：
            - config.actor_rollout_ref.rollout.n: 每个 prompt 生成的 response 数量（整数）
            - config.actor_rollout_ref.rollout.skip_dump_dir: 缓存目录路径（字符串或 Path）
            - config.data.experiment_name: 实验名称，用于生成唯一的缓存文件名
            - config.data.project_name: 项目名称，用于生成唯一的缓存文件名
            - config.data.gen_batch_size: 生成批次大小（整数）
        rollout_wg: Rollout worker group 对象，包含 generate_sequences 方法
            - 类型：WorkerGroup（来自 verl.single_controller.base.worker_group）
            - 必须包含 generate_sequences(batch, **kwargs) 方法

    副作用：
        - 创建缓存目录（如果不存在）
        - 可能会打印警告信息（如果缓存路径不推荐使用）
        - 通过 wrap_generate_sequences() 方法修改 rollout_wg.generate_sequences 的行为

    异常/边界条件：
        - 如果缓存目录无法创建（权限不足等），会抛出 OSError
        - 如果配置参数缺失，会抛出 KeyError 或 AttributeError
        - 如果 rollout_wg 没有 generate_sequences 方法，wrap_generate_sequences() 会失败

    最小示例：
        >>> from verl.utils.rollout_skip import RolloutSkip
        >>> # 假设 config 和 rollout_wg 已准备好
        >>> rollout_skip = RolloutSkip(config, rollout_wg)
        >>> rollout_skip.wrap_generate_sequences()
        >>> # 之后调用 rollout_wg.generate_sequences 会自动尝试加载缓存
        >>> outputs = rollout_wg.generate_sequences(batch)

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/rollout_skip.py
        - 类：RolloutSkip

    典型调用路径：
        verl/trainer/main_ppo.py (入口脚本)
        -> verl/trainer/ppo/ray_trainer.py::PPOTrainer.__init__()
        -> （可选）初始化 RolloutSkip(config, rollout_wg)
        -> rollout_skip.wrap_generate_sequences()
        -> 训练循环中调用 rollout_wg.generate_sequences()

    被谁调用：
        - verl/trainer/ppo/ray_trainer.py（可选启用）
        - 用户自定义训练脚本

    调用了谁（项目内）：
        - verl.protocol.DataProto.load_from_disk()（加载缓存）
        - verl.protocol.DataProto.save_to_disk()（保存缓存）

    调用了谁（外部依赖）：
        - pathlib.Path（路径操作）
        - print（日志输出）

    记忆提示：
        - 缓存文件名 = {exp_name}_{project_name}_GBS{gbs}__N{n}
        - 参数变化时会自动生成新的缓存文件
        - 通过 monkey patch 包装 generate_sequences 方法
    """  # 注释：类 docstring 结束，下面是类变量与方法定义

    # 注释：类变量，用于在所有实例的打印信息中添加统一的前缀标记
    print_mark = "[RolloutSkip()]"  # 注释：打印标记，便于在日志中识别来自 RolloutSkip 的消息

    def __init__(self, config, rollout_wg):  # 注释：构造函数，初始化 RolloutSkip 实例
        """（构造函数说明：初始化 RolloutSkip 对象，设置缓存路径和相关参数）

        参数：
            config: 训练配置对象
            rollout_wg: Rollout worker group 对象

        副作用：
            - 创建缓存目录（如果不存在）
            - 打印缓存路径信息
            - 如果缓存路径在 /tmp/ray/session* 下，打印警告
        """  # 注释：构造函数 docstring 结束

        # 注释：提取 rollout 相关配置（如 n、skip_dump_dir 等）
        self.rollout_config = config.actor_rollout_ref.rollout  # 注释：rollout_config 包含生成参数和缓存配置

        # 注释：提取实验名称，用于生成唯一的缓存文件名（默认为空字符串）
        self.exp_name = config.data.get("experiment_name", "")  # 注释：experiment_name 用于区分不同实验的缓存文件

        # 注释：提取项目名称，用于生成唯一的缓存文件名（默认为空字符串）
        self.project_name = config.data.get("project_name", "")  # 注释：project_name 用于区分不同项目的缓存文件

        # 注释：提取生成数量参数 n（每个 prompt 生成几个 response），转换为整数（默认为 0）
        self.n = int(self.rollout_config.get("n", 0))  # 注释：n 是 rollout 的关键参数，不同 n 值会生成不同的缓存文件

        # 注释：提取生成批次大小（gen_batch_size），如果没有则使用 train_batch_size，转换为整数（默认为 0）
        self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))  # 注释：gbs 影响缓存文件名，确保批次大小匹配

        # 注释：提取缓存目录路径（skip_dump_dir），默认为 /tmp/verl/rollout_dump，转换为 Path 对象
        self.dumped_dir = Path(self.rollout_config.get("skip_dump_dir", "/tmp/verl/rollout_dump"))  # 注释：dumped_dir 是缓存文件的存储目录

        # 注释：创建缓存目录（如果不存在），parents=True 表示创建所有父目录，exist_ok=True 表示目录已存在时不报错
        self.dumped_dir.mkdir(parents=True, exist_ok=True)  # 注释：确保缓存目录存在，避免后续保存时失败

        # 注释：检查缓存路径是否在 Ray 临时目录下（/tmp/ray/session*）
        if str(self.dumped_dir.absolute()).startswith("/tmp/ray/session"):  # 注释：Ray 会话结束后可能删除该目录，导致缓存丢失
            # 注释：打印黄色警告信息（\033[33m 开始，\033[0m 结束）
            print(
                f"\033[33m{self.print_mark} Warning: \nUsing dump path ",
                f"'{self.dumped_dir.absolute()}' is not recommended ",
                "as it's located in /tmp/ray/session*\033[0m",
                flush=True,  # 注释：flush=True 确保立即输出，不缓冲
            )

        # 注释：打印缓存路径信息（正常颜色）
        print(
            f"{self.print_mark} Rollout skip dump path set to: ",
            f"{self.dumped_dir.absolute()}",  # 注释：使用绝对路径，避免相对路径歧义
            flush=True,
        )

        # 注释：保存 rollout worker group 对象的引用，后续用于包装其 generate_sequences 方法
        self._rollout_wg = rollout_wg  # 注释：_rollout_wg 是内部属性，存储 worker group 对象

    @property  # 注释：装饰器，将方法转换为只读属性，调用时无需括号
    def curr_path_dump(self):  # 注释：属性方法，返回当前配置对应的缓存文件路径
        """（属性说明：返回当前配置对应的缓存文件路径）

        返回：
            Path: 缓存文件的绝对路径，格式为 {dumped_dir}/{exp_name}_{project_name}_GBS{gbs}__N{n}

        最小示例：
            >>> rollout_skip = RolloutSkip(config, rollout_wg)
            >>> # 假设 exp_name="debug", project_name="gsm8k", gbs=4, n=8
            >>> print(rollout_skip.curr_path_dump)
            /tmp/verl/rollout_dump/debug_gsm8k_GBS4__N8
        """  # 注释：属性 docstring 结束

        # 注释：拼接缓存目录和文件名，生成完整的缓存文件路径
        # 注释：文件名格式：{exp_name}_{project_name}_GBS{gbs}__N{n}，确保不同配置使用不同的缓存文件
        return self.dumped_dir.joinpath(f"{self.exp_name}_{self.project_name}_GBS{self.gbs}__N{self.n}").absolute()  # 注释：返回绝对路径，避免相对路径问题

    def wrap_generate_sequences(self):  # 注释：方法定义，包装 rollout_wg 的 generate_sequences 方法
        """（方法说明：包装 rollout_wg 的 generate_sequences 方法，添加缓存加载和保存逻辑）

        本方法通过 monkey patch 方式修改 rollout_wg.generate_sequences 的行为：
        1. 在原始方法调用前，尝试加载缓存数据
        2. 如果缓存存在，直接返回缓存结果，跳过生成
        3. 如果缓存不存在，调用原始方法生成数据，并将结果保存到缓存

        副作用：
            - 修改 self._rollout_wg.generate_sequences 的实现
            - 打印成功或失败信息

        异常/边界条件：
            - 如果 rollout_wg 没有 generate_sequences 方法，会抛出 AttributeError
            - 如果包装过程失败，会抛出 RuntimeError

        最小示例：
            >>> rollout_skip = RolloutSkip(config, rollout_wg)
            >>> rollout_skip.wrap_generate_sequences()
            [RolloutSkip()] Successfully patched `actor_rollout_wg.generate_sequences()`
        """  # 注释：方法 docstring 结束

        try:  # 注释：try 块，捕获可能的异常
            # 注释：调用 wrap_generate_sequences 函数（定义在模块末尾），将原始方法包装为新方法
            self._rollout_wg.generate_sequences = wrap_generate_sequences(self, self._rollout_wg)  # 注释：替换 rollout_wg.generate_sequences 为包装后的版本

            # 注释：打印成功信息
            print(
                f"{self.print_mark} Successfully patched `actor_rollout_wg.generate_sequences()`",
                flush=True,
            )

        except Exception as e:  # 注释：捕获任何异常（如 AttributeError、TypeError 等）
            # 注释：抛出 RuntimeError，包含错误信息和原始异常
            raise RuntimeError(
                f"{self.print_mark} Failed to patch `actor_rollout_wg.generate_sequences()`",  # 注释：修正格式化字符串错误
                flush=True,
            ) from e  # 注释：使用 from e 保留原始异常的堆栈信息

    def try_load(self):  # 注释：方法定义，尝试从磁盘加载缓存数据
        """（方法说明：尝试从磁盘加载之前缓存的 rollout 数据）

        返回：
            DataProto | None:
                - 如果缓存文件存在且加载成功，返回 DataProto 对象
                - 如果缓存文件不存在或加载失败，返回 None

        副作用：
            - 打印加载状态信息（成功、失败或文件不存在）

        最小示例：
            >>> rollout_skip = RolloutSkip(config, rollout_wg)
            >>> cached_data = rollout_skip.try_load()
            >>> if cached_data is None:
            ...     print("缓存不存在，需要重新生成")
            ... else:
            ...     print(f"成功加载缓存，包含 {len(cached_data)} 个样本")
        """  # 注释：方法 docstring 结束

        # 注释：检查缓存文件是否存在
        if not self.curr_path_dump.exists():  # 注释：使用 Path.exists() 检查文件是否存在
            # 注释：打印提示信息，说明缓存不存在，将自动生成并保存
            print(
                f"{self.print_mark} No data dump found at {self.curr_path_dump}.",
                "The trainer will generate and automatically dump the data for this first run.",
                flush=True,
            )
            # 注释：返回 None，表示缓存不存在
            return None  # 注释：调用者需要根据 None 判断是否需要重新生成

        try:  # 注释：try 块，尝试加载缓存数据
            # 注释：调用 DataProto.load_from_disk 静态方法，从磁盘加载缓存数据
            ret_batch = DataProto.load_from_disk(self.curr_path_dump)  # 注释：load_from_disk 返回 DataProto 对象

            # 注释：打印绿色成功信息（\033[32m 开始，\033[0m 结束）
            print(
                f"\033[32m{self.print_mark} Successfully load pre-generated data from {self.curr_path_dump}\033[0m",
                flush=True,
            )

            # 注释：返回加载的 DataProto 对象
            return ret_batch  # 注释：包含 prompt、response、logprobs 等完整的 rollout 数据

        except Exception as e:  # 注释：捕获加载过程中的任何异常（如文件损坏、反序列化失败等）
            # 注释：打印红色错误信息（\033[31m 开始，\033[0m 结束）
            print(
                f"\033[31m{self.print_mark} Failed to load pre-generated data from {self.curr_path_dump}",
                f"Error: {str(e)}\033[0m",  # 注释：输出异常详细信息
                flush=True,
            )
            # 注释：返回 None，表示加载失败，需要重新生成
            return None  # 注释：调用者会自动回退到正常生成流程

    def dump(self, outputs: DataProto):  # 注释：方法定义，将 rollout 结果保存到磁盘
        """（方法说明：将 rollout 生成结果保存到磁盘缓存）

        参数：
            outputs (DataProto): rollout 生成的数据，包含 prompt、response、logprobs 等

        副作用：
            - 将 outputs 保存到 self.curr_path_dump 对应的磁盘文件
            - 打印保存状态信息（成功或失败）

        异常/边界条件：
            - 如果磁盘空间不足，保存会失败
            - 如果权限不足，保存会失败
            - 异常不会向上抛出，只打印错误信息

        最小示例：
            >>> rollout_skip = RolloutSkip(config, rollout_wg)
            >>> outputs = rollout_wg.generate_sequences(batch)
            >>> rollout_skip.dump(outputs)
            [RolloutSkip()] Successfully dump data in /tmp/verl/rollout_dump/...
        """  # 注释：方法 docstring 结束

        try:  # 注释：try 块，尝试保存数据
            # 注释：调用 DataProto.save_to_disk 方法，将数据序列化并保存到磁盘
            outputs.save_to_disk(self.curr_path_dump)  # 注释：save_to_disk 会将 DataProto 对象序列化为文件

            # 注释：打印绿色成功信息（\033[32m 开始，\033[0m 结束）
            print(
                f"\033[32m{self.print_mark} Successfully dump data in {self.curr_path_dump}\033[0m",
                flush=True,
            )

        except Exception as e:  # 注释：捕获保存过程中的任何异常（如磁盘空间不足、权限不足等）
            # 注释：打印红色错误信息（\033[31m 开始，\033[0m 结束）
            print(
                f"\033[31m{self.print_mark} Failed to dump data in {self.curr_path_dump}: {e}\033[0m",
                flush=True,
            )
            # 注释：异常不向上抛出，只记录日志，避免中断训练流程


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):  # 注释：模块级函数，创建包装后的 generate_sequences 函数
    """（函数说明：创建包装版本的 generate_sequences 函数，添加缓存逻辑）

    本函数是一个高阶函数（higher-order function），接收 RolloutSkip 对象和 rollout_wg，
    返回一个新函数 warp_fn，该函数会在调用原始 generate_sequences 前尝试加载缓存。

    参数：
        rolloutskip (RolloutSkip): RolloutSkip 对象，提供 try_load 和 dump 方法
        rollout_wg: Rollout worker group 对象，包含原始的 generate_sequences 方法

    返回：
        function: 包装后的 generate_sequences 函数，签名与原始函数相同

    工作原理：
        1. 保存原始 generate_sequences 方法的引用
        2. 定义新函数 warp_fn，在调用原始方法前尝试加载缓存
        3. 如果缓存存在，直接返回缓存结果
        4. 如果缓存不存在，调用原始方法生成数据，并保存到缓存
        5. 返回生成或加载的数据

    调用路径依赖：

    所在位置：
        - 路径：verl/utils/rollout_skip.py
        - 函数：wrap_generate_sequences(rolloutskip, rollout_wg)

    典型调用路径：
        RolloutSkip.__init__()
        -> RolloutSkip.wrap_generate_sequences()
        -> wrap_generate_sequences(self, self._rollout_wg)  # 当前函数
        -> 返回 warp_fn
        -> 替换 rollout_wg.generate_sequences

    被谁调用：
        - RolloutSkip.wrap_generate_sequences()

    调用了谁（项目内）：
        - rolloutskip.try_load()（尝试加载缓存）
        - rolloutskip.dump()（保存缓存）
        - rollout_wg.generate_sequences()（原始生成方法）

    调用了谁（外部依赖）：
        - 无

    最小示例：
        >>> # 通常不直接调用，由 RolloutSkip.wrap_generate_sequences() 调用
        >>> wrapped_fn = wrap_generate_sequences(rollout_skip, rollout_wg)
        >>> outputs = wrapped_fn(batch, temperature=0.7)
    """  # 注释：函数 docstring 结束

    # 注释：保存原始 generate_sequences 方法的引用，后续需要调用
    generate_sequences = rollout_wg.generate_sequences  # 注释：这是原始的、未包装的生成函数

    def warp_fn(batch, **kwargs):  # 注释：定义包装函数，签名与原始 generate_sequences 相同
        """（内部函数说明：包装后的 generate_sequences 函数，添加缓存逻辑）

        参数：
            batch: 输入批次数据（prompt 数据）
            **kwargs: 传递给原始 generate_sequences 的其他参数（如 temperature、top_p 等）

        返回：
            DataProto: rollout 生成的数据（可能来自缓存或新生成）
        """  # 注释：内部函数 docstring 结束

        # 注释：步骤1：尝试从磁盘加载缓存数据
        gen_batch_output = rolloutskip.try_load()  # 注释：如果缓存存在且加载成功，返回 DataProto 对象；否则返回 None

        # 注释：检查缓存是否加载成功
        if gen_batch_output is None:  # 注释：缓存不存在或加载失败
            # 注释：步骤2：调用原始 generate_sequences 方法生成新数据
            gen_batch_output = generate_sequences(batch, **kwargs)  # 注释：调用原始方法，传递所有参数

            # 注释：步骤3：将新生成的数据保存到磁盘缓存
            rolloutskip.dump(gen_batch_output)  # 注释：保存数据，下次运行时可以直接加载

        # 注释：返回生成或加载的数据（调用者无需关心数据来源）
        return gen_batch_output  # 注释：DataProto 对象，包含 prompt、response、logprobs 等完整数据

    # 注释：返回包装后的函数，该函数会替换 rollout_wg.generate_sequences
    return warp_fn  # 注释：warp_fn 保持原始函数的签名，但添加了缓存逻辑
