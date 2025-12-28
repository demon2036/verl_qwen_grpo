# Copyright 2024 Bytedance Ltd. and/or its affiliates  # 注释:版权声明
#  # 注释:保留注释符号,保证该行有中文说明
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释:声明Apache 2.0许可证
# you may not use this file except in compliance with the License.  # 注释:使用需遵守许可证
# You may obtain a copy of the License at  # 注释:提示许可证链接
#  # 注释:保留注释符号,保证该行有中文说明
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释:Apache 2.0许可证地址
#  # 注释:保留注释符号,保证该行有中文说明
# Unless required by applicable law or agreed to in writing, software  # 注释:免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释:软件按原样提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释:不提供担保
# See the License for the specific language governing permissions and  # 注释:更多许可条款
# limitations under the License.  # 注释:许可限制说明

"""（模块说明:序列长度负载均衡工具,用于在多设备/微批次间平衡计算量）

模块用途:
    本模块提供序列长度均衡分片(sequence length balancing)功能,主要用于RLHF/PPO训练中的动态批次划分。
    核心功能包括:
    1. 将变长序列分配到K个分片,使每个分片的计算负载尽可能均衡
    2. 根据序列长度估算Transformer块的FLOPs(浮点运算量)
    3. 使用Karmarkar-Karp最大差分法(LDM)进行负载均衡分片
    4. 支持动态微批次划分,减少GPU空闲时间(bubble)
    5. 提供分片前后的负载统计与日志记录

输入/输出:
    - 输入:序列长度列表(list[int]),分片数K,是否等大小约束
          例如:[128, 256, 512, 1024], K=2, equal_size=False
    - 输出:K个分片的索引列表(list[list[int]]),每个分片包含原序列的下标
          例如:[[0, 2], [1, 3]]表示第1片包含序列0和2,第2片包含序列1和3

关键依赖:
    - heapq:优先队列,用于Karmarkar-Karp算法的状态管理
    - torch:计算序列长度和,分布式通信(可选)
    - verl.protocol.DataProto:RLHF数据协议,用于批次打包与还原

典型用法:
    >>> from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions
    >>> # 场景1:将4个变长序列分到2个分片,使计算负载均衡
    >>> seqlen_list = [128, 256, 512, 1024]  # 序列长度
    >>> partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions=2, equal_size=False)
    >>> print(partitions)
    [[0, 3], [1, 2]]  # 第1片:128+1024=1152, 第2片:256+512=768 (相对均衡)

    >>> # 场景2:动态微批次划分,用于Actor训练
    >>> from verl.utils.seqlen_balancing import prepare_dynamic_batch
    >>> micro_batches, batch_idx_list = prepare_dynamic_batch(
    ...     data=data_proto,
    ...     max_token_len=2048,  # 每个微批次最多2048个token
    ...     dp_group=None,
    ...     use_dynamic_bsz_balance=True
    ... )
    >>> # micro_batches包含多个DataProto,每个包含一个微批次
    >>> # batch_idx_list记录每个微批次对应原批次的哪些样本

调用路径概览:
    入口脚本(如verl/trainer/main_ppo.py)
    -> verl.trainer.ppo.ray_trainer.PPOTrainer.fit()
    -> verl.workers.actor.dp_actor.update_policy()
    -> verl.workers.fsdp_workers.FSDPWorker.forward()
    -> verl.utils.seqlen_balancing.prepare_dynamic_batch()
    -> verl.utils.seqlen_balancing.rearrange_micro_batches()
    -> verl.utils.seqlen_balancing.get_seqlen_balanced_partitions()
    -> verl.utils.seqlen_balancing.karmarkar_karp()
    -> 返回均衡分片的索引列表

所在位置:
    - 路径:verl/utils/seqlen_balancing.py
    - 模块:verl.utils.seqlen_balancing

被谁调用:
    - verl/trainer/ppo/ray_trainer.py(PPO训练循环中动态批次划分)
    - verl/workers/actor/dp_actor.py(Actor模型前向传播时划分微批次)
    - verl/workers/fsdp_workers.py(FSDP worker中序列长度均衡)

调用了谁(项目内):
    - verl.protocol.DataProto(数据协议,用于批次打包)
    - verl.utils.tensordict_utils.index_select_tensor_dict(根据索引选择张量)
    - verl.utils.device.get_device_name(获取设备名称)

调用了谁(外部依赖):
    - heapq:优先队列(heappush, heappop)
    - torch:张量运算(sum, tensor, distributed.all_reduce等)
    - itertools.chain:展平嵌套列表

注意事项:
    1. Karmarkar-Karp算法的时间复杂度为O(N log N),N为序列数量
    2. equal_size=True时要求len(seqlen_list) % k_partitions == 0
    3. 动态批次划分会破坏原批次顺序,需要通过batch_idx_list还原
    4. FLOPs估算公式假设hidden_size=4096(7B模型),不同模型需调整
    5. 分布式训练时,dp_group参数用于同步微批次数量,避免死锁

记忆提示:
    - 核心算法:Karmarkar-Karp(最大差分法)
    - 口诀:大小配对,差值最小,负载均衡
    - 应用场景:变长序列->多设备/微批次->均衡计算
"""  # 注释:模块级docstring结束,下面是依赖导入

# 注释:导入copy模块,用于深拷贝对象(如索引映射)
import copy  # 注释:提供deepcopy函数,避免修改原对象
# 注释:导入heapq模块,提供优先队列(最小堆)实现,用于Karmarkar-Karp算法
import heapq  # 注释:heappush/heappop用于维护状态优先队列
# 注释:导入itertools.chain,用于展平嵌套列表(如合并多个分片的索引)
from itertools import chain  # 注释:chain.from_iterable([list1, list2, ...])展平为单层列表
# 注释:空行说明:分隔标准库与第三方库导入
# 注释:导入torch库,提供张量运算与分布式通信功能
import torch  # 注释:用于计算序列长度和,创建张量,分布式reduce等
# 注释:导入torch.distributed,用于跨进程同步微批次数量
from torch import distributed as dist  # 注释:dist.all_reduce同步全局最大值
# 注释:空行说明:分隔第三方库与项目内模块导入
# 注释:导入DataProto数据协议类,RLHF统一的数据封装格式
from verl.protocol import DataProto  # 注释:DataProto.from_dict用于打包微批次数据
# 注释:导入tensordict工具函数,用于根据索引选择字典中的张量
from verl.utils import tensordict_utils as tu  # 注释:tu.index_select_tensor_dict用于索引选择
# 注释:导入设备工具函数,获取当前设备名称(cuda/npu/cpu)
from verl.utils.device import get_device_name  # 注释:get_device_name()返回设备字符串


def calculate_workload(seqlen_list: list[int]):  # 注释:函数签名,接收序列长度列表(int或tensor),返回FLOPs估算值
    """（函数说明:根据序列长度估算Transformer块的计算量FLOPs）

    本函数使用简化的FLOPs公式估算密集Transformer块的计算负载:
    FLOPs = 12 * hidden_size^2 * seqlen + 2 * hidden_size * seqlen^2
    硬编码假设hidden_size=4096(7B模型),简化为:
    FLOPs ≈ 6 * 4096 * seqlen + seqlen^2 = 24576 * seqlen + seqlen^2

    参数:
        seqlen_list (list[int] | torch.Tensor): 序列长度列表或张量
            - 类型:可以是Python list或torch.Tensor
            - 元素类型:int(序列长度,如128, 256, 512等)
            - 长度:任意正整数
            - 示例:[128, 256, 512]或torch.tensor([128, 256, 512])

    返回:
        torch.Tensor | list[float]: 每个序列的FLOPs估算值
            - 类型:与输入类型一致(list输入返回标量运算结果,tensor输入返回tensor)
            - 元素类型:float(FLOPs值)
            - 形状:与输入seqlen_list相同
            - 示例:输入[128, 256, 512]返回[3145728.0, 6356992.0, 12845056.0]

    副作用:
        - 无(纯函数,不修改输入,不改变全局状态)

    异常/边界条件:
        - seqlen_list为空列表时,返回空列表或空张量
        - 元素为负数时,结果无意义但不会报错(调用者需保证输入合法)
        - 元素过大时(如seqlen>100000)可能溢出,但实际训练不会出现

    最小示例(手算验证):
        >>> # 输入:3个序列,长度分别为128, 256, 512
        >>> seqlen_list = [128, 256, 512]
        >>> # 手算:
        >>> # seq0: 24576*128 + 128^2 = 3145728 + 16384 = 3162112
        >>> # seq1: 24576*256 + 256^2 = 6291456 + 65536 = 6356992
        >>> # seq2: 24576*512 + 512^2 = 12582912 + 262144 = 12845056
        >>> workloads = calculate_workload(seqlen_list)
        >>> # 预期输出:[3162112, 6356992, 12845056]
        >>> assert workloads[0] == 24576 * 128 + 128**2
        >>> assert workloads[1] == 24576 * 256 + 256**2
        >>> assert workloads[2] == 24576 * 512 + 512**2

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:calculate_workload(seqlen_list: list[int])

    典型调用路径:
        rearrange_micro_batches()
        -> calculate_workload(seq_len_effective)
        -> 返回FLOPs估算值,用于分片决策

    被谁调用:
        - rearrange_micro_batches()(本文件内,动态微批次划分的核心逻辑)

    调用了谁(项目内):
        - 无(纯数学计算,不依赖其他项目模块)

    调用了谁(外部依赖):
        - Python算术运算符(*, +)
        - **运算符(平方,等价于seqlen_list * seqlen_list)

    记忆提示:
        - FLOPs公式:线性项(hidden_size^2 * seqlen) + 二次项(seqlen^2)
        - 口诀:序列越长,计算量平方增长(注意力机制O(n^2)复杂度)
        - 简化假设:hidden_size=4096固定,适用于7B模型
    """  # 注释:函数docstring结束,下面进入函数体

    # 注释:计算FLOPs = 24576 * seqlen + seqlen^2
    # 注释:24576 = 6 * 4096,其中4096是假设的hidden_size(7B模型)
    # 注释:第一项24576*seqlen来自线性层计算(12*h^2*s简化后的系数)
    # 注释:第二项seqlen**2来自注意力机制的O(n^2)复杂度
    return 24576 * seqlen_list + seqlen_list**2  # 注释:返回FLOPs估算值,支持list和tensor输入


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):  # 注释:函数签名,实现Karmarkar-Karp最大差分法
    """（函数说明:使用Karmarkar-Karp算法将序列长度列表分成K个负载均衡的分片）

    本函数实现Karmarkar-Karp(KK)算法,也称为最大差分法(Largest Differencing Method, LDM)。
    算法原理:
    1. 初始化K个空集合(分片)
    2. 每次从优先队列中取出负载最大和次大的两个状态
    3. 将这两个状态合并(负载大的分片分配给负载小的状态)
    4. 重复步骤2-3,直到队列中只剩一个状态

    时间复杂度:O(N log N),其中N为序列数量
    空间复杂度:O(N),存储优先队列和状态

    参数:
        seqlen_list (list[int]): 序列长度列表
            - 元素类型:int(序列长度)
            - 长度:>= k_partitions
            - 示例:[128, 256, 512, 1024]
        k_partitions (int): 目标分片数量
            - 取值范围:[1, len(seqlen_list)]
            - 示例:2(分成2个分片)
        equal_size (bool): 是否强制每个分片包含相同数量的序列
            - True:每个分片包含len(seqlen_list) // k_partitions个序列
            - False:允许分片大小不等,只关注负载均衡
            - 示例:False(允许不等大小)

    返回:
        list[list[int]]: K个分片的索引列表
            - 外层列表长度:k_partitions
            - 内层列表元素:原seqlen_list的下标(int)
            - 内层列表长度:不固定(equal_size=True时固定为len(seqlen_list)//k_partitions)
            - 示例:[[0, 3], [1, 2]]表示第1片包含序列0和3,第2片包含序列1和2

    副作用:
        - 无(不修改输入列表)

    异常/边界条件:
        - 断言失败:equal_size=True时,len(seqlen_list) % k_partitions != 0会触发assert
        - 空列表:seqlen_list=[]时,返回k_partitions个空列表

    最小示例(手算验证):
        >>> # 输入:4个序列,分成2片,允许不等大小
        >>> seqlen_list = [128, 256, 512, 1024]
        >>> k_partitions = 2
        >>> equal_size = False
        >>> # 算法步骤(简化描述):
        >>> # 1. 初始状态:4个单元素集合{128}, {256}, {512}, {1024}
        >>> # 2. 第1次合并:{1024}和{512}的最小分片合并->{512,1024}, {256}, {128}
        >>> # 3. 第2次合并:{512,1024}和{256}的最小分片合并->{256,512,1024}, {128}
        >>> # 4. 第3次合并:{256,512,1024}和{128}合并->最终2片
        >>> # 预期输出:[[0, 3], [1, 2]]或类似负载均衡的分片
        >>> partitions = karmarkar_karp(seqlen_list, k_partitions, equal_size)
        >>> # 验证:分片1总长度=128+1024=1152, 分片2总长度=256+512=768
        >>> sum1 = sum(seqlen_list[i] for i in partitions[0])
        >>> sum2 = sum(seqlen_list[i] for i in partitions[1])
        >>> assert abs(sum1 - sum2) <= max(seqlen_list)  # 差值在可接受范围内

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:karmarkar_karp(seqlen_list, k_partitions, equal_size)

    典型调用路径:
        get_seqlen_balanced_partitions()
        -> karmarkar_karp(seqlen_list, k_partitions, equal_size)
        -> 返回均衡分片索引列表

    被谁调用:
        - get_seqlen_balanced_partitions()(本文件内,对外暴露的主接口)

    调用了谁(项目内):
        - 无(独立算法实现)

    调用了谁(外部依赖):
        - heapq.heappush():将状态加入优先队列
        - heapq.heappop():从优先队列中取出最小(spread最大)状态
        - sorted():对序列和集合排序

    记忆提示:
        - 算法核心:贪心策略,每次合并负载差最大的两个状态
        - 口诀:大小配对,差值递减,最终均衡
        - 与贪心算法区别:KK考虑全局差值,贪心只看当前最小
    """  # 注释:函数docstring结束,下面定义内部类

    # see: https://en.wikipedia.org/wiki/Largest_differencing_method  # 注释:保留原始注释,提供算法参考链接
    class Set:  # 注释:内部类Set,表示一个分片(包含多个序列的集合)
        """（内部类说明:表示一个分片集合,记录总负载和包含的序列索引）

        属性:
            sum (int): 当前分片的总负载(所有序列长度之和)
            items (list[tuple[int, int]]): 包含的序列列表,每项为(原索引, 序列长度)
        """  # 注释:内部类docstring结束

        def __init__(self) -> None:  # 注释:Set构造函数,初始化空分片
            """（构造函数说明:初始化空分片）"""  # 注释:构造函数docstring
            self.sum = 0  # 注释:初始化总负载为0
            self.items = []  # 注释:初始化序列列表为空

        def add(self, idx: int, val: int):  # 注释:向分片添加一个序列
            """（方法说明:向分片添加序列）

            参数:
                idx (int): 序列在原列表中的索引
                val (int): 序列长度
            """  # 注释:方法docstring结束
            self.items.append((idx, val))  # 注释:将(索引,长度)元组加入items列表
            self.sum += val  # 注释:累加总负载

        def merge(self, other):  # 注释:合并另一个Set到当前Set
            """（方法说明:合并另一个分片到当前分片）

            参数:
                other (Set): 待合并的分片
            """  # 注释:方法docstring结束
            for idx, val in other.items:  # 注释:遍历待合并分片的所有序列
                self.items.append((idx, val))  # 注释:将序列加入当前分片
                self.sum += val  # 注释:累加总负载

        def __lt__(self, other):  # 注释:定义小于运算符,用于优先队列排序
            """（方法说明:定义Set之间的比较规则,用于优先队列排序）

            比较规则(优先级从高到低):
            1. 总负载sum更大的Set更小(优先级更高)
            2. sum相同时,包含序列更多的Set更小
            3. 都相同时,按items字典序比较
            """  # 注释:方法docstring结束
            if self.sum != other.sum:  # 注释:首先比较总负载
                return self.sum < other.sum  # 注释:总负载小的在优先队列中靠前
            if len(self.items) != len(other.items):  # 注释:总负载相同时比较序列数量
                return len(self.items) < len(other.items)  # 注释:序列少的在优先队列中靠前
            return self.items < other.items  # 注释:都相同时按items字典序比较

    class State:  # 注释:内部类State,表示当前K个分片的状态(用于KK算法的迭代过程)
        """（内部类说明:表示KK算法中的一个状态,包含K个分片集合）

        属性:
            k (int): 分片数量
            sets (list[Set]): K个分片集合,按负载降序排列
        """  # 注释:内部类docstring结束

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:  # 注释:State构造函数
            """（构造函数说明:初始化状态,创建K个分片）

            参数:
                items (list[tuple[int, int]]): 初始序列列表,每项为(原索引, 序列长度)
                    - 长度必须为1或k(单序列状态或K分片状态)
                k (int): 分片数量
            """  # 注释:构造函数docstring结束
            self.k = k  # 注释:保存分片数量
            # sets should always be decreasing order  # 注释:保留原注释,说明sets始终降序排列
            self.sets = [Set() for _ in range(k)]  # 注释:初始化K个空分片
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"  # 注释:断言items长度合法
            for i, (idx, seqlen) in enumerate(items):  # 注释:遍历输入序列
                self.sets[i].add(idx=idx, val=seqlen)  # 注释:将第i个序列添加到第i个分片
            self.sets = sorted(self.sets, reverse=True)  # 注释:按负载降序排列分片(负载大的在前)

        def get_partitions(self):  # 注释:提取K个分片的索引列表
            """（方法说明:提取当前状态的分片索引列表）

            返回:
                list[list[int]]: K个分片的索引列表
            """  # 注释:方法docstring结束
            partitions = []  # 注释:初始化结果列表
            for i in range(len(self.sets)):  # 注释:遍历K个分片
                cur_partition = []  # 注释:当前分片的索引列表
                for idx, _ in self.sets[i].items:  # 注释:遍历分片中的序列
                    cur_partition.append(idx)  # 注释:提取序列索引
                partitions.append(cur_partition)  # 注释:加入结果列表
            return partitions  # 注释:返回K个分片的索引列表

        def merge(self, other):  # 注释:合并另一个State到当前State
            """（方法说明:合并两个State,将other的分片逆序合并到当前State）

            参数:
                other (State): 待合并的状态
            """  # 注释:方法docstring结束
            for i in range(self.k):  # 注释:遍历K个分片
                self.sets[i].merge(other.sets[self.k - 1 - i])  # 注释:将other的第k-1-i个分片合并到当前第i个分片(逆序合并)
            self.sets = sorted(self.sets, reverse=True)  # 注释:合并后重新降序排列

        @property  # 注释:定义属性方法,计算当前状态的负载差(spread)
        def spread(self) -> int:  # 注释:返回最大分片与最小分片的负载差
            """（属性说明:计算最大分片与最小分片的负载差）

            返回:
                int: 负载差(spread),值越小表示越均衡
            """  # 注释:属性docstring结束
            return self.sets[0].sum - self.sets[-1].sum  # 注释:最大负载-最小负载=spread

        def __lt__(self, other):  # 注释:定义State之间的比较规则,用于优先队列排序
            """（方法说明:定义State之间的比较规则,用于优先队列排序）

            比较规则:
            1. spread更大的State更小(优先级更高,优先处理负载不均衡的状态)
            2. spread相同时,最大分片更大的State更小

            注意:这是最小堆,__lt__返回True表示self优先级更高
            """  # 注释:方法docstring结束
            # least heap, let the state with largest spread to be popped first,  # 注释:保留原注释
            # if the spread is the same, let the state who has the largest set  # 注释:保留原注释
            # to be popped first.  # 注释:保留原注释
            if self.spread != other.spread:  # 注释:首先比较spread
                return self.spread > other.spread  # 注释:spread大的优先级高(在最小堆中返回值相反)
            return self.sets[0] > other.sets[0]  # 注释:spread相同时比较最大分片

        def __repr__(self) -> str:  # 注释:定义State的字符串表示,用于调试和日志
            """（方法说明:返回State的字符串表示,格式为[{s1,s2,...},{...},...]）"""  # 注释:方法docstring结束
            repr_str = "["  # 注释:初始化字符串
            for i in range(self.k):  # 注释:遍历K个分片
                if i > 0:  # 注释:非第一个分片前添加逗号
                    repr_str += ","  # 注释:分片间分隔符
                repr_str += "{"  # 注释:分片开始符号
                for j, (_, seqlen) in enumerate(self.sets[i].items):  # 注释:遍历分片中的序列
                    if j > 0:  # 注释:非第一个序列前添加逗号
                        repr_str += ","  # 注释:序列间分隔符
                    repr_str += str(seqlen)  # 注释:添加序列长度
                repr_str += "}"  # 注释:分片结束符号
            repr_str += "]"  # 注释:整体结束符号
            return repr_str  # 注释:返回格式化字符串

    # 注释:KK算法主流程开始
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])  # 注释:将序列长度与索引配对并排序(升序)
    states_pq = []  # 注释:初始化优先队列(存储State对象)
    if equal_size:  # 注释:如果要求等大小分片
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"  # 注释:断言序列总数可被K整除
        for offset in range(0, len(sorted_seqlen_list), k_partitions):  # 注释:每K个序列分成一组
            items = []  # 注释:当前组的序列列表
            for i in range(k_partitions):  # 注释:取K个序列
                seqlen, idx = sorted_seqlen_list[offset + i]  # 注释:提取序列长度和索引
                items.append((idx, seqlen))  # 注释:加入当前组
            heapq.heappush(states_pq, State(items=items, k=k_partitions))  # 注释:将K个序列初始化为一个State并加入优先队列
    else:  # 注释:如果允许不等大小分片
        for seqlen, idx in sorted_seqlen_list:  # 注释:遍历每个序列
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))  # 注释:每个序列初始化为一个单元素State并加入优先队列

    # 注释:KK算法迭代合并阶段
    while len(states_pq) > 1:  # 注释:当优先队列中有多于1个State时继续迭代
        state0 = heapq.heappop(states_pq)  # 注释:弹出spread最大的State(优先级最高)
        state1 = heapq.heappop(states_pq)  # 注释:弹出spread次大的State(优先级第二)
        # merge states  # 注释:保留原注释
        state0.merge(state1)  # 注释:将state1合并到state0
        heapq.heappush(states_pq, state0)  # 注释:将合并后的state0重新加入优先队列

    # 注释:提取最终分片结果
    final_state = states_pq[0]  # 注释:队列中唯一剩余的State即为最终结果
    partitions = final_state.get_partitions()  # 注释:提取K个分片的索引列表
    if equal_size:  # 注释:如果要求等大小分片,验证结果
        for i, partition in enumerate(partitions):  # 注释:遍历每个分片
            assert len(partition) * k_partitions == len(seqlen_list), (  # 注释:断言每个分片大小=总数/K
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"  # 注释:错误信息
            )  # 注释:断言结束
    return partitions  # 注释:返回K个分片的索引列表


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):  # 注释:函数签名,实现贪心分片算法(备用方案)
    """（函数说明:使用贪心算法将序列分成K个分片,作为Karmarkar-Karp的替代方案）

    本函数实现简单的贪心分片算法:
    1. 将序列按长度排序
    2. 每次将当前序列分配给负载最小的分片
    3. 重复步骤2,直到所有序列分配完毕

    贪心算法比Karmarkar-Karp简单但效果稍差,主要用于调试或特殊场景。

    参数:
        seqlen_list (list[int]): 序列长度列表
        k_partitions (int): 目标分片数量
        equal_size (bool): 是否强制等大小分片(与KK算法一致)

    返回:
        list[list[int]]: K个分片的索引列表

    副作用:
        - 无

    异常/边界条件:
        - 与karmarkar_karp相同

    最小示例:
        >>> seqlen_list = [128, 256, 512, 1024]
        >>> partitions = greedy_partition(seqlen_list, k_partitions=2, equal_size=False)
        >>> # 预期输出:[[3], [0, 1, 2]]或类似结果(贪心策略下的分片)

    调用路径依赖:
        - 未被主流程调用,可作为备用方案

    记忆提示:
        - 算法核心:局部最优(每次选负载最小分片)
        - 与KK区别:KK考虑全局差值,贪心只看当前
    """  # 注释:函数docstring结束
    bias = sum(seqlen_list) + 1 if equal_size else 0  # 注释:等大小模式下添加bias,强制优先填满分片
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]  # 注释:序列长度排序(加bias)
    partitions = [[] for _ in range(k_partitions)]  # 注释:初始化K个空分片
    partition_sums = [0 for _ in range(k_partitions)]  # 注释:记录每个分片的当前负载
    for seqlen, i in sorted_seqlen:  # 注释:遍历排序后的序列
        min_idx = None  # 注释:负载最小分片的索引
        for j in range(k_partitions):  # 注释:遍历所有分片,找负载最小的
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:  # 注释:更新最小索引
                min_idx = j  # 注释:记录最小分片索引
        partitions[min_idx].append(i)  # 注释:将当前序列分配给负载最小的分片
        partition_sums[min_idx] += seqlen  # 注释:更新分片负载
    if equal_size:  # 注释:验证等大小约束
        for i, partition in enumerate(partitions):  # 注释:遍历分片
            assert len(partition) * k_partitions == len(seqlen_list), (  # 注释:断言分片大小正确
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"  # 注释:错误信息
            )  # 注释:断言结束
    return partitions  # 注释:返回分片结果


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):  # 注释:函数签名,对外暴露的主接口
    """（函数说明:计算序列长度均衡的分片,对外暴露的主接口）

    本函数是序列长度均衡分片的主入口,内部调用Karmarkar-Karp算法,并进行结果验证。
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
            - 元素类型:int(序列长度)
            - 长度:>= k_partitions
            - 示例:[128, 256, 512, 1024]
        k_partitions (int): The desired number of partitions.
            - 取值范围:[1, len(seqlen_list)]
            - 示例:2
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.
            - True:强制等大小
            - False:允许不等大小
            - 示例:False

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.
            - 外层列表长度:k_partitions
            - 内层列表元素:原seqlen_list的索引(int)
            - 内层列表已排序(升序)
            - 示例:[[0, 2], [1, 3]]

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
            - 序列数量少于分片数量时触发
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
            - 等大小模式下序列数量无法整除分片数量时触发
        AssertionError: If any resulting partition is empty.
            - 任何分片为空时触发(不应发生,防御性断言)

    副作用:
        - 无(纯函数)

    异常/边界条件:
        - len(seqlen_list) < k_partitions:断言失败
        - equal_size=True且len(seqlen_list) % k_partitions != 0:断言失败
        - 任何分片为空:断言失败

    最小示例(手算验证):
        >>> # 输入:4个序列,分成2片
        >>> seqlen_list = [128, 256, 512, 1024]
        >>> k_partitions = 2
        >>> equal_size = False
        >>> partitions = get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size)
        >>> # 预期:分片1=[0,3](128+1024=1152), 分片2=[1,2](256+512=768)
        >>> # 或类似负载均衡的结果
        >>> assert len(partitions) == 2
        >>> assert all(idx in range(4) for partition in partitions for idx in partition)
        >>> # 验证每个索引只出现一次
        >>> all_indices = [idx for partition in partitions for idx in partition]
        >>> assert sorted(all_indices) == [0, 1, 2, 3]

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:get_seqlen_balanced_partitions(seqlen_list, k_partitions, equal_size)

    典型调用路径:
        rearrange_micro_batches()
        -> get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)
        -> 返回均衡分片索引列表

    被谁调用:
        - rearrange_micro_batches()(本文件内,动态微批次划分)

    调用了谁(项目内):
        - karmarkar_karp(seqlen_list, k_partitions, equal_size)(本文件内,KK算法实现)

    调用了谁(外部依赖):
        - sorted():对分片内索引排序

    记忆提示:
        - 主入口函数,封装KK算法并验证结果
        - 口诀:对外接口->KK算法->结果验证->返回分片
    """  # 注释:函数docstring结束
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"  # 注释:断言序列数量>=分片数量

    def _check_and_sort_partitions(partitions):  # 注释:内部函数,验证分片结果并排序
        """（内部函数说明:验证分片结果的正确性并对索引排序）

        验证内容:
        1. 分片数量==k_partitions
        2. 每个分片非空
        3. 所有索引覆盖[0, len(seqlen_list))且不重复

        参数:
            partitions (list[list[int]]): 待验证的分片

        返回:
            list[list[int]]: 验证通过并排序后的分片
        """  # 注释:内部函数docstring结束
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"  # 注释:断言分片数量正确
        seen_idx = set()  # 注释:记录已出现的索引,用于检查重复
        sorted_partitions = [None] * k_partitions  # 注释:初始化排序后的分片列表
        for i, partition in enumerate(partitions):  # 注释:遍历每个分片
            assert len(partition) > 0, f"the {i}-th partition is empty"  # 注释:断言分片非空
            for idx in partition:  # 注释:遍历分片中的索引
                seen_idx.add(idx)  # 注释:记录索引
            sorted_partitions[i] = sorted(partition)  # 注释:对分片内索引排序(升序)
        assert seen_idx == set(range(len(seqlen_list)))  # 注释:断言所有索引覆盖[0,len-1]且无重复
        return sorted_partitions  # 注释:返回验证通过并排序后的分片

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)  # 注释:调用KK算法计算分片
    return _check_and_sort_partitions(partitions)  # 注释:验证并排序分片结果后返回


def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix):  # 注释:函数签名,计算并记录分片前后的负载统计信息
    """（函数说明:计算分片前后的负载统计指标,用于日志记录和性能分析）

    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    本函数计算以下指标:
    1. 分片前:简单分批(按顺序分成K批)的最小/最大/总负载
    2. 分片后:均衡分片后的最小/最大负载
    3. 负载差值:最大-最小(值越小越均衡)

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
            - 元素类型:int(序列长度)
            - 示例:[128, 256, 512, 1024]
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
            - 格式:[[idx1, idx2, ...], [...], ...]
            - 示例:[[0, 3], [1, 2]]
        prefix (str): A prefix to be added to each metric key in the returned dictionary.
            - 用途:区分不同场景的指标(如"train"、"rollout"等)
            - 示例:"seqlen_balance"

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance.
            - 键格式:"{prefix}/指标名"
            - 指标说明:
              * {prefix}/min:分片前最小批次负载
              * {prefix}/max:分片前最大批次负载
              * {prefix}/minmax_diff:分片前负载差(max-min)
              * {prefix}/balanced_min:分片后最小分片负载
              * {prefix}/balanced_max:分片后最大分片负载
              * {prefix}/mean:平均负载(总负载/分片数)
            - 示例:{"seqlen_balance/min": 640, "seqlen_balance/max": 1280, ...}

    副作用:
        - 无(纯计算函数)

    异常/边界条件:
        - partitions为空:返回空字典或引发异常
        - seqlen_list为空:返回全0指标

    最小示例(手算验证):
        >>> # 输入:4个序列,2个分片
        >>> seqlen_list = [128, 256, 512, 1024]
        >>> partitions = [[0, 3], [1, 2]]  # 分片1=128+1024=1152, 分片2=256+512=768
        >>> metrics = log_seqlen_unbalance(seqlen_list, partitions, "test")
        >>> # 分片前(简单分批):batch0=[128,256]=384, batch1=[512,1024]=1536
        >>> assert metrics["test/min"] == 384
        >>> assert metrics["test/max"] == 1536
        >>> assert metrics["test/minmax_diff"] == 1536 - 384
        >>> # 分片后(均衡分片):partition0=1152, partition1=768
        >>> assert metrics["test/balanced_min"] == 768
        >>> assert metrics["test/balanced_max"] == 1152
        >>> assert metrics["test/mean"] == (384 + 1536) / 2  # 总负载/分片数

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:log_seqlen_unbalance(seqlen_list, partitions, prefix)

    典型调用路径:
        rearrange_micro_batches()或训练循环
        -> log_seqlen_unbalance(seqlen_list, partitions, "seqlen_balance")
        -> 返回指标字典,用于日志记录

    被谁调用:
        - 训练循环或性能分析脚本(可选,用于监控负载均衡效果)

    调用了谁(项目内):
        - 无

    调用了谁(外部依赖):
        - sum():计算序列长度总和
        - min()/max():计算最小/最大值

    记忆提示:
        - 指标分两类:分片前(min/max/diff)和分片后(balanced_min/balanced_max)
        - 口诀:对比前后,评估优化效果
    """  # 注释:函数docstring结束
    # Get the number of partitions  # 注释:保留原注释
    k_partition = len(partitions)  # 注释:获取分片数量
    # assert len(seqlen_list) % k_partition == 0  # 注释:保留原注释(已注释掉,允许不等大小)
    batch_size = len(seqlen_list) // k_partition  # 注释:计算简单分批的批次大小(向下取整)
    min_sum_seqlen = None  # 注释:初始化分片前最小批次负载
    max_sum_seqlen = None  # 注释:初始化分片前最大批次负载
    total_sum_seqlen = 0  # 注释:初始化总负载

    # Iterate over each batch of sequence lengths  # 注释:保留原注释
    for offset in range(0, len(seqlen_list), batch_size):  # 注释:遍历每个简单分批的批次(按顺序划分)
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])  # 注释:计算当前批次的总负载
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:  # 注释:更新最小负载
            min_sum_seqlen = cur_sum_seqlen  # 注释:记录最小值
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:  # 注释:更新最大负载
            max_sum_seqlen = cur_sum_seqlen  # 注释:记录最大值
        total_sum_seqlen += cur_sum_seqlen  # 注释:累加总负载

    balanced_sum_seqlen_list = []  # 注释:初始化分片后的负载列表
    for partition in partitions:  # 注释:遍历均衡分片后的每个分片
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])  # 注释:计算当前分片的总负载
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)  # 注释:记录到列表
    # print("balanced_sum_seqlen_list: ", balanced_sum_seqlen_list)  # 注释:保留原注释(调试用,已注释)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)  # 注释:计算分片后最小负载
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)  # 注释:计算分片后最大负载

    return {  # 注释:返回指标字典
        f"{prefix}/min": min_sum_seqlen,  # 注释:分片前最小批次负载
        f"{prefix}/max": max_sum_seqlen,  # 注释:分片前最大批次负载
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,  # 注释:分片前负载差
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,  # 注释:分片后最小分片负载
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,  # 注释:分片后最大分片负载
        f"{prefix}/mean": total_sum_seqlen / len(partitions),  # 注释:平均负载(总负载/分片数)
    }  # 注释:字典结束


def ceildiv(a, b):  # 注释:函数签名,计算向上取整除法
    """（函数说明:计算向上取整除法,等价于math.ceil(a/b)但避免浮点数）

    本函数使用整数运算实现向上取整除法:
    ceildiv(a, b) = -(a // -b)
    原理:
    - Python的//运算符是向下取整
    - -a // -b会先取相反数,向下取整后再取相反数,等价于向上取整

    参数:
        a (int): 被除数
        b (int): 除数(必须>0)

    返回:
        int: 向上取整的商

    副作用:
        - 无

    异常/边界条件:
        - b=0:引发ZeroDivisionError
        - b<0:结果可能不符合预期(未定义行为)

    最小示例(手算验证):
        >>> # 示例1:整除
        >>> assert ceildiv(10, 5) == 2  # 10/5=2(整除)
        >>> # 示例2:有余数
        >>> assert ceildiv(11, 5) == 3  # 11/5=2.2->向上取整=3
        >>> assert ceildiv(9, 5) == 2   # 9/5=1.8->向上取整=2
        >>> # 手算验证:-(11 // -5) = -(-3) = 3 ✓

    调用路径依赖:
        - 被rearrange_micro_batches调用,计算微批次数量

    记忆提示:
        - 公式:ceildiv(a,b) = -(a // -b)
        - 口诀:负负得正,向下变向上
    """  # 注释:函数docstring结束
    return -(a // -b)  # 注释:使用双重取反实现向上取整除法


def roundup_divisible(a, b):  # 注释:函数签名,将a向上舍入到b的倍数
    """（函数说明:将a向上舍入到b的倍数）

    本函数计算大于等于a的最小b的倍数:
    roundup_divisible(a, b) = ((a + b - 1) // b) * b

    参数:
        a (int): 待舍入的数
        b (int): 目标倍数

    返回:
        int: 大于等于a的最小b的倍数

    副作用:
        - 无

    异常/边界条件:
        - b=0:引发ZeroDivisionError
        - b<0:结果可能不符合预期

    最小示例(手算验证):
        >>> # 示例1:a是b的倍数
        >>> assert roundup_divisible(10, 5) == 10  # 10已是5的倍数
        >>> # 示例2:a不是b的倍数
        >>> assert roundup_divisible(11, 5) == 15  # 11->向上到15(5*3)
        >>> assert roundup_divisible(9, 5) == 10   # 9->向上到10(5*2)
        >>> # 手算验证:((11+5-1)//5)*5 = (15//5)*5 = 3*5 = 15 ✓

    调用路径依赖:
        - 被rearrange_micro_batches调用,对齐虚拟管道并行大小

    记忆提示:
        - 公式:((a+b-1)//b)*b
        - 口诀:先加b-1向上偏移,再整除取整,最后乘回倍数
    """  # 注释:函数docstring结束
    return ((a + b - 1) // b) * b  # 注释:向上舍入到b的倍数


def rearrange_micro_batches(  # 注释:函数签名,动态微批次划分的核心函数
    batch,  # 注释:输入批次(TensorDict格式)
    max_token_len,  # 注释:每个微批次的最大token数量
    dp_group=None,  # 注释:数据并行组(用于同步微批次数量)
    num_batches_divided_by=None,  # 注释:虚拟管道并行大小(Megatron专用)
    same_micro_num_in_dp=True,  # 注释:是否在DP组内同步微批次数量
    min_num_micro_batch=None,  # 注释:最小微批次数量(用于管道并行)
    use_dynamic_bsz_balance=True,  # 注释:是否使用动态负载均衡排序
):  # 注释:函数签名结束
    """（函数说明:将批次按总token数划分为多个微批次,支持DP同步和负载均衡）

    Split a batch into micro-batches by total token count, with optional DP sync and padding.

    本函数是动态微批次划分的核心实现,主要功能:
    1. 计算每个样本的有效token数(去除padding)
    2. 根据max_token_len将批次划分为多个微批次
    3. 使用Karmarkar-Karp算法进行负载均衡分片
    4. 可选:在DP组内同步微批次数量,避免死锁
    5. 可选:对微批次重排序,减少管道并行的bubble

    Args:
        batch (TensorDict): must include "attention_mask" (B*S); other fields are sliced similarly.
            - 必须包含"attention_mask"字段(形状[batch_size, seq_len])
            - 其他字段(如input_ids)会按相同索引切片
            - 类型:TensorDict或dict[str, torch.Tensor]
        max_token_len (int): max sum of attention_mask per micro-batch.
            - 每个微批次的最大token总数(去除padding后)
            - 示例:2048(每个微批次最多2048个有效token)
        dp_group (optional): torch.distributed group for data-parallel sync.
            - 数据并行组,用于同步微批次数量
            - None表示不同步
        num_batches_divided_by (optional): virtual pipeline parallel size, for megatron.
            - 虚拟管道并行大小,微批次数量必须是其倍数
            - 用于Megatron的交错管道并行(interleaved pipeline)
        same_micro_num_in_dp (bool): if True and dp_group set, pad all ranks to the same count.
            - True:在DP组内同步微批次数量到最大值
            - False:允许不同rank有不同数量的微批次
        min_num_micro_batch (int, optional): force at least this many splits (pads empty ones).
            - 最小微批次数量,用于管道并行对齐
            - 不足时会填充空微批次
        use_dynamic_bsz_balance (bool, optional): balance the computational workload between micro-batches
            - True:对微批次按负载排序,减少bubble
            - False:保持KK算法的原始顺序

    Returns:
        List[TensorDict]: the micro-batches.
            - 每个元素是一个微批次(TensorDict格式)
            - 长度:num_micro_batches
            - 示例:[micro_batch0, micro_batch1, ...]
        List[List[int]]: index lists mapping each micro-batch back to original positions.
            - 每个元素是一个索引列表,记录该微批次包含原批次的哪些样本
            - 长度:num_micro_batches
            - 示例:[[0, 3], [1, 2]]表示micro_batch0包含原batch的样本0和3

    副作用:
        - 可能修改全局分布式状态(all_reduce操作)

    异常/边界条件:
        - max_token_len < max_seq_len:断言失败(微批次容不下最长序列)
        - batch为空:返回空列表
        - num_micro_batches计算结果为0:可能引发错误

    最小示例(手算验证):
        >>> # 输入:4个样本,有效长度分别为128, 256, 512, 1024
        >>> # max_token_len=1024,期望划分为2个微批次
        >>> batch = {
        ...     "input_ids": torch.randint(0, 1000, (4, 1024)),
        ...     "attention_mask": torch.tensor([
        ...         [1]*128 + [0]*896,   # 样本0:128个有效token
        ...         [1]*256 + [0]*768,   # 样本1:256个有效token
        ...         [1]*512 + [0]*512,   # 样本2:512个有效token
        ...         [1]*1024             # 样本3:1024个有效token
        ...     ])
        ... }
        >>> micro_batches, batch_idx_list = rearrange_micro_batches(
        ...     batch, max_token_len=1024, use_dynamic_bsz_balance=False
        ... )
        >>> # 预期:2个微批次
        >>> # micro_batch0包含样本[0,3](128+1024=1152>1024,但0单独<1024)
        >>> # 实际KK算法会尽量均衡,可能是[[0,2], [1,3]]等
        >>> assert len(micro_batches) == 2
        >>> assert len(batch_idx_list) == 2

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:rearrange_micro_batches(batch, max_token_len, ...)

    典型调用路径:
        prepare_dynamic_batch()
        -> rearrange_micro_batches(data.batch, max_token_len, ...)
        -> 返回微批次列表和索引映射

    被谁调用:
        - prepare_dynamic_batch()(本文件内,DataProto包装函数)
        - 训练循环或worker中直接调用(可选)

    调用了谁(项目内):
        - calculate_workload(seq_len_effective):估算每个样本的FLOPs
        - get_seqlen_balanced_partitions(workloads, num_micro_batches, False):负载均衡分片
        - tu.index_select_tensor_dict(batch, partition):按索引选择张量

    调用了谁(外部依赖):
        - torch.sum():计算有效token数
        - torch.tensor():创建张量
        - torch.distributed.all_reduce():同步微批次数量

    记忆提示:
        - 核心流程:计算有效长度->估算FLOPs->均衡分片->重排序
        - 口诀:变长输入->固定容量->均衡负载->减少bubble
    """  # 注释:函数docstring结束
    # this is per local micro_bsz  # 注释:保留原注释,说明这是本地微批次大小
    input_ids = batch["input_ids"]  # 注释:提取input_ids字段
    if input_ids.is_nested:  # 注释:如果是嵌套张量(NestedTensor,Flash Attention格式)
        seq_len_effective: torch.Tensor = input_ids.offsets().diff()  # 注释:从offset计算有效长度(diff操作)
        max_seq_len = max(seq_len_effective)  # 注释:获取最大有效长度
    else:  # 注释:普通张量格式
        max_seq_len = batch["attention_mask"].shape[-1]  # 注释:从attention_mask获取序列长度
        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)  # 注释:计算每个样本的有效token数(去除padding)

    assert max_token_len >= max_seq_len, (  # 注释:断言微批次容量足够容纳最长序列
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"  # 注释:错误信息
    )  # 注释:断言结束
    total_seqlen = seq_len_effective.sum().item()  # 注释:计算总有效token数
    # NOTE: num_microbatches <= batch_size, so take the min of this two.  # 注释:保留原注释,说明微批次数量不超过batch_size
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))  # 注释:计算微批次数量=min(batch_size, ceil(总token/容量))
    if min_num_micro_batch is not None:  # 注释:如果指定了最小微批次数量
        # used to support pp  # 注释:保留原注释,用于支持管道并行
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)  # 注释:取最大值,确保满足管道并行要求
    if dist.is_initialized() and same_micro_num_in_dp:  # 注释:如果启用分布式且需要在DP组内同步
        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())  # 注释:转为张量,移到设备上
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)  # 注释:在DP组内同步到最大值(避免死锁)
        num_micro_batches = num_micro_batches.cpu().item()  # 注释:转回CPU标量
    if num_batches_divided_by is not None:  # 注释:如果指定了虚拟管道并行大小
        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)  # 注释:向上舍入到虚拟管道并行大小的倍数

    assert num_micro_batches <= len(seq_len_effective)  # 注释:断言微批次数量不超过batch_size

    # note that seq_len_effective is a GPU tensor. We need to make it a list to avoid D2H!  # 注释:保留原注释,避免GPU->CPU传输
    workloads = calculate_workload(seq_len_effective).cpu().tolist()  # 注释:估算每个样本的FLOPs,转为CPU列表(一次性D2H)
    micro_bsz_idx = get_seqlen_balanced_partitions(workloads, num_micro_batches, equal_size=False)  # 注释:使用KK算法进行负载均衡分片

    if use_dynamic_bsz_balance:  # 注释:如果启用动态负载均衡排序
        # Use the sum of squared sequence lengths to approximate attention computation workload  # 注释:保留原注释
        micro_bsz_idx.sort(  # 注释:对微批次按负载排序
            key=lambda partition: (  # 注释:排序键:先按总负载,再按第一个元素索引
                sum(workloads[idx] for idx in partition),  # 注释:计算分片总负载
                partition[0] if partition else 0,  # 注释:第一个元素索引(分片为空时用0)
            ),  # 注释:排序键结束
            reverse=True,  # 注释:降序排列(负载大的在前)
        )  # 注释:排序结束
        # Place smaller micro-batches at both ends to reduce the bubbles exposed during the warm-up and cool-down.  # 注释:保留原注释,说明重排序策略
        micro_bsz_idx = micro_bsz_idx[::2][::-1] + micro_bsz_idx[1::2]  # 注释:重排序:偶数索引逆序+奇数索引正序(减少管道并行bubble)

    micro_batches = []  # 注释:初始化微批次列表

    for partition in micro_bsz_idx:  # 注释:遍历每个分片(微批次的索引列表)
        curr_micro_batch = tu.index_select_tensor_dict(batch, partition)  # 注释:根据索引从原batch中选择张量,构造微批次
        micro_batches.append(curr_micro_batch)  # 注释:添加到微批次列表

    return micro_batches, micro_bsz_idx  # 注释:返回微批次列表和索引映射


def get_reverse_idx(idx_map):  # 注释:函数签名,构建索引映射的逆映射
    """（函数说明:构建索引映射的逆映射）

    Build the inverse of an index mapping.

    本函数将正向索引映射(idx_map[i]=j)转换为逆映射(reverse[j]=i)。

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.
            - 元素类型:int(目标索引)
            - 长度:任意
            - 示例:[3, 1, 0, 2]表示0->3, 1->1, 2->0, 3->2

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
            - 元素类型:int(源索引)
            - 长度:与idx_map相同
            - 示例:输入[3,1,0,2]返回[2,1,3,0](逆映射)

    副作用:
        - 无(创建新列表)

    异常/边界条件:
        - idx_map包含重复值:后出现的会覆盖先出现的(未定义行为)
        - idx_map包含超出范围的值:会导致IndexError

    最小示例(手算验证):
        >>> # 输入:正向映射[3, 1, 0, 2]
        >>> idx_map = [3, 1, 0, 2]
        >>> reverse_idx = get_reverse_idx(idx_map)
        >>> # 手算:
        >>> # idx_map[0]=3 => reverse[3]=0
        >>> # idx_map[1]=1 => reverse[1]=1
        >>> # idx_map[2]=0 => reverse[0]=2
        >>> # idx_map[3]=2 => reverse[2]=3
        >>> # 预期输出:[2, 1, 3, 0]
        >>> assert reverse_idx == [2, 1, 3, 0]
        >>> # 验证逆映射:reverse[idx_map[i]]==i对所有i成立
        >>> assert all(reverse_idx[idx_map[i]] == i for i in range(len(idx_map)))

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:get_reverse_idx(idx_map)

    典型调用路径:
        restore_dynamic_batch()
        -> get_reverse_idx(indices)
        -> 返回逆映射,用于还原原始顺序

    被谁调用:
        - restore_dynamic_batch()(本文件内,还原微批次数据)

    调用了谁(项目内):
        - 无

    调用了谁(外部依赖):
        - copy.deepcopy():深拷贝列表(避免修改原列表)

    记忆提示:
        - 核心操作:正向映射->逆映射
        - 口诀:i->j变成j->i
    """  # 注释:函数docstring结束
    reverse_idx_map = copy.deepcopy(idx_map)  # 注释:深拷贝idx_map,避免修改原列表

    for i, idx in enumerate(idx_map):  # 注释:遍历正向映射
        reverse_idx_map[idx] = i  # 注释:构造逆映射:reverse[idx]=i

    return reverse_idx_map  # 注释:返回逆映射列表


def prepare_dynamic_batch(  # 注释:函数签名,DataProto格式的动态批次划分函数
    data: DataProto,  # 注释:输入数据(DataProto格式)
    max_token_len: int,  # 注释:每个微批次的最大token数量
    dp_group=None,  # 注释:数据并行组
    num_batches_divided_by=None,  # 注释:虚拟管道并行大小
    same_micro_num_in_dp=True,  # 注释:是否在DP组内同步微批次数量
    min_num_micro_batch=None,  # 注释:最小微批次数量
    use_dynamic_bsz_balance=True,  # 注释:是否使用动态负载均衡排序
) -> tuple[list[DataProto], list[list[int]]]:  # 注释:返回类型:微批次列表和索引映射
    """（函数说明:对DataProto格式的数据进行动态批次划分）

    Prepare a batch for dynamic batching.

    本函数是rearrange_micro_batches的DataProto包装版本,将输入DataProto划分为多个微批次DataProto。

    Args:
        data (DataProto): The input data.
            - 包含batch(TensorDict)、non_tensor_batch(dict)、meta_info(dict)
        max_token_len (int): The maximum token length for dynamic batching.
            - 每个微批次的最大token总数

    Returns:
        Tuple[List[DataProto], List[List[int]]]: A tuple containing a list of DataProto objects
        and a list of index lists.
            - 第1个元素:微批次列表(每个是DataProto对象)
            - 第2个元素:索引映射列表(用于还原)

    副作用:
        - 无(创建新DataProto对象)

    异常/边界条件:
        - 与rearrange_micro_batches相同

    最小示例:
        >>> # 输入:包含4个样本的DataProto
        >>> data = DataProto.from_dict(
        ...     tensors={"input_ids": ..., "attention_mask": ...},
        ...     non_tensors={"prompts": ["p1", "p2", "p3", "p4"]},
        ...     meta_info={"version": "1.0"}
        ... )
        >>> micro_batches, batch_idx_list = prepare_dynamic_batch(
        ...     data, max_token_len=1024
        ... )
        >>> # 预期:micro_batches包含若干DataProto,每个包含一个微批次

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:prepare_dynamic_batch(data, max_token_len, ...)

    典型调用路径:
        训练循环或worker
        -> prepare_dynamic_batch(data, max_token_len, ...)
        -> rearrange_micro_batches(data.batch, ...)
        -> 返回微批次DataProto列表

    被谁调用:
        - verl/workers/actor/dp_actor.py(Actor训练)
        - verl/workers/fsdp_workers.py(FSDP worker)

    调用了谁(项目内):
        - rearrange_micro_batches(data.batch, ...):核心划分逻辑
        - DataProto.from_dict():构造微批次DataProto

    调用了谁(外部依赖):
        - copy.deepcopy():深拷贝meta_info

    记忆提示:
        - 核心功能:DataProto包装版的rearrange_micro_batches
        - 口诀:单批次DataProto->多微批次DataProto列表
    """  # 注释:函数docstring结束
    batch, batch_idx_list = rearrange_micro_batches(  # 注释:调用核心划分函数
        data.batch,  # 注释:提取TensorDict部分
        max_token_len=max_token_len,  # 注释:传递最大token数
        dp_group=dp_group,  # 注释:传递DP组
        num_batches_divided_by=num_batches_divided_by,  # 注释:传递虚拟管道并行大小
        same_micro_num_in_dp=same_micro_num_in_dp,  # 注释:传递DP同步标志
        min_num_micro_batch=min_num_micro_batch,  # 注释:传递最小微批次数量
        use_dynamic_bsz_balance=use_dynamic_bsz_balance,  # 注释:传递负载均衡标志
    )  # 注释:rearrange_micro_batches调用结束
    micro_batches = []  # 注释:初始化微批次列表
    for i, batch_idx in enumerate(batch_idx_list):  # 注释:遍历每个微批次的索引列表
        tensors = dict(batch[i])  # 注释:提取第i个微批次的张量(TensorDict转dict)
        non_tensors = {key: value[batch_idx] for key, value in data.non_tensor_batch.items()}  # 注释:根据索引选择非张量数据(如prompts)
        meta_info = copy.deepcopy(data.meta_info)  # 注释:深拷贝meta_info(每个微批次独立)
        micro_batches.append(DataProto.from_dict(tensors, non_tensors, meta_info=meta_info))  # 注释:构造微批次DataProto并添加到列表

    return micro_batches, batch_idx_list  # 注释:返回微批次DataProto列表和索引映射


def restore_dynamic_batch(data: torch.Tensor, batch_idx_list: list[list[int]]) -> torch.Tensor:  # 注释:函数签名,还原动态批次划分的数据
    """（函数说明:将动态批次划分后的数据还原为原始顺序）

    Restore a batch from dynamic batching.

    本函数将prepare_dynamic_batch划分后的微批次数据还原为原始批次顺序。

    Args:
        data (torch.Tensor): The input data.
            - 微批次拼接后的张量(已按微批次顺序排列)
            - 形状:[total_batch_size, ...]
        batch_idx_list (List[List[int]]): The list of index lists.
            - 索引映射列表,由prepare_dynamic_batch返回
            - 示例:[[0, 3], [1, 2]]

    Returns:
        torch.Tensor: The restored data.
            - 还原为原始顺序的张量
            - 形状:与data相同

    副作用:
        - 无(创建新张量)

    异常/边界条件:
        - len(indices) != batch_size:断言失败
        - 索引超出范围:引发IndexError

    最小示例(手算验证):
        >>> # 输入:微批次划分后的数据
        >>> batch_idx_list = [[0, 3], [1, 2]]  # 微批次0包含原样本0和3,微批次1包含原样本1和2
        >>> # 拼接后的顺序:[0, 3, 1, 2]
        >>> data = torch.tensor([10, 40, 20, 30])  # 按拼接顺序排列的数据
        >>> restored = restore_dynamic_batch(data, batch_idx_list)
        >>> # 预期:还原为原始顺序[0, 1, 2, 3] -> [10, 20, 30, 40]
        >>> assert restored.tolist() == [10, 20, 30, 40]

    调用路径依赖:

    所在位置:
        - 路径:verl/utils/seqlen_balancing.py
        - 函数:restore_dynamic_batch(data, batch_idx_list)

    典型调用路径:
        训练循环或worker
        -> restore_dynamic_batch(outputs, batch_idx_list)
        -> get_reverse_idx(indices)
        -> 返回还原后的张量

    被谁调用:
        - verl/workers/actor/dp_actor.py(Actor训练后还原logits)
        - verl/workers/fsdp_workers.py(FSDP worker还原输出)

    调用了谁(项目内):
        - get_reverse_idx(indices):构造逆映射

    调用了谁(外部依赖):
        - itertools.chain.from_iterable():展平嵌套列表
        - torch.tensor():创建索引张量
        - torch.nested.as_nested_tensor():构造嵌套张量(如果输入是嵌套张量)

    记忆提示:
        - 核心操作:微批次顺序->原始顺序
        - 口诀:展平索引->逆映射->重排张量
    """  # 注释:函数docstring结束
    indices = list(chain.from_iterable(batch_idx_list))  # 注释:展平嵌套索引列表,得到微批次拼接后的顺序
    batch_size = data.shape[0]  # 注释:获取批次大小
    assert len(indices) == batch_size, f"{len(indices)} vs. {batch_size}"  # 注释:断言索引数量与批次大小一致
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)  # 注释:构造逆映射索引张量

    if data.is_nested:  # 注释:如果是嵌套张量(NestedTensor)
        data_lst = data.unbind()  # 注释:解绑为列表
        tensors = [data_lst[i] for i in revert_indices]  # 注释:按逆映射索引重排
        reverted_data = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)  # 注释:重新构造嵌套张量
    else:  # 注释:普通张量
        reverted_data = data[revert_indices]  # 注释:按逆映射索引重排(高级索引)

    return reverted_data  # 注释:返回还原后的张量
