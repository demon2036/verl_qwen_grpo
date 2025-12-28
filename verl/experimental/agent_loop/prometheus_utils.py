# Copyright 2025 Meituan Ltd. and/or its affiliates  # 注释：版权声明
#  # 注释：空行占位
# Licensed under the Apache License, Version 2.0 (the "License");  # 注释：Apache 2.0 许可证声明
# you may not use this file except in compliance with the License.  # 注释：使用需遵守许可证
# You may obtain a copy of the License at  # 注释：许可证获取提示
#  # 注释：空行占位
#     http://www.apache.org/licenses/LICENSE-2.0  # 注释：许可证链接
#  # 注释：空行占位
# Unless required by applicable law or agreed to in writing, software  # 注释：免责声明开头
# distributed under the License is distributed on an "AS IS" BASIS,  # 注释：按现状提供
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 注释：无担保声明
# See the License for the specific language governing permissions and  # 注释：更多许可条款
# limitations under the License.  # 注释：许可限制说明

"""
模块用途：更新 Prometheus 配置并在 Ray 集群节点上触发 reload。  # 注释：模块用途
输入：PrometheusConfig（含 config.file/port）与 rollout server 地址列表。  # 注释：输入说明
输出：写入配置文件并触发 Prometheus reload（无显式返回）。  # 注释：输出说明
关键依赖：ray、yaml、os、logging。  # 注释：依赖说明
典型用法：  # 注释：用法标题
- update_prometheus_config(config, ["ip:port", ...])  # 注释：最小示例
调用路径概览：  # 注释：调用路径标题
- AgentLoopManager / Rollout 管理 -> update_prometheus_config。  # 注释：调用链
"""  # 注释：模块 docstring 结束

import logging  # 注释：标准库，日志输出
import os  # 注释：标准库，文件与路径操作

import ray  # 注释：第三方库，Ray 任务调度
import yaml  # 注释：第三方库，YAML 序列化

from verl.workers.config.rollout import PrometheusConfig  # 注释：项目内配置类型

logger = logging.getLogger(__file__)  # 注释：获取模块级 logger
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))  # 注释：设置日志级别


def update_prometheus_config(config: PrometheusConfig, server_addresses: list[str]):  # 注释：更新 Prometheus 配置并触发 reload
    """
    功能：根据 rollout server 地址列表生成 Prometheus 配置并在所有 Ray 节点上写入/重载。  # 注释：函数用途
    参数：  # 注释：参数说明标题
    - config (PrometheusConfig)：包含 config.file 与 port 的配置对象。  # 注释：参数含义
    - server_addresses (list[str])：vLLM/sglang 服务器地址列表。  # 注释：参数含义
    返回：  # 注释：返回值说明标题
    - None（无显式返回）。  # 注释：返回值语义
    副作用：  # 注释：副作用说明标题
    - 在各节点写入 YAML 文件并向 Prometheus 发起 reload 请求。  # 注释：副作用说明
    异常/边界条件：  # 注释：异常说明标题
    - 地址为空时直接 warning 返回；异常捕获后记录错误日志。  # 注释：异常说明
    最小示例：  # 注释：最小示例标题
    - update_prometheus_config(cfg, ["127.0.0.1:9090"])。  # 注释：示例
    调用路径依赖：  # 注释：调用路径说明标题
    - 所在位置：verl/experimental/agent_loop/prometheus_utils.py::update_prometheus_config。  # 注释：函数位置
    - 典型调用路径：AgentLoopManager._initialize_llm_servers -> update_prometheus_config。  # 注释：调用链
    - 被谁调用：verl/experimental/agent_loop/agent_loop.py。  # 注释：调用方说明
    - 调用了谁（项目内）：PrometheusConfig。  # 注释：项目内依赖
    - 调用了谁（关键外部依赖）：ray、yaml、os。  # 注释：外部依赖
    """  # 注释：函数 docstring 结束

    """
    Update Prometheus configuration file with server addresses and reload on first node.

    server_addresses: vllm or sglang server addresses
    """  # 注释：保留原英文说明（便于对照）

    if not server_addresses:  # 注释：地址列表为空
        logger.warning("No server addresses available to update Prometheus config")  # 注释：告警并返回
        return  # 注释：提前退出

    try:  # 注释：异常捕获开始
        # Get Prometheus config file path from environment or use default  # 注释：原注释保留（配置文件说明）
        prometheus_config_json = {  # 注释：构造 Prometheus 配置字典
            "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},  # 注释：全局采样/评估间隔
            "scrape_configs": [  # 注释：采集配置列表
                {  # 注释：Ray metrics 采集
                    "job_name": "ray",  # 注释：任务名
                    "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],  # 注释：Ray 服务发现文件
                },  # 注释：ray job 配置结束
                {"job_name": "rollout", "static_configs": [{"targets": server_addresses}]},  # 注释：rollout 静态目标
            ],  # 注释：scrape_configs 结束
        }  # 注释：prometheus_config_json 结束

        # Write configuration file to all nodes  # 注释：原注释保留（写配置到所有节点）
        @ray.remote(num_cpus=0)  # 注释：Ray 远程函数（不占 CPU）
        def write_config_file(config_data, config_path):  # 注释：写配置文件到指定路径
            os.makedirs(os.path.dirname(config_path), exist_ok=True)  # 注释：确保目录存在
            with open(config_path, "w") as f:  # 注释：打开文件写入
                yaml.dump(config_data, f, default_flow_style=False, indent=2)  # 注释：写入 YAML
            return True  # 注释：返回成功标记

        # Reload Prometheus on all nodes. Only master node should succeed, skip errors on other nodes.  # 注释：原注释保留（重载说明）
        @ray.remote(num_cpus=0)  # 注释：Ray 远程函数
        def reload_prometheus(port):  # 注释：触发 Prometheus reload
            import socket  # 注释：标准库，获取主机信息
            import subprocess  # 注释：标准库，调用 curl

            hostname = socket.gethostname()  # 注释：获取主机名
            ip_address = socket.gethostbyname(hostname)  # 注释：解析 IP 地址

            reload_url = f"http://{ip_address}:{port}/-/reload"  # 注释：拼接 reload URL

            try:  # 注释：捕获 reload 异常
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)  # 注释：调用 curl 触发重载
                print(f"Reloading Prometheus on node: {reload_url}")  # 注释：打印重载日志
            except Exception:  # 注释：捕获异常
                # Skip errors on non-master nodes  # 注释：原注释保留（忽略非主节点错误）
                pass  # 注释：忽略异常

        # Get all available nodes and schedule tasks on each node  # 注释：原注释保留（获取节点）
        nodes = ray.nodes()  # 注释：获取 Ray 节点列表
        alive_nodes = [node for node in nodes if node["Alive"]]  # 注释：过滤存活节点

        # Write config files on all nodes  # 注释：原注释保留（写配置）
        write_tasks = []  # 注释：任务列表
        for node in alive_nodes:  # 注释：遍历存活节点
            node_ip = node["NodeManagerAddress"]  # 注释：节点 IP
            task = write_config_file.options(  # 注释：配置任务调度资源
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node  # 注释：绑定到指定节点
            ).remote(prometheus_config_json, config.file)  # 注释：提交写配置任务
            write_tasks.append(task)  # 注释：记录任务句柄

        ray.get(write_tasks)  # 注释：等待写配置完成

        print(f"Updated Prometheus configuration at {config.file} with {len(server_addresses)} VLLM servers")  # 注释：打印完成信息

        # Reload Prometheus on all nodes  # 注释：原注释保留（重载）
        reload_tasks = []  # 注释：重载任务列表
        for node in alive_nodes:  # 注释：遍历存活节点
            node_ip = node["NodeManagerAddress"]  # 注释：节点 IP
            task = reload_prometheus.options(  # 注释：配置重载任务调度资源
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node  # 注释：绑定到指定节点
            ).remote(config.port)  # 注释：提交重载任务
            reload_tasks.append(task)  # 注释：记录任务句柄

        ray.get(reload_tasks)  # 注释：等待重载完成

    except Exception as e:  # 注释：捕获总体异常
        logger.error(f"Failed to update Prometheus configuration: {e}")  # 注释：记录错误日志
