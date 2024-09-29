import os  # 导入操作系统接口模块，用于处理文件和目录路径
from isaacgym import gymapi  # 从 isaacgym 库中导入 gymapi，用于与 Isaac Gym 进行交互
from isaacgym import gymutil  # 从 isaacgym 库中导入 gymutil，提供辅助工具函数
from datetime import datetime  # 从 datetime 模块中导入 datetime 类，用于处理日期和时间
from typing import Tuple  # 从 typing 模块中导入 Tuple，用于类型注解

from learning.env import VecEnv  # 从 learning.env 模块中导入 VecEnv 类，表示向量化环境
from learning.runners import OnPolicyRunner  # 从 learning.runners 模块中导入 OnPolicyRunner 类，用于策略运行
from learning.utils import set_discount_from_horizon  # 从 learning.utils 模块中导入 set_discount_from_horizon 函数，用于设置折扣率

from gym import LEGGED_GYM_ROOT_DIR  # 从 gym 模块中导入 LEGGED_GYM_ROOT_DIR，表示项目根目录
from .helpers import \
    update_cfg_from_args, class_to_dict, get_load_path, set_seed  # 从当前包的 helpers 模块中导入多个辅助函数
from gym.envs.base.legged_robot_config import (
    LeggedRobotCfg, LeggedRobotRunnerCfg)  # 从 gym.envs.base.legged_robot_config 模块中导入配置类
from gym.envs.base.base_config import BaseConfig  # 从 gym.envs.base.base_config 模块中导入 BaseConfig 类，作为基础配置
from gym.envs.base.sim_config import SimCfg  # 从 gym.envs.base.sim_config 模块中导入 SimCfg 类，表示仿真配置


class TaskRegistry():
    """任务注册器类，用于注册和管理不同的任务及其配置"""

    def __init__(self):
        """初始化 TaskRegistry 实例，设置任务类、环境配置和训练配置的字典"""
        self.task_classes = {}  # 存储任务名称与对应任务类的映射
        self.env_cfgs = {}  # 存储任务名称与对应环境配置的映射
        self.train_cfgs = {}  # 存储任务名称与对应训练配置的映射
        self.sim_cfg = class_to_dict(SimCfg)  # 将 SimCfg 类转换为字典格式的仿真配置
        self.sim = {}  # 存储仿真相关的配置和参数

    def register(self, name: str, task_class: VecEnv,
                 env_cfg: BaseConfig, train_cfg: LeggedRobotRunnerCfg):
        """注册一个新的任务，包括任务类、环境配置和训练配置

        Args:
            name (str): 任务的名称
            task_class (VecEnv): 任务对应的环境类
            env_cfg (BaseConfig): 环境配置
            train_cfg (LeggedRobotRunnerCfg): 训练配置
        """
        self.task_classes[name] = task_class  # 将任务类添加到任务类字典中
        self.env_cfgs[name] = env_cfg  # 将环境配置添加到环境配置字典中
        self.train_cfgs[name] = train_cfg  # 将训练配置添加到训练配置字典中

    def get_task_class(self, name: str) -> VecEnv:
        """根据任务名称获取对应的任务类

        Args:
            name (str): 任务的名称

        Returns:
            VecEnv: 对应的任务类
        """
        return self.task_classes[name]  # 返回指定任务名称的任务类

    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotRunnerCfg]:
        """获取指定任务的环境配置和训练配置

        Args:
            name (str): 任务的名称

        Returns:
            Tuple[LeggedRobotCfg, LeggedRobotRunnerCfg]: 环境配置和训练配置的元组
        """
        env_cfg = self.env_cfgs[name]  # 获取指定任务的环境配置
        train_cfg = self.train_cfgs[name]  # 获取指定任务的训练配置
        # * 复制种子值，将训练配置中的种子赋值给环境配置
        env_cfg.seed = train_cfg.seed  
        return env_cfg, train_cfg  # 返回环境配置和训练配置

    def create_cfgs(self, args):
        """创建并更新环境配置和训练配置

        Args:
            args: 命令行参数或其他参数对象

        Returns:
            Tuple[LeggedRobotCfg, LeggedRobotRunnerCfg]: 更新后的环境配置和训练配置
        """
        env_cfg, train_cfg = self.get_cfgs(name=args.task)  # 获取指定任务的配置
        self.update_and_parse_cfgs(env_cfg, train_cfg, args)  # 更新并解析配置
        self.set_log_dir_name(train_cfg)  # 设置日志目录名称
        return env_cfg, train_cfg  # 返回更新后的配置

    def set_log_dir_name(self, train_cfg, log_root="default"):
        """设置训练日志的目录名称

        Args:
            train_cfg: 训练配置对象
            log_root (str, optional): 日志根目录。默认为 "default"
        """
        if log_root == "default":  # 如果使用默认的日志根目录
            log_root = os.path.join(
                LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)  # 拼接日志根目录路径
            log_dir = os.path.join(
                log_root, datetime.now().strftime('%b%d_%H-%M-%S')
                + '_' + train_cfg.runner.run_name)  # 生成带有时间戳和运行名称的日志目录路径
        elif log_root is None:  # 如果日志根目录为 None
            log_dir = None  # 日志目录设置为 None
        train_cfg.log_dir = log_dir  # 将生成的日志目录路径赋值给训练配置

    def update_and_parse_cfgs(self, env_cfg, train_cfg, args):
        """更新并解析环境配置和训练配置

        Args:
            env_cfg: 环境配置对象
            train_cfg: 训练配置对象
            args: 命令行参数或其他参数对象
        """
        update_cfg_from_args(env_cfg, train_cfg, args)  # 根据参数更新配置
        self.convert_frequencies_to_params(env_cfg, train_cfg)  # 转换频率参数
        self.update_sim_cfg(args)  # 更新仿真配置

    def convert_frequencies_to_params(self, env_cfg, train_cfg):
        """将控制频率转换为仿真参数，并设置折扣率

        Args:
            env_cfg: 环境配置对象
            train_cfg: 训练配置对象
        """
        self.set_control_and_sim_dt(env_cfg, train_cfg)  # 设置控制和仿真时间步长
        self.set_discount_rates(train_cfg, env_cfg.control.ctrl_dt)  # 设置折扣率

    def set_control_and_sim_dt(self, env_cfg, train_cfg):
        """设置控制频率和仿真时间步长

        Args:
            env_cfg: 环境配置对象
            train_cfg: 训练配置对象
        """
        env_cfg.control.decimation = int(env_cfg.control.desired_sim_frequency
                                         / env_cfg.control.ctrl_frequency)  # 计算 decimation 值
        env_cfg.control.ctrl_dt = 1.0 / env_cfg.control.ctrl_frequency  # 计算控制时间步长
        env_cfg.sim_dt = env_cfg.control.ctrl_dt / env_cfg.control.decimation  # 计算仿真时间步长
        self.sim_cfg["dt"] = env_cfg.sim_dt  # 更新仿真配置中的时间步长
        if env_cfg.sim_dt != 1.0 / env_cfg.control.desired_sim_frequency:  # 如果仿真时间步长不等于期望值
            print(f'****** Simulation dt adjusted from '
                  f'{1.0/env_cfg.control.desired_sim_frequency}'
                  f' to {env_cfg.sim_dt}.')  # 打印调整后的仿真时间步长

    def set_discount_rates(self, train_cfg, dt):
        """根据时间步长设置折扣率

        Args:
            train_cfg: 训练配置对象
            dt (float): 时间步长
        """
        if hasattr(train_cfg.algorithm, 'discount_horizon'):  # 如果训练算法有 discount_horizon 属性
            hrzn = train_cfg.algorithm.discount_horizon  # 获取 discount_horizon 值
            train_cfg.algorithm.gamma = set_discount_from_horizon(dt, hrzn)  # 设置 gamma 折扣率

        if hasattr(train_cfg.algorithm, 'GAE_bootstrap_horizon'):  # 如果训练算法有 GAE_bootstrap_horizon 属性
            hrzn = train_cfg.algorithm.GAE_bootstrap_horizon  # 获取 GAE_bootstrap_horizon 值
            train_cfg.algorithm.lam = set_discount_from_horizon(dt, hrzn)  # 设置 lam 折扣率

    def update_sim_cfg(self, args):
        """更新仿真配置参数

        Args:
            args: 命令行参数或其他参数对象
        """
        self.sim["sim_device"] = args.sim_device  # 设置仿真设备
        self.sim["sim_device_id"] = args.sim_device_id  # 设置仿真设备 ID
        self.sim["graphics_device_id"] = args.graphics_device_id  # 设置图形设备 ID
        self.sim["physics_engine"] = args.physics_engine  # 设置物理引擎类型
        self.sim["headless"] = args.headless  # 设置是否无头模式
        if self.sim["headless"]:  # 如果是无头模式
            self.sim["graphics_device_id"] = -1  # 设置图形设备 ID 为 -1，表示不使用图形
        self.sim["params"] = gymapi.SimParams()  # 创建新的仿真参数对象
        self.sim["params"].physx.use_gpu = args.use_gpu  # 设置是否使用 GPU
        self.sim["params"].physx.num_subscenes = args.subscenes  # 设置物理引擎的子场景数量
        self.sim["params"].use_gpu_pipeline = args.use_gpu_pipeline  # 设置是否使用 GPU 管线
        gymutil.parse_sim_config(self.sim_cfg, self.sim["params"])  # 解析并应用仿真配置到仿真参数

    def make_gym_and_sim(self):
        """创建 Gym 实例和仿真实例"""
        self.make_gym()  # 创建 Gym 实例
        self.make_sim()  # 创建仿真实例

    def make_gym(self):
        """获取并设置 Gym 实例"""
        self._gym = gymapi.acquire_gym()  # 通过 gymapi 获取 Gym 实例并赋值给 _gym 属性

    def make_sim(self):
        """创建仿真实例并设置到 _sim 属性"""
        self._sim = self._gym.create_sim(
            self.sim["sim_device_id"],  # 仿真设备 ID
            self.sim["graphics_device_id"],  # 图形设备 ID
            self.sim["physics_engine"],  # 物理引擎类型
            self.sim["params"])  # 仿真参数

    def make_env(self, name, env_cfg) -> VecEnv:
        """根据任务名称和环境配置创建环境实例

        Args:
            name (str): 任务的名称
            env_cfg: 环境配置对象

        Returns:
            VecEnv: 创建的环境实例
        """
        if name in self.task_classes:  # 如果任务名称已注册
            task_class = self.get_task_class(name)  # 获取对应的任务类
        else:
            raise ValueError(f"Task with name: {name} was not registered")  # 未注册任务则抛出错误
        set_seed(env_cfg.seed)  # 设置随机种子
        env = task_class(gym=self._gym, sim=self._sim, cfg=env_cfg,
                         sim_params=self.sim["params"],
                         sim_device=self.sim["sim_device"],
                         headless=self.sim["headless"])  # 创建环境实例
        return env  # 返回创建的环境实例

    def make_alg_runner(self, env, train_cfg):
        """创建算法运行器实例，并根据需要加载预训练模型

        Args:
            env: 环境实例
            train_cfg: 训练配置对象

        Returns:
            OnPolicyRunner: 创建的算法运行器实例
        """
        train_cfg_dict = class_to_dict(train_cfg)  # 将训练配置类转换为字典
        runner = OnPolicyRunner(env, train_cfg_dict, train_cfg.runner.device)  # 创建 OnPolicyRunner 实例

        # * 在创建新的日志目录之前保存恢复路径
        if train_cfg.runner.resume:  # 如果需要恢复训练
            resume_path = get_load_path(name=train_cfg.runner.experiment_name,
                                        load_run=train_cfg.runner.load_run,
                                        checkpoint=train_cfg.runner.checkpoint)  # 获取恢复路径
            print(f"Loading model from: {resume_path}")  # 打印恢复路径
            runner.load(resume_path)  # 加载预训练模型
        return runner  # 返回创建的算法运行器实例


# 创建全局的任务注册器实例
task_registry = TaskRegistry()  # 实例化 TaskRegistry，用于全局任务管理
