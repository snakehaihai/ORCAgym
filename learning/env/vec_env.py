# 导入需要的模块
from abc import ABC, abstractmethod  # 导入抽象基类模块和抽象方法装饰器，用于定义接口类
import torch  # 导入PyTorch库，用于张量操作
from typing import Tuple, Union  # 导入类型注解工具，帮助定义函数输入和返回值的类型
# 定义一个抽象类 VecEnv，表示一个向量化的强化学习环境接口

class VecEnv(ABC):
    """
    VecEnv 是一个强化学习环境接口的抽象类，继承自Python的ABC（Abstract Base Class）模块，
    该类规定了所有环境类必须实现的基本方法和属性。它主要用于处理多个并行环境，
    即向量化环境。这在强化学习中非常常见，比如多环境并行的情况下加速训练。
    """
    
    # 属性定义
    num_envs: int  # 表示并行运行的环境数量，通常是多个环境实例
    num_obs: int  # 观测值的数量，每个环境中可能有多个观测值，如传感器数据
    num_privileged_obs: int  # 特权观测值的数量（如果有），用于强化学习中的部分可见或完全可见信息
    num_actions: int  # 动作数量，表示每个环境可以采取的动作数
    max_episode_length: int  # 每个环境中的最大回合长度（episode length），即回合能持续的最大时间步
    privileged_obs_buf: torch.Tensor  # 存储特权观测值的缓冲区，类型为PyTorch张量
    obs_buf: torch.Tensor  # 存储普通观测值的缓冲区，类型为PyTorch张量
    rew_buf: torch.Tensor  # 存储奖励的缓冲区，每个时间步都会更新环境的奖励信息
    reset_buf: torch.Tensor  # 存储环境重置标志的缓冲区，当某个环境需要重置时会设置对应的标志位
    episode_length_buf: torch.Tensor  # 记录当前回合的时长，每个环境独立跟踪自己的回合时长
    extras: dict  # 用于存储额外的环境信息，字典格式，可能包含如环境状态、调试信息等
    device: torch.device  # 指定环境运行的设备（CPU或GPU），PyTorch的设备类型

    # step 函数：执行环境中的一步
    @abstractmethod
    def step(self, actions: torch.Tensor) \
        -> Tuple[
            torch.Tensor, Union[torch.Tensor, None],
            torch.Tensor, torch.Tensor, dict]:
        """
        step() 是环境中的一个抽象方法，用于执行环境中的一步动作。
        子类必须实现该方法，以根据传入的动作更新环境的状态，并返回一系列信息。

        参数:
            actions (torch.Tensor): 包含多个环境的动作，动作的形状取决于环境的动作空间。

        返回:
            Tuple[
                torch.Tensor: 每个环境的观测值张量，
                Union[torch.Tensor, None]: 特权观测值（如果环境不提供特权观测值，则返回None），
                torch.Tensor: 每个环境的奖励，
                torch.Tensor: 重置标志，指示哪些环境需要在下一步被重置，
                dict: 包含额外信息的字典（例如调试信息或其他环境状态）。
            ]
        """
        pass  # 这是抽象方法，子类需要实现具体逻辑

    # reset 函数：重置环境
    @abstractmethod
    def reset(self, env_ids: Union[list, torch.Tensor]):
        """
        reset() 是环境的抽象方法，用于在某些条件下重置指定的环境。
        当某个环境完成一个回合（episode）时，需要重置它的状态。

        参数:
            env_ids (Union[list, torch.Tensor]): 需要重置的环境ID，可以是列表或张量形式。
        
        功能：
        重置指定ID的环境，通常是将环境返回到初始状态，例如重新随机化起始位置等。
        """
        pass  # 这是抽象方法，子类需要实现具体逻辑

    # get_observations 函数：获取环境的当前观测值
    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        """
        get_observations() 是环境的抽象方法，用于获取每个环境的当前观测值。
        通常每个环境的观测值可以是传感器数据、图像、物理状态（如速度、位置）等。

        返回:
            torch.Tensor: 当前所有环境的观测值，形状取决于环境的观测空间。
        """
        pass  # 这是抽象方法，子类需要实现具体逻辑

    # get_privileged_observations 函数：获取特权观测值
    @abstractmethod
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        """
        get_privileged_observations() 是一个抽象方法，返回环境的特权观测值。
        特权观测值通常是完全信息，可能对智能体不可见，主要用于仿真、训练或评估。

        返回:
            Union[torch.Tensor, None]: 返回包含特权观测值的张量，如果没有特权观测值，则返回None。
        """
        pass  # 这是抽象方法，子类需要实现具体逻辑
