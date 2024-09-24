import torch.nn as nn  # 导入PyTorch的神经网络模块

from .actor import Actor  # 从当前包中导入Actor模块
from .critic import Critic  # 从当前包中导入Critic模块


class ActorCritic(nn.Module):
    """
    Actor-Critic模型，结合了Actor和Critic两个网络，用于强化学习中的策略和价值评估。
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation="relu",
                 init_noise_std=1.0,
                 **kwargs):
        """
        初始化ActorCritic类。

        参数:
            num_actor_obs (int): Actor输入的观测维度
            num_critic_obs (int): Critic输入的观测维度
            num_actions (int): 动作空间的维度
            actor_hidden_dims (list): Actor网络隐藏层的维度列表
            critic_hidden_dims (list): Critic网络隐藏层的维度列表
            activation (str): 激活函数类型
            init_noise_std (float): 初始化噪声的标准差
            **kwargs: 其他未使用的关键字参数
        """
        if kwargs:
            print(
                "ActorCritic.__init__ 接收到未预期的参数，"
                "这些参数将被忽略: "
                + str([key for key in kwargs.keys()]))
        
        super(ActorCritic, self).__init__()  # 调用父类的构造函数

        # 初始化Actor网络
        self.actor = Actor(num_actor_obs,
                           num_actions,
                           actor_hidden_dims,
                           activation,
                           init_noise_std)

        # 初始化Critic网络
        self.critic = Critic(num_critic_obs,
                             critic_hidden_dims,
                             activation)

        # 打印Actor和Critic的多层感知机结构
        print(f"Actor MLP: {self.actor.NN}")
        print(f"Critic MLP: {self.critic.NN}")

    @property
    def action_mean(self):
        """
        获取Actor输出的动作均值。
        """
        return self.actor.action_mean

    @property
    def action_std(self):
        """
        获取Actor输出的动作标准差。
        """
        return self.actor.action_std

    @property
    def entropy(self):
        """
        获取Actor输出的熵值，用于衡量策略的随机性。
        """
        return self.actor.entropy

    @property
    def std(self):
        """
        获取Actor的标准差。
        """
        return self.actor.std

    def update_distribution(self, observations):
        """
        更新Actor的动作分布，根据新的观测值。

        参数:
            observations: 当前的观测值
        """
        self.actor.update_distribution(observations)

    def act(self, observations, **kwargs):
        """
        根据观测值选择动作。

        参数:
            observations: 当前的观测值
            **kwargs: 其他可选参数

        返回:
            选择的动作
        """
        return self.actor.act(observations)

    def get_actions_log_prob(self, actions):
        """
        获取给定动作的对数概率。

        参数:
            actions: 动作

        返回:
            动作的对数概率
        """
        return self.actor.get_actions_log_prob(actions)

    def act_inference(self, observations):
        """
        在推理模式下根据观测值选择动作（通常为确定性动作）。

        参数:
            observations: 当前的观测值

        返回:
            选择的动作
        """
        return self.actor.act_inference(observations)

    def evaluate(self, critic_observations, **kwargs):
        """
        评估Critic网络的价值估计。

        参数:
            critic_observations: 用于Critic的观测值
            **kwargs: 其他可选参数

        返回:
            价值估计
        """
        return self.critic.evaluate(critic_observations)

    def export_policy(self, path):
        """
        导出Actor的策略网络到指定路径。

        参数:
            path (str): 文件保存路径
        """
        self.actor.export(path)
