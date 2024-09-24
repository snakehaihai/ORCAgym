import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from .utils import create_MLP  # 从当前包的utils模块中导入create_MLP函数
from .utils import RunningMeanStd  # 从当前包的utils模块中导入RunningMeanStd类


class Critic(nn.Module):
    """
    Critic类，用于估计给定状态的价值，通常在强化学习中的Actor-Critic方法中使用。
    """
    def __init__(self,
                 num_obs,
                 hidden_dims,
                 activation="relu",
                 normalize_obs=True,
                 **kwargs):
        """
        初始化Critic类。

        参数:
            num_obs (int): 输入观测的维度。
            hidden_dims (list): 隐藏层的维度列表。
            activation (str): 激活函数类型，默认为"relu"。
            normalize_obs (bool): 是否对输入观测进行归一化，默认为True。
            **kwargs: 其他未使用的关键字参数。
        """
        if kwargs:
            print("Critic.__init__ 接收到未预期的参数，"
                  "这些参数将被忽略: "
                  + str([key for key in kwargs.keys()]))
        
        super().__init__()  # 调用父类的构造函数

        # 创建一个多层感知机（MLP）作为Critic网络，输出维度为1（价值估计）
        self.NN = create_MLP(num_obs, 1, hidden_dims, activation)

        self._normalize_obs = normalize_obs  # 是否对观测进行归一化
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)  # 初始化运行中的均值和标准差，用于归一化

    def evaluate(self, critic_observations):
        """
        评估给定观测的价值。

        参数:
            critic_observations (torch.Tensor): 输入的观测张量。

        返回:
            torch.Tensor: 估计的价值。
        """
        if self._normalize_obs:
            observations = self.norm_obs(critic_observations)  # 对观测进行归一化
        return self.NN(observations).squeeze()  # 通过网络计算价值并压缩维度

    def norm_obs(self, observation):
        """
        对观测进行归一化处理。

        参数:
            observation (torch.Tensor): 原始观测张量。

        返回:
            torch.Tensor: 归一化后的观测张量。
        """
        with torch.no_grad():  # 禁用梯度计算，以节省内存和计算资源
            return self.obs_rms(observation) if self._normalize_obs else observation
