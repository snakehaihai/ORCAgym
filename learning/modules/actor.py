import torch  # 导入 PyTorch 库，主要用于张量操作和深度学习模型构建
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.distributions import Normal  # 导入正态分布类，用于动作生成中的噪声处理
from .utils import RunningMeanStd  # 从 utils 模块中导入 RunningMeanStd 类，用于观测值的归一化
from .utils import create_MLP  # 从 utils 模块中导入 create_MLP 函数，用于创建多层感知器
from .utils import export_network  # 从 utils 模块中导入 export_network 函数，用于导出神经网络权重


class Actor(nn.Module):
    """ 
    Actor 类继承自 PyTorch 的 nn.Module 类，表示策略网络，
    用于根据观测值生成动作，同时使用高斯分布对动作进行建模。
    """

    def __init__(self,
                 num_obs,  # 观测值的维度
                 num_actions,  # 动作的维度
                 hidden_dims,  # 隐藏层的维度，用于构建神经网络
                 activation="relu",  # 激活函数，默认为 ReLU
                 init_noise_std=1.0,  # 初始动作噪声标准差
                 normalize_obs=True,  # 是否归一化观测值
                 **kwargs):
        """
        初始化 Actor 类。
        
        参数：
        - num_obs: 观测值的维度
        - num_actions: 动作空间的维度
        - hidden_dims: 神经网络隐藏层的维度
        - activation: 激活函数类型
        - init_noise_std: 初始化动作噪声的标准差
        - normalize_obs: 是否对观测值进行归一化
        """
        
        # 如果有不期望的参数传入，发出警告
        if kwargs:
            print("Actor.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()  # 调用父类 nn.Module 的初始化函数

        self._normalize_obs = normalize_obs  # 是否归一化观测值的标志
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)  # 创建 RunningMeanStd 对象，用于观测值的归一化

        self.num_obs = num_obs  # 观测值维度
        self.num_actions = num_actions  # 动作维度
        # 创建神经网络，输入维度是观测值维度，输出维度是动作维度
        self.NN = create_MLP(num_obs, num_actions, hidden_dims, activation)

        # 初始化动作噪声的标准差为可训练参数
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None  # 动作分布初始化为 None
        # 禁用分布参数的验证以加速计算
        Normal.set_default_validate_args = False

    # 动作标准差
    @property
    def action_std(self):
        return self.std

    # 动作均值
    @property
    def action_mean(self):
        return self.distribution.mean

    # 动作分布的标准差
    @property
    def action_std(self):
        return self.distribution.stddev

    # 动作分布的熵
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # 更新动作分布
    def update_distribution(self, observations):
        if self._normalize_obs:
            observations = self.norm_obs(observations)  # 如果需要，对观测值进行归一化
        mean = self.NN(observations)  # 通过神经网络生成动作的均值
        self.distribution = Normal(mean, mean * 0. + self.std)  # 生成一个正态分布，均值是动作，标准差是训练的参数

    # 生成动作
    def act(self, observations):
        self.update_distribution(observations)  # 更新动作分布
        return self.distribution.sample()  # 从分布中采样动作

    # 计算动作的对数概率
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)  # 计算动作的对数概率，用于策略梯度更新

    # 推理时生成动作
    def act_inference(self, observations):
        if self._normalize_obs:
            observations = self.norm_obs(observations)  # 归一化观测值
        actions_mean = self.NN(observations)  # 生成动作均值
        return actions_mean  # 返回动作均值（不加噪声）

    # 导出模型
    def export(self, path):
        export_network(self.NN, "policy", path, self.num_obs)  # 将策略网络的权重导出到指定路径

    # 归一化观测值
    def norm_obs(self, observation):
        with torch.no_grad():  # 在不进行梯度更新的情况下对观测值进行归一化
            return self.obs_rms(observation) if self._normalize_obs else observation
