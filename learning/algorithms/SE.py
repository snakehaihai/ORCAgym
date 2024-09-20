import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块

from learning.modules import StateEstimatorNN  # 从自定义模块中导入状态估计神经网络类
from learning.storage import SERolloutStorage  # 从自定义模块中导入状态估计的回滚存储类

# 定义状态估计器类，用于学习状态估计
class StateEstimator:
    """ 该类提供了一个经过学习的状态估计器。
        使用监督学习进行训练，只使用从回滚存储中收集的策略数据。
        predict() 函数提供了用于强化学习的状态估计
        update() 函数用于优化神经网络参数
        process_env_step() 函数用于将值存储在回滚存储中
    """
    state_estimator: StateEstimatorNN  # 定义状态估计神经网络

    # 初始化函数，设置各类参数和组件
    def __init__(self,
                 state_estimator,    # 神经网络模块
                 learning_rate=1e-3,  # 学习率，默认值为1e-3
                 num_mini_batches=1,  # 每次更新的最小批次数量
                 num_learning_epochs=1,  # 学习周期数
                 device='cpu',  # 运行设备，默认使用CPU
                 **kwargs
                 ):

        # 通用参数
        self.device = device  # 设置设备
        self.learning_rate = learning_rate  # 设置学习率
        self.num_mini_batches = num_mini_batches  # 设置最小批次数量
        self.num_learning_epochs = num_learning_epochs  # 设置学习周期数

        # 状态估计存储
        self.transition = SERolloutStorage.Transition()  # 初始化存储过渡
        self.storage = None  # 存储器初始化为空

        # 状态估计网络和优化器
        self.state_estimator = state_estimator  # 初始化状态估计神经网络
        self.state_estimator.to(self.device)  # 将神经网络移动到指定设备上
        self.optimizer = optim.Adam(self.state_estimator.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.SE_loss_fn = nn.MSELoss()  # 使用均方误差（MSE）作为损失函数

    # 初始化存储器，指定环境数量、每个环境的转移次数、观察形状和状态估计形状
    def init_storage(self, num_envs, num_transitions_per_env,
                     obs_shape, se_shape):
        # 初始化回滚存储器，用于保存强化学习过程中的状态和观察
        self.storage = SERolloutStorage(num_envs, num_transitions_per_env,
                                        obs_shape, se_shape,
                                        device=self.device)

    # 预测函数，根据观测值预测状态
    def predict(self, obs):
        return self.state_estimator.evaluate(obs)  # 使用状态估计器评估观察并返回结果

    # 处理环境中的一步，将当前的观测和状态估计目标存储到回滚存储中
    def process_env_step(self, obs, SE_targets):
        # 记录当前的转移
        self.transition.SE_targets = SE_targets  # 保存状态估计目标
        self.transition.observations = obs  # 保存当前的观察
        self.storage.add_transitions(self.transition)  # 将转移添加到存储中
        self.transition.clear()  # 清空过渡

    # 更新函数，用于通过监督学习更新神经网络的权重
    def update(self):
        """ 通过监督学习更新状态估计神经网络的权重 """
        # 从存储中生成最小批次的训练数据
        generator = self.storage.mini_batch_generator(self.num_mini_batches,
                                                      self.num_learning_epochs)
        mean_loss = 0  # 初始化平均损失为0
        # 遍历生成器，获取批次数据并进行训练
        for obs_batch, SE_target_batch in generator:
            SE_estimate_batch = self.state_estimator.evaluate(obs_batch)  # 预测状态估计
            SE_loss = self.SE_loss_fn(SE_estimate_batch, SE_target_batch)  # 计算损失

            self.optimizer.zero_grad()  # 清除梯度
            SE_loss.backward()  # 反向传播
            self.optimizer.step()  # 更新模型参数

            mean_loss += SE_loss.item()  # 累加损失

        num_updates = self.num_learning_epochs * self.num_mini_batches  # 计算总的更新次数
        mean_loss /= num_updates  # 计算平均损失
        self.storage.clear()  # 清空存储器

        return mean_loss  # 返回平均损失

    # 导出模型的权重到指定路径
    def export(self, path):
        self.state_estimator.export(path)  # 导出状态估计神经网络的权重
