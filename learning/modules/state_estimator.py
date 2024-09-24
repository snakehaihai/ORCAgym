import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from .utils import create_MLP  # 从当前包的utils模块中导入create_MLP函数
from .utils import export_network  # 从当前包的utils模块中导入export_network函数


class StateEstimatorNN(nn.Module):
    """
    设置用于状态估计的神经网络。

    关键字参数:
        num_inputs (int): 使用的观测数量
        num_outputs (int): 估计的状态数量
        hidden_dims (list): 隐藏层及其大小的列表（默认值 [256, 128]）
        activation (str): 激活函数类型（默认值 'elu'）
        dropouts (list): 每层的dropout率列表（默认值 None）
    """
    def __init__(self, num_inputs, num_outputs, hidden_dims=[256, 128],
                 activation='elu', dropouts=None, **kwargs):
        """
        初始化StateEstimatorNN类。

        参数:
            num_inputs (int): 输入观测的维度。
            num_outputs (int): 输出估计状态的维度。
            hidden_dims (list): 隐藏层的维度列表。
            activation (str): 激活函数类型。
            dropouts (list): 每层的dropout率列表。
            **kwargs: 其他未使用的关键字参数。
        """
        if kwargs:
            print("StateEstimator.__init__ 接收到未预期的参数，"
                  "这些参数将被忽略: "
                  + str([key for key in kwargs.keys()]))
        
        super().__init__()  # 调用父类的构造函数

        self.num_inputs = num_inputs  # 输入观测的维度
        self.num_outputs = num_outputs  # 输出状态的维度
        # 使用create_MLP函数创建多层感知机（MLP）作为状态估计网络
        self.NN = create_MLP(num_inputs, num_outputs, hidden_dims,
                             activation, dropouts)
        print(f"状态估计器 MLP: {self.NN}")  # 打印创建的MLP结构

    def evaluate(self, observations):
        """
        评估给定观测的状态估计。

        参数:
            observations (torch.Tensor): 输入的观测张量。

        返回:
            torch.Tensor: 估计的状态张量。
        """
        return self.NN(observations)  # 通过网络进行前向传播，得到状态估计

    def export(self, path):
        """
        导出状态估计网络到指定路径。

        参数:
            path (str): 文件保存路径。
        """
        export_network(self.NN, "state_estimator", path, self.num_inputs)  # 调用export_network函数导出网络
