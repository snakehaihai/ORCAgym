import torch  # 导入PyTorch库


def split_and_pad_trajectories(tensor, dones):
    """
    在完成标志（dones）索引处分割轨迹。然后将它们连接起来，并用零填充到最长轨迹的长度。
    返回对应于轨迹有效部分的掩码。

    示例:
        输入: [ [a1, a2, a3, a4 | a5, a6],
               [b1, b2 | b3, b4, b5 | b6]
              ]

        输出:[ [a1, a2, a3, a4], | [  [True, True, True, True],
               [a5, a6, 0, 0],   |    [True, True, False, False],
               [b1, b2, 0, 0],   |    [True, True, False, False],
               [b3, b4, b5, 0],  |    [True, True, True, False],
               [b6, 0, 0, 0]     |    [True, False, False, False],
              ]                  | ]

    假设输入的维度顺序如下:
    [时间步, 环境数量, 其他维度]
    """
    dones = dones.clone()  # 克隆dones张量以避免修改原始数据
    dones[-1] = 1  # 将最后一个时间步的done标志设置为True，确保所有轨迹都被正确分割
    # * 转置缓冲区以获得顺序 (环境数量, 每个环境的时间步, 其他维度)，以便正确重塑
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # * 通过计算连续非完成状态的数量来获取轨迹的长度
    done_indices = torch.cat(
        (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]  # 计算每个轨迹的长度
    trajectory_lengths_list = trajectory_lengths.tolist()  # 转换为Python列表

    # * 提取单个轨迹
    trajectories = torch.split(
        tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list
    )
    # 使用pad_sequence对轨迹进行填充，使所有轨迹具有相同的长度
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)

    # 创建轨迹的掩码，标记有效的部分
    trajectory_masks = trajectory_lengths > torch.arange(
        0, tensor.shape[0], device=tensor.device
    ).unsqueeze(1)
    return padded_trajectories, trajectory_masks  # 返回填充后的轨迹和对应的掩码


def unpad_trajectories(trajectories, masks):
    """ 
    执行 split_and_pad_trajectories() 的逆操作，将填充的轨迹和掩码还原为原始轨迹。
    
    参数:
        trajectories (torch.Tensor): 填充后的轨迹张量。
        masks (torch.Tensor): 对应的掩码张量，标记轨迹中的有效部分。
    
    返回:
        torch.Tensor: 原始轨迹张量，去除了填充的部分。
    """
    # * 在掩码前后进行转置以确保正确的重塑
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(
        -1, trajectories.shape[0], trajectories.shape[-1]
    ).transpose(1, 0)


def remove_zero_weighted_rewards(reward_weights):
    """
    移除奖励权重字典中权重为零的奖励项。
    
    参数:
        reward_weights (dict): 包含奖励名称和对应权重的字典。
    """
    for name in list(reward_weights.keys()):
        if reward_weights[name] == 0:
            reward_weights.pop(name)  # 移除权重为零的奖励项


def set_discount_from_horizon(dt, horizon):
    """ 
    根据期望的折扣视界（horizon）和时间步长（dt）计算折扣因子。
    
    参数:
        dt (float): 时间步长，必须大于0。
        horizon (float): 期望的折扣视界，必须大于等于dt。
    
    返回:
        float: 计算得到的折扣因子。
    """
    assert (dt > 0), "Invalid time-step"  # 确保时间步长有效
    if horizon == 0:
        discount_factor = 0  # 如果视界为0，则折扣因子为0
    else:
        assert (horizon >= dt), "Invalid discounting horizon"  # 确保视界大于等于时间步长
        discrete_time_horizon = int(horizon / dt)  # 计算离散时间视界
        discount_factor = 1 - 1 / discrete_time_horizon  # 计算折扣因子

    return discount_factor  # 返回计算得到的折扣因子
