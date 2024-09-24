import torch  # 导入PyTorch库，用于深度学习和张量操作

from gym.envs.base.legged_robot import LeggedRobot  # 从Gym的基类模块导入LeggedRobot类


class MiniCheetah(LeggedRobot):
    """
    MiniCheetah类继承自LeggedRobot，用于定义迷你猎豹机器人的具体行为和奖励函数。
    """
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        """
        初始化MiniCheetah实例。

        参数:
            gym: Gym环境实例。
            sim: 模拟器实例。
            cfg: 配置字典，包含机器人和环境的参数。
            sim_params: 模拟参数。
            sim_device: 模拟设备（如CPU或GPU）。
            headless (bool): 是否以无头模式运行（不显示图形界面）。
        """
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)  # 调用父类的初始化方法

    def _init_buffers(self):
        """
        初始化用于存储机器人状态的缓冲区。
        """
        super()._init_buffers()  # 调用父类的方法初始化缓冲区

    def _reward_lin_vel_z(self):
        """
        奖励函数：惩罚机器人在z轴的线速度，使用平方指数函数。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        return self._sqrdexp(self.base_lin_vel[:, 2] / self.scales["base_lin_vel"])

    def _reward_ang_vel_xy(self):
        """
        奖励函数：惩罚机器人在xy轴的角速度，使用平方指数函数。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        error = self._sqrdexp(self.base_ang_vel[:, :2] / self.scales["base_ang_vel"])
        return torch.sum(error, dim=1)

    def _reward_orientation(self):
        """
        奖励函数：惩罚机器人非平坦的基础姿态，使用平方指数函数。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        error = (torch.square(self.projected_gravity[:, :2]) / self.cfg.reward_settings.tracking_sigma)
        return torch.sum(torch.exp(-error), dim=1)

    def _reward_min_base_height(self):
        """
        奖励函数：惩罚基础高度低于目标值的情况，使用平方指数函数。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        error = (self.base_height - self.cfg.reward_settings.base_height_target)
        error /= self.scales["base_height"]
        error = torch.clamp(error, max=0, min=None).flatten()
        return self._sqrdexp(error)

    def _reward_tracking_lin_vel(self):
        """
        奖励函数：跟踪线速度命令（xy轴），使用平方误差并通过指数函数转换。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        # 计算线速度误差
        error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        # 按(1 + |cmd|)缩放误差，如果cmd=0，则无缩放
        error *= 1. / (1. + torch.abs(self.commands[:, :2]))
        # 计算平方误差的和
        error = torch.sum(torch.square(error), dim=1)
        # 使用指数函数转换为奖励值
        return torch.exp(-error / self.cfg.reward_settings.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        奖励函数：跟踪角速度命令（偏航），使用平方误差并通过指数函数转换。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        ang_vel_error = torch.square((self.commands[:, 2] - self.base_ang_vel[:, 2]) / 5.)
        return self._sqrdexp(ang_vel_error)

    def _reward_dof_vel(self):
        """
        奖励函数：惩罚关节速度，使用平方指数函数。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        return torch.sum(
            self._sqrdexp(self.dof_vel / self.scales["dof_vel"]), dim=1)

    def _reward_dof_near_home(self):
        """
        奖励函数：惩罚关节位置远离初始位置的情况，使用平方指数函数。

        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        return torch.sum(
            self._sqrdexp(
                (self.dof_pos - self.default_dof_pos) / self.scales["dof_pos_obs"]),
            dim=1)


# 辅助函数定义（假设在父类或其他地方定义）
def _sqrdexp(x):
    """
    辅助函数：计算平方指数函数。

    参数:
        x (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 计算后的张量。
    """
    return torch.square(torch.exp(-x))


# 如果需要，可以在这里添加更多辅助函数或类的定义
