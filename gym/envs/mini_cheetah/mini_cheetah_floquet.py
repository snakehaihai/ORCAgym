import torch  # 导入PyTorch库，用于深度学习和张量操作
import pandas as pd  # 导入Pandas库，用于数据处理
from isaacgym.torch_utils import torch_rand_float, to_torch  # 从Isaac Gym的torch工具中导入随机浮点数生成和张量转换函数

from gym.utils.math import exp_avg_filter  # 从Gym的数学工具中导入指数移动平均滤波函数

from gym import LEGGED_GYM_ROOT_DIR  # 导入LeGGed Gym的根目录路径
from gym.envs.mini_cheetah.mini_cheetah_osc import MiniCheetahOsc  # 从Gym的迷你猎豹环境中导入MiniCheetahOsc类

MINI_CHEETAH_MASS = 8.292 * 9.81  # 定义迷你猎豹的质量（牛顿），质量乘以重力加速度

class MiniCheetahFloquet(MiniCheetahOsc):
    """
    MiniCheetahFloquet类继承自MiniCheetahOsc，用于定义迷你猎豹机器人的Floquet分析相关功能。
    """
    
    def _init_buffers(self):
        """
        初始化缓冲区，用于存储上一周期的相位信息、交叉点和Floquet奖励等。
        """
        super()._init_buffers()  # 调用父类的方法初始化基础缓冲区

        # 初始化用于存储各个周期的相位信息的缓冲区
        self.last_phase_1 = torch.zeros(self.num_envs, 4, 2, device=self.device)
        self.last_phase_2 = torch.zeros(self.num_envs, 4, 2, device=self.device)
        self.last_phase_3 = torch.zeros(self.num_envs, 4, 2, device=self.device)
        self.last_phase_4 = torch.zeros(self.num_envs, 4, 2, device=self.device)
        
        # 初始化交叉点缓冲区，用于标记振荡器是否跨越2π
        self.crossings = torch.zeros_like(self.oscillators, dtype=torch.bool)
        
        # 初始化Floquet奖励缓冲区
        self.floquet_reward = torch.zeros_like(self.crossings, dtype=torch.float32)

        # 初始化平均频率缓冲区，初始值为振荡器的角频率
        self.average_frequency = (torch.ones_like(self.oscillators) * self.osc_omega)
        
        # 初始化最大频率差异缓冲区
        self.max_freq_diff = torch.zeros(self.num_envs, device=self.device)
        
        # 初始化最后一次交叉点的步数缓冲区
        self.last_cross = torch.zeros_like(self.oscillators, dtype=torch.long)

    def _pre_physics_step(self):
        """
        物理步前的预处理步骤。
        """
        super()._pre_physics_step()  # 调用父类的预物理步方法

    def _post_physics_step(self):
        """
        物理步后的后处理步骤，计算最大频率差异。
        """
        super()._post_physics_step()  # 调用父类的后物理步方法
        # 计算每个环境中振荡器的最大频率差异
        self.max_freq_diff = self.average_frequency.max(dim=1)[0] - self.average_frequency.min(dim=1)[0]

    def _step_oscillators(self, dt=None):
        """
        更新振荡器的相位和速度，并处理相位跨越。
        
        参数:
            dt (float, optional): 时间步长，默认为环境的时间步长self.dt。
        """
        if dt is None:
            dt = self.dt  # 如果未提供时间步长，使用默认的时间步长

        # 计算局部反馈，基于振荡器的相位和偏移量
        local_feedback = self.osc_coupling * (torch.cos(self.oscillators) + self.osc_offset)
        
        # 计算地面反作用力
        grf = self._compute_grf()
        
        # 更新振荡器的速度，基于角频率和地面反作用力的反馈
        self.oscillators_vel = self.osc_omega - grf * local_feedback
        
        # 添加过程噪声到振荡器速度
        self.oscillators_vel += (torch.randn(self.oscillators_vel.shape, device=self.device) 
                                  * self.cfg.osc.process_noise_std)

        # 将振荡器速度转换为弧度/秒
        self.oscillators_vel *= 2 * torch.pi
        
        # 更新振荡器的相位
        self.oscillators += self.oscillators_vel * dt
        
        # 检测振荡器是否跨越2π
        self.crossings = self.oscillators >= 2 * torch.pi
        
        # 将振荡器相位限制在[0, 2π)范围内
        self.oscillators = torch.remainder(self.oscillators, 2 * torch.pi)

        # 遍历每个振荡器，处理相位跨越事件
        for osc_id in range(4):
            env_id = self.crossings[:, osc_id].nonzero().flatten()  # 获取跨越2π的环境ID
            if len(env_id) == 0:
                continue  # 如果没有环境跨越2π，继续下一个振荡器

            # 根据振荡器ID，更新对应的相位缓冲区
            if osc_id == 0:
                self.last_phase_1.roll(1, dims=2)  # 循环移动缓冲区
                self.last_phase_1[env_id, :, 0] = self.oscillators[env_id, :]
            if osc_id == 1:
                self.last_phase_2.roll(1, dims=2)
                self.last_phase_2[env_id, :, 0] = self.oscillators[env_id, :]
            if osc_id == 2:
                self.last_phase_3.roll(1, dims=2)
                self.last_phase_3[env_id, :, 0] = self.oscillators[env_id, :]
            if osc_id == 3:
                self.last_phase_4.roll(1, dims=2)
                self.last_phase_4[env_id, :, 0] = self.oscillators[env_id, :]

            # 更新平均频率
            self._update_avg_frequency(env_id, osc_id)

        # 更新振荡器的观测值，使用余弦和正弦表示相位
        self.oscillator_obs = torch.cat((torch.cos(self.oscillators),
                                         torch.sin(self.oscillators)), dim=1)

    def _update_avg_frequency(self, env_id, osc_id):
        """
        更新平均频率，使用指数移动平均滤波器。
        
        参数:
            env_id (torch.Tensor): 环境ID张量。
            osc_id (int): 振荡器ID。
        """
        # 计算自上次跨越以来的时间步数
        time_steps = (self.common_step_counter - self.last_cross[env_id, osc_id])
        
        # 计算当前的频率，并应用指数移动平均滤波
        self.average_frequency[env_id, osc_id] = exp_avg_filter(
            1.0 / (time_steps * self.dt),
            self.average_frequency[env_id, osc_id]
        )
        
        # 更新最后一次跨越的步数
        self.last_cross[env_id, osc_id] = self.common_step_counter

    def _reward_floquet(self):
        """
        计算Floquet奖励，基于相位差和速度误差。
        
        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        # 计算每个周期的相位差
        phase_diff = torch.cat((
            (self.last_phase_1[:, :, 0] - self.last_phase_1[:, :, 1]).norm(dim=1).unsqueeze(1),
            (self.last_phase_2[:, :, 0] - self.last_phase_2[:, :, 1]).norm(dim=1).unsqueeze(1),
            (self.last_phase_3[:, :, 0] - self.last_phase_3[:, :, 1]).norm(dim=1).unsqueeze(1),
            (self.last_phase_4[:, :, 0] - self.last_phase_4[:, :, 1]).norm(dim=1).unsqueeze(1)
        ), dim=1)
        
        # 将相位差归一化到[0, 1)范围内
        phase_diff /= 2 * torch.pi
        
        # 更新Floquet奖励，当振荡器跨越2π时，使用相位差
        self.floquet_reward = torch.where(self.crossings, phase_diff, self.floquet_reward)
        
        # 计算线速度误差（xy轴）
        vel_error = (self.commands[:, :2] - self.base_lin_vel[:, :2]).norm(dim=1)
        vel_error = self._sqrdexp(vel_error / (self.scales["base_lin_vel"] / 2.))
        
        # 当跟踪误差较小时，惩罚Floquet乘子的大偏差
        reward = -self.floquet_reward.sum(dim=1) * vel_error
        return reward

    def _reward_locked_frequency(self):
        """
        计算锁定频率的奖励，基于最大频率差异。
        
        返回:
            torch.Tensor: 每个环境的奖励值。
        """
        return -self.max_freq_diff  # 奖励为负的最大频率差异，差异越大，奖励越低
