import torch  # 导入 PyTorch 库，用于张量操作和深度学习
import pandas as pd  # 导入 Pandas 库，用于数据处理
from isaacgym.torch_utils import torch_rand_float, to_torch  # 从 isaacgym 库导入随机浮点数生成函数和张量转换工具

from gym import LEGGED_GYM_ROOT_DIR  # 从 gym 模块中导入项目根目录路径
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah  # 从 mini_cheetah 模块中导入 MiniCheetah 类


class MiniCheetahRef(MiniCheetah):
    """MiniCheetahRef 类继承自 MiniCheetah，使用参考轨迹控制机器人"""

    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        """初始化 MiniCheetahRef 类"""
        # 读取参考轨迹 CSV 文件
        csv_path = cfg.init_state.ref_traj.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self.leg_ref = to_torch(pd.read_csv(csv_path).to_numpy(),
                                device=sim_device)  # 将读取的数据转换为 PyTorch 张量
        self.omega = 2 * torch.pi * cfg.control.gait_freq  # 计算步态频率对应的角速度
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)  # 调用父类的初始化方法

    def _init_buffers(self):
        """初始化缓冲区"""
        super()._init_buffers()  # 调用父类的缓冲区初始化方法
        self.phase = torch.zeros(self.num_envs, 1, dtype=torch.float,
                                 device=self.device)  # 初始化相位张量，形状为 (num_envs, 1)
        self.phase_obs = torch.zeros(self.num_envs, 2, dtype=torch.float,
                                     device=self.device)  # 初始化相位观测，存储正弦和余弦值

    def _reset_system(self, env_ids):
        """重置系统"""
        super()._reset_system(env_ids)  # 调用父类的重置系统方法
        # 随机初始化相位
        self.phase[env_ids] = torch_rand_float(0, torch.pi,
                                               shape=self.phase[env_ids].shape,
                                               device=self.device)

    def _post_physics_step(self):
        """更新所有不由物理引擎管理的状态"""
        super()._post_physics_step()  # 调用父类的后物理步方法
        # 更新相位，确保相位始终在 [0, 2*pi] 范围内
        self.phase = torch.fmod(self.phase + self.dt * self.omega, 2 * torch.pi)
        # 计算相位的正弦和余弦，作为相位观测
        self.phase_obs = torch.cat((torch.sin(self.phase),
                                    torch.cos(self.phase)), dim=1)

    def _resample_commands(self, env_ids):
        """重新采样部分环境的指令"""
        super()._resample_commands(env_ids)  # 调用父类的指令重新采样方法
        # 以 10% 的概率将命令重置为 0
        rand_ids = torch_rand_float(0, 1, (len(env_ids), 1),
                                    device=self.device).squeeze(1)
        self.commands[env_ids, :3] *= (rand_ids < 0.9).unsqueeze(1)

    def _switch(self):
        """根据指令速度计算一个切换因子"""
        c_vel = torch.linalg.norm(self.commands, dim=1)  # 计算指令速度的范数
        # 当速度接近 0 时，返回一个小的值，否则返回较大的值
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel),
                                                 c_vel - 0.1)) / 0.1)

    def _reward_swing_grf(self):
        """奖励摆动期间非零的地面反作用力"""
        # 检查脚是否接触地面
        in_contact = torch.gt(torch.norm(self.contact_forces[:, self.feet_indices, :],
                                         dim=-1), 50.)
        # 检查相位是否在 [0, pi] 范围内（摆动期）
        ph_off = torch.lt(self.phase, torch.pi)
        # 计算奖励，只有在摆动期接触地面时会受到惩罚
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)
        return -torch.sum(rew.float(), dim=1) * (1 - self._switch())

    def _reward_stance_grf(self):
        """奖励站立期间非零的地面反作用力"""
        # 检查脚是否接触地面
        in_contact = torch.gt(torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1), 50.)
        # 检查相位是否在 [pi, 2*pi] 范围内（站立期）
        ph_off = torch.gt(self.phase, torch.pi)
        # 计算奖励，在站立期时，接触地面会得到奖励
        rew = in_contact * torch.cat((ph_off, ~ph_off, ~ph_off, ph_off), dim=1)

        return torch.sum(rew.float(), dim=1) * (1 - self._switch())

    def _reward_reference_traj(self):
        """基于参考轨迹的奖励函数"""
        # 计算每条腿的关节位置误差
        error = self._get_ref() + self.default_dof_pos - self.dof_pos
        error /= self.scales['dof_pos']  # 归一化误差
        # 计算奖励，惩罚大的误差
        reward = torch.mean(self._sqrdexp(error) - torch.abs(error) * 0.2, dim=1)
        # 只有在速度较高时才计算奖励
        return reward * (1 - self._switch())

    def _get_ref(self):
        """获取参考轨迹中每条腿的参考关节位置"""
        leg_frame = torch.zeros_like(self.torques)  # 初始化腿部参考关节位置
        # 相位偏移量，确保步态为对角交替步
        ph_off = torch.fmod(self.phase + torch.pi, 2 * torch.pi)
        # 根据相位索引参考轨迹中的关节位置
        phd_idx = (torch.round(
            self.phase * (self.leg_ref.size(dim=0) / (2 * torch.pi) - 1))).long()
        pho_idx = (torch.round(
            ph_off * (self.leg_ref.size(dim=0) / (2 * torch.pi) - 1))).long()
        # 为每条腿分配参考关节位置
        leg_frame[:, 0:3] += self.leg_ref[phd_idx.squeeze(), :]
        leg_frame[:, 3:6] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 6:9] += self.leg_ref[pho_idx.squeeze(), :]
        leg_frame[:, 9:12] += self.leg_ref[phd_idx.squeeze(), :]
        return leg_frame

    def _reward_stand_still(self):
        """当指令为零时，惩罚机器人运动"""
        # 规范化关节角度，确保关节在合理范围内
        rew_pos = torch.mean(self._sqrdexp(
            (self.dof_pos - self.default_dof_pos) / torch.pi * 36), dim=1)
        # 惩罚关节速度和基座的线速度和角速度
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)
        return (rew_vel + rew_pos - rew_base_vel) * self._switch()

    def _reward_tracking_lin_vel(self):
        """追踪线速度命令的奖励函数"""
        # 使用父类的线速度追踪奖励函数
        reward = super()._reward_tracking_lin_vel()
        return reward * (1 - self._switch())
