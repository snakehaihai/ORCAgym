import torch  # 导入 PyTorch 库，用于张量操作和深度学习
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作
from isaacgym.torch_utils import torch_rand_float, to_torch  # 从 isaacgym 库中导入 PyTorch 工具函数

from gym import LEGGED_GYM_ROOT_DIR  # 从 gym 模块中导入 LEGGED_GYM_ROOT_DIR，表示项目根目录
from gym.envs.mini_cheetah.mini_cheetah import MiniCheetah  # 从 gym.envs.mini_cheetah.mini_cheetah 模块中导入 MiniCheetah 类

from isaacgym import gymtorch, gymapi  # 从 isaacgym 库中导入 gymtorch 和 gymapi，用于与 Isaac Gym 交互

MINI_CHEETAH_MASS = 8.292 * 9.81  # 定义 Mini Cheetah 的质量，单位为牛顿（N）

class MiniCheetahOsc(MiniCheetah):
    """MiniCheetahOsc 类继承自 MiniCheetah，用于添加振荡器控制功能"""

    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        """初始化 MiniCheetahOsc 实例"""
        super().__init__(gym, sim, cfg, sim_params, sim_device, headless)  # 调用父类初始化方法
        self.process_noise_std = self.cfg.osc.process_noise_std  # 获取振荡器过程噪声标准差

    def _init_buffers(self):
        """初始化缓冲区，包括振荡器、接触力等"""
        super()._init_buffers()  # 调用父类的缓冲区初始化方法
        self.oscillators = torch.zeros(self.num_envs, 4, device=self.device)  # 初始化振荡器相位
        self.oscillator_obs = torch.zeros(self.num_envs, 8, device=self.device)  # 初始化振荡器观测
        self.oscillators_vel = torch.zeros_like(self.oscillators)  # 初始化振荡器速度
        self.grf = torch.zeros(self.num_envs, 4, device=self.device)  # 初始化地面反作用力（GRF）
        self.osc_omega = self.cfg.osc.omega * torch.ones(self.num_envs, 1, device=self.device)  # 初始化振荡器频率
        self.osc_coupling = self.cfg.osc.coupling * torch.ones(self.num_envs, 1, device=self.device)  # 初始化振荡器耦合强度
        self.osc_offset = self.cfg.osc.offset * torch.ones(self.num_envs, 1, device=self.device)  # 初始化振荡器偏移量

    def _reset_oscillators(self, env_ids):
        """重置指定环境的振荡器相位"""
        if len(env_ids) == 0:  # 如果没有环境需要重置
            return  # 直接返回
        # 根据配置选择初始化方式
        if self.cfg.osc.init_to == 'random':
            self.oscillators[env_ids] = torch_rand_float(
                0, 2*torch.pi, shape=self.oscillators[env_ids].shape,
                device=self.device)  # 随机初始化振荡器相位
        elif self.cfg.osc.init_to == 'standing':
            self.oscillators[env_ids] = 3*torch.pi/2  # 将振荡器相位设置为站立姿态
        elif self.cfg.osc.init_to == 'trot':
            self.oscillators[env_ids] = torch.tensor([0., torch.pi, torch.pi, 0.],
                                                     device=self.device)  # 设置为 trot 步态
        elif self.cfg.osc.init_to == 'pace':
            self.oscillators[env_ids] = torch.tensor([0., torch.pi, 0., torch.pi],
                                                     device=self.device)  # 设置为 pace 步态
        elif self.cfg.osc.init_to == 'pronk':
            self.oscillators[env_ids, :] *= 0.  # 设置为 pronk 步态，相位归零
        elif self.cfg.osc.init_to == 'bound':
            self.oscillators[env_ids, :] = torch.tensor([torch.pi, torch.pi, 0., 0.], device=self.device)  # 设置为 bound 步态
        else:
            raise NotImplementedError  # 如果初始化方式未实现，抛出异常

        # 如果配置要求初始化偏移量
        if self.cfg.osc.init_w_offset:
            self.oscillators[env_ids, :] += \
                torch.rand_like(self.oscillators[env_ids, 0]).unsqueeze(1) * 2 * torch.pi  # 添加随机偏移量
        self.oscillators = torch.remainder(self.oscillators, 2*torch.pi)  # 将相位限制在 [0, 2π) 范围内

    def _reset_system(self, env_ids):
        """重置系统状态，包括振荡器和环境"""
        if len(env_ids) == 0:  # 如果没有环境需要重置
            return  # 直接返回
        self._reset_oscillators(env_ids)  # 重置振荡器相位

        # 更新振荡器观测，将相位转换为余弦和正弦值
        self.oscillator_obs = torch.cat((torch.cos(self.oscillators),
                                         torch.sin(self.oscillators)), dim=1)

        # 保持部分机器人在相同的起始状态
        timed_out_subset = (self.timed_out & ~self.terminated) * \
            (torch.rand(self.num_envs, device=self.device)
             < self.cfg.init_state.timeout_reset_ratio)  # 选择需要重置的环境
        env_ids = (self.terminated | timed_out_subset).nonzero().flatten()  # 获取需要重置的环境ID
        if len(env_ids) == 0:  # 如果没有环境需要重置
            return  # 直接返回
        super()._reset_system(env_ids)  # 调用父类的重置系统方法

    def _pre_physics_step(self):
        """在物理步骤之前执行的操作"""
        super()._pre_physics_step()  # 调用父类的方法
        # self.grf = self._compute_grf()  # 计算地面反作用力（被注释掉）
        if not self.cfg.osc.randomize_osc_params:
            self.compute_osc_slope()  # 计算振荡器斜率

    def compute_osc_slope(self):
        """根据命令调整振荡器的斜率和耦合参数"""
        cmd_x = torch.abs(self.commands[:, 0:1]) - self.cfg.osc.stop_threshold  # 计算命令的x轴分量
        stop = (cmd_x < 0)  # 判断是否需要停止

        # 根据是否停止调整振荡器偏移量和频率
        self.osc_offset = stop * self.cfg.osc.offset
        self.osc_omega = stop * self.cfg.osc.omega_stop \
            + torch.randn_like(self.osc_omega) * self.cfg.osc.omega_var  # 添加随机波动
        self.osc_coupling = stop * self.cfg.osc.coupling_stop \
            + torch.randn_like(self.osc_coupling) * self.cfg.osc.coupling_var  # 添加随机波动

        # 根据命令调整振荡器频率和耦合强度
        self.osc_omega += (~stop) * torch.clamp(cmd_x*self.cfg.osc.omega_slope
                                                + self.cfg.osc.omega_step,
                                                min=0.,
                                                max=self.cfg.osc.omega_max)
        self.osc_coupling += \
            (~stop) * torch.clamp(cmd_x*self.cfg.osc.coupling_slope
                                  + self.cfg.osc.coupling_step,
                                  min=0.,
                                  max=self.cfg.osc.coupling_max)

        self.osc_omega = torch.clamp_min(self.osc_omega, 0.1)  # 确保振荡器频率最低为0.1
        self.osc_coupling = torch.clamp_min(self.osc_coupling, 0)  # 确保耦合强度非负

    def _process_rigid_body_props(self, props, env_id):
        """处理刚体属性，包括质量和质心位置的随机化"""
        if env_id == 0:
            # 初始化用于领域随机化的缓冲区
            self.mass = torch.zeros(self.num_envs, 1, device=self.device)  # 初始化质量缓冲区
            self.com = torch.zeros(self.num_envs, 3, device=self.device)  # 初始化质心缓冲区

        # 如果配置要求随机化基础质量
        if self.cfg.domain_rand.randomize_base_mass:
            lower = self.cfg.domain_rand.lower_mass_offset  # 获取质量偏移下限
            upper = self.cfg.domain_rand.upper_mass_offset  # 获取质量偏移上限
            # 随机调整质量
            props[0].mass += np.random.uniform(lower, upper)
            self.mass[env_id] = props[0].mass  # 更新质量缓冲区
            # 随机调整质心位置
            lower = self.cfg.domain_rand.lower_z_offset  # 获取质心z轴偏移下限
            upper = self.cfg.domain_rand.upper_z_offset  # 获取质心z轴偏移上限
            props[0].com.z += np.random.uniform(lower, upper)  # 调整质心z轴位置
            self.com[env_id, 2] = props[0].com.z  # 更新质心z轴缓冲区

            lower = self.cfg.domain_rand.lower_x_offset  # 获取质心x轴偏移下限
            upper = self.cfg.domain_rand.upper_x_offset  # 获取质心x轴偏移上限
            props[0].com.x += np.random.uniform(lower, upper)  # 调整质心x轴位置
            self.com[env_id, 0] = props[0].com.x  # 更新质心x轴缓冲区
        return props  # 返回更新后的刚体属性

    def _post_physics_step(self):
        """在物理步骤之后更新所有未在 PhysX 中处理的状态"""
        super()._post_physics_step()  # 调用父类的方法
        self.grf = self._compute_grf()  # 计算地面反作用力
        # self._step_oscillators()  # 更新振荡器（被注释掉）

    def _post_torque_step(self):
        """在扭矩步骤之后更新振荡器"""
        super()._post_torque_step()  # 调用父类的方法
        self._step_oscillators(self.dt/self.cfg.control.decimation)  # 更新振荡器状态
        return None  # 返回空值

    def _step_oscillators(self, dt=None):
        """根据时间步长更新振荡器状态"""
        if dt is None:
            dt = self.dt  # 如果未提供时间步长，使用默认步长

        # 计算局部反馈
        local_feedback = self.osc_coupling * (torch.cos(self.oscillators) + self.osc_offset)
        grf = self._compute_grf()  # 计算地面反作用力
        self.oscillators_vel = self.osc_omega - grf * local_feedback  # 更新振荡器速度
        # 添加过程噪声
        self.oscillators_vel += (torch.randn(self.oscillators_vel.shape,
                                             device=self.device)
                                 * self.cfg.osc.process_noise_std)
        self.oscillators_vel *= 2*torch.pi  # 缩放振荡器速度
        self.oscillators += self.oscillators_vel * dt  # 更新振荡器相位
        self.oscillators = torch.remainder(self.oscillators, 2*torch.pi)  # 将相位限制在 [0, 2π) 范围内
        # 更新振荡器观测
        self.oscillator_obs = torch.cat((torch.cos(self.oscillators),
                                         torch.sin(self.oscillators)), dim=1)

    def _resample_commands(self, env_ids):
        """随机重新采样指定环境的命令"""
        if len(env_ids) == 0:  # 如果没有环境需要重新采样
            return  # 直接返回
        super()._resample_commands(env_ids)  # 调用父类的方法重新采样命令
        possible_commands = torch.tensor(self.command_ranges["lin_vel_x"],
                                         device=self.device)  # 获取可能的线速度命令
        self.commands[env_ids, 0:1] = possible_commands[torch.randint(
            0, len(possible_commands), (len(env_ids), 1),
            device=self.device)]  # 随机选择线速度命令
        # 为命令添加高斯噪声
        self.commands[env_ids, 0:1] += torch.randn((len(env_ids), 1),
                                                   device=self.device) \
                                        * self.cfg.commands.var

        # 如果线速度x包含0
        if (0 in self.cfg.commands.ranges.lin_vel_x):
            # 20% 的概率，除了前进方向外，其他命令归零
            self.commands[env_ids, 1:] *= (torch_rand_float(0, 1, (len(env_ids), 1),
                device=self.device).squeeze(1) < 0.8).unsqueeze(1)
            # 20% 的概率，除了旋转方向外，其他命令归零
            self.commands[env_ids, :2] *= (torch_rand_float(0, 1, (len(env_ids), 1),
                device=self.device).squeeze(1) < 0.8).unsqueeze(1)
            # 10% 的概率，所有命令归零
            self.commands[env_ids, :] *= (torch_rand_float(0, 1, (len(env_ids), 1),
                device=self.device).squeeze(1) < 0.9).unsqueeze(1)

        # 如果配置要求随机化振荡器参数
        if self.cfg.osc.randomize_osc_params:
            self._resample_osc_params(env_ids)  # 重新采样振荡器参数

    def _resample_osc_params(self, env_ids):
        """重新采样振荡器参数"""
        if (len(env_ids) > 0):  # 如果有环境需要重新采样
            self.osc_omega[env_ids, 0] = torch_rand_float(self.cfg.osc.omega_range[0],
                                                        self.cfg.osc.omega_range[1],
                                                        (len(env_ids), 1),
                                                        device=self.device).squeeze(1)  # 重新采样振荡器频率
            self.osc_coupling[env_ids, 0] = torch_rand_float(self.cfg.osc.coupling_range[0],
                                                            self.cfg.osc.coupling_range[1],
                                                            (len(env_ids), 1),
                                                            device=self.device).squeeze(1)  # 重新采样振荡器耦合强度
            self.osc_offset[env_ids, 0] = torch_rand_float(self.cfg.osc.offset_range[0],
                                                        self.cfg.osc.offset_range[1],
                                                        (len(env_ids), 1),
                                                        device=self.device).squeeze(1)  # 重新采样振荡器偏移量

    def perturb_base_velocity(self, velocity_delta, env_ids=None):
        """扰动基础速度，用于模拟外部干扰

        Args:
            velocity_delta (torch.Tensor): 速度增量
            env_ids (List[int], optional): 需要扰动的环境ID。如果为 None，则扰动所有环境
        """
        if env_ids is None:
            env_ids = [range(self.num_envs)]  # 如果未指定环境ID，扰动所有环境
        self.root_states[env_ids, 7:10] += velocity_delta  # 增加速度增量到根状态
        self.gym.set_actor_root_state_tensor(self.sim,
                                    gymtorch.unwrap_tensor(self.root_states))  # 更新仿真中的根状态

    def _compute_grf(self, grf_norm=True):
        """计算地面反作用力（GRF）

        Args:
            grf_norm (bool, optional): 是否归一化GRF。默认为 True

        Returns:
            torch.Tensor: 计算得到的GRF
        """
        grf = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)  # 计算接触力的范数
        if grf_norm:
            return torch.clamp_max(grf / MINI_CHEETAH_MASS, 1.0)  # 归一化GRF并限制最大值为1.0
        else:
            return grf  # 返回原始GRF

    def _switch(self):
        """根据命令速度计算切换因子，用于奖励函数"""
        c_vel = torch.linalg.norm(self.commands, dim=1)  # 计算命令速度的范数
        return torch.exp(-torch.square(torch.max(torch.zeros_like(c_vel),
                                                 c_vel-0.1))
                         / self.cfg.reward_settings.switch_scale)  # 计算切换因子

    def _reward_cursorial(self):
        """对关节位置偏离0的情况进行惩罚"""
        return -torch.mean(torch.square(self.dof_pos[:, 0:12:3]
                                        /self.scales["dof_pos"][0]), dim=1)  # 计算关节位置的平方差并取平均

    def _reward_swing_grf(self):
        """奖励在摆动期有非零GRF"""
        rew = self.get_swing_grf(self.cfg.osc.osc_bool, self.cfg.osc.grf_bool)  # 获取摆动期的GRF奖励
        return -torch.sum(rew, dim=1)  # 计算总奖励并取负值

    def _reward_stance_grf(self):
        """奖励在站立期有非零GRF"""
        rew = self.get_stance_grf(self.cfg.osc.osc_bool, self.cfg.osc.grf_bool)  # 获取站立期的GRF奖励
        return torch.sum(rew, dim=1)  # 计算总奖励

    def get_swing_grf(self, osc_bool=False, contact_bool=False):
        """获取摆动期的GRF

        Args:
            osc_bool (bool, optional): 是否使用振荡器。默认为 False
            contact_bool (bool, optional): 是否考虑接触。默认为 False

        Returns:
            torch.Tensor: 摆动期的GRF
        """
        if osc_bool:
            phase = torch.lt(self.oscillators, torch.pi).int()  # 如果使用振荡器，获取摆动期的相位
        else:
            phase = torch.maximum(torch.zeros_like(self.oscillators),
                               torch.sin(self.oscillators))  # 否则，根据正弦值判断摆动期
        if contact_bool:
            return phase * torch.gt(self._compute_grf(),
                                  self.cfg.osc.grf_threshold)  # 如果考虑接触，过滤GRF低于阈值的部分
        else:
            return phase * self._compute_grf()  # 否则，直接返回GRF

    def get_stance_grf(self, osc_bool=False, contact_bool=False):
        """获取站立期的GRF

        Args:
            osc_bool (bool, optional): 是否使用振荡器。默认为 False
            contact_bool (bool, optional): 是否考虑接触。默认为 False

        Returns:
            torch.Tensor: 站立期的GRF
        """
        if osc_bool:
            phase = torch.gt(self.oscillators, torch.pi).int()  # 如果使用振荡器，获取站立期的相位
        else:
            phase = torch.maximum(torch.zeros_like(self.oscillators),
                               - torch.sin(self.oscillators))  # 否则，根据负正弦值判断站立期
        if contact_bool:
            return phase * torch.gt(self._compute_grf(),
                                  self.cfg.osc.grf_threshold)  # 如果考虑接触，过滤GRF低于阈值的部分
        else:
            return phase * self._compute_grf()  # 否则，直接返回GRF

    def _reward_coupled_grf(self):
        """
        奖励函数，结合摆动期和站立期的GRF，惩罚不良行为（摆动期有GRF，站立期无GRF）
        """
        swing_rew = self.get_swing_grf()  # 获取摆动期的GRF奖励
        stance_rew = self.get_stance_grf()  # 获取站立期的GRF奖励
        combined_rew = self._sqrdexp(swing_rew*2) + stance_rew  # 结合摆动期和站立期的奖励
        prod = torch.prod(torch.clip(combined_rew, 0, 1), dim=1)  # 计算奖励的乘积
        return prod - torch.ones_like(prod)  # 返回最终奖励

    def _reward_dof_vel(self):
        """奖励函数，基于关节速度"""
        return super()._reward_dof_vel() * self._switch()  # 调用父类方法并乘以切换因子

    def _reward_dof_near_home(self):
        """奖励函数，基于关节位置接近初始位置"""
        return super()._reward_dof_near_home() * self._switch()  # 调用父类方法并乘以切换因子

    def _reward_stand_still(self):
        """奖励函数，惩罚在零命令时的运动"""
        # 归一化角度，确保在5度范围内
        rew_pos = torch.mean(self._sqrdexp(
            (self.dof_pos - self.default_dof_pos)/torch.pi*36), dim=1)  # 计算关节位置的奖励
        rew_vel = torch.mean(self._sqrdexp(self.dof_vel), dim=1)  # 计算关节速度的奖励
        rew_base_vel = torch.mean(torch.square(self.base_lin_vel), dim=1)  # 计算基础线速度的惩罚
        rew_base_vel += torch.mean(torch.square(self.base_ang_vel), dim=1)  # 计算基础角速度的惩罚
        return (rew_vel + rew_pos - rew_base_vel) * self._switch()  # 综合奖励

    def _reward_standing_torques(self):
        """奖励函数，惩罚在零命令时的扭矩"""
        return super()._reward_torques() * self._switch()  # 调用父类方法并乘以切换因子

    # * gait similarity scores
    def angle_difference(self, theta1, theta2):
        """计算两个角度之间的最小差异

        Args:
            theta1 (torch.Tensor): 第一个角度
            theta2 (torch.Tensor): 第二个角度

        Returns:
            torch.Tensor: 最小角度差异
        """
        diff = torch.abs(theta1 - theta2) % (2 * torch.pi)  # 计算绝对角度差异并取模2π
        return torch.min(diff, 2*torch.pi - diff)  # 返回最小差异

    def _reward_trot(self):
        """奖励函数，基于 trot 步态的相似性"""
        # 计算前右和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 3])
        similarity = self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前左和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前左和前右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)  # 计算相似性
        # 计算后左和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)  # 计算相似性
        return similarity  # 返回总相似性

    def _reward_pronk(self):
        """奖励函数，基于 pronk 步态的相似性"""
        # 计算前右和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 3])
        similarity = self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前左和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前右和前左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前右和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle, torch.pi)  # 计算相似性
        return similarity  # 返回总相似性

    def _reward_pace(self):
        """奖励函数，基于 pace 步态的相似性"""
        # 计算前右和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity = self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前左和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前右和前左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)  # 计算相似性
        # 计算后右和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)  # 计算相似性

        return similarity  # 返回总相似性

    def _reward_any_symm_gait(self):
        """奖励函数，奖励任何对称步态（trot、pace、bound）"""
        rew_trot = self._reward_trot()  # 获取 trot 步态奖励
        rew_pace = self._reward_pace()  # 获取 pace 步态奖励
        rew_bound = self._reward_bound()  # 获取 bound 步态奖励
        return torch.max(torch.max(rew_trot, rew_pace), rew_bound)  # 返回最大的步态奖励

    def _reward_enc_pace(self):
        """奖励函数，enc pace 步态"""
        return self._reward_pace()  # 调用 pace 步态奖励

    def _reward_bound(self):
        """奖励函数，基于 bound 步态的相似性"""
        # 计算前右和前左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity = self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算后右和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle, torch.pi)  # 计算相似性
        # 计算前右和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)  # 计算相似性
        # 计算前左和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 3])
        similarity *= self._sqrdexp(angle - torch.pi, torch.pi)  # 计算相似性
        return similarity  # 返回总相似性

    def _reward_asymettric(self):
        """奖励函数，基于非对称步态的相似性"""
        # 计算后左和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 2],
                                      self.oscillators[:, 3])
        similarity = (1 - self._sqrdexp(angle, torch.pi))  # 计算相似性
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))  # 进一步计算相似性
        # 计算前左和后左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 3])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))  # 计算相似性
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))  # 进一步计算相似性
        # 计算前右和后右腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 2])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))  # 计算相似性
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))  # 进一步计算相似性
        # 计算前右和前左腿的角度差异
        angle = self.angle_difference(self.oscillators[:, 0],
                                      self.oscillators[:, 1])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))  # 计算相似性
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))  # 进一步计算相似性
        # 计算前左和后右腿的角度差异（对角线）
        angle = self.angle_difference(self.oscillators[:, 1],
                                      self.oscillators[:, 2])
        similarity *= (1 - self._sqrdexp(angle, torch.pi))  # 计算相似性
        similarity *= (1 - self._sqrdexp((torch.pi - angle), torch.pi))  # 进一步计算相似性
        return similarity  # 返回总相似性
