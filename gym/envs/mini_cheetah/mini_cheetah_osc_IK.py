import torch  # 导入 PyTorch 库，用于张量操作和深度学习
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from isaacgym.torch_utils import torch_rand_float, to_torch  # 从 isaacgym 库中导入 PyTorch 工具函数

from gym import LEGGED_GYM_ROOT_DIR  # 从 gym 模块中导入 LEGGED_GYM_ROOT_DIR，表示项目根目录
from gym.envs.mini_cheetah.mini_cheetah_osc import MiniCheetahOsc  # 从 gym.envs.mini_cheetah.mini_cheetah_osc 模块中导入 MiniCheetahOsc 类

MINI_CHEETAH_MASS = 8.292 * 9.81  # 定义 Mini Cheetah 的质量，单位为牛顿（N）

class MiniCheetahOscIK(MiniCheetahOsc):
    """MiniCheetahOscIK 类继承自 MiniCheetahOsc，用于添加逆运动学（IK）控制功能"""

    def _init_buffers(self):
        """初始化缓冲区，包括逆运动学目标位置"""
        super()._init_buffers()  # 调用父类的缓冲区初始化方法
        self.ik_pos_target = torch.zeros(self.num_envs, 12,  # 初始化逆运动学目标位置张量，形状为 (num_envs, 12)
                                         dtype=torch.float, device=self.device)  # 数据类型为 float，存储在指定设备上

    def _pre_physics_step(self):
        """在物理步之前执行的操作，包括计算逆运动学目标关节位置"""
        super()._pre_physics_step()  # 调用父类的预物理步方法

        # 定义 IK 计算的默认偏移量
        ik_defaults_L = torch.tensor([0., -0.019, -0.5], device=self.device)  # 左腿默认偏移
        ik_defaults_R = torch.tensor([0., 0.019, -0.5], device=self.device)  # 右腿默认偏移

        # 计算每条腿的逆运动学关节位置
        joints_ik_fr = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 0:3] + ik_defaults_R)  # 前右腿
        joints_ik_fl = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 3:6] + ik_defaults_L)  # 前左腿
        joints_ik_br = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 6:9] + ik_defaults_R)  # 后右腿
        joints_ik_bl = self.IK_leg_3DOF(p_hr2ft_hr=self.ik_pos_target[:, 9:12] + ik_defaults_L)  # 后左腿

        # 将所有腿的 IK 关节位置拼接成一个张量，并赋值给 dof_pos_target
        self.dof_pos_target = torch.cat(
            (joints_ik_fr, joints_ik_fl, joints_ik_br, joints_ik_bl), dim=1)  # 拼接后形状为 (num_envs, 12)

    def IK_leg_3DOF(self, p_hr2ft_hr):
        """
        计算三自由度腿的逆运动学关节角度

        Args:
            p_hr2ft_hr (torch.Tensor): 从髋关节到脚趾的目标位置，形状为 (num_envs, 3)

        Returns:
            torch.Tensor: 计算得到的关节角度，形状为 (num_envs, 3)
        """
        # 定义从髋关节到膝关节和膝关节到脚趾的向量
        r_hr2hp = torch.tensor([0.0, -0.019, 0.0], device=self.device)  # 髋滚转到髋俯仰的向量
            .unsqueeze(0).repeat(self.num_envs, 1)  # 复制到每个环境
        r_hp2kp = torch.tensor([0.0, 0.0, -0.2085], device=self.device)  # 髋俯仰到膝关节的向量
            .unsqueeze(0).repeat(self.num_envs, 1)  # 复制到每个环境
        r_kp2ft = torch.tensor([0.0, 0.0, -0.22], device=self.device)  # 膝关节到脚趾的向量
            .unsqueeze(0).repeat(self.num_envs, 1)  # 复制到每个环境

        L1 = torch.abs(r_hp2kp[:, 2])  # 计算 L1 长度（髋俯仰到膝关节的绝对 z 方向距离）
        L2 = torch.abs(r_kp2ft[:, 2])  # 计算 L2 长度（膝关节到脚趾的绝对 z 方向距离）

        # 计算髋滚转角度 q1
        L_hr2ft_yz = torch.norm(p_hr2ft_hr[:, 1:3], dim=1).clamp(min=1e-6)  # 髋到脚趾在 yz 平面的距离，防止除零
        alpha_1 = torch.arcsin(
            torch.clamp(p_hr2ft_hr[:, 1] / L_hr2ft_yz, min=-1, max=1))  # 计算 alpha_1
        beta_1 = torch.arcsin(
            torch.clamp(torch.abs(r_hr2hp[:, 1])/L_hr2ft_yz, min=-1, max=1))  # 计算 beta_1
        q1 = alpha_1 - beta_1  # 髋滚转角度

        # 计算髋俯仰角度 q2
        p_hr2hp_hr = torch.zeros(self.num_envs, 3).to(self.device)  # 初始化髋到髋俯仰的向量
        p_hr2hp_hr[:, 1] = r_hr2hp[:, 1]*torch.cos(q1)  # 更新 y 方向
        p_hr2hp_hr[:, 2] = r_hr2hp[:, 1]*torch.sin(q1)  # 更新 z 方向
        L_hp2ft_hr = torch.clamp(
            torch.norm(p_hr2ft_hr - p_hr2hp_hr, dim=1), min=1e-6)  # 髋俯仰到脚趾的距离，防止除零
        vec = p_hr2ft_hr - p_hr2hp_hr  # 计算向量
        p_hp2ft_hp = torch.stack(
            (vec[:, 0],
             vec[:, 1]*torch.cos(q1) + vec[:, 2]*torch.sin(q1),
             vec[:, 2]*torch.cos(q1) - vec[:, 1]*torch.sin(q1)), dim=1)  # 旋转向量以适应 q1
        alpha_2 = torch.arccos(
            torch.clamp(p_hp2ft_hp[:, 0] / L_hp2ft_hr, min=-1, max=1))  # 计算 alpha_2
        cos_angle_2 = (L1**2 + L_hp2ft_hr**2 - L2**2)/(2*L1*L_hp2ft_hr)  # 余弦定理计算 angle_2
        cos_angle_2 = torch.clamp(cos_angle_2, min=-1, max=1)  # 防止数值溢出
        beta_2 = torch.arccos(cos_angle_2)  # 计算 beta_2
        q2 = alpha_2 + beta_2 - torch.pi / 2  # 髋俯仰角度

        # 计算膝关节俯仰角度 q3
        cos_angle_3 = (L1**2 + L2**2 - L_hp2ft_hr**2)/(2*L1*L2)  # 余弦定理计算 angle_3
        cos_angle_3 = torch.clamp(cos_angle_3, min=-1, max=1)  # 防止数值溢出
        acos_angle_3 = torch.arccos(cos_angle_3)  # 计算 acos(angle_3)
        q3 = -(torch.pi - acos_angle_3)  # 膝关节俯仰角度

        return torch.stack((q1, -q2, -q3), dim=1)  # 返回关节角度张量，形状为 (num_envs, 3)
