import torch  # 导入PyTorch库，用于深度学习和张量操作

from gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotRunnerCfg
# 从Gym的基类模块导入LeggedRobotCfg和LeggedRobotRunnerCfg配置类，用于定义机器人和训练的配置


BASE_HEIGHT_REF = 0.33  # 定义基础高度参考值，用于奖励函数中的目标高度


class MiniCheetahCfg(LeggedRobotCfg):
    """
    MiniCheetahCfg类继承自LeggedRobotCfg，用于定义迷你猎豹机器人环境的具体配置。
    """
    
    class env(LeggedRobotCfg.env):
        """
        环境配置类，继承自LeggedRobotCfg.env。
        """
        num_envs = 2**12  # 设置并行环境的数量为4096
        num_actuators = 12  # 定义机器人的关节数量
        episode_length_s = 10  # 每个回合的持续时间为10秒

    class terrain(LeggedRobotCfg.terrain):
        """
        地形配置类，继承自LeggedRobotCfg.terrain。
        """
        mesh_type = 'plane'  # 设置地形类型为平面

    class init_state(LeggedRobotCfg.init_state):
        """
        初始状态配置类，继承自LeggedRobotCfg.init_state。
        """
        default_joint_angles = {
            "haa": 0.0,        # Hip Abduction/Adduction角度
            "hfe": -0.785398,  # Hip Flexion/Extension角度
            "kfe": 1.596976,    # Knee Flexion/Extension角度
        }

        # * 重置设置，选择初始条件的方式
        # * "reset_to_basic" = 单一位置
        # * "reset_to_range" = 从下面定义的范围中均匀随机选择
        reset_mode = "reset_to_basic"  # 设置重置模式为“reset_to_basic”

        # * 基本初始化的默认质心（COM）位置和状态
        pos = [0.0, 0.0, 0.33]  # 基础位置 [x, y, z]，单位为米
        rot = [0.0, 0.0, 0.0, 1.0]  # 基础旋转，使用四元数 [x, y, z, w]
        lin_vel = [0.0, 0.0, 0.0]  # 基础线速度 [x, y, z]，单位为米/秒
        ang_vel = [0.0, 0.0, 0.0]  # 基础角速度 [x, y, z]，单位为弧度/秒

        # * 随机范围初始化设置
        dof_pos_range = {'haa': [-0.05, 0.05],
                         'hfe': [-0.85, -0.6],
                         'kfe': [-1.45, 1.72]}  # 各关节位置的随机范围
        dof_vel_range = {'haa': [0., 0.],
                         'hfe': [0., 0.],
                         'kfe': [0., 0.]}  # 各关节速度的随机范围
        root_pos_range = [[0., 0.],       # 基础位置x轴范围
                          [0., 0.],       # 基础位置y轴范围
                          [0.37, 0.4],    # 基础位置z轴范围
                          [0., 0.],       # Roll角范围
                          [0., 0.],       # Pitch角范围
                          [0., 0.]]       # Yaw角范围
        root_vel_range = [[-0.05, 0.05],  # 基础线速度x轴范围
                          [0., 0.],       # 基础线速度y轴范围
                          [-0.05, 0.05],  # 基础线速度z轴范围
                          [0., 0.],       # 基础角速度Roll范围
                          [0., 0.],       # 基础角速度Pitch范围
                          [0., 0.]]       # 基础角速度Yaw范围

    class control(LeggedRobotCfg.control):
        """
        控制参数配置类，继承自LeggedRobotCfg.control。
        """
        # * PD控制器的参数设置
        stiffness = {'haa': 20., 'hfe': 20., 'kfe': 20.}  # 各关节的刚度
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}   # 各关节的阻尼
        ctrl_frequency = 100  # 控制器频率，单位为Hz
        desired_sim_frequency = 1000  # 期望的模拟器频率，单位为Hz

    class commands:
        """
        命令参数配置类。
        """
        resampling_time = 10.0  # 命令重采样时间，单位为秒

        class ranges:
            """
            命令范围配置类。
            """
            lin_vel_x = [-1.0, 1.0]  # 线速度x轴的最小值和最大值，单位为米/秒
            lin_vel_y = 1.0           # 线速度y轴的最大值，单位为米/秒
            yaw_vel = 1.0             # 偏航速度的最大值，单位为弧度/秒

    class push_robots:
        """
        推送机器人的配置类，用于外部扰动。
        """
        toggle = False            # 是否启用推送机器人功能
        interval_s = 1.0          # 推送间隔时间，单位为秒
        max_push_vel_xy = 0.5     # 推送的最大速度，单位为米/秒

    class domain_rand(LeggedRobotCfg.domain_rand):
        """
        域随机化配置类，继承自LeggedRobotCfg.domain_rand。
        """
        randomize_friction = True           # 是否随机化摩擦系数
        friction_range = [0.5, 1.25]        # 摩擦系数的随机范围
        randomize_base_mass = True          # 是否随机化基础质量
        added_mass_range = [-1., 1.]        # 添加质量的随机范围

    class asset(LeggedRobotCfg.asset):
        """
        资产配置类，继承自LeggedRobotCfg.asset，用于定义机器人模型和物理属性。
        """
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/" \
               + "mini_cheetah/urdf/mini_cheetah_simple.urdf"  # 机器人URDF文件路径
        foot_name = "foot"  # 机器人的脚部名称
        penalize_contacts_on = []  # 惩罚接触的部位
        terminate_after_contacts_on = ["base", "thigh"]  # 接触这些部位后终止回合
        end_effector_names = ['foot']  # 末端执行器的名称列表
        collapse_fixed_joints = False  # 是否合并通过固定关节连接的刚体
        self_collisions = 1  # 自碰撞设置，1为禁用，0为启用
        flip_visual_attachments = False  # 是否翻转视觉附件
        disable_gravity = False  # 是否禁用重力
        disable_motors = False  # 是否禁用电机
        joint_damping = 0.1  # 关节阻尼系数
        rotor_inertia = [0.002268, 0.002268, 0.005484] * 4  # 转子惯性

    class reward_settings(LeggedRobotCfg.reward_settings):
        """
        奖励设置配置类，继承自LeggedRobotCfg.reward_settings。
        """
        soft_dof_pos_limit = 0.9       # 关节位置的软限制
        soft_dof_vel_limit = 0.9       # 关节速度的软限制
        soft_torque_limit = 0.9         # 力矩的软限制
        max_contact_force = 600.        # 最大接触力
        base_height_target = BASE_HEIGHT_REF  # 基础高度目标值
        tracking_sigma = 0.25            # 跟踪误差的标准差

    class scaling(LeggedRobotCfg.scaling):
        """
        缩放配置类，继承自LeggedRobotCfg.scaling，用于对不同变量进行缩放。
        """
        base_ang_vel = 1.       # 基础角速度的缩放因子
        base_lin_vel = 1.       # 基础线速度的缩放因子
        commands = 1.            # 命令的缩放因子
        dof_vel = 100.           # 关节速度的缩放因子，约等于预期的最大速度
        base_height = 1.         # 基础高度的缩放因子
        dof_pos = 1.             # 关节位置的缩放因子
        dof_pos_obs = dof_pos    # 关节位置观测值的缩放因子
        # * 动作缩放
        dof_pos_target = dof_pos  # 关节位置目标值的缩放因子
        # tau_ff = 4*[18, 18, 28]  # hip-abd, hip-pitch, knee（已注释）
        clip_actions = 1000.      # 动作裁剪阈值


class MiniCheetahRunnerCfg(LeggedRobotRunnerCfg):
    """
    MiniCheetahRunnerCfg类继承自LeggedRobotRunnerCfg，用于定义迷你猎豹机器人的训练配置。
    """
    seed = -1  # 随机种子，-1表示不固定种子

    class policy(LeggedRobotRunnerCfg.policy):
        """
        策略配置类，继承自LeggedRobotRunnerCfg.policy。
        """
        actor_hidden_dims = [256, 256, 256]  # Actor网络的隐藏层维度
        critic_hidden_dims = [256, 256, 256]  # Critic网络的隐藏层维度
        activation = 'elu'  # 激活函数，可以选择'elu', 'relu', 'selu', 'crelu', 'lrelu', 'tanh', 'sigmoid'

        actor_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel"
            ]  # Actor网络的观测变量
        critic_obs = [
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel"
            ]  # Critic网络的观测变量

        actions = ["dof_pos_target"]  # 策略输出的动作
        add_noise = False  # 是否在动作中添加噪声

        class noise:
            """
            噪声配置类，用于定义在动作中添加的噪声量。
            """
            dof_pos_obs = 0.005  # 关节位置观测值的噪声
            dof_vel = 0.005      # 关节速度的噪声
            base_ang_vel = 0.05  # 基础角速度的噪声
            projected_gravity = 0.02  # 投影重力的噪声

        class reward(LeggedRobotRunnerCfg.policy.reward):
            """
            奖励函数配置类，继承自LeggedRobotRunnerCfg.policy.reward。
            """
            class weights(LeggedRobotRunnerCfg.policy.reward.weights):
                """
                奖励权重配置类，继承自LeggedRobotRunnerCfg.policy.reward.weights。
                """
                tracking_lin_vel = 5.0      # 线速度跟踪奖励的权重
                tracking_ang_vel = 5.0      # 角速度跟踪奖励的权重
                lin_vel_z = 0.0             # z轴线速度奖励的权重
                ang_vel_xy = 0.0            # xy轴角速度奖励的权重
                orientation = 1.0           # 姿态奖励的权重
                torques = 5.e-7             # 力矩奖励的权重
                dof_vel = 1.0               # 关节速度奖励的权重
                base_height = 1.0           # 基础高度奖励的权重
                action_rate = 0.001         # 动作变化率奖励的权重
                action_rate2 = 0.0001       # 动作变化率2奖励的权重
                stand_still = 0.0           # 静止奖励的权重
                dof_pos_limits = 0.0        # 关节位置限制奖励的权重
                feet_contact_forces = 0.0   # 脚部接触力奖励的权重
                dof_near_home = 1.0         # 关节接近初始位置奖励的权重

            class termination_weight:
                """
                终止条件奖励权重配置类。
                """
                termination = 0.01  # 终止条件的权重

    class algorithm(LeggedRobotRunnerCfg.algorithm):
        """
        算法配置类，继承自LeggedRobotRunnerCfg.algorithm，用于定义训练算法的参数。
        """
        value_loss_coef = 1.0          # 值函数损失系数
        use_clipped_value_loss = True  # 是否使用裁剪后的值函数损失
        clip_param = 0.2                # PPO裁剪参数
        entropy_coef = 0.01             # 熵正则化系数
        num_learning_epochs = 6         # 学习的迭代次数
        num_mini_batches = 6            # 每次更新的小批量数量
        learning_rate = 1.e-4           # 学习率
        schedule = 'adaptive'           # 学习率调度策略，可以选择'adaptive'或'fixed'
        discount_horizon = 1.0          # 折扣范围，单位为秒
        GAE_bootstrap_horizon = 1.0     # GAE引导范围，单位为秒
        desired_kl = 0.01               # 期望的KL散度
        max_grad_norm = 1.0             # 梯度裁剪的最大范数

    class runner(LeggedRobotRunnerCfg.runner):
        """
        运行器配置类，继承自LeggedRobotRunnerCfg.runner，用于定义训练运行的参数。
        """
        run_name = ''                  # 运行名称
        experiment_name = 'mini_cheetah'  # 实验名称
        max_iterations = 1000          # 最大策略更新次数
        algorithm_class_name = 'PPO'   # 训练算法的类名称，此处为PPO（Proximal Policy Optimization）
        num_steps_per_env = 24         # 每个环境每次更新的步数
