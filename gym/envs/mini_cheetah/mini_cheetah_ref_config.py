from gym.envs.mini_cheetah.mini_cheetah_config \
    import MiniCheetahCfg, MiniCheetahRunnerCfg  # 从 mini_cheetah_config 模块中导入配置类

BASE_HEIGHT_REF = 0.33  # 定义基准高度参考值 [米]

# 定义 MiniCheetahRefCfg 类，继承自 MiniCheetahCfg，用于配置 MiniCheetahRef 环境
class MiniCheetahRefCfg(MiniCheetahCfg):
    class env(MiniCheetahCfg.env):
        num_envs = 4096  # 设置环境数量为 4096
        num_actuators = 12  # 设置驱动器数量为 12
        episode_length_s = 15.0  # 设置每个回合的长度为 15 秒

    class terrain(MiniCheetahCfg.terrain):
        pass  # 继承父类的地形配置，不做额外修改

    class init_state(MiniCheetahCfg.init_state):
        reset_mode = "reset_to_basic"  # 设置重置模式为 "reset_to_basic"
        # * 基础初始化的重心（COM）位置和姿态
        pos = [0.0, 0.0, 0.33]  # 初始位置 [x, y, z]，单位为米
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始旋转 [x, y, z, w]，四元数表示
        lin_vel = [0.0, 0.0, 0.0]  # 初始线速度 [x, y, z]，单位为米/秒
        ang_vel = [0.0, 0.0, 0.0]  # 初始角速度 [x, y, z]，单位为弧度/秒
        default_joint_angles = {
            "haa": 0.0,        # 髋滚转关节角度
            "hfe": -0.785398,  # 髋俯仰关节角度
            "kfe": 1.596976,   # 膝关节角度
        }
        # * 关节位置和速度的范围，用于随机初始化
        dof_pos_range = {'haa': [-0.05, 0.05],  # 髋滚转关节位置范围
                         'hfe': [-0.85, -0.6],  # 髋俯仰关节位置范围
                         'kfe': [-1.45, 1.72]}  # 膝关节位置范围
        dof_vel_range = {'haa': [0.0, 0.0],  # 髋滚转关节速度范围
                         'hfe': [0.0, 0.0],  # 髋俯仰关节速度范围
                         'kfe': [0.0, 0.0]}  # 膝关节速度范围
        # * 根部（base）位置和速度的范围，用于随机初始化
        root_pos_range = [[0.0, 0.0],       # x 位置范围
                          [0.0, 0.0],       # y 位置范围
                          [0.35, 0.4],      # z 位置范围
                          [0.0, 0.0],       # roll 角度范围
                          [0.0, 0.0],       # pitch 角度范围
                          [0.0, 0.0]]       # yaw 角度范围
        root_vel_range = [[-0.5, 2.0],     # x 速度范围
                          [0.0, 0.0],       # y 速度范围
                          [-0.05, 0.05],    # z 速度范围
                          [0.0, 0.0],       # roll 角速度范围
                          [0.0, 0.0],       # pitch 角速度范围
                          [0.0, 0.0]]       # yaw 角速度范围
    
        ref_traj = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/trajectories/single_leg.csv")  # 定义参考轨迹文件路径
    
    class control(MiniCheetahCfg.control):
        # * PD 驱动参数，用于关节控制
        stiffness = {'haa': 20.0, 'hfe': 20.0, 'kfe': 20.0}  # 刚度参数
        damping = {'haa': 0.5, 'hfe': 0.5, 'kfe': 0.5}       # 阻尼参数
        gait_freq = 4.0  # 步态频率，单位为赫兹
        ctrl_frequency = 100  # 控制频率，单位为赫兹
        desired_sim_frequency = 1000  # 期望的仿真频率，单位为赫兹
    
    class commands:
        resampling_time = 4.0  # 指令重新采样时间间隔，单位为秒
    
        class ranges:
            lin_vel_x = [0.0, 3.0]  # 线速度 x 方向的范围 [最小值, 最大值]，单位为米/秒
            lin_vel_y = 1.0         # 线速度 y 方向的最大值，单位为米/秒
            yaw_vel = 3.14 / 2.0    # 旋转速度的最大值，单位为弧度/秒
    
    class push_robots:
        toggle = True  # 是否启用推送机器人功能
        interval_s = 10  # 推送机器人的时间间隔，单位为秒
        max_push_vel_xy = 0.2  # 推送机器人的最大速度，单位为米/秒
    
    class domain_rand(MiniCheetahCfg.domain_rand):
        randomize_friction = True  # 是否随机化摩擦系数
        friction_range = [0.75, 1.05]  # 摩擦系数的随机范围
        randomize_base_mass = True  # 是否随机化基座质量
        added_mass_range = [-1.0, 3.0]  # 基座质量增加的范围
        friction_range = [0.0, 1.0]  # 更新后的摩擦系数范围
    
    class asset(MiniCheetahCfg.asset):
        file = (
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/"
            + "mini_cheetah/urdf/mini_cheetah_rotor.urdf")  # 机器人 URDF 文件路径
        foot_name = "foot"  # 定义脚部的名称
        penalize_contacts_on = ["shank"]  # 惩罚接触的部件
        terminate_after_contacts_on = ["base", "thigh"]  # 接触到这些部件后终止回合
        collapse_fixed_joints = False  # 是否折叠固定关节
        fix_base_link = False  # 是否固定基座链接
        self_collisions = 1  # 自碰撞设置，1 禁用，0 启用
        flip_visual_attachments = False  # 是否翻转视觉附件
        disable_gravity = False  # 是否禁用重力
        disable_motors = False  # 是否禁用电机（所有扭矩设置为 0）
    
    class reward_settings(MiniCheetahCfg.reward_settings):
        soft_dof_pos_limit = 0.9  # 关节位置的软限制
        soft_dof_vel_limit = 0.9  # 关节速度的软限制
        soft_torque_limit = 0.9  # 扭矩的软限制
        max_contact_force = 600.0  # 最大接触力
        base_height_target = BASE_HEIGHT_REF  # 基座高度目标
        tracking_sigma = 0.3  # 追踪误差的标准差
    
    class scaling(MiniCheetahCfg.scaling):
        base_ang_vel = 3.14 / (BASE_HEIGHT_REF / 9.81) ** 0.5  # 基座角速度的缩放因子
        base_lin_vel = BASE_HEIGHT_REF  # 基座线速度的缩放因子
        dof_vel = 100.0  # 关节速度的缩放因子，接近预期的最大速度
        base_height = BASE_HEIGHT_REF  # 基座高度的缩放因子
        dof_pos = 4 * [0.1, 1.0, 2.0]  # 关节位置的缩放因子（髋滚转、髋俯仰、膝关节）
        dof_pos_obs = dof_pos  # 关节位置观测的缩放因子
        dof_pos_target = 0.75  # 动作缩放因子
        tau_ff = 4 * [18, 18, 28]  # 前馈扭矩缩放因子（髋滚转、髋俯仰、膝关节）
        commands = [base_lin_vel, base_lin_vel, base_ang_vel]  # 命令缩放因子
    
    # 定义 MiniCheetahRefRunnerCfg 类，继承自 MiniCheetahRunnerCfg，用于配置 MiniCheetahRef 的运行器
    class MiniCheetahRefRunnerCfg(MiniCheetahRunnerCfg):
        seed = -1  # 设置随机种子，-1 表示不固定种子
    
        class policy(MiniCheetahRunnerCfg.policy):
            actor_hidden_dims = [256, 256, 128]  # Actor 网络隐藏层尺寸
            critic_hidden_dims = [256, 256, 128]  # Critic 网络隐藏层尺寸
            # * 激活函数，可以选择 elu, relu, selu, crelu, lrelu, tanh, sigmoid
            activation = 'elu'
    
            actor_obs = ["base_height",
                         "base_lin_vel",
                         "base_ang_vel",
                         "projected_gravity",
                         "commands",
                         "dof_pos_obs",
                         "dof_vel",
                         "phase_obs"
                         ]  # Actor 网络的观测输入列表
    
            critic_obs = actor_obs  # Critic 网络的观测输入与 Actor 相同
    
            actions = ["dof_pos_target"]  # 动作输出列表
    
            class noise:
                dof_pos_obs = 0.0  # 关节位置观测的噪声标准差
                dof_vel = 0.0      # 关节速度的噪声标准差
                ang_vel = 0.0      # 角速度的噪声标准差
                base_ang_vel = 0.0 # 基座角速度的噪声标准差
                dof_pos = 0.0      # 关节位置动作的噪声标准差
                dof_vel = 0.0      # 关节速度动作的噪声标准差
                lin_vel = 0.0      # 线速度动作的噪声标准差
                ang_vel = 0.0      # 角速度动作的噪声标准差
                gravity_vec = 0.0  # 重力向量的噪声标准差
    
            class reward:
                class weights(MiniCheetahRunnerCfg.policy.reward.weights):
                    tracking_lin_vel = 4.0        # 线速度追踪的权重
                    tracking_ang_vel = 1.0        # 角速度追踪的权重
                    lin_vel_z = 0.0                # z 方向线速度的权重
                    ang_vel_xy = 0.0               # xy 平面角速度的权重
                    orientation = 1.75             # 姿态的权重
                    torques = 5.e-7                # 扭矩的权重
                    dof_vel = 0.0                   # 关节速度的权重
                    min_base_height = 1.5          # 最小基座高度的权重
                    collision = 0.25                # 碰撞的权重
                    action_rate = 0.01              # 动作频率的权重
                    action_rate2 = 0.001             # 动作频率的二级权重
                    stand_still = 0.0                # 静止状态的权重
                    dof_pos_limits = 0.0              # 关节位置限制的权重
                    feet_contact_forces = 0.0         # 脚部接触力的权重
                    dof_near_home = 0.0               # 关节接近初始位置的权重
                    reference_traj = 0.5              # 参考轨迹的权重
                    swing_grf = 3.0                    # 摆动期地面反作用力的权重
                    stance_grf = 3.0                   # 支撑期地面反作用力的权重
    
                class termination_weight:
                    termination = 0.15  # 终止条件的权重
    
        class algorithm(MiniCheetahRunnerCfg.algorithm):
            # 训练参数
            value_loss_coef = 1.0  # 值函数损失的系数
            use_clipped_value_loss = True  # 是否使用裁剪的值函数损失
            clip_param = 0.2  # 裁剪参数
            entropy_coef = 0.01  # 熵正则化的系数
            num_learning_epochs = 6  # 学习的轮数
            # mini batch size = num_envs * nsteps / nminibatches
            num_mini_batches = 6  # 小批量数量
            learning_rate = 5.e-5  # 学习率
            schedule = 'adaptive'  # 学习率调度方式，可以是 'adaptive' 或 'fixed'
            discount_horizon = 1.0  # 折扣视界，单位为秒
            GAE_bootstrap_horizon = 1.0  # GAE 引导视界，单位为秒
            desired_kl = 0.01  # 期望的 KL 散度
            max_grad_norm = 1.0  # 最大梯度范数
    
        class runner(MiniCheetahRunnerCfg.runner):
            run_name = ''  # 运行名称
            experiment_name = 'mini_cheetah_ref'  # 实验名称
            max_iterations = 1000  # 最大策略更新次数
            algorithm_class_name = 'PPO'  # 算法名称，使用 PPO
            num_steps_per_env = 32  # 每个环境的步骤数
   
