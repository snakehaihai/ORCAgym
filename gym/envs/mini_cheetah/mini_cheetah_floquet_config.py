from gym.envs.mini_cheetah.mini_cheetah_osc_config \
    import MiniCheetahOscCfg, MiniCheetahOscRunnerCfg  # 从 mini_cheetah_osc_config 模块中导入配置类

BASE_HEIGHT_REF = 0.33  # 定义基座高度参考值，单位为米


class MiniCheetahFloquetCfg(MiniCheetahOscCfg):
    """MiniCheetahFloquetCfg 类继承自 MiniCheetahOscCfg，用于定义 Floquet 控制的配置"""

    class env(MiniCheetahOscCfg.env):
        """环境配置类，继承自 MiniCheetahOscCfg.env"""
        num_envs = 4096  # 环境数量
        num_actuators = 12  # 执行器数量
        episode_length_s = 1.  # 每个回合的长度，单位为秒

    class terrain(MiniCheetahOscCfg.terrain):
        """地形配置类，继承自 MiniCheetahOscCfg.terrain"""
        pass  # 使用父类的配置，不做额外修改

    class init_state(MiniCheetahOscCfg.init_state):
        """初始状态配置类，继承自 MiniCheetahOscCfg.init_state"""
        pass  # 使用父类的配置，不做额外修改

    class control(MiniCheetahOscCfg.control):
        """控制配置类，继承自 MiniCheetahOscCfg.control"""
        pass  # 使用父类的配置，不做额外修改

    class osc(MiniCheetahOscCfg.osc):
        """振荡器配置类，继承自 MiniCheetahOscCfg.osc"""
        pass  # 使用父类的配置，不做额外修改

    class commands(MiniCheetahOscCfg.commands):
        """命令配置类，继承自 MiniCheetahOscCfg.commands"""
        pass  # 使用父类的配置，不做额外修改
        # resampling_time = 4.  # 命令重新采样时间，单位为秒（已注释）

        class ranges(MiniCheetahOscCfg.commands.ranges):
            """命令范围配置类，继承自 MiniCheetahOscCfg.commands.ranges"""
            pass  # 使用父类的配置，不做额外修改
            # lin_vel_x = [-1., 4.]  # 线速度 x 轴的最小和最大值，单位为 m/s（已注释）
            # lin_vel_y = 1.  # 线速度 y 轴的最大值，单位为 m/s（已注释）
            # yaw_vel = 6    # 偏航速度的最大值，单位为 rad/s（已注释）

    class push_robots:
        """推机器人配置类"""
        toggle = True  # 是否启用推机器人功能
        interval_s = 10  # 推机器人的时间间隔，单位为秒
        max_push_vel_xy = 0.2  # 推机器人的最大速度，单位为 m/s

    class domain_rand(MiniCheetahOscCfg.domain_rand):
        """领域随机化配置类，继承自 MiniCheetahOscCfg.domain_rand"""
        pass  # 使用父类的配置，不做额外修改

    class asset(MiniCheetahOscCfg.asset):
        """资产配置类，继承自 MiniCheetahOscCfg.asset"""
        pass  # 使用父类的配置，不做额外修改

    class reward_settings(MiniCheetahOscCfg.reward_settings):
        """奖励设置配置类，继承自 MiniCheetahOscCfg.reward_settings"""
        pass  # 使用父类的配置，不做额外修改

    class scaling(MiniCheetahOscCfg.scaling):
        """缩放配置类，继承自 MiniCheetahOscCfg.scaling"""
        pass  # 使用父类的配置，不做额外修改


class MiniCheetahFloquetRunnerCfg(MiniCheetahOscRunnerCfg):
    """MiniCheetahFloquetRunnerCfg 类继承自 MiniCheetahOscRunnerCfg，用于定义 Floquet 控制的运行配置"""
    seed = -1  # 随机种子，-1 表示不固定

    class policy(MiniCheetahOscRunnerCfg.policy):
        """策略配置类，继承自 MiniCheetahOscRunnerCfg.policy"""
        actor_hidden_dims = [256, 256, 128]  # Actor 网络的隐藏层维度
        critic_hidden_dims = [256, 256, 128]  # Critic 网络的隐藏层维度
        activation = 'elu'  # 激活函数，可选 'elu', 'relu', 'selu', 'crelu', 'lrelu', 'tanh', 'sigmoid'

        actor_obs = [  # Actor 网络的观察空间
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "dof_pos_target"
        ]

        critic_obs = [  # Critic 网络的观察空间
            "base_height",
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos_obs",
            "dof_vel",
            "oscillator_obs",
            "oscillators_vel",
            "dof_pos_target"
        ]

        actions = ["dof_pos_target"]  # Actor 网络的动作空间

        class noise(MiniCheetahOscRunnerCfg.policy.noise):
            """噪声配置类，继承自 MiniCheetahOscRunnerCfg.policy.noise"""
            pass  # 使用父类的配置，不做额外修改

        class reward(MiniCheetahOscRunnerCfg.policy.reward):
            """奖励函数配置类，继承自 MiniCheetahOscRunnerCfg.policy.reward"""

            class weights(MiniCheetahOscRunnerCfg.policy.reward.weights):
                """奖励权重配置类，继承自 MiniCheetahOscRunnerCfg.policy.reward.weights"""
                floquet = 0.  # Floquet 奖励权重
                locked_frequency = 0.  # 锁定频率奖励权重

            class termination_weight:
                """终止条件权重配置类"""
                pass
                # termination = 15./100.  # 终止条件权重（已注释）

    class algorithm(MiniCheetahOscRunnerCfg.algorithm):
        """算法配置类，继承自 MiniCheetahOscRunnerCfg.algorithm"""
        pass  # 使用父类的配置，不做额外修改

    class runner(MiniCheetahOscRunnerCfg.runner):
        """运行器配置类，继承自 MiniCheetahOscRunnerCfg.runner"""
        run_name = 'floquet'  # 运行名称
        experiment_name = 'osc'  # 实验名称
        max_iterations = 3000  # 最大迭代次数，即策略更新次数
        algorithm_class_name = 'PPO'  # 使用的算法类名称，这里为 PPO
        num_steps_per_env = 32  # 每个环境的步数
