from gym.envs.mini_cheetah.mini_cheetah_osc_config \
    import MiniCheetahOscCfg, MiniCheetahOscRunnerCfg  # 从 mini_cheetah_osc_config 模块中导入配置类

BASE_HEIGHT_REF = 0.33  # 定义基础高度参考值，单位为米（m）

class MiniCheetahOscIKCfg(MiniCheetahOscCfg):
    """MiniCheetahOscIKCfg 类继承自 MiniCheetahOscCfg，用于配置 Mini Cheetah 逆运动学（IK）相关参数"""

    class env(MiniCheetahOscCfg.env):
        """环境配置类，继承自 MiniCheetahOscCfg.env"""
        num_envs = 4096  # 定义环境数量为 4096 个
        num_actuators = 12  # 定义执行器数量为 12 个
        episode_length_s = 1.0  # 定义每个回合的长度为 1 秒

    class terrain(MiniCheetahOscCfg.terrain):
        """地形配置类，继承自 MiniCheetahOscCfg.terrain"""
        pass  # 暂不添加额外配置

    class init_state(MiniCheetahOscCfg.init_state):
        """初始状态配置类，继承自 MiniCheetahOscCfg.init_state"""
        pass  # 暂不添加额外配置

    class control(MiniCheetahOscCfg.control):
        """控制配置类，继承自 MiniCheetahOscCfg.control"""
        pass  # 暂不添加额外配置

    class osc(MiniCheetahOscCfg.osc):
        """振荡器配置类，继承自 MiniCheetahOscCfg.osc"""
        pass  # 暂不添加额外配置

    class commands:
        """命令配置类，用于定义机器人运动的命令参数"""
        resampling_time = 4.0  # 命令重采样时间，单位为秒（s）

        class ranges:
            """命令范围配置类，定义线速度和角速度的范围"""
            lin_vel_x = [-1.0, 4.0]  # 线速度 x 轴的最小值和最大值，单位为米/秒（m/s）
            lin_vel_y = 1.0  # 线速度 y 轴的最大值，单位为米/秒（m/s）
            yaw_vel = 3.14    # 角速度的最大值，单位为弧度/秒（rad/s）

    class push_robots:
        """推送机器人配置类，用于定义机器人被外力推送的参数"""
        toggle = False  # 是否启用推送机器人功能
        interval_s = 10  # 推送间隔时间，单位为秒（s）
        max_push_vel_xy = 0.2  # 推送时的最大速度，单位为米/秒（m/s）

    class domain_rand(MiniCheetahOscCfg.domain_rand):
        """领域随机化配置类，继承自 MiniCheetahOscCfg.domain_rand"""
        pass  # 暂不添加额外配置

    class asset(MiniCheetahOscCfg.asset):
        """资产配置类，继承自 MiniCheetahOscCfg.asset"""
        pass  # 暂不添加额外配置

    class reward_settings(MiniCheetahOscCfg.reward_settings):
        """奖励设置配置类，继承自 MiniCheetahOscCfg.reward_settings"""
        pass  # 暂不添加额外配置

    class scaling(MiniCheetahOscCfg.scaling):
        """缩放配置类，继承自 MiniCheetahOscCfg.scaling"""
        ik_pos_target = 0.015  # 逆运动学目标位置的缩放因子，单位为米（m）


class MiniCheetahOscIKRunnerCfg(MiniCheetahOscRunnerCfg):
    """MiniCheetahOscIKRunnerCfg 类继承自 MiniCheetahOscRunnerCfg，用于配置运行器相关参数"""
    seed = -1  # 定义随机种子，-1 表示使用默认种子

    class policy(MiniCheetahOscRunnerCfg.policy):
        """策略配置类，继承自 MiniCheetahOscRunnerCfg.policy"""
        actor_hidden_dims = [256, 256, 128]  # 定义策略网络的隐藏层维度
        critic_hidden_dims = [256, 256, 128]  # 定义价值网络的隐藏层维度
        activation = 'elu'  # 定义激活函数类型，可以是 'elu', 'relu', 'selu', 'crelu', 'lrelu', 'tanh', 'sigmoid' 等

        actor_obs = [
            "base_ang_vel",  # 基座角速度
            "projected_gravity",  # 投影重力
            "commands",  # 命令
            "dof_pos_obs",  # 自由度位置观测
            "dof_vel",  # 自由度速度
            "oscillator_obs"  # 振荡器观测
            # "oscillators_vel",  # 振荡器速度（注释掉）
            # "grf",  # 地面反作用力（注释掉）
            # "osc_coupling",  # 振荡器耦合（注释掉）
            # "osc_offset"  # 振荡器偏移（注释掉）
        ]  # 定义用于策略网络的观测输入

        critic_obs = [
            "base_height",  # 基座高度
            "base_lin_vel",  # 基座线速度
            "base_ang_vel",  # 基座角速度
            "projected_gravity",  # 投影重力
            "commands",  # 命令
            "dof_pos_obs",  # 自由度位置观测
            "dof_vel",  # 自由度速度
            "oscillator_obs",  # 振荡器观测
            "oscillators_vel"  # 振荡器速度
        ]  # 定义用于价值网络的观测输入

        actions = ["ik_pos_target"]  # 定义策略网络的动作输出，这里为逆运动学目标位置

        class noise(MiniCheetahOscRunnerCfg.policy.noise):
            """噪声配置类，继承自 MiniCheetahOscRunnerCfg.policy.noise"""
            pass  # 暂不添加额外配置

        class reward(MiniCheetahOscRunnerCfg.policy.reward):
            """奖励配置类，继承自 MiniCheetahOscRunnerCfg.policy.reward"""

            class weights(MiniCheetahOscRunnerCfg.policy.reward.weights):
                """奖励权重配置类，继承自 MiniCheetahOscRunnerCfg.policy.reward.weights"""
                pass  # 暂不添加额外配置

            class termination_weight:
                """终止奖励权重配置类"""
                termination = 15.0 / 100.0  # 终止条件的奖励权重，15%

    class algorithm(MiniCheetahOscRunnerCfg.algorithm):
        """算法配置类，继承自 MiniCheetahOscRunnerCfg.algorithm"""
        pass  # 暂不添加额外配置

    class runner(MiniCheetahOscRunnerCfg.runner):
        """运行器配置类，继承自 MiniCheetahOscRunnerCfg.runner"""
        run_name = 'IK'  # 运行名称，标识为 'IK'（逆运动学）
        experiment_name = 'oscIK'  # 实验名称，标识为 'oscIK'
        max_iterations = 1000  # 最大迭代次数，表示最多进行 1000 次策略更新
        algorithm_class_name = 'PPO'  # 算法类名称，使用 PPO（Proximal Policy Optimization）算法
        num_steps_per_env = 32  # 每个环境的步数，表示每个环境在一次更新中执行 32 步

