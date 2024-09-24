def adjust_settings(toggle, env_cfg, train_cfg):
    """
    根据切换标志（toggle）调整环境配置（env_cfg）和训练配置（train_cfg）。
    
    参数:
        toggle (str): 一个长度为3的字符串，每个字符为'0'或'1'，用于控制不同的配置选项。
                      例如，'010'表示第一项关闭，第二项开启，第三项关闭。
        env_cfg (dict): 环境配置字典，包含与环境相关的各种参数。
        train_cfg (dict): 训练配置字典，包含与训练过程相关的各种参数。
    
    返回:
        tuple: 更新后的环境配置和训练配置。
    """
    # * 设置实验名称和运行名称
    if train_cfg.runner.experiment_name == '':
        train_cfg.runner.experiment_name = "ORC_" + toggle  # 如果没有指定实验名称，则设置为"ORC_"加上toggle
    else:
        train_cfg.runner.experiment_name = "ORC_" + toggle + '_' \
                                            + train_cfg.runner.experiment_name  # 否则，在现有名称前加上"ORC_"和toggle
    train_cfg.runner.run_name = "ORC_" + toggle  # 设置运行名称为"ORC_"加上toggle
    
    # 将toggle字符串转换为列表，方便逐位检查
    toggle = [x for x in toggle]
    
    # 检查toggle的第一位是否为'0'
    if toggle[0] == '0':
        # * 禁用振荡器观测
        train_cfg.policy.actor_obs.remove('oscillator_obs')  # 从Actor的观测中移除'oscillator_obs'
        train_cfg.policy.critic_obs.remove('oscillator_obs')  # 从Critic的观测中移除'oscillator_obs'
        train_cfg.policy.critic_obs.remove('oscillators_vel')  # 从Critic的观测中移除'oscillators_vel'
    
    # 检查toggle的第二位是否为'0'
    if toggle[1] == '0':
        # * 禁用奖励
        train_cfg.policy.reward.weights.swing_grf = 0.  # 将'swing_grf'奖励权重设置为0
        train_cfg.policy.reward.weights.stance_grf = 0.  # 将'stance_grf'奖励权重设置为0
    
    # 检查toggle的第三位是否为'0'
    if toggle[2] == '0':
        # * 禁用耦合
        env_cfg.osc.coupling = 0.  # 将耦合系数设置为0
        env_cfg.osc.coupling_range = [0., 0.]  # 将耦合范围设置为[0., 0.]
        coupling_stop = 0.  # 将耦合停止时间设置为0
        coupling_step = 0.  # 将耦合步长设置为0
        coupling_slope = 0.  # 将耦合斜率设置为0
        env_cfg.osc.coupling_max = 0.  # 将最大耦合值设置为0
    
    return env_cfg, train_cfg  # 返回更新后的环境配置和训练配置
