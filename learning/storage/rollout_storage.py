import torch  # 导入PyTorch库

from .base_storage import BaseStorage  # 从当前包中导入BaseStorage基类


class RolloutStorage(BaseStorage):
    """ 
    标准的Rollout存储类，实现用于PPO算法。
    """
    
    class Transition:
        """ 
        Transition存储类。
        即存储所有代理每一步的数据。
        """
        def __init__(self):
            """ 
            初始化Transition实例，定义需要存储的数据。
            """
            self.observations = None  # 当前观测
            self.critic_observations = None  # Critic使用的观测
            self.actions = None  # 执行动作
            self.rewards = None  # 收到的奖励
            self.dones = None  # 完成标志（是否结束）
            self.values = None  # Critic评估的值
            self.actions_log_prob = None  # 动作的对数概率
            self.action_mean = None  # 动作的均值（用于高斯策略）
            self.action_sigma = None  # 动作的标准差（用于高斯策略）

        def clear(self):
            """ 
            清除存储的数据，通过重新调用初始化方法。
            """
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape,
                 privileged_obs_shape, actions_shape, device='cpu'):
        """
        初始化RolloutStorage类。

        参数:
            num_envs (int): 并行环境的数量。
            num_transitions_per_env (int): 每个环境要存储的过渡数量。
            obs_shape (tuple): 观测的形状。
            privileged_obs_shape (tuple or list): 额外（特权）观测的形状，用于Critic。
            actions_shape (tuple): 动作的形状。
            device (str): 计算设备，默认为'cpu'，可选'cuda'。
        """
        self.device = device  # 设置计算设备

        self.obs_shape = obs_shape  # 观测的形状
        self.privileged_obs_shape = privileged_obs_shape  # 特权观测的形状
        self.actions_shape = actions_shape  # 动作的形状

        # * 核心存储
        self.observations = torch.zeros(num_transitions_per_env, num_envs,
                                        *obs_shape, device=self.device)  # 存储观测
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape,
                device=self.device)  # 存储特权观测
        else:
            self.privileged_observations = None  # 如果没有特权观测，则设置为None

        self.rewards = torch.zeros(num_transitions_per_env, num_envs,
                                   device=self.device)  # 存储奖励
        self.actions = torch.zeros(num_transitions_per_env, num_envs,
                                   *actions_shape, device=self.device)  # 存储动作
        self.dones = torch.zeros(num_transitions_per_env, num_envs,
                                 device=self.device).byte()  # 存储完成标志

        # * 用于PPO的额外存储
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs,
                                            1, device=self.device)  # 存储动作的对数概率
        self.values = torch.zeros(num_transitions_per_env, num_envs,
                                  device=self.device)  # 存储Critic评估的值
        self.returns = torch.zeros(num_transitions_per_env, num_envs,
                                   device=self.device)  # 存储计算得到的回报
        self.advantages = torch.zeros(num_transitions_per_env, num_envs,
                                      device=self.device)  # 存储优势值
        self.mu = torch.zeros(num_transitions_per_env, num_envs,
                              *actions_shape, device=self.device)  # 存储动作均值
        self.sigma = torch.zeros(num_transitions_per_env, num_envs,
                                 *actions_shape, device=self.device)  # 存储动作标准差

        self.num_transitions_per_env = num_transitions_per_env  # 每个环境的过渡数量
        self.num_envs = num_envs  # 环境数量
        self.fill_count = 0  # 填充计数，跟踪当前存储中填充的数据量

    def add_transitions(self, transition: Transition):
        """
        将一个Transition实例添加到存储中。

        参数:
            transition (Transition): 当前的Transition实例。
        
        异常:
            AssertionError: 当存储已满时抛出。
        """
        if self.fill_count >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")  # 防止存储溢出

        # 将Transition中的数据复制到相应的存储位置
        self.observations[self.fill_count].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.fill_count].copy_(
                transition.critic_observations)
        self.actions[self.fill_count].copy_(transition.actions)
        self.rewards[self.fill_count].copy_(transition.rewards)
        self.dones[self.fill_count].copy_(transition.dones)
        self.values[self.fill_count].copy_(transition.values)
        self.actions_log_prob[self.fill_count].copy_(
            transition.actions_log_prob.view(-1, 1))
        self.mu[self.fill_count].copy_(transition.action_mean)
        self.sigma[self.fill_count].copy_(transition.action_sigma)
        self.fill_count += 1  # 增加填充计数

    def clear(self):
        """
        清除存储中的所有数据，将填充计数重置为0。
        """
        self.fill_count = 0  # 重置填充计数

    def compute_returns(self, last_values, gamma, lam):
        """
        计算回报和优势值，用于PPO的策略更新。

        参数:
            last_values (torch.Tensor): 最后一步的值估计。
            gamma (float): 折扣因子。
            lam (float): GAE的衰减因子。
        """
        advantage = 0
        for fill_count in reversed(range(self.num_transitions_per_env)):
            if fill_count == self.num_transitions_per_env - 1:
                next_values = last_values  # 如果是最后一个时间步，使用最后的值估计
            else:
                next_values = self.values[fill_count + 1]  # 否则，使用下一步的值估计
            next_is_not_terminal = 1.0 - self.dones[fill_count].float()  # 检查是否非终止状态
            delta = (self.rewards[fill_count]
                     + next_is_not_terminal * gamma * next_values
                     - self.values[fill_count])  # 计算TD误差
            advantage = delta + next_is_not_terminal * gamma * lam * advantage  # 计算优势值
            self.returns[fill_count] = advantage + self.values[fill_count]  # 计算回报

        # * 计算并标准化优势值
        self.advantages = self.returns - self.values  # 优势值 = 回报 - 值
        self.advantages = ((self.advantages - self.advantages.mean())
                           / (self.advantages.std() + 1e-8))  # 标准化优势值

    def get_statistics(self):
        """
        获取存储中的统计信息，如平均轨迹长度和平均奖励。

        返回:
            tuple: (平均轨迹长度, 平均奖励)
        """
        done = self.dones
        done[-1] = 1  # 确保最后一个时间步标记为完成
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)  # 重排并展平完成标志
        done_indices = torch.cat((flat_dones.new_tensor([-1],
                                                        dtype=torch.int64),
                                  flat_dones.nonzero()[:, 0]))  # 获取完成的索引
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])  # 计算轨迹长度
        return trajectory_lengths.float().mean(), self.rewards.mean()  # 返回平均轨迹长度和平均奖励

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        生成用于学习的小批量数据。

        参数:
            num_mini_batches (int): 小批量的数量。
            num_epochs (int): 每个小批量要迭代的次数（默认为8）。

        生成器:
            每个小批量的数据，包括观测、特权观测、动作、目标值、优势值、回报、旧动作对数概率、旧动作均值和旧动作标准差。
        """
        batch_size = self.num_envs * self.num_transitions_per_env  # 总批量大小
        mini_batch_size = batch_size // num_mini_batches  # 每个小批量的大小
        indices = torch.randperm(num_mini_batches * mini_batch_size,
                                 requires_grad=False,
                                 device=self.device)  # 生成随机索引

        # 展平特征
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]  # 获取当前小批量的索引

                # 根据索引获取小批量的数据
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # 生成并返回小批量数据
                yield obs_batch, critic_observations_batch, actions_batch, \
                    target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
