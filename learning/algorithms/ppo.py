# 导入必要的库和模块
import torch  # PyTorch主库
import torch.nn as nn  # 用于构建神经网络模块
import torch.optim as optim  # 优化器模块

from learning.modules import ActorCritic  # 从自定义模块中导入ActorCritic类
from learning.storage import RolloutStorage  # 从自定义模块中导入回滚存储类，用于存储环境交互数据

# 定义PPO类（Proximal Policy Optimization，近端策略优化算法）
class PPO:
    actor_critic: ActorCritic  # 声明actor_critic是ActorCritic类型的对象

    # 初始化函数，设置PPO算法的各个参数
    def __init__(self,
                 actor_critic,  # ActorCritic模型，用于生成动作和计算状态值
                 num_learning_epochs=1,  # 学习周期数
                 num_mini_batches=1,  # 每次更新的最小批次数量
                 clip_param=0.2,  # 裁剪参数，用于策略损失的裁剪
                 gamma=0.998,  # 折扣因子，用于计算回报
                 lam=0.95,  # GAE的衰减因子
                 value_loss_coef=1.0,  # 值函数损失的权重
                 entropy_coef=0.0,  # 熵正则项的权重，用于鼓励探索
                 learning_rate=1e-3,  # 学习率
                 max_grad_norm=1.0,  # 梯度裁剪的最大范数
                 use_clipped_value_loss=True,  # 是否使用裁剪的值函数损失
                 schedule="fixed",  # 学习率调度方式
                 desired_kl=0.01,  # 目标KL散度
                 device='cpu',  # 使用的设备，默认为CPU
                 **kwargs  # 其他可选参数
                 ):

        # 设置设备，学习率，KL散度等PPO核心参数
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # * PPO组件初始化
        self.actor_critic = actor_critic  # ActorCritic网络
        self.actor_critic.to(self.device)  # 将actor_critic模型移动到指定设备
        self.storage = None  # 存储器在后续步骤初始化
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)  # Adam优化器
        self.transition = RolloutStorage.Transition()  # 存储过渡状态

        # * PPO超参数
        self.clip_param = clip_param  # 策略损失的裁剪系数
        self.num_learning_epochs = num_learning_epochs  # 学习周期数
        self.num_mini_batches = num_mini_batches  # 每次更新使用的最小批次数量
        self.value_loss_coef = value_loss_coef  # 值函数损失权重
        self.entropy_coef = entropy_coef  # 熵损失权重
        self.gamma = gamma  # 折扣因子
        self.lam = lam  # GAE因子
        self.max_grad_norm = max_grad_norm  # 梯度裁剪的最大范数
        self.use_clipped_value_loss = use_clipped_value_loss  # 是否使用裁剪的值函数损失

    # 初始化存储器，存储环境交互的数据
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape,
                     critic_obs_shape, action_shape):
        # 使用RolloutStorage存储每个环境的交互数据
        self.storage = RolloutStorage(num_envs, num_transitions_per_env,
                                      actor_obs_shape, critic_obs_shape,
                                      action_shape, self.device)

    # 切换到测试模式，不更新网络权重
    def test_mode(self):
        self.actor_critic.test()

    # 切换到训练模式，可以更新网络权重
    def train_mode(self):
        self.actor_critic.train()

    # 根据观测值生成动作
    def act(self, obs, critic_obs):
        # * 计算动作和状态值
        self.transition.actions = self.actor_critic.act(obs).detach()  # 生成动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()  # 评估当前状态的值
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()  # 获取动作的对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach()  # 动作均值
        self.transition.action_sigma = self.actor_critic.action_std.detach()  # 动作标准差
        # * 在执行环境步之前记录观测值
        self.transition.observations = obs  # 保存当前观测值
        self.transition.critic_observations = critic_obs  # 保存当前状态值的观测
        return self.transition.actions  # 返回生成的动作

    # 处理环境中的一步，包括奖励和是否完成
    def process_env_step(self, rewards, dones, timed_out=None):
        self.transition.rewards = rewards.clone()  # 记录奖励
        self.transition.dones = dones  # 记录是否完成
        # * 处理超时时的奖励引导
        if timed_out is not None:
            self.transition.rewards += self.gamma * self.transition.values * timed_out  # 对超时的情况进行奖励引导

        # * 记录过渡
        self.storage.add_transitions(self.transition)  # 将过渡添加到存储器
        self.transition.clear()  # 清除当前的过渡数据

    # 计算回报，使用最后一个观测的状态值
    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()  # 计算最后状态的值
        self.storage.compute_returns(last_values, self.gamma, self.lam)  # 计算每个状态的回报

    # 更新PPO模型，进行策略优化
    def update(self):
        mean_value_loss = 0  # 平均值函数损失
        mean_surrogate_loss = 0  # 平均代理损失
        # 从存储器中生成最小批次训练数据
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # 遍历生成的批次
        for (obs_batch, critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch) in generator:

            # * 更新策略
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)  # 计算新的动作概率
            value_batch = self.actor_critic.evaluate(critic_obs_batch)  # 评估状态值
            mu_batch = self.actor_critic.action_mean  # 获取动作均值
            sigma_batch = self.actor_critic.action_std  # 获取动作标准差
            entropy_batch = self.actor_critic.entropy  # 计算熵

            # * KL散度自适应调整学习率
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():  # 不计算梯度
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                                   + (torch.square(old_sigma_batch)
                                      + torch.square(old_mu_batch - mu_batch))
                                   / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)  # 计算KL散度
                    kl_mean = torch.mean(kl)  # 计算KL散度的均值

                    # 自适应调整学习率
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # 更新优化器中的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # * 代理损失（策略损失）
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))  # 计算策略的比率
            surrogate = -torch.squeeze(advantages_batch) * ratio  # 计算未裁剪的策略损失
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)  # 计算裁剪的策略损失
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()  # 取最大值并计算均值

            # * 值函数损失
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
