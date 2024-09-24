
import time  # 导入时间模块，用于记录时间
import os  # 导入操作系统接口模块，用于路径操作
import wandb  # 导入Weights & Biases库，用于实验追踪和可视化
import torch  # 导入PyTorch库，用于深度学习
from isaacgym.torch_utils import torch_rand_float  # 从isaacgym库中导入随机浮点数生成函数

from learning.algorithms import PPO  # 从自定义模块中导入PPO算法类
from learning.modules import ActorCritic  # 从自定义模块中导入ActorCritic类
from learning.env import VecEnv  # 从自定义模块中导入向量化环境类
from learning.utils import remove_zero_weighted_rewards  # 从自定义模块中导入实用函数
from learning.utils import Logger  # 从自定义模块中导入日志记录器类


class OnPolicyRunner:
    """
    OnPolicyRunner类，用于运行基于策略的强化学习算法（如PPO），管理环境交互、策略更新和日志记录。
    """

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 device='cpu'):
        """
        初始化OnPolicyRunner类。

        参数:
            env (VecEnv): 向量化的环境实例。
            train_cfg (dict): 训练配置字典，包含runner、algorithm和policy的配置。
            device (str): 计算设备，默认为'cpu'，可选'cuda'。
        """
        self.device = device  # 设置计算设备
        self.env = env  # 赋值环境实例
        self.parse_train_cfg(train_cfg)  # 解析训练配置

        # 获取Actor和Critic的观测维度及动作维度
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])
        num_actions = self.get_action_size(self.policy_cfg["actions"])

        # 初始化ActorCritic网络，并移动到指定设备
        actor_critic = ActorCritic(num_actor_obs,
                                   num_critic_obs,
                                   num_actions,
                                   **self.policy_cfg).to(self.device)

        # 动态获取算法类（例如PPO）
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic,
                                  device=self.device, **self.alg_cfg)

        # 设置训练参数
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # 每个环境的步数
        self.save_interval = self.cfg["save_interval"]  # 保存模型的间隔
        self.tot_timesteps = 0  # 总时间步数
        self.tot_time = 0  # 总时间
        self.it = 0  # 当前迭代次数

        # 初始化存储和模型
        self.init_storage()

        # 初始化日志记录
        self.log_dir = train_cfg["log_dir"]  # 日志目录
        self.SE_path = os.path.join(self.log_dir, 'SE')  # 状态估计器的日志路径
        self.logger = Logger(self.log_dir, self.env.max_episode_length_s,
                             self.device)  # 创建日志记录器实例

        # 设置需要记录的奖励键
        reward_keys_to_log = \
            list(self.policy_cfg["reward"]["weights"].keys()) \
            + list(self.policy_cfg["reward"]["termination_weight"].keys())

        reward_keys_to_log += ["Total_reward"]

        # 设置步态跟踪奖励权重
        self.gait_weights = {'trot': 1.,
                             'pace': 1.,
                             'pronk': 1.,
                             'bound': 1,
                             'asymettric': 1.}
        reward_keys_to_log += list(self.gait_weights.keys())  # 添加步态相关的奖励键
        reward_keys_to_log += ['Gait_score']
        self.logger.initialize_buffers(self.env.num_envs, reward_keys_to_log)  # 初始化日志缓冲区

    def parse_train_cfg(self, train_cfg):
        """
        解析训练配置。

        参数:
            train_cfg (dict): 训练配置字典。
        """
        self.cfg = train_cfg['runner']  # 运行器配置
        self.alg_cfg = train_cfg['algorithm']  # 算法配置

        # 移除奖励中权重为零的项
        remove_zero_weighted_rewards(train_cfg['policy']['reward']['weights'])
        self.policy_cfg = train_cfg['policy']  # 策略配置

    def init_storage(self):
        """
        初始化算法的存储，用于存储观测、动作、奖励等数据。
        """
        num_actor_obs = self.get_obs_size(self.policy_cfg["actor_obs"])  # Actor观测维度
        num_critic_obs = self.get_obs_size(self.policy_cfg["critic_obs"])  # Critic观测维度
        num_actions = self.get_action_size(self.policy_cfg["actions"])  # 动作维度

        # 初始化存储结构
        self.alg.init_storage(self.env.num_envs,
                              self.num_steps_per_env,
                              actor_obs_shape=[num_actor_obs],
                              critic_obs_shape=[num_critic_obs],
                              action_shape=[num_actions])

    def attach_to_wandb(self, wandb, log_freq=100, log_graph=True):
        """
        将模型附加到Weights & Biases进行监控和记录。

        参数:
            wandb: Weights & Biases实例。
            log_freq (int): 日志记录频率。
            log_graph (bool): 是否记录计算图。
        """
        wandb.watch((self.alg.actor_critic.actor,
                    self.alg.actor_critic.critic),
                    log_freq=log_freq,
                    log_graph=log_graph)  # 监控Actor和Critic网络

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        执行学习过程，包括环境交互、策略更新和日志记录。

        参数:
            num_learning_iterations (int): 学习的迭代次数。
            init_at_random_ep_len (bool): 是否以随机的初始回合长度开始，默认为False。
        """
        if init_at_random_ep_len:
            # 以随机的初始回合长度开始
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length))

        # 获取初始观测
        actor_obs = self.get_obs(self.policy_cfg["actor_obs"])
        critic_obs = self.get_obs(self.policy_cfg["critic_obs"])
        self.alg.actor_critic.train()  # 设置ActorCritic为训练模式
        self.num_learning_iterations = num_learning_iterations  # 设置学习迭代次数
        self.tot_iter = self.it + num_learning_iterations  # 计算总迭代次数

        self.save()  # 保存初始模型

        # 获取奖励权重和终止权重
        reward_weights = self.policy_cfg['reward']['weights']
        termination_weight = self.policy_cfg['reward']['termination_weight']
        rewards = 0. * self.get_rewards(reward_weights)  # 初始化奖励

        # 模拟2秒，让机器人倒下
        for i in range(int(2 / self.env.dt)):
            self.env.step()

        # 开始学习循环
        for self.it in range(self.it + 1, self.tot_iter + 1):
            start = time.time()  # 记录开始时间
            # 收集Rollout
            with torch.inference_mode():  # 禁用梯度计算，加快推理速度

                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(actor_obs, critic_obs)  # 获取动作
                    self.set_actions(actions)  # 设置动作到环境

                    self.env.step()  # 执行动作

                    # 获取新的观测
                    actor_obs = self.get_noisy_obs(
                        self.policy_cfg['actor_obs'],
                        self.policy_cfg['noise'])
                    critic_obs = self.get_obs(self.policy_cfg['critic_obs'])

                    # 获取超时和终止标志
                    timed_out = self.get_timed_out()
                    terminated = self.get_terminated()
                    dones = timed_out | terminated  # 计算完成标志

                    # 计算并记录奖励
                    rewards += self.get_and_log_rewards(reward_weights,
                                                        modifier=self.env.dt,
                                                        mask=~terminated)
                    rewards += self.get_and_log_rewards(termination_weight,
                                                        mask=terminated)
                    self.logger.log_current_reward('Total_reward', rewards)

                    # 记录步态相关的奖励
                    gait_score = self.get_and_log_rewards(self.gait_weights,
                                                          modifier=self.env.dt,
                                                          mask=~terminated)
                    self.logger.log_current_reward('Gait_score', gait_score)

                    # 处理环境步
                    self.alg.process_env_step(rewards, dones, timed_out)
                    self.logger.update_episode_buffer(dones)
                    rewards *= 0.  # 重置奖励

                stop = time.time()  # 记录结束时间
                self.collection_time = stop - start  # 计算收集时间

                # 学习步骤
                start = stop
                self.alg.compute_returns(critic_obs)  # 计算回报

            # 更新算法
            self.mean_value_loss, self.mean_surrogate_loss = self.alg.update()
            stop = time.time()  # 记录学习结束时间
            self.learn_time = stop - start  # 计算学习时间
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs  # 更新总时间步数
            self.tot_time += self.collection_time + self.learn_time  # 更新总时间

            self.log()  # 记录日志

            if self.it % self.save_interval == 0:
                self.save()  # 按照保存间隔保存模型

        # 训练结束后保存模型
        self.save()

    def get_noise(self, obs_list, noise_dict):
        """
        生成噪声向量，用于对观测进行扰动。

        参数:
            obs_list (list): 观测名称列表。
            noise_dict (dict): 噪声配置字典，包含各观测的噪声幅度和缩放因子。

        返回:
            torch.Tensor: 生成的噪声张量。
        """
        # 初始化噪声向量
        noise_vec = torch.zeros(self.get_obs_size(obs_list),
                                device=self.device)
        obs_index = 0  # 观测索引
        for obs in obs_list:
            obs_size = self.get_obs_size([obs])  # 当前观测的尺寸
            if obs in noise_dict.keys():
                # 根据噪声字典生成噪声张量
                noise_tensor = torch.ones(obs_size).to(self.device) \
                               * torch.tensor(noise_dict[obs]).to(self.device)
                if obs in self.env.scales.keys():
                    noise_tensor /= self.env.scales[obs]  # 根据环境缩放因子调整噪声
                noise_vec[obs_index:obs_index + obs_size] = noise_tensor  # 更新噪声向量
            obs_index += obs_size  # 更新观测索引

        # 生成随机噪声并应用缩放
        return torch_rand_float(-1., 1., (self.env.num_envs, len(noise_vec)),
                                self.device) * noise_vec * noise_dict["scale"]








def get_noisy_obs(self, obs_list, noise_dict):
    """
    获取带噪声的观测值。

    参数:
        obs_list (list): 需要获取的观测名称列表。
        noise_dict (dict): 噪声配置字典，包含各观测的噪声幅度和缩放因子。

    返回:
        torch.Tensor: 添加噪声后的观测张量。
    """
    observation = self.get_obs(obs_list)  # 获取原始观测
    return observation + self.get_noise(obs_list, noise_dict)  # 添加噪声并返回

def get_obs(self, obs_list):
    """
    获取指定观测的当前状态。

    参数:
        obs_list (list): 需要获取的观测名称列表。

    返回:
        torch.Tensor: 当前观测的张量。
    """
    observation = self.env.get_states(obs_list).to(self.device)  # 从环境中获取观测并移动到指定设备
    return observation  # 返回观测张量

def set_actions(self, actions):
    """
    将动作设置到环境中执行。

    参数:
        actions (torch.Tensor): 要执行的动作张量。
    """
    if self.policy_cfg['disable_actions']:
        return  # 如果配置中禁用了动作，则不执行任何操作

    # 如果环境配置中有动作裁剪的设置，则进行动作裁剪
    if hasattr(self.env.cfg.scaling, "clip_actions"):
        actions = torch.clip(actions,
                             -self.env.cfg.scaling.clip_actions,
                             self.env.cfg.scaling.clip_actions)

    # 将动作设置到环境中
    self.env.set_states(self.policy_cfg["actions"], actions)

def get_timed_out(self):
    """
    获取是否超时的标志。

    返回:
        torch.Tensor: 超时标志的张量。
    """
    return self.env.get_states(['timed_out']).to(self.device)  # 从环境中获取'timed_out'状态并移动到设备

def get_terminated(self):
    """
    获取是否终止的标志。

    返回:
        torch.Tensor: 终止标志的张量。
    """
    return self.env.get_states(['terminated']).to(self.device)  # 从环境中获取'terminated'状态并移动到设备

def get_obs_size(self, obs_list):
    """
    获取指定观测的维度大小。

    参数:
        obs_list (list): 需要获取维度的观测名称列表。

    返回:
        int: 观测的维度大小。
    """
    return self.get_obs(obs_list)[0].shape[0]  # 获取观测的第一个样本的维度大小

def get_action_size(self, action_list):
    """
    获取指定动作的维度大小。

    参数:
        action_list (list): 需要获取维度的动作名称列表。

    返回:
        int: 动作的维度大小。
    """
    return self.env.get_states(action_list)[0].shape[0]  # 获取动作的第一个样本的维度大小

def get_and_log_rewards(self, reward_weights, modifier=1, mask=None):
    """
    计算每个奖励，记录日志，并返回总奖励。

    参数:
        reward_weights (dict): 包含奖励名称和对应权重的字典。
        modifier (float): 额外的权重系数，应用于所有奖励（默认值为1）。
        mask (torch.Tensor or None): 布尔张量，用于控制哪些奖励被计算（默认值为None，表示所有奖励都被计算）。

    返回:
        torch.Tensor: 计算得到的总奖励张量。
    """
    if mask is None:
        mask = 1.0  # 如果没有提供掩码，则所有奖励都被计算

    # 初始化总奖励张量
    total_rewards = torch.zeros(self.env.num_envs,
                                device=self.device, dtype=torch.float)

    # 遍历所有奖励类型，计算并累加奖励
    for name, weight in reward_weights.items():
        reward = mask * self.get_rewards({name: weight}, modifier)  # 计算单个奖励
        total_rewards += reward  # 累加到总奖励
        self.logger.log_current_reward(name, reward)  # 记录当前奖励

    return total_rewards  # 返回总奖励

def get_rewards(self, reward_weights, modifier=1):
    """
    计算指定权重的奖励。

    参数:
        reward_weights (dict): 包含奖励名称和对应权重的字典。
        modifier (float): 额外的权重系数，应用于所有奖励（默认值为1）。

    返回:
        torch.Tensor: 计算得到的奖励张量。
    """
    return modifier * self.env.compute_reward(reward_weights).to(self.device)  # 计算并返回奖励

def log(self):
    """
    记录和打印当前的训练状态和统计信息。
    """
    # 计算每秒帧数（FPS）
    fps = int(self.num_steps_per_env * self.env.num_envs
              / (self.collection_time + self.learn_time))
    
    # 计算Actor的噪声标准差的平均值
    mean_noise_std = self.alg.actor_critic.std.mean().item()
    
    # 添加日志信息
    self.logger.add_log(self.logger.mean_rewards)
    self.logger.add_log({
        'Loss/value_function': self.mean_value_loss,
        'Loss/surrogate': self.mean_surrogate_loss,
        'Loss/learning_rate': self.alg.learning_rate,
        'Policy/mean_noise_std': mean_noise_std,
        'Perf/total_fps': fps,
        'Perf/collection_time': self.collection_time,
        'Perf/learning_time': self.learn_time,
        'Train/mean_reward': self.logger.total_mean_reward,
        'Train/mean_episode_length': self.logger.mean_episode_length,
        'Train/total_timesteps': self.tot_timesteps,
        'Train/iteration_time': self.collection_time + self.learn_time,
        'Train/time': self.tot_time,
    })
    
    # 更新迭代次数
    self.logger.update_iterations(self.it, self.tot_iter,
                                  self.num_learning_iterations)
    
    # 如果使用了Weights & Biases，记录到wandb
    if wandb.run is not None:
        self.logger.log_to_wandb()
    
    # 打印日志到终端
    self.logger.print_to_terminal()

def get_infos(self):
    """
    获取环境的额外信息。

    返回:
        dict: 环境的额外信息字典。
    """
    return self.env.extras  # 返回环境的额外信息

def save(self):
    """
    保存当前模型和优化器的状态到文件。
    """
    # 构建保存路径
    path = os.path.join(self.log_dir, 'model_{}.pt'.format(self.it))
    
    # 保存模型状态字典和优化器状态字典
    torch.save({
        'model_state_dict': self.alg.actor_critic.state_dict(),
        'optimizer_state_dict': self.alg.optimizer.state_dict(),
        'iter': self.it
    }, path)

def load(self, path, load_optimizer=True):
    """
    从文件加载模型和优化器的状态。

    参数:
        path (str): 模型文件的路径。
        load_optimizer (bool): 是否加载优化器的状态（默认值为True）。
    """
    loaded_dict = torch.load(path)  # 加载保存的字典
    self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])  # 加载模型状态
    if load_optimizer:
        self.alg.optimizer.load_state_dict(
            loaded_dict['optimizer_state_dict'])  # 加载优化器状态
    self.it = loaded_dict['iter']  # 更新当前迭代次数

def switch_to_eval(self):
    """
    将ActorCritic网络切换到评估模式。
    """
    self.alg.actor_critic.eval()  # 设置ActorCritic为评估模式

def get_inference_actions(self):
    """
    获取推理模式下的动作。

    返回:
        torch.Tensor: 推理模式下生成的动作张量。
    """
    # 获取带噪声的观测
    obs = self.get_noisy_obs(self.policy_cfg["actor_obs"],
                             self.policy_cfg['noise'])
    return self.alg.actor_critic.actor.act_inference(obs)  # 使用Actor的推理方法生成动作


    def export(self, path):
        self.alg.actor_critic.export_policy(path)
