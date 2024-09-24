import os  # 导入操作系统接口模块，用于路径操作
import wandb  # 导入Weights & Biases库，用于实验追踪和可视化
import torch  # 导入PyTorch库，用于深度学习
import shutil  # 导入高级文件操作模块，用于复制文件和目录
import fnmatch  # 导入文件名匹配模块，用于处理文件匹配模式
from collections import deque  # 导入双端队列，用于高效的队列操作
from statistics import mean  # 导入统计模块，用于计算平均值


class Logger:
    def __init__(self, log_dir, max_episode_length_s, device):
        """
        初始化Logger类，用于记录训练过程中的各种日志信息。

        参数:
            log_dir (str): 日志保存的目录路径。
            max_episode_length_s (float): 每个回合的最大持续时间（秒）。
            device (str): 计算设备，默认为'cpu'，可选'cuda'。
        """
        self.log_dir = log_dir  # 设置日志目录
        self.device = device  # 设置计算设备
        self.avg_window = 100  # 设置用于计算平均值的窗口大小
        self.log = {}  # 初始化日志字典
        self.it = 0  # 当前迭代次数
        self.tot_iter = 0  # 总迭代次数
        self.learning_iter = 0  # 学习迭代次数
        self.mean_episode_length = 0.  # 平均回合长度
        self.total_mean_reward = 0.  # 总平均奖励
        self.max_episode_length_s = max_episode_length_s  # 设置最大回合持续时间

    def initialize_buffers(self, num_envs, reward_keys):
        """
        初始化用于存储奖励和回合长度的缓冲区。

        参数:
            num_envs (int): 并行环境的数量。
            reward_keys (list): 需要记录的奖励键列表。
        """
        self.current_episode_return = {
            name: torch.zeros(
                num_envs, dtype=torch.float,
                device=self.device, requires_grad=False)
            for name in reward_keys
        }  # 初始化当前回合的奖励字典，每个奖励键对应一个零张量
        self.current_episode_length = torch.zeros(
            num_envs,
            dtype=torch.float, device=self.device
        )  # 初始化当前回合的长度张量
        self.avg_return_buffer = {
            name: deque(maxlen=self.avg_window)
            for name in reward_keys
        }  # 初始化用于计算平均奖励的缓冲区
        self.avg_length_buffer = deque(maxlen=self.avg_window)  # 初始化用于计算平均回合长度的缓冲区
        self.mean_rewards = {"Episode/" + name: 0. for name in reward_keys}  # 初始化平均奖励字典

    def log_to_wandb(self):
        """
        将当前日志记录到Weights & Biases（wandb）。
        """
        wandb.log(self.log)  # 使用wandb记录日志

    def add_log(self, log_dict):
        """
        将新的日志信息添加到当前日志字典中。

        参数:
            log_dict (dict): 包含需要记录的日志信息的字典。
        """
        self.log.update(log_dict)  # 更新日志字典

    def update_iterations(self, it, tot_iter, learning_iter):
        """
        更新当前迭代次数、总迭代次数和学习迭代次数。

        参数:
            it (int): 当前迭代次数。
            tot_iter (int): 总迭代次数。
            learning_iter (int): 学习迭代次数。
        """
        self.it = it  # 更新当前迭代次数
        self.tot_iter = tot_iter  # 更新总迭代次数
        self.learning_iter = learning_iter  # 更新学习迭代次数

    def log_current_reward(self, name, reward):
        """
        记录当前回合的奖励。

        参数:
            name (str): 奖励的名称。
            reward (torch.Tensor): 当前奖励的张量。
        """
        if name in self.current_episode_return.keys():
            self.current_episode_return[name] += reward  # 累加当前回合的奖励

    def update_episode_buffer(self, dones):
        """
        更新回合缓冲区，当回合结束时，将当前回合的奖励和长度添加到缓冲区。

        参数:
            dones (torch.Tensor): 完成标志的张量，指示哪些回合已经结束。
        """
        self.current_episode_length += 1  # 增加当前回合的长度
        terminated_ids = torch.where(dones == True)[0]  # 获取已结束回合的索引
        for name in self.current_episode_return.keys():
            # 将已结束回合的奖励添加到平均奖励缓冲区
            self.avg_return_buffer[name].extend(
                self.current_episode_return[name]
                [terminated_ids].cpu().numpy().tolist()
            )
            self.current_episode_return[name][terminated_ids] = 0.  # 重置已结束回合的奖励

        # 将已结束回合的长度添加到平均长度缓冲区
        self.avg_length_buffer.extend(
            self.current_episode_length[terminated_ids].cpu().numpy().tolist()
        )
        self.current_episode_length[terminated_ids] = 0  # 重置已结束回合的长度

        if (len(self.avg_length_buffer) > 0):
            self.calculate_reward_avg()  # 计算平均奖励和回合长度

    def calculate_reward_avg(self):
        """
        计算平均回合长度和平均奖励。
        """
        self.mean_episode_length = mean(self.avg_length_buffer)  # 计算平均回合长度
        self.mean_rewards = {
            "Episode/" + name:
                mean(self.avg_return_buffer[name]) / self.max_episode_length_s
            for name in self.current_episode_return.keys()
        }  # 计算并标准化每种奖励的平均值
        self.total_mean_reward = sum(list(self.mean_rewards.values()))  # 计算总平均奖励

    def print_to_terminal(self):
        """
        打印当前的训练状态和统计信息到终端。
        """
        width = 80  # 设置输出宽度
        pad = 35  # 设置对齐填充
        header = f" \033[1m Learning iteration {self.it}/{self.tot_iter} \033[0m "  # 格式化标题

        log_string = (
            f"""{'#' * width}\n"""
            f"""{header.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {
                self.log['Perf/total_fps']:.0f} steps/s (collection: {
                self.log['Perf/collection_time']:.3f}s, learning {
                self.log['Perf/learning_time']:.3f}s)\n"""
            f"""{'Value function loss:':>{pad}} {
                self.log['Loss/value_function']:.4f}\n"""
            f"""{'Surrogate loss:':>{pad}} {
                self.log['Loss/surrogate']:.4f}\n"""
            f"""{'Mean action noise std:':>{pad}} {
                self.log['Policy/mean_noise_std']:.2f}\n"""
            f"""{'Mean episode length:':>{pad}} {
                self.log['Train/mean_episode_length']:.2f}\n"""
            f"""{'Mean reward:':>{pad}} {self.total_mean_reward:.2f}\n"""
        )
        log_string += f"""{'-' * width}\n"""  # 添加分隔线

        for key, value in self.mean_rewards.items():
            log_string += f"""{
                f'Mean episode {key[8:]}:':>{pad}} {value:.4f}\n"""  # 记录每种奖励的平均值

        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {
                self.log['Train/total_timesteps']}\n"""
            f"""{'Iteration time:':>{pad}} {
                self.log['Train/iteration_time']:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.log['Train/time']:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.log['Train/time'] / (self.it + 1) * (
                self.learning_iter - self.it):.1f}s\n"""
        )

        print(log_string)  # 打印日志字符串到终端

    def configure_local_files(self, save_paths):
        """
        配置并复制本地文件到日志目录中，用于记录和备份。

        参数:
            save_paths (list): 包含需要保存的文件或目录信息的字典列表。
        """
        def create_ignored_pattern_except(*patterns):
            """
            创建一个忽略模式，仅保留指定的模式。

            参数:
                *patterns: 需要保留的文件或目录模式。

            返回:
                函数: 用于shutil.copytree的ignore参数。
            """
            def _ignore_patterns(path, names):
                keep = set(name for pattern in patterns for name in
                           fnmatch.filter(names, pattern))
                ignore = set(name for name in names if name not in keep and
                             not os.path.isdir(os.path.join(path, name)))
                return ignore
            return _ignore_patterns

        def remove_empty_folders(path, removeRoot=True):
            """
            移除空文件夹。

            参数:
                path (str): 需要检查和移除的目录路径。
                removeRoot (bool): 是否移除根目录，默认为True。
            """
            if not os.path.isdir(path):
                return
            # 递归移除空子文件夹
            files = os.listdir(path)
            if len(files):
                for f in files:
                    fullpath = os.path.join(path, f)
                    if os.path.isdir(fullpath):
                        remove_empty_folders(fullpath)
            # 如果文件夹为空，则删除它
            files = os.listdir(path)
            if len(files) == 0 and removeRoot:
                os.rmdir(path)

        # 将相关的源文件复制到本地日志目录中以备记录
        save_dir = self.log_dir + '/files/'  # 设置保存目录
        for save_path in save_paths:
            if save_path['type'] == 'file':
                os.makedirs(save_dir + save_path['target_dir'],
                            exist_ok=True)  # 创建目标目录
                shutil.copy2(save_path['source_file'],
                             save_dir + save_path['target_dir'])  # 复制文件
            elif save_path['type'] == 'dir':
                shutil.copytree(
                    save_path['source_dir'],
                    save_dir + save_path['target_dir'],
                    ignore=create_ignored_pattern_except(
                        *save_path['include_patterns']))  # 复制目录，忽略不需要的文件
            else:
                print('WARNING: uncaught save path type:', save_path['type'])  # 未处理的保存类型
        remove_empty_folders(save_dir)  # 移除空文件夹
