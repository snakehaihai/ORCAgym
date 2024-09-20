import sys
from isaacgym import gymapi
from isaacgym import gymutil
import torch

#BaseTask类：这个类是强化学习任务的基类，定义了任务环境的基本操作，包括重置、状态获取、奖励计算、渲染等方法。
#缓冲区：to_be_reset、terminated、timed_out等缓冲区用来记录当前环境的状态信息。
#渲染与交互：如果不是headless模式，将创建viewer，并且订阅键盘事件用于控制程序的退出或者同步操作。
#奖励计算：奖励通过调用 _eval_reward 方法动态计算，并根据不同的奖励项进行加权。
# * Base class for RL tasks
class BaseTask():
    def __init__(self, gym, sim, cfg, sim_params, sim_device, headless):
        # 初始化任务时保存 gym 环境、模拟器、配置、设备信息等
        self.gym = gym
        self.sim = sim
        self.sim_params = sim_params
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = \
            gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # * 判断模拟器是否运行在 GPU 上，并设置设备为 GPU 或 CPU
        # * 如果使用 GPU 管道且设备是 CUDA，则设备设置为 GPU，否则为 CPU
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # * 设置图形设备 ID，用于渲染，-1 表示不渲染
        self.graphics_device_id = self.sim_device_id

        # 从配置中读取环境的数量和执行器的数量
        self.num_envs = cfg.env.num_envs
        self.num_actuators = cfg.env.num_actuators

        # * 优化 PyTorch JIT 运行时的配置
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # 分配缓冲区，用于保存环境状态信息
        self.to_be_reset = torch.ones(self.num_envs,
                                      device=self.device,
                                      dtype=torch.bool)
        self.terminated = torch.ones(self.num_envs,
                                     device=self.device,
                                     dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs,
                                              device=self.device,
                                              dtype=torch.long)
        self.timed_out = torch.zeros(self.num_envs,
                                     device=self.device,
                                     dtype=torch.bool)

        self.extras = {}

        # todo: 读取配置中的同步设置，默认开启 viewer 同步
        self.enable_viewer_sync = True
        self.viewer = None

        # * 如果不是 headless 模式，设置 viewer 和键盘快捷键
        if self.headless is False:
            # 创建 viewer 来渲染模拟环境
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            # * 订阅键盘事件，ESC 退出，V 切换同步状态
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_states(self, obs_list):
        # 返回一个连接在一起的观测状态 tensor
        return torch.cat([self.get_state(obs) for obs in obs_list], dim=-1)

    def get_state(self, name):
        # 获取某个状态的值，如果该状态有缩放因子，则返回缩放后的值
        if name in self.scales.keys():
            return getattr(self, name)/self.scales[name]
        else:
            return getattr(self, name)

    def set_states(self, state_list, values):
        # 设置多个状态的值，并确保维度匹配
        idx = 0
        for state in state_list:
            state_dim = getattr(self, state).shape[1]
            self.set_state(state, values[:, idx:idx+state_dim])
            idx += state_dim
        assert (idx == values.shape[1]), "Actions don't equal tensor shapes"

    def set_state(self, name, value):
        # 设置某个状态的值，如果有缩放因子，则应用缩放
        try:
            if name in self.scales.keys():
                setattr(self, name, value*self.scales[name])
            else:
                setattr(self, name, value)
        except AttributeError:
            print("Value for " + name + " does not match tensor shape")

    def _reset_idx(self, env_ids):
        """重置选中的环境（机器人）"""
        raise NotImplementedError

    def reset(self):
        """重置所有环境（机器人）"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self.step()

    def _reset_buffers(self):
        # 重置标志缓冲区
        self.to_be_reset[:] = False
        self.terminated[:] = False
        self.timed_out[:] = False

    def compute_reward(self, reward_weights):
        '''计算并返回奖励 tensor
        reward_weights: 一个包含奖励名称和对应权重的字典
        '''
        reward = torch.zeros(self.num_envs,
                             device=self.device, dtype=torch.float)
        for name, weight in reward_weights.items():
            reward += weight*self._eval_reward(name)
        return reward

    def _eval_reward(self, name):
        # 计算指定的奖励值，通过调用相应的奖励函数
        return eval('self._reward_'+name+'()')

    def _check_terminations_and_timeouts(self):
        """检查环境是否需要重置"""
        contact_forces = \
            self.contact_forces[:, self.termination_contact_indices, :]
        # 如果接触力超过阈值，环境结束
        self.terminated = \
            torch.any(torch.norm(contact_forces, dim=-1) > 1., dim=1)
        # 如果达到最大步长，超时
        self.timed_out = self.episode_length_buf > self.max_episode_length
        # 标记需要重置的环境
        self.to_be_reset = self.timed_out | self.terminated

    def step(self, actions):
        # 执行一步，具体实现由子类定义
        raise NotImplementedError

    def _render(self, sync_frame_time=True):
        # 渲染环境
        if self.viewer:
            # * 如果 viewer 关闭，退出程序
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # * 处理键盘事件
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # * 获取仿真结果
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # * 渲染图形
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
                # * 渲染图像（未启用的部分）
                # self.gym.draw_viewer(self.viewer, self.sim, True)
