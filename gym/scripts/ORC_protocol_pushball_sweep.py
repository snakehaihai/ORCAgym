import os  # 导入操作系统接口模块，用于路径操作

from gym.envs import __init__  # 从gym.envs导入初始化模块
from gym import LEGGED_GYM_ROOT_DIR  # 从gym导入LEGGED_GYM_ROOT_DIR常量，表示LeGGed Gym的根目录
from gym.utils import get_args, task_registry  # 从gym.utils导入get_args函数和task_registry模块
from gym.utils import KeyboardInterface, GamepadInterface  # 从gym.utils导入键盘和游戏手柄接口
from ORC import *  # 从ORC模块导入所有内容
# torch需要在isaacgym导入后导入
import torch  # 导入PyTorch库，用于深度学习和张量操作
from torch.multiprocessing import Process  # 从torch.multiprocessing导入Process类，用于创建新进程
from torch.multiprocessing import set_start_method  # 从torch.multiprocessing导入set_start_method函数，用于设置多进程启动方法
import numpy as np  # 导入NumPy库，用于数值计算
import glob  # 导入glob模块，用于文件模式匹配
import imageio  # 导入imageio库，用于图像处理和GIF创建
from datetime import datetime  # 从datetime模块导入datetime类，用于时间操作

RECORD_VID = False  # 设置是否记录视频的标志

def get_run_names(experiment_name):
    """
    获取指定实验名称下的所有运行名称。

    参数:
        experiment_name (str): 实验名称。

    返回:
        list: 所有运行名称的列表。
    """
    experiment_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name)  # 构建实验路径
    # 遍历实验路径下的所有文件夹，返回文件夹名称列表
    return [folder_name for folder_name in os.listdir(experiment_path)
            if os.path.isdir(os.path.join(experiment_path, folder_name))]

def create_logging_dict(runner, test_total_timesteps):
    """
    创建用于日志记录的字典，用于存储不同观测状态的数据。

    参数:
        runner: 运行器实例，包含环境和策略信息。
        test_total_timesteps (int): 测试期间的总时间步数。

    返回:
        tuple: 包含要记录的状态列表和对应的字典。
    """
    # 定义要记录的状态变量
    states_to_log = [
        'commands',
        'base_lin_vel',
        'base_ang_vel',
        'oscillators',
        'base_height'
    ]
    states_to_log_dict = {}

    # 为每个状态变量初始化一个零张量，用于存储数据
    for state in states_to_log:
        array_dim = runner.get_obs_size([state, ])  # 获取状态变量的维度大小
        states_to_log_dict[state] = torch.zeros((runner.env.num_envs,
                                                 test_total_timesteps,
                                                 array_dim),
                                                device=runner.env.device)
    return states_to_log, states_to_log_dict

def setup(toggle, run_name):
    """
    根据切换标志和运行名称设置环境配置和训练配置，并创建运行器实例。

    参数:
        toggle (str): 切换标志，控制不同配置的开关。
        run_name (str): 运行名称，用于标识不同的运行。

    返回:
        tuple: 包含环境实例（env）、运行器实例（runner）和训练配置（train_cfg）的元组。
    """
    args = get_args()  # 获取命令行参数
    env_cfg, train_cfg = task_registry.create_cfgs(args)  # 创建环境配置和训练配置
    env_cfg.env.num_envs = 1800  # 设置并行环境的数量为1800
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False  # 如果配置中有push_robots，禁用它
    env_cfg.commands.resampling_time = 999999  # 设置命令的重采样时间
    env_cfg.env.episode_length_s = 100.  # 设置每个回合的长度（秒）
    env_cfg.init_state.timeout_reset_ratio = 1.  # 设置超时重置比例
    env_cfg.domain_rand.randomize_base_mass = False  # 禁用基础质量随机化
    env_cfg.domain_rand.randomize_friction = False  # 禁用摩擦力随机化
    env_cfg.terrain.mesh_type = "plane"  # 设置地形网格类型为平面
    env_cfg.osc.init_to = 'random'  # 设置振荡器初始化方式为随机
    env_cfg.osc.init_w_offset = False  # 禁用振荡器初始相位偏移
    env_cfg.osc.process_noise = 0.  # 设置振荡器过程噪声为0
    env_cfg.osc.omega_var = 0.  # 设置振荡器角频率方差为0
    env_cfg.osc.coupling_var = 0.  # 设置振荡器耦合方差为0
    env_cfg.commands.ranges.lin_vel_x = [0., 0.]  # 设置线速度x的范围为[0, 0]
    env_cfg.commands.ranges.lin_vel_y = 0.  # 设置线速度y的范围为0
    env_cfg.commands.ranges.yaw_vel = 0.  # 设置偏航速度的范围为0
    env_cfg.commands.var = 0.  # 设置命令的方差为0
    env_cfg.env.env_spacing = 3.0  # 设置环境间距为3.0
    train_cfg.policy.noise.scale = 0.0  # 设置策略噪声的缩放因子为0.0
    train_cfg.runner.run_name = run_name  # 设置运行器的运行名称
    train_cfg.runner.load_run = run_name  # 设置运行器加载的运行名称
    env_cfg, train_cfg = adjust_settings(toggle=toggle,  # 根据切换标志调整配置
                                         env_cfg=env_cfg,
                                         train_cfg=train_cfg)
    env_cfg.init_state.reset_mode = "reset_to_basic"  # 设置初始化状态的重置模式为"reset_to_basic"
    task_registry.set_log_dir_name(train_cfg)  # 设置日志目录名称

    task_registry.make_gym_and_sim()  # 创建Gym环境和模拟器实例
    env = task_registry.make_env(args.task, env_cfg)  # 创建环境实例
    train_cfg.runner.resume = True  # 设置运行器为继续模式
    runner = task_registry.make_alg_runner(env, train_cfg)  # 创建算法运行器实例

    # * 切换到评估模式（例如禁用Dropout）
    runner.switch_to_eval()
    # ... [其余的设置代码]
    return env, runner, train_cfg  # 返回环境、运行器和训练配置

def play(env, runner, train_cfg, log_file_path, run_name):
    """
    执行推送实验，对环境施加不同幅度的推力，并记录状态数据。

    参数:
        env: 环境实例。
        runner: 运行器实例，包含策略和算法信息。
        train_cfg: 训练配置字典。
        log_file_path (str): 数据保存的文件路径。
        run_name (str): 运行名称，用于标识不同的运行。

    返回:
        无，函数执行过程中进行数据记录和保存。
    """
    push_mags = torch.arange(0.5, 4., 0.5)  # 定义推力幅度范围，从0.5到3.5，每次增加0.5

    for push_mag in push_mags:
        push_interval = 0.5  # 推送间隔时间（秒）
        stagger_interval = 0.01  # 推送错位间隔时间（秒）
        num_staggers = int(push_interval / stagger_interval)  # 计算错位次数
        stagger_timesteps = int(stagger_interval / env.cfg.control.ctrl_dt)  # 计算错位的时间步数
        n_directions = 36  # 推送方向的数量
        n_trials = num_staggers * n_directions  # 计算试验次数，必须与环境数量匹配
        # print(f"number of trials: {n_trials}")
        # print(f"number of loaded environments: {env.num_envs}")
        assert n_trials == env.num_envs, f"number of trials: {n_trials}, number of loaded environments: {env.num_envs}"  # 确保试验次数与环境数量匹配
        env.commands[:, 0] = 3.0  # 设置命令的第一个维度为3.0，可能代表某个控制信号
        angles = torch.linspace(0, 2 * np.pi, n_directions, device=env.device)  # 生成推送方向的角度
        push_ball = torch.stack((torch.cos(angles),
                                 torch.sin(angles),
                                 torch.zeros_like(angles)), dim=1) * push_mag  # 计算推送向量

        # 运行推送协议，记录状态数据
        states_to_log_dict = run_protocol(env, runner, push_ball,
                                          num_staggers, stagger_timesteps,
                                          n_directions)

        # * 保存数据
        states_to_log_dict_cpu = {k: v.detach().cpu().numpy()
                                   for k, v in states_to_log_dict.items()}  # 将记录的数据从GPU转移到CPU，并转换为NumPy数组

        save_path = os.path.join(log_file_path, f"{push_mag:.1f}")  # 构建保存路径，按推力幅度分类
        os.makedirs(save_path, exist_ok=True)  # 创建保存路径文件夹
        save_name = os.path.join(save_path, f"{run_name}_data.npz")  # 构建保存文件名
        np.savez_compressed(save_name, **states_to_log_dict_cpu)  # 压缩保存记录的数据
        env.reset()  # 重置环境，为下一个推力幅度准备

def run_protocol(env, runner, push_ball, num_staggers, stagger_timesteps,
                 n_directions):
    """
    执行具体的推送协议，包括强迫同步阶段和测试阶段，记录状态数据。

    参数:
        env: 环境实例。
        runner: 运行器实例，包含策略和算法信息。
        push_ball (torch.Tensor): 推送向量。
        num_staggers (int): 错位次数。
        stagger_timesteps (int): 每次错位的时间步数。
        n_directions (int): 推送方向的数量。

    返回:
        dict: 记录的状态数据字典。
    """
    kdx = 0  # 试验计数器
    entrainment_time = 5.0  # 强迫同步时间（秒）

    test_start_time = 1.0  # 测试开始时间（秒）
    test_total_time = 9.0 + test_start_time  # 测试总时间（秒）
    test_total_timesteps = int(test_total_time / env.cfg.control.ctrl_dt)  # 测试总时间步数

    states_to_log, states_to_log_dict = create_logging_dict(runner,
                                                            test_total_timesteps)  # 创建日志记录字典

    # ** 强迫同步阶段 **
    for t in range(int(entrainment_time / env.cfg.control.ctrl_dt)):
        # * 模拟一步
        runner.set_actions(runner.get_inference_actions())  # 设置动作
        env.step()  # 执行一步模拟

    # ** 测试阶段 **
    for t in range(int(test_total_time / env.cfg.control.ctrl_dt)):

        # * 执行推送操作
        if t == int(test_start_time / env.cfg.control.ctrl_dt) + stagger_timesteps * kdx:
            env_ids = torch.arange(kdx * n_directions,
                                   (kdx + 1) * n_directions,
                                   dtype=torch.int64,
                                   device=env.device)  # 获取要推送的环境ID
            env.perturb_base_velocity(push_ball, env_ids)  # 执行推送操作
            kdx += 1  # 增加试验计数
        if kdx == num_staggers:  # 当达到错位次数时，重置计数器
            kdx = 0
        # * 模拟一步
        runner.set_actions(runner.get_inference_actions())  # 设置动作
        env.step()  # 执行一步模拟

        # * 记录状态
        for state in states_to_log:
            states_to_log_dict[state][:, t, :] = getattr(env, state)  # 获取并记录当前状态

    return states_to_log_dict  # 返回记录的状态数据字典

def worker(toggle, run_name):
    """
    工作进程函数，负责设置环境、运行测试并记录数据。

    参数:
        toggle (str): 切换标志，控制不同配置的开关。
        run_name (str): 运行名称，用于标识不同的运行。
    """
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        env, runner, train_cfg = setup(toggle, run_name)  # 设置环境和运行器实例
        # ... [其余的工作进程代码]

        # 调整log_file_path以包含run_name
        log_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'gym', 'scripts',
                                     "FS_data", train_cfg.runner.run_name)

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 创建日志文件夹

        png_folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'temp', run_name)  # 构建PNG图像文件夹路径
        os.makedirs(png_folder, exist_ok=True)  # 创建PNG图像文件夹

        play(env, runner=runner, train_cfg=train_cfg,
             log_file_path=log_file_path, run_name=run_name)  # 执行推送实验并记录数据

def save_gif(png_folder, gif_folder, frame_rate):
    """
    将PNG图像序列保存为GIF动画。

    参数:
        png_folder (str): 存储PNG图像的文件夹路径。
        gif_folder (str): 保存GIF动画的文件夹路径。
        frame_rate (int): GIF的帧率。
    """
    png_files = sorted(glob.glob(f"{png_folder}/*.png"),
                       key=os.path.getmtime)  # 获取所有PNG文件并按修改时间排序
    images = [imageio.imread(f) for f in png_files]  # 读取所有PNG图像
    gif_path = os.path.join(gif_folder, 'output.gif')  # 构建GIF文件路径
    imageio.mimsave(gif_path, images, fps=frame_rate)  # 保存为GIF动画

if __name__ == '__main__':
    set_start_method('spawn')  # 设置多进程启动方法为'spawn'
    # all_toggles = ['111']
    all_toggles = ['000', '111', '110', '100', '101', '001', '010', '011']  # 定义所有的切换配置
    experiment_name = "ORC_" + all_toggles[0] + "_FullSend"  # 构建第一个实验名称
    run_names = get_run_names(experiment_name)  # 获取第一个实验的所有运行名称
    worker(all_toggles[0], run_names[0])  # 运行第一个工作进程

    # 遍历所有切换配置和对应的运行名称，启动工作进程
    for toggle in all_toggles:
        experiment_name = "ORC_" + toggle + "_FullSend"  # 构建当前实验名称
        run_names = get_run_names(experiment_name)  # 获取当前实验的所有运行名称

        for run_name in run_names:
            p = Process(target=worker, args=(toggle, run_name))  # 创建一个工作进程
            p.start()  # 启动工作进程
            p.join()  # 等待工作进程完成

            # * 如果需要，释放任何资源
            p.terminate()  # 终止工作进程（通常在p.join()后不需要）
            p.close()  # 关闭进程对象，释放资源

    print("All learning runs are completed.")  # 打印所有学习任务完成的消息
