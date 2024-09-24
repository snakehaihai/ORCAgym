import os  # 导入操作系统接口模块，用于路径操作

from gym.envs import __init__  # 从gym.envs导入初始化模块
from gym import LEGGED_GYM_ROOT_DIR  # 从gym导入LEGGED_GYM_ROOT_DIR常量，表示LeGGed Gym的根目录
from gym.utils import get_args, task_registry  # 从gym.utils导入get_args函数和task_registry模块
from gym.utils import KeyboardInterface, GamepadInterface  # 从gym.utils导入键盘和游戏手柄接口
from ORC import adjust_settings  # 从ORC模块导入adjust_settings函数
# torch需要在isaacgym导入后导入
import torch  # 导入PyTorch库，用于深度学习和张量操作


def setup(args):
    """
    设置环境配置和训练配置，初始化环境和策略运行器。

    参数:
        args: 从命令行获取的参数对象，包含实验配置。

    返回:
        tuple: 包含环境实例（env）、策略运行器实例（runner）和训练配置（train_cfg）的元组。
    """
    # 创建环境配置和训练配置
    env_cfg, train_cfg = task_registry.create_cfgs(args)
    env_cfg.env.num_envs = 50  # 设置并行环境的数量为50
    if hasattr(env_cfg, "push_robots"):
        env_cfg.push_robots.toggle = False  # 如果配置中包含push_robots，则禁用它
    env_cfg.commands.resampling_time = 9999  # 设置命令的重采样时间
    env_cfg.env.episode_length_s = 9999  # 设置每个回合的长度（秒）
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

    train_cfg.policy.noise.scale = 1.0  # 设置策略噪声的缩放因子为1.0

    # 根据切换标志调整配置
    env_cfg, train_cfg = adjust_settings(toggle=args.ORC_toggle,
                                         env_cfg=env_cfg,
                                         train_cfg=train_cfg)
    env_cfg.init_state.reset_mode = "reset_to_basic"  # 设置初始化状态的重置模式为"reset_to_basic"
    task_registry.set_log_dir_name(train_cfg)  # 设置日志目录名称

    task_registry.make_gym_and_sim()  # 创建Gym环境和模拟器实例
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)  # 创建环境实例
    train_cfg.runner.resume = True  # 设置运行器为继续模式
    runner = task_registry.make_alg_runner(env, train_cfg)  # 创建算法运行器实例

    # 切换到评估模式（例如禁用Dropout）
    runner.switch_to_eval()
    if EXPORT_POLICY:
        # 如果需要导出策略，将策略保存到指定路径
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs',
                            train_cfg.runner.experiment_name, 'exported')
        runner.export(path)
    return env, runner, train_cfg  # 返回环境、运行器和训练配置


def play(env, runner, train_cfg):
    """
    执行环境的交互步骤，记录状态并（可选）保存视频帧。

    参数:
        env: 环境实例。
        runner: 运行器实例，包含策略和算法信息。
        train_cfg: 训练配置字典。

    返回:
        无，函数执行过程中进行环境交互和数据记录。
    """
    RECORD_FRAMES = False  # 设置是否记录帧的标志
    # 设置接口：可以选择GamepadInterface(env)或KeyboardInterface(env)
    COMMANDS_INTERFACE = hasattr(env, "commands")  # 检查环境是否有commands属性
    if COMMANDS_INTERFACE:
        # 选择键盘接口进行控制
        interface = KeyboardInterface(env)
    img_idx = 0  # 图像计数器

    for i in range(10 * int(env.max_episode_length)):
        if RECORD_FRAMES:
            if i % 5:
                # 构建图像文件名路径
                filename = os.path.join(LEGGED_GYM_ROOT_DIR,
                                        'gym', 'scripts', 'gifs',
                                        train_cfg.runner.experiment_name,
                                        f"{img_idx}.png")
                # 将当前视图保存为PNG文件
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1  # 增加图像计数

        if COMMANDS_INTERFACE:
            interface.update(env)  # 更新接口，处理用户输入
        runner.set_actions(runner.get_inference_actions())  # 设置动作
        env.step()  # 执行一步模拟


if __name__ == '__main__':
    EXPORT_POLICY = True  # 设置是否导出策略的标志
    args = get_args()  # 获取命令行参数
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        env, runner, train_cfg = setup(args)  # 调用setup函数，获取环境、运行器和训练配置
        play(env, runner, train_cfg)  # 调用play函数，开始环境交互
