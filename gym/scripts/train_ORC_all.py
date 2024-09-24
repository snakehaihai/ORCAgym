from gym.envs import __init__  # 从gym.envs导入初始化模块
from gym.utils import get_args, task_registry  # 从gym.utils导入get_args函数和task_registry模块
from gym.utils.logging_and_saving import local_code_save_helper, wandb_singleton  # 从gym.utils.logging_and_saving导入日志和保存辅助函数
from ORC import adjust_settings  # 从ORC模块导入adjust_settings函数

from torch.multiprocessing import Process  # 从torch.multiprocessing导入Process类，用于创建新进程
from torch.multiprocessing import set_start_method  # 从torch.multiprocessing导入set_start_method函数，用于设置多进程启动方法

def setup(toggle):
    """
    根据切换标志设置环境配置和训练配置，初始化wandb日志记录，并创建环境和策略运行器实例。
    
    参数:
        toggle (str): 切换标志，用于控制不同配置的开关。
    
    返回:
        tuple: 包含训练配置（train_cfg）和策略运行器（policy_runner）的元组。
    """
    args = get_args()  # 获取命令行参数
    wandb_helper = wandb_singleton.WandbSingleton()  # 初始化wandb单例帮助器，用于日志记录
    
    # * 准备环境配置和训练配置
    env_cfg, train_cfg = task_registry.create_cfgs(args)  # 根据命令行参数创建环境配置和训练配置
    env_cfg, train_cfg = adjust_settings(toggle=toggle,  # 根据切换标志调整配置
                                         env_cfg=env_cfg,
                                         train_cfg=train_cfg)
    task_registry.set_log_dir_name(train_cfg)  # 设置日志目录名称，确保日志被保存到正确的位置
    
    task_registry.make_gym_and_sim()  # 创建Gym环境和模拟器实例
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)  # 设置wandb日志记录
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)  # 创建环境实例
    # * 创建策略运行器
    policy_runner = task_registry.make_alg_runner(env, train_cfg)  # 创建算法运行器实例，用于执行学习算法
    
    # * 保存本地代码和配置
    local_code_save_helper.log_and_save(env, env_cfg, train_cfg, policy_runner)  # 记录和保存本地代码及配置，确保实验可重现
    wandb_helper.attach_runner(policy_runner=policy_runner)  # 将策略运行器附加到wandb，确保训练过程中的数据被记录和可视化
    
    return train_cfg, policy_runner  # 返回训练配置和策略运行器

def train(train_cfg, policy_runner):
    """
    执行训练过程，执行策略学习，并关闭wandb日志记录。
    
    参数:
        train_cfg (dict): 训练配置字典，包含训练过程中的各种参数。
        policy_runner: 策略运行器实例，负责执行学习算法。
    """
    wandb_helper = wandb_singleton.WandbSingleton()  # 获取wandb单例帮助器
    
    # 执行学习过程
    policy_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,  # 设置学习的迭代次数
        init_at_random_ep_len=True  # 初始化时是否以随机回合长度开始，增加训练的多样性
    )
    
    wandb_helper.close_wandb()  # 关闭wandb日志记录，确保所有数据被正确保存

def worker(toggle):
    """
    工作进程函数，负责设置环境、执行训练并记录数据。
    
    参数:
        toggle (str): 切换标志，用于控制不同配置的开关。
    """
    train_cfg, policy_runner = setup(toggle)  # 调用setup函数，获取训练配置和策略运行器实例
    train(train_cfg=train_cfg, policy_runner=policy_runner)  # 调用train函数，开始训练过程

if __name__ == '__main__':
    all_toggles = ['000', '010', '011', '100', '101', '110', '111']  # 定义所有的切换配置
    set_start_method('spawn')  # 设置多进程启动方法为'spawn'，确保跨平台兼容性
    
    processes = []  # 初始化进程列表
    for toggle in all_toggles:
        p = Process(target=worker, args=(toggle,))  # 创建一个工作进程，目标函数为worker，传入切换标志作为参数
        p.start()  # 启动工作进程
        p.join()  # 等待工作进程完成
        
        # * 释放资源（通常在进程完成后不需要显式终止）
        p.terminate()  # 终止工作进程（通常在p.join()之后不需要）
        p.close()  # 关闭进程对象，释放资源
    
    print("All learning runs are completed.")  # 打印所有学习任务完成的消息
