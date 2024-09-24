from gym.envs import __init__  # 从gym.envs导入初始化模块
from gym.utils import get_args, task_registry  # 从gym.utils导入get_args函数和task_registry模块
from gym.utils.logging_and_saving import local_code_save_helper, wandb_singleton  # 从gym.utils.logging_and_saving导入日志和保存辅助函数

def setup():
    """
    设置环境配置、训练配置，并初始化Weights & Biases（wandb）日志记录。
    
    返回:
        tuple: 包含训练配置（train_cfg）和策略运行器（policy_runner）的元组。
    """
    args = get_args()  # 获取命令行参数
    wandb_helper = wandb_singleton.WandbSingleton()  # 初始化wandb单例帮助器

    # * 准备环境配置和训练配置
    env_cfg, train_cfg = task_registry.create_cfgs(args)  # 创建环境配置和训练配置
    task_registry.make_gym_and_sim()  # 创建Gym环境和模拟器实例
    wandb_helper.setup_wandb(env_cfg=env_cfg, train_cfg=train_cfg, args=args)  # 设置wandb日志记录
    env = task_registry.make_env(name=args.task, env_cfg=env_cfg)  # 创建环境实例
    # * 创建策略运行器
    policy_runner = task_registry.make_alg_runner(env, train_cfg)  # 创建算法运行器实例

    # * 保存本地代码和配置
    local_code_save_helper.log_and_save(env, env_cfg, train_cfg, policy_runner)  # 记录和保存本地代码及配置
    wandb_helper.attach_runner(policy_runner=policy_runner)  # 将策略运行器附加到wandb

    return train_cfg, policy_runner  # 返回训练配置和策略运行器

def train(train_cfg, policy_runner):
    """
    进行训练过程，执行策略学习，并关闭wandb日志记录。
    
    参数:
        train_cfg (dict): 训练配置字典。
        policy_runner: 策略运行器实例，负责执行学习算法。
    """
    wandb_helper = wandb_singleton.WandbSingleton()  # 获取wandb单例帮助器

    # 执行学习过程
    policy_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,  # 设置学习的迭代次数
        init_at_random_ep_len=True  # 初始化时是否以随机回合长度开始
    )

    wandb_helper.close_wandb()  # 关闭wandb日志记录

if __name__ == '__main__':
    train_cfg, policy_runner = setup()  # 调用setup函数，获取训练配置和策略运行器
    train(train_cfg=train_cfg, policy_runner=policy_runner)  # 开始训练过程
