import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import os  # 导入操作系统接口模块，用于路径操作
from entro import sliding_window_entropy_multi_trial  # 从entro模块导入滑动窗口熵计算函数
import osc_plotters as oscplt  # 导入自定义的振荡器绘图模块
import gc  # 导入垃圾回收模块，用于内存管理
import glob  # 导入glob模块，用于文件模式匹配


def get_run_names(experiment_name, base_path):
    """
    获取指定实验名称下的所有运行名称。

    参数:
        experiment_name (str): 实验名称，例如"ORC_000"。
        base_path (str): 基础路径，指向数据所在的根目录。

    返回:
        list: 所有运行名称的列表，去除了'_data.npz'后缀。
    """
    experiment_path = os.path.join(base_path, data_folder, experiment_name)  # 构建实验路径
    # 使用glob查找所有符合模式的文件，并提取运行名称
    return [os.path.basename(f).replace('_data.npz', '') for f in glob.glob(os.path.join(experiment_path, '*_data.npz'))]


# 设置参数
all_toggles = ["000", "001", "100", "101", "010", "110", "011", "111"]  # 定义所有的切换配置
all_pushmags = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5']  # 定义所有的推力幅度
colors = ['lightsteelblue', 'mediumaquamarine', 'yellowgreen', 'forestgreen',
          'sandybrown', 'peru', 'indianred', 'crimson']  # 每个切换配置对应的颜色
toggle_settings = dict(zip(all_toggles, colors))  # 创建切换配置与颜色的映射字典
base_path = '/home/heim/Repos/trigym/gym/scripts/'  # 设置基础路径
data_folder = 'FS_data'  # 数据文件夹名称，根据实际情况调整
obs2use = ['base_lin_vel', 'base_ang_vel', 'commands']  # 定义需要使用的观测变量

# 初始化数据字典
data_dict = {}

# 创建绘图对象
fig, ax = plt.subplots()

# 初始化失败率字典，嵌套字典结构：fail_rate[toggle][pushmag]
fail_rate = {toggle: {pushmag: [] for pushmag in all_pushmags}
             for toggle in all_toggles}

# 遍历所有的切换配置
for toggle in all_toggles:
    for pushmag in all_pushmags:
        subpath_name = f"ORC_{toggle}"  # 构建子路径名称，例如"ORC_000"
        experiment_name = pushmag  # 实验名称为当前的推力幅度
        save_name = subpath_name + "_" + experiment_name  # 构建保存名称，例如"ORC_000_0.5"
        # 获取当前实验路径下的所有运行名称
        run_names = get_run_names(os.path.join(subpath_name, experiment_name),
                                  base_path)

        # 遍历所有的运行名称
        for run_name in run_names:
            # 构建数据文件的路径
            data_path = os.path.join(base_path, data_folder,
                                     subpath_name, experiment_name,
                                     run_name + "_data.npz")
            # 加载数据文件
            with np.load(data_path, allow_pickle=True) as loaded_data:
                # 提取需要的观测变量（已注释掉的部分为可选的其他观测变量）
                # data_dict = {key: loaded_data[key] for key in obs2use}
                # 计算失败次数，假设commands的倒数第二个时间步的第一个维度小于0.1为失败
                fails = np.sum(loaded_data['commands'][:, -2, 0] < 0.1)

                # * 绘图和分析
                # print(f"Toggle: {toggle}, Run: {run_name}, Failed: {fails}")

                # 绘制base_lin_vel的第一个维度随时间的变化曲线
                # ax.plot(loaded_data['base_lin_vel'][:, :, 0].T,
                #         linewidth=1,
                #         color=toggle_settings[toggle],
                #         alpha=0.5)

                # 创建保存绘图的文件夹
                # os.makedirs("FS_plots", exist_ok=True)
                # 保存当前运行的绘图
                # plt.savefig(os.path.join("FS_plots", f"{save_name}.png"))

                # 将当前运行的失败次数添加到对应的fail_rate字典中
                fail_rate[toggle][pushmag].append(fails)

            # 清除当前绘图，准备绘制下一个运行的图表（已注释掉的部分为可选的绘图清理步骤）
            ax.clear()
            # print(' ')

# 设置环境数量
num_envs = 1800

# 计算并打印每个切换配置和推力幅度的失败率均值和标准差
for dataset_name, fails_for_dataset in fail_rate.items():
    print(f"{dataset_name}")  # 打印当前切换配置名称
    for pushmag, fails in fails_for_dataset.items():
        a = np.array(fails)  # 将失败次数列表转换为NumPy数组
        mean_val = a.mean() / num_envs * 100  # 计算失败率的均值，并转换为百分比
        std_val = a.std() / num_envs * 100  # 计算失败率的标准差，并转换为百分比
        print(f"{pushmag}: {mean_val:.4f}% +/- {std_val:.4f}%")  # 打印结果
    print(' ')  # 打印空行，分隔不同切换配置的输出

# * 保存整个fail_rate字典，便于后续加载和分析
os.makedirs("FS_plots", exist_ok=True)  # 确保保存路径存在
np.savez_compressed(os.path.join("FS_plots", "fail_rate.npz"), **fail_rate)  # 压缩保存fail_rate字典
