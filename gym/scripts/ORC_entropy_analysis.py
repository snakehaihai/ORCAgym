import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib  # 导入Matplotlib库，用于绘图
import matplotlib.pyplot as plt  # 导入Matplotlib的pyplot模块，用于绘图
import os  # 导入操作系统接口模块，用于路径操作
import glob  # 导入文件模式匹配模块，用于查找符合特定模式的文件
from entro import sliding_window_entropy_multi_trial  # 从entro模块导入滑动窗口熵计算函数
import osc_plotters as oscplt  # 导入自定义的振荡器绘图模块


def get_run_names(experiment_name, base_path, data_folder):
    """
    获取指定实验名称下所有运行的名称。

    参数:
        experiment_name (str): 实验名称。
        base_path (str): 基础路径，包含数据文件夹。
        data_folder (str): 数据文件夹名称。

    返回:
        list: 所有运行名称的列表。
    """
    experiment_path = os.path.join(base_path, data_folder, experiment_name)  # 构建实验路径
    # 使用glob查找所有以'_data.npz'结尾的文件，并提取运行名称
    return [os.path.basename(f).replace('_data.npz', '') for f in glob.glob(os.path.join(experiment_path, '*_data.npz'))]


# 初始化变量以存储每个变量的全局最小值和最大值
global_min = None
global_max = None

# 所有的切换配置
all_toggles = ['111', '110', '100', '000', '001', '010', '011', '101']

# 定义不同切换配置对应的绘图选项
plotting_options = {
    '000': {'color': 'cornflowerblue',
            'linestyle': (0, (3, 1, 1, 1, 1, 1)),
            'linewidth': 2},
    '100': {'color': 'yellowgreen',
            'linestyle': 'solid',
            'linewidth': 3},
    '110': {'color': 'goldenrod',
            'linestyle': (0, (3, 1, 1, 1, 1, 1)),
            'linewidth': 2},
    '111': {'color': 'crimson',
            'linestyle': 'solid',
            'linewidth': 3},
}

base_path = '/home/heim/Repos/trigym/gym/scripts/'  # 基础路径
data_folder = 'FS_data'  # 数据文件夹名称
obs2use = ['base_lin_vel', 'base_ang_vel', 'base_height']  # 需要使用的观测变量

# 使用字典存储所有切换配置下的数据，初始化为空字典
data_dict = {toggle: {} for toggle in all_toggles}

# 加载所有数据并找到每个变量的全局最小值和最大值
for toggle in all_toggles:
    experiment_name = "ORC_" + toggle  # 构建实验名称
    run_names = get_run_names(experiment_name, base_path, data_folder)  # 获取所有运行名称

    for run_name in run_names:
        data_path = os.path.join(base_path, data_folder, 'ORC_' + toggle, f"{run_name}_data.npz")  # 构建数据文件路径
        with np.load(data_path, allow_pickle=True) as loaded_data:
            if len(obs2use) == 1:
                data = loaded_data[obs2use[0]]  # 如果只有一个观测变量，直接获取
            else:
                # 如果有多个观测变量，将它们沿第三维拼接
                data = np.concatenate([loaded_data[name] for name in obs2use], axis=2)

        # 将数据存储在嵌套字典中
        data_dict[toggle][run_name] = data

        # 更新全局最小值和最大值
        if global_min is None:
            global_min = np.min(data, axis=(0, 1))  # 初始时设置为当前数据的最小值
            global_max = np.max(data, axis=(0, 1))  # 初始时设置为当前数据的最大值
        else:
            global_min = np.minimum(global_min, np.min(data, axis=(0, 1)))  # 更新全局最小值
            global_max = np.maximum(global_max, np.max(data, axis=(0, 1)))  # 更新全局最大值

# 创建基于全局最小值和最大值的通用箱边缘
num_bins = [10] * data.shape[2]  # 为每个观测变量设置10个箱
custom_bin_edges = [np.linspace(global_min[i], global_max[i], num_bins[i] + 1)
                    for i in range(len(global_min))]  # 生成每个变量的箱边缘

fig, ax = plt.subplots(1, 1, figsize=(12, 8))  # 创建一个12x8英寸的图和一个轴
window_size = 10  # 滑动窗口大小
step_size = 4  # 滑动步长
sample_rate = 100 / step_size  # 采样率

# 初始化一个字典来存储所有切换配置下的熵值
all_entropy_values = {toggle: [] for toggle in all_toggles}

# 遍历所有切换配置进行绘图
idx = 0
for toggle in all_toggles:
    for run_name, data in data_dict[toggle].items():

        assert len(num_bins) == data.shape[2], "Check your bins"  # 确保箱的数量与数据的维度匹配

        # 计算多次试验的滑动窗口熵值
        entropy_values_multi_trial = sliding_window_entropy_multi_trial(
            data, window_size, step_size, 'digitize', custom_bin_edges)

        options = plotting_options.get(toggle, {})  # 获取当前切换配置的绘图选项
        oscplt.entropy(ax, entropy_values_multi_trial,
                       f'Entropy for pushball perturbation',
                       color=options.get('color', 'black'),
                       linestyle=options.get('linestyle', 'solid'),
                       linewidth=options.get('linewidth', 1),
                       sample_rate=sample_rate)

        all_entropy_values[toggle].append(entropy_values_multi_trial)  # 存储熵值
        idx += 1
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # 调整布局，使右侧留出空间放置文本

# 将所有熵值保存到磁盘
save_path = os.path.join(base_path, "all_entropy_values_ORC.npz")
np.savez(save_path, **all_entropy_values)

# 手动设置轴的位置，为图例留出空间
ax.set_position([0.1, 0.1, 0.6, 0.8])  # [左, 底, 宽, 高]

# 创建图例的句柄
legend_handles = [matplotlib.lines.Line2D([0], [0],
                                          color=opts['color'],
                                          linestyle=opts['linestyle'],
                                          linewidth=opts['linewidth'])
                  for opts in plotting_options.values()]
# 创建图例，并将其放置在图外
legend = ax.legend(handles=legend_handles,
                   labels=['ORC_' + toggle for toggle in plotting_options.keys()],
                   loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

# 在图中添加观测变量的文本
observations_text = "Observations used:\n" + "\n".join(obs2use)
fig.text(0.73, 0.6, observations_text, ha='left', va='top',
         bbox={'facecolor': 'white', 'edgecolor': 'grey',
               'boxstyle': 'round, pad=0.5'})

# 手动设置轴的位置，再次调整以确保图例不重叠
ax.set_position([0.1, 0.1, 0.6, 0.8])  # [左, 底, 宽, 高]

# 调整布局以避免与图例重叠
plt.subplots_adjust(right=0.6, left=0.1, top=0.9, bottom=0.1)

# 保存并显示图像
save_path = os.path.join(base_path, "entropy_plot_all_ORC.png")
plt.savefig(save_path, bbox_extra_artists=(legend,), bbox_inches='tight')  # 保存图像，包含图例
print(f"Saved plot to {save_path}")  # 打印保存路径
plt.show()  # 显示图像
