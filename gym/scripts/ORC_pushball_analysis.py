import numpy as np
import matplotlib.pyplot as plt
import os
from entro import sliding_window_entropy_multi_trial
import osc_plotters as oscplt
import gc
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_run_names(experiment_name, base_path):
    experiment_path = os.path.join(base_path, data_folder, experiment_name)
    return [os.path.basename(f).replace('_data.npz', '') for f in glob.glob(os.path.join(experiment_path, '*_data.npz'))]

def create_logging_dict(runner, test_total_timesteps):
    states_to_log = [
        'commands',
        'base_lin_vel',
        'base_ang_vel',
        'oscillators',
        'base_height'
    ]
    states_to_log_dict = {}
    for state in states_to_log:
        array_dim = runner.get_obs_size([state, ])
        states_to_log_dict[state] = torch.zeros((runner.env.num_envs,
                                                 test_total_timesteps,
                                                 array_dim),
                                                device=runner.env.device)
    return states_to_log, states_to_log_dict

def process_run(toggle, run_name, base_path, data_folder, obs2use, toggle_settings):
    """
    处理单个运行的数据加载、失败次数计算和绘图保存。

    参数:
        toggle (str): 切换标志。
        run_name (str): 运行名称。
        base_path (str): 基础路径。
        data_folder (str): 数据文件夹名称。
        obs2use (list): 需要使用的观测变量。
        toggle_settings (dict): 切换配置与颜色的映射字典。

    返回:
        tuple: (toggle, fails, run_name, base_lin_vel_data)
    """
    data_path = os.path.join(base_path, data_folder, 'ORC_'+toggle, f"{run_name}_data.npz")
    try:
        with np.load(data_path, allow_pickle=True) as loaded_data:
            data_dict = {key: loaded_data[key] for key in obs2use}
    except Exception as e:
        print(f"Error loading {data_path}: {e}")
        return (toggle, None, run_name, None)

    fails = np.sum(data_dict['commands'][:, -2, 0] < 0.1)
    base_lin_vel_data = data_dict['base_lin_vel'][:, :, 0].T

    # 绘图
    fig, ax = plt.subplots()
    ax.plot(base_lin_vel_data,
            linewidth=1,
            color=toggle_settings[toggle],
            alpha=0.5)
    os.makedirs("FS_plots", exist_ok=True)
    plt.savefig(os.path.join("FS_plots", f"{run_name}.png"))
    plt.close(fig)  # 关闭绘图，释放内存

    return (toggle, fails, run_name, base_lin_vel_data)

def main():
    # 设置参数
    all_toggles = ['000', '001', '010', '011', '100', '101', '110', '111']
    colors = ['lightsteelblue', 'mediumaquamarine', 'yellowgreen', 'forestgreen', 'sandybrown', 'peru', 'indianred', 'crimson']
    toggle_settings = dict(zip(all_toggles, colors))
    base_path = '/home/heim/Repos/trigym/gym/scripts/'
    data_folder = 'FS_data'  # 根据实际情况调整
    obs2use = ['base_lin_vel', 'base_ang_vel', 'commands']

    # 初始化失败率字典
    fail_rate = {toggle: [] for toggle in all_toggles}

    # 使用多进程并行处理
    with ProcessPoolExecutor(max_workers=4) as executor:  # 根据CPU核心数调整max_workers
        futures = []
        for toggle in all_toggles:
            experiment_name = "ORC_" + toggle
            run_names = get_run_names(experiment_name, base_path)
            for run_name in run_names:
                futures.append(executor.submit(process_run, toggle, run_name, base_path, data_folder, obs2use, toggle_settings))
        
        for future in as_completed(futures):
            toggle, fails, run_name, base_lin_vel_data = future.result()
            if fails is not None:
                fail_rate[toggle].append(fails)

    # 统计并打印失败率
    for dataset_name, fails in fail_rate.items():
        if len(fails) == 0:
            print(f"{dataset_name}: No data available.")
            continue
        a = np.array(fails)
        mean_val = a.mean()
        std_val = a.std()
        print(f"{dataset_name}: {mean_val:.4f} +/- {std_val:.4f}")

if __name__ == '__main__':
    main()
