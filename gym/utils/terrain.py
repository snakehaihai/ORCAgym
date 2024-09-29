import numpy as np  # 导入 NumPy 库，用于数值计算和数组操作

from isaacgym import terrain_utils  # 从 isaacgym 库中导入 terrain_utils 模块，用于地形生成和处理
from gym.envs.base.legged_robot_config import LeggedRobotCfg  # 从 gym.envs.base.legged_robot_config 模块中导入 LeggedRobotCfg 配置类


class Terrain:
    """Terrain 类用于生成和管理仿真环境中的地形"""

    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        """
        Terrain 类的初始化方法

        Args:
            cfg (LeggedRobotCfg.terrain): 地形配置对象，包含地形的各种参数
            num_robots (int): 机器人数量，用于调整地形生成参数
        """
        self.cfg = cfg  # 保存地形配置
        self.num_robots = num_robots  # 保存机器人数量
        self.type = cfg.mesh_type  # 获取地形类型（例如 'none', 'plane', 'trimesh' 等）

        if self.type in ["none", "plane"]:  # 如果地形类型为 'none' 或 'plane'
            return  # 不需要生成复杂地形，直接返回

        self.env_length = cfg.terrain_length  # 获取地形长度
        self.env_width = cfg.terrain_width  # 获取地形宽度

        # 检查地形比例是否总和为 1.0，用于后续地形类型的选择
        if np.sum(cfg.terrain_proportions) != 1.0:
            raise ValueError("Terrain proportions provided do not sum to 1.0")  # 如果比例不正确，抛出错误

        # 计算每种地形类型的累积比例，用于随机选择地形类型
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1])
                            for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols  # 计算子地形的总数（行数 * 列数）
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))  # 初始化每个子地形的原点坐标

        # 计算每个子地形在像素单位下的宽度和长度
        self.width_per_env_pixels = int(
            self.env_width / cfg.horizontal_scale)  # 每个子地形的宽度（像素）
        self.length_per_env_pixels = int(
            self.env_length / cfg.horizontal_scale)  # 每个子地形的长度（像素）

        # 计算边界大小（像素）
        self.border = int(
            cfg.border_size / self.cfg.horizontal_scale)
        # 计算整个地形图的总列数，包括边界
        self.tot_cols = int(
            cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # 计算整个地形图的总行数，包括边界
        self.tot_rows = int(
            cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 初始化高度场原始数据为全零矩阵
        self.height_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16)

        # 根据配置决定使用哪种地形生成方法
        if cfg.curriculum:
            self.curiculum()  # 使用课程式地形生成
        elif cfg.selected:
            self.selected_terrain()  # 使用预选地形生成
        else:
            self.randomized_terrain()  # 使用随机地形生成

        self.heightsamples = self.height_field_raw  # 保存高度样本数据

        # 如果地形类型为 'trimesh'，将高度场转换为三角网格
        if self.type == "trimesh":
            self.vertices, self.triangles = \
                terrain_utils.convert_heightfield_to_trimesh(
                    self.height_field_raw,  # 高度场数据
                    self.cfg.horizontal_scale,  # 水平缩放因子
                    self.cfg.vertical_scale,  # 垂直缩放因子
                    self.cfg.slope_treshold)  # 坡度阈值


    def randomized_terrain(self):
        """生成随机地形，遍历每个子地形区域并随机生成地形类型"""
        for k in range(self.cfg.num_sub_terrains):  # 遍历所有子地形
            # 将线性索引 k 转换为二维坐标 (i, j)
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)  # 随机选择一个浮点数，用于决定地形类型
            difficulty = np.random.choice([0.5, 0.75, 0.9])  # 随机选择一个难度等级
            terrain = self.make_terrain(choice, difficulty)  # 生成地形
            self.add_terrain_to_map(terrain, i, j)  # 将生成的地形添加到地形图中


    def curiculum(self):
        """生成课程式地形，根据行列位置逐步增加难度"""
        for j in range(self.cfg.num_cols):  # 遍历所有列
            for i in range(self.cfg.num_rows):  # 遍历所有行
                difficulty = i / self.cfg.num_rows  # 难度随行数增加而增加
                choice = j / self.cfg.num_cols + 0.001  # 难度随列数变化，增加小偏移量避免边界问题

                terrain = self.make_terrain(choice, difficulty)  # 生成地形
                self.add_terrain_to_map(terrain, i, j)  # 将生成的地形添加到地形图中


    def selected_terrain(self):
        """生成预选地形，根据配置中指定的地形类型和参数生成地形"""
        terrain_type = self.cfg.terrain_kwargs.pop('type')  # 获取并移除地形类型参数
        for k in range(self.cfg.num_sub_terrains):  # 遍历所有子地形
            # 将线性索引 k 转换为二维坐标 (i, j)
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            # 创建一个 SubTerrain 对象，用于生成具体地形
            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,  # 子地形宽度
                length=self.length_per_env_pixels,  # 子地形长度
                vertical_scale=self.vertical_scale,  # 垂直缩放因子
                horizontal_scale=self.horizontal_scale)  # 水平缩放因子

            # 根据地形类型动态调用相应的地形生成函数，并传入参数
            eval(terrain_type)(
                terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)  # 将生成的地形添加到地形图中


    def make_terrain(self, choice, difficulty):
        """
        根据选择和难度生成地形

        Args:
            choice (float): 用于决定地形类型的随机值
            difficulty (float): 地形难度等级

        Returns:
            SubTerrain: 生成的地形对象
        """
        # 创建一个 SubTerrain 对象，用于生成具体地形
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,  # 子地形宽度
            length=self.length_per_env_pixels,  # 子地形长度
            vertical_scale=self.cfg.vertical_scale,  # 垂直缩放因子
            horizontal_scale=self.cfg.horizontal_scale)  # 水平缩放因子

        # 根据难度调整坡度、台阶高度等参数
        slope = difficulty * 0.2  # 坡度
        step_height = 0.05 + 0.05 * difficulty  # 台阶高度
        discrete_obstacles_height = 0.05 + difficulty * 0.1  # 离散障碍物高度
        stepping_stones_size = 1.5 * (1.05 - difficulty)  # 踏步石大小
        stone_distance = 0.05 if difficulty == 0 else 0.1  # 踏步石间距
        gap_size = .3 * difficulty  # 缺口大小
        pit_depth = .3 * difficulty  # 坑深度

        # 根据选择值决定生成哪种地形类型
        if choice < self.proportions[0]:
            # 如果选择值小于第一个比例的一半，生成反坡地形
            if choice < self.proportions[0] / 2:
                slope *= -1  # 坡度取反
            # 生成金字塔坡度地形
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            # 如果选择值在第一个比例和第二个比例之间，生成随机均匀地形
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-0.05, max_height=0.05,
                step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            # 如果选择值在第二个比例和第四个比例之间，生成金字塔台阶地形
            if choice < self.proportions[2]:
                step_height *= -1  # 台阶高度取反
            terrain_utils.pyramid_stairs_terrain(
                terrain, step_width=0.31,
                step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            # 如果选择值在第四个比例和第五个比例之间，生成离散障碍物地形
            num_rectangles = 20  # 矩形数量
            rectangle_min_size = 1.  # 矩形最小尺寸
            rectangle_max_size = 2.  # 矩形最大尺寸
            terrain_utils.discrete_obstacles_terrain(
                terrain, discrete_obstacles_height, rectangle_min_size,
                rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            # 如果选择值在第五个比例和第六个比例之间，生成踏步石地形
            terrain_utils.stepping_stones_terrain(
                terrain, stone_size=stepping_stones_size,
                stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            # 如果选择值在第六个比例和第七个比例之间，生成缺口地形
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            # 如果选择值大于第七个比例，生成坑地形
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)

        return terrain  # 返回生成的地形对象


    def add_terrain_to_map(self, terrain, row, col):
        """
        将生成的地形添加到地形图中指定的位置

        Args:
            terrain (SubTerrain): 生成的地形对象
            row (int): 地形所在的行号
            col (int): 地形所在的列号
        """
        i = row  # 行索引
        j = col  # 列索引

        # 计算地形在全局地形图中的起始和结束坐标
        start_x = self.border + i * self.length_per_env_pixels  # 起始 x 坐标
        end_x = self.border + (i + 1) * self.length_per_env_pixels  # 结束 x 坐标
        start_y = self.border + j * self.width_per_env_pixels  # 起始 y 坐标
        end_y = self.border + (j + 1) * self.width_per_env_pixels  # 结束 y 坐标

        # 将生成的地形高度场数据复制到全局高度场中
        self.height_field_raw[start_x: end_x, start_y:end_y] = \
            terrain.height_field_raw

        # 计算环境的原点坐标
        env_origin_x = (i + 0.5) * self.env_length  # 环境中心的 x 坐标
        env_origin_y = (j + 0.5) * self.env_width  # 环境中心的 y 坐标

        # 计算环境中心区域的 z 坐标（高度）
        x1 = int((self.env_length / 2. - 1) / terrain.horizontal_scale)  # 中心区域起始 x
        x2 = int((self.env_length / 2. + 1) / terrain.horizontal_scale)  # 中心区域结束 x
        y1 = int((self.env_width / 2. - 1) / terrain.horizontal_scale)  # 中心区域起始 y
        y2 = int((self.env_width / 2. + 1) / terrain.horizontal_scale)  # 中心区域结束 y
        env_origin_z = np.max(
            terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale  # 计算环境中心的 z 坐标

        # 将环境原点坐标保存到 env_origins 数组中
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    """
    在地形中生成一个缺口

    Args:
        terrain (SubTerrain): 目标地形对象
        gap_size (float): 缺口的大小
        platform_size (float, optional): 平台的大小。默认为 1.0
    """
    gap_size = int(gap_size / terrain.horizontal_scale)  # 缩放缺口大小
    platform_size = int(platform_size / terrain.horizontal_scale)  # 缩放平台大小

    center_x = terrain.length // 2  # 地形中心的 x 坐标
    center_y = terrain.width // 2  # 地形中心的 y 坐标
    x1 = (terrain.length - platform_size) // 2  # 缺口起始 x 坐标
    x2 = x1 + gap_size  # 缺口结束 x 坐标
    y1 = (terrain.width - platform_size) // 2  # 缺口起始 y 坐标
    y2 = y1 + gap_size  # 缺口结束 y 坐标

    # 在缺口区域设置高度为 -1000，表示深坑
    terrain.height_field_raw[
        center_x - x2: center_x + x2, center_y - y2: center_y + y2] = -1000
    # 在平台区域设置高度为 0，表示平坦地面
    terrain.height_field_raw[
        center_x - x1: center_x + x1, center_y - y1: center_y + y1] = 0


def pit_terrain(terrain, depth, platform_size=1.):
    """
    在地形中生成一个坑

    Args:
        terrain (SubTerrain): 目标地形对象
        depth (float): 坑的深度
        platform_size (float, optional): 平台的大小。默认为 1.0
    """
    depth = int(depth / terrain.vertical_scale)  # 缩放坑深度
    platform_size = int(platform_size / terrain.horizontal_scale / 2)  # 缩放平台大小

    x1 = terrain.length // 2 - platform_size  # 坑区域起始 x 坐标
    x2 = terrain.length // 2 + platform_size  # 坑区域结束 x 坐标
    y1 = terrain.width // 2 - platform_size  # 坑区域起始 y 坐标
    y2 = terrain.width // 2 + platform_size  # 坑区域结束 y 坐标

    # 在坑区域设置高度为负深度值，表示深坑
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
