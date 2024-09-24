""" 
存储不同循环的数据
设计考虑：将原始的“拥有”关系（LT->rollout->transition）改为并行关系。Transition
存储每一步所需的所有数据，并在每一步后被清除/更新，然后传递给rollout和longterm。
Rollout存储每次迭代的数据，生成用于学习的批次。
Longterm存储来自Transition的必要数据，并可能也生成批次。
"""

class BaseStorage:
    """ 
    这是一个任意存储类型的骨架类。
    """

    class Transition:
        """ 
        Transition存储类。
        即存储所有代理每一步的数据。
        """
        def __init__(self):
            """ 
            在初始化方法中定义所有需要存储的数据。
            """
            raise NotImplementedError

        def clear(self):
            """ 
            清除存储的数据，通过重新调用初始化方法。
            """
            self.__init__()

    def __init__(self, max_storage, device='cpu'):
        """
        初始化BaseStorage类。

        参数:
            max_storage (int): 存储的最大容量。
            device (str): 计算设备，默认为'cpu'，可选'cuda'。
        """
        self.device = device  # 设置计算设备
        self.max_storage = max_storage  # 设置存储的最大容量
        # fill_count用于跟踪已填充的存储量
        # fill_count之后的任何数据都是陈旧的，应被忽略
        self.fill_count = 0

    def add_transitions(self, transition: Transition):
        """ 
        将当前的Transition添加到长期存储中。
        根据初始化方法存储变量。

        参数:
            transition (Transition): 当前的Transition实例。
        """
        self.fill_count += 1  # 增加填充计数
        raise NotImplementedError

    def clear(self):
        """ 
        清除存储的数据，将填充计数重置为0。
        """
        self.fill_count = 0

    def mini_batch_generator(self):
        """ 
        生成用于学习的小批量数据。
        """
        raise NotImplementedError
