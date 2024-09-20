import inspect

class BaseConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively.
        Ignores all names starting with '__' (built-in methods)."""
        # 调用下面定义的静态方法来初始化成员类
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # 遍历所有属性名
        for key in dir(obj):
            # 忽略内置的属性名称
            if key == "__class__":
                continue
            # 获取对象obj中属性名为key的属性值
            var = getattr(obj, key)
            # 检查这个属性是否为类
            if inspect.isclass(var):
                # 如果是类，则实例化这个类
                i_var = var()
                # 将实例化的对象设置为该属性的新值
                setattr(obj, key, i_var)
                # 递归地初始化这个新实例化的类的成员
                BaseConfig.init_member_classes(i_var)
