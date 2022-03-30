# 导入需要库
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


# 定义LeNet网络结构
class LeNet:
    # 初始化函数
    def __init__(self, width, height, depth, optimizer, loss_function, metrics, classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.classes = classes

    # 定义实际的LeNet网络架构
    def create_model(self):
        # 初始化模型
        model = Sequential()
        inputShape = (self.height, self.width, self.depth)

        # 如果使用的顺序是通道有限，那么depth在第一个位置上
        if K.image_data_format() == "channel_first":
            inputShape = (self.depth, self.height, self.width)

        # 第一层的conv -> relu -> pool定义如下
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 继续定义新的一层conv->relu->pool层如下
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 然后输入量被压平，应用到一个包含500个节点的FC层
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # 接下来是最终的分类器
        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        # 在控制台打印模型结构
        model.summary()

        # 最后返回该网络架构
        return model
