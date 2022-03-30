# 导入库
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


# 建立网络架构
class MiniVGGNet:

    # 初始化函数
    def __init__(self, width, height, depth, optimizer, loss_function, metrics, classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.classes = classes

    def create_model(self):
        # 初始化模型设置输入格式
        model = Sequential()
        inputShape = (self.height, self.width, self.depth)
        # 设置通道尺寸
        chanDim = -1

        # 对通道有限的像素输入实现泛化
        if K.image_data_format() == "channel_first":
            inputShape = (self.depth, self.height, self.width)
            chanDim = 1  # 当通道首先出现的时候，设置BN归一化的位置在index = 1处

        # 定义第一层
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 定义第二层
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 第一个全连接层
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 使用softmax分类器
        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        # 在控制台打印模型结构
        model.summary()

        return model

    





