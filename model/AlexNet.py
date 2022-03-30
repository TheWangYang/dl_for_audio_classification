# 导入需要库
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.regularizers import l2
from keras import backend as K


# 定义AlexNet网络结构
class AlexNet:
    def __init__(self, width, height, depth, optimizer, loss_function, metrics, classes, reg=0.0002):
        self.width = width
        self.height = height
        self.depth = depth
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.reg = reg
        self.classes = classes

    def create_model(self):
        # 初始化模型
        model = Sequential()

        inputShape = (self.height, self.width, self.depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (self.depth, self.height, self.width)
            chanDim = -1

        # 定义网络中的第一个层
        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape,
                         padding="same", kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.25))

        # Block #2: second CONV => RELU => POOL layer set
        model.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        # Block #3: CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        # Block #5: second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(self.reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        # softmax classifier，设置最后的分类结果
        model.add(Dense(self.classes, kernel_regularizer=l2(self.reg)))
        model.add(Activation("softmax"))

        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        model.summary()

        # 返回模型
        return model



