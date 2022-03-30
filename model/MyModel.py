from keras.layers.convolutional import Conv2D
from keras import Sequential
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten


class Model1:
    def __init__(self, input_dim, optimizer, loss_function, metrics):
        self.input_dim = input_dim
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def create_model(self):
        model = Sequential()
        # 设置卷积层
        model.add(Conv2D(64, (3, 3), padding="same", activation="tanh", input_shape=self.input_dim))  # 卷积层
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化
        model.add(Conv2D(128, (3, 3), padding="same", activation="tanh"))  # 卷积层
        model.add(MaxPooling2D(pool_size=(2, 2)))  # 最大池化层
        model.add(Dropout(0.1))
        model.add(Flatten())  # 展开
        model.add(Dense(1024, activation="tanh"))
        model.add(Dense(10, activation="softmax"))  # 输出层：20个units输出20个类的概率

        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=['accuracy'])

        model.summary()

        return model

