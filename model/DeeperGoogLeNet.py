# 导入需要的库
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import concatenate
from keras.regularizers import l2
from keras import backend as K


# 创建更深层的googlenet网络类
class DeeperGoogLeNet:

    # 初始化函数
    def __init__(self, width, height, depth, optimizer, loss_function, metrics, classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.classes = classes

    def conv_module(x, K, kX, kY, stride, chanDim, padding="same", reg=0.0005, name=None):
        # 初始化卷积层和BN和RELU
        (convName, bnName, actName) = (None, None, None)

        if name is not None:
            convName = name + "_conv"
            bnName = name + "_bn"
            actName = name + "_act"

        # 定义一个conv -> BN->relu
        x = Conv2D(K, (kX, kY), strides=stride, padding=padding, kernel_regularizer=l2(reg), name=convName)(x)
        x = BatchNormalization(axis=chanDim, name=bnName)(x)
        x = Activation("relu", name=actName)(x)
        return x

    def inception_module(x, num1x1, num3x3Reduce, num3x3, num5x5Reduce, num5x5, num1x1Proj,
                         chanDim, stage, reg=0.0005):
        # 定义第一个分支
        first = DeeperGoogLeNet.conv_module(x, num1x1, 1, 1,
                                            (1, 1), chanDim, reg=reg, name=stage + "_first")

        # 定义第二个分支
        # define the second branch of the Inception module which
        # consists of 1x1 and 3x3 convolutions
        second = DeeperGoogLeNet.conv_module(x, num3x3Reduce, 1, 1,
                                             (1, 1), chanDim, reg=reg, name=stage + "_second1")
        second = DeeperGoogLeNet.conv_module(second, num3x3, 3, 3,
                                             (1, 1), chanDim, reg=reg, name=stage + "_second2")

        # define the third branch of the Inception module which
        # are our 1x1 and 5x5 convolutions
        third = DeeperGoogLeNet.conv_module(x, num5x5Reduce, 1, 1,
                                            (1, 1), chanDim, reg=reg, name=stage + "_third1")
        third = DeeperGoogLeNet.conv_module(third, num5x5, 5, 5,
                                            (1, 1), chanDim, reg=reg, name=stage + "_third2")

        # define the fourth branch of the Inception module which
        # is the POOL projection
        fourth = MaxPooling2D((3, 3), strides=(1, 1), padding="same", name=stage + "_pool")(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num1x1Proj,
        1, 1, (1, 1), chanDim, reg=reg, name=stage + "_fourth")

        # 沿着通道维度进行合并
        x = concatenate([first, second, third, fourth], axis=chanDim, name=stage+"_mixed")
        return x

    def create_model(self, reg=0.0005):
        # 初始化输入形式
        inputShape = (self.height, self.width, self.depth)
        chanDim = -1
        if K.image_data_format() == "channel_first":
            inputShape = (self.depth, self.height, self.width)
            chanDim = -1

        # define the model input, followed by a sequence of CONV =>
        # POOL => (CONV * 2) => POOL layers
        inputs = Input(shape=inputShape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1),
        chanDim, reg=reg, name="block1")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chanDim, reg=reg, name="block2")
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chanDim, reg=reg, name="block3")
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool2")(x)

        # apply two Inception modules followed by a POOL
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chanDim, "3a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chanDim, "3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        # apply five Inception modules followed by POOL
        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chanDim, "4a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chanDim, "4b", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chanDim, "4c", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chanDim, "4d", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chanDim, "4e", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool4")(x)

        # apply a POOL layer (average) followed by dropout
        x = AveragePooling2D((4, 4), name="pool5", padding="same")(x)
        x = Dropout(0.4, name="do")(x)
        # softmax classifier
        x = Flatten(name="flatten")(x)
        x = Dense(self.classes, kernel_regularizer=l2(reg), name="labels")(x)
        x = Activation("softmax", name="softmax")(x)
        # create the model
        model = Model(inputs, x, name="googlenet")
        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        # 在控制台打印模型结构
        model.summary()
        # 返回创建的模型结构
        return model






























