# 导入需要库
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers.advanced_activations import ReLU
from keras.layers import AveragePooling2D
from keras.layers.core import Activation
from keras.layers import Input
from keras.layers import Dense
from keras import backend as K
from keras.models import Model
from keras.layers import Flatten


class MobileNetV3:
    # 设置的初始化函数
    def __init__(self, width, height, depth, optimizer, loss_function, metrics, classes, reg=0.0002):
        self.width = width
        self.height = height
        self.depth = depth
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.reg = reg
        self.classes = classes

    # 设置网络组件函数
    def depth_point_conv2d(x, s=[1, 1, 2, 1], channel=[64, 128]):
        """
        s:the strides of the conv
        channel: the depth of pointwiseconvolutions
        """
        dw1 = DepthwiseConv2D((3, 3), strides=s[0], padding='same')(x)
        bn1 = BatchNormalization()(dw1)
        relu1 = ReLU()(bn1)
        pw1 = Conv2D(channel[0], (1, 1), strides=s[1], padding='same')(relu1)
        bn2 = BatchNormalization()(pw1)
        relu2 = ReLU()(bn2)
        dw2 = DepthwiseConv2D((3, 3), strides=s[2], padding='same')(relu2)
        bn3 = BatchNormalization()(dw2)
        relu3 = ReLU()(bn3)
        pw2 = Conv2D(channel[1], (1, 1), strides=s[3], padding='same')(relu3)
        bn4 = BatchNormalization()(pw2)
        relu4 = ReLU()(bn4)

        return relu4

    def repeat_conv(x, s=[1, 1], channel=512):
        dw1 = DepthwiseConv2D((3, 3), strides=s[0], padding='same')(x)
        bn1 = BatchNormalization()(dw1)
        relu1 = ReLU()(bn1)
        pw1 = Conv2D(channel, (1, 1), strides=s[1], padding='same')(relu1)
        bn2 = BatchNormalization()(pw1)
        relu2 = ReLU()(bn2)
        return relu2

    def create_model(self):

        inputShape = (self.height, self.width, self.depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (self.depth, self.height, self.width)
            chanDim = -1

        h0 = Input(shape=inputShape)
        h1 = Conv2D(32, (3, 3), strides=2, padding="same")(h0)
        h2 = BatchNormalization()(h1)
        h3 = ReLU()(h2)
        h4 = MobileNetV3.depth_point_conv2d(h3, s=[1, 1, 2, 1], channel=[64, 128])
        h5 = MobileNetV3.depth_point_conv2d(h4, s=[1, 1, 2, 1], channel=[128, 256])
        h6 = MobileNetV3.depth_point_conv2d(h5, s=[1, 1, 2, 1], channel=[256, 512])
        h7 = MobileNetV3.repeat_conv(h6)
        h8 = MobileNetV3.repeat_conv(h7)
        h9 = MobileNetV3.repeat_conv(h8)
        h10 = MobileNetV3.repeat_conv(h9)
        h11 = MobileNetV3.depth_point_conv2d(h10, s=[1, 1, 2, 1], channel=[512, 1024])
        h12 = MobileNetV3.repeat_conv(h11, channel=1024)
        h13 = AveragePooling2D((7, 7), padding="same")(h12)
        # 维度归一化
        h14 = Flatten()(h13)
        h15 = Dense(self.classes)(h14)
        h16 = Activation("softmax")(h15)
        model = Model(inputs=h0, outputs=h16)
        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        model.summary()

        return model


