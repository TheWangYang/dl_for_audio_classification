# 导入需要的库
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K


# 创建ResNet类
class ResNet:

    # 初始化函数
    def __init__(self, width, height, depth, optimizer, loss_function, metrics, classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.classes = classes

    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # 模型的快捷方式应该被初始化为输入（身份）数据
        shortcut = data
        # 第一个模块是：1x1 CONVs
        # 定义一个批量正则化层
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        # 定义激活层
        act1 = Activation("relu")(bn1)
        # 定义卷积层
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # ResNet模块中的第二个卷积层为：3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

        # 第三个卷积层为一组1x1的卷积
        # CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    def create_model(self, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (self.height, self.width, self.depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (self.depth, self.height, self.width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        # if dataset == "cifar":
        #     # 应用一个单一的卷积层
        #     x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        # elif dataset == "tiny_imagenet":
        #     # 应用CONV => BN => ACT => POOL减小空间大小
        #     x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        #     x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        #     x = Activation("relu")(x)
        #     x = ZeroPadding2D((1, 1))(x)
        #     x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red = True, bnEps = bnEps, bnMom = bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8), padding="same")(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(self.classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="resnet")
        # 编译模型，设置损失函数，优化方法以及评价标准
        model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)
        # 在控制台打印模型结构
        model.summary()
        # 返回模型
        return model


    


























