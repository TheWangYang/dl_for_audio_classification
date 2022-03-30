import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os
import librosa
import librosa.display
import glob

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras import Sequential
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


# 建立特征提取及数据集的建立
feature = []
label = []
# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2,'candied_fruits':3, 'carrots': 4, 'chips':5,
                  'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream':11,
                  'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon':17,
                  'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

from tqdm import tqdm


def extract_features(parent_dir, sub_dirs, max_file=10, file_ext="*.wav"):
    c = 0
    label, feature = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[:max_file]):  # 遍历数据集的所有文件

            # segment_log_specgrams, segment_labels = [], []
            # sound_clip,sr = librosa.load(fn)
            # print(fn)
            label_name = fn.split('/')[-2]
            label.extend([label_dict[label_name]])
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])

    return [feature, label]


parent_dir = './dataset/train_sample/'
save_dir = "./dataset/"
folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

# 获取特征feature以及类别的label
temp = extract_features(parent_dir, sub_dirs, max_file=100)

temp = np.array(temp)
data = temp.transpose()

# 获取特征
X = np.vstack(data[:, 0])

# 获取标签
Y = np.array(data[:, 1])
print('X的特征尺寸是：', X.shape)
print('Y的特征尺寸是：', Y.shape)

# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = to_categorical(Y)

'''最终数据'''
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, stratify=Y)
print('训练集的大小', len(X_train))
print('测试集的大小',len(X_test))

X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)

# 建立模型，搭建CNN网络
model = Sequential()

# 输入的大小
input_dim = (16, 8, 1)

model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))# 卷积层
model.add(MaxPool2D(pool_size=(2, 2)))# 最大池化
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh")) #卷积层
model.add(MaxPool2D(pool_size=(2, 2))) # 最大池化层
model.add(Dropout(0.1))
model.add(Flatten()) # 展开
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(20, activation = "softmax")) # 输出层：20个units输出20个类的概率

# 编译模型，设置损失函数，优化方法以及评价标准
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()


# 训练模型
model.fit(X_train, Y_train, epochs = 20, batch_size = 15, validation_data = (X_test, Y_test))


def extract_features(test_dir, file_ext="*.wav"):
    feature = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]): # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn,res_type='kaiser_fast')
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0) # 计算梅尔频谱(mel spectrogram),并把它作为特征
        feature.extend([mels])
    return feature


X_test = extract_features('./dataset/test_a/')
X_test = np.vstack(X_test)
predictions = model.predict(X_test.reshape(-1, 16, 8, 1))

preds = np.argmax(predictions, axis=1)
preds = [label_dict_inv[x] for x in preds]

path = glob.glob('./test_a/*.wav')
result = pd.DataFrame({'name': path, 'label': preds})

result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
result.to_csv('official_submit.csv', index=None)













