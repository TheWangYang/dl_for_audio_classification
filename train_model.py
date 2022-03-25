from model.model_1 import Model1
from network_files.extract_features import ExtractFeatures
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from keras.utils import np_utils


# 初始化模型
input_dim = (16, 8, 1)
optimizer = "adam"
loss_function = "categorical_crossentropy"
metrics = ['accuracy']
model_1 = Model1(input_dim, optimizer, loss_function, metrics)


parent_dir = '.\\dataset\\train_sample\\'
save_dir = ".\\dataset\\"

folds = sub_dirs = np.array(['aloe','burger','cabbage','candied_fruits',
                             'carrots','chips','chocolate','drinks','fries',
                            'grapes','gummies','ice-cream','jelly','noodles','pickles',
                            'pizza','ribs','salmon','soup','wings'])

# 建立类别标签，不同类别对应不同的数字。
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2,'candied_fruits':3, 'carrots': 4, 'chips':5,
                  'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream':11,
                  'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon':17,
                  'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}

# 获得数据集
ef = ExtractFeatures(label_dict, parent_dir, sub_dirs, max_file=100)
# 获取特征feature以及类别的label
index = ef.extract_features()
index = np.array(index)
data = index.transpose()

# 获取特征
X = np.vstack(data[:, 0])
# 获取标签
Y = np.array(data[:, 1])
print('X的特征尺寸是：', X.shape)
print('Y的特征尺寸是：', Y.shape)


# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = np_utils.to_categorical(Y)

'''最终数据'''
print(X.shape)
print(Y.shape)


# 划分数据集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1, stratify=Y)
print('训练集的大小', len(X_train))
print('测试集的大小',len(X_test))

# reshape
X_train = X_train.reshape(-1, 16, 8, 1)
X_test = X_test.reshape(-1, 16, 8, 1)

# 得到模型
train_model = model_1.create_model()

# 模型训练
train_model.fit(X_train, Y_train, epochs=20, batch_size=15, validation_data=(X_test, Y_test))


X_test = ef.extract_test_features('.\\dataset\\test_a\\')
X_test = np.vstack(X_test)

# 获得预测结果
predictions = train_model.predict(X_test.reshape(-1, 16, 8, 1))


# 结果后处理
preds = np.argmax(predictions, axis=1)
preds = [label_dict_inv[x] for x in preds]

path = glob.glob('.\\dataset\\test_a\\*.wav')
result = pd.DataFrame({'name': path, 'label': preds})

result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
result.to_csv('submit.csv', index=None)











