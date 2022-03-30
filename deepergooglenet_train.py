from model.DeeperGoogLeNet import DeeperGoogLeNet
from utils.extract_audio_features import ExtractAudioFeatures
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import pandas as pd
from keras.utils import np_utils
import argparse
import matplotlib.pyplot as plt


# 初始化模型
input_dim = (16, 8, 1)
optimizer = "adam"
loss_function = "categorical_crossentropy"
metrics = ['accuracy']
classes = 10

# 得到初始化模型
deepergooglenetModel = DeeperGoogLeNet(16, 8, 1, optimizer, loss_function, metrics, classes)

root_dir = '.\\audio_dataset\\train\\'

# 建立类别标签，不同类别对应不同的数字。
label_dict = {
    'loose_core_parts': 0,
    'internal_breakdown_short_circuit': 1,
    'severe_internal_short_circuit': 2,
    'heavy_load_startup_or_internal_short_circuit': 3,
    'single_phase_ground': 4,
    'silicon_steel_sheet_or_coil_loose': 5,
    'after_partial_discharge': 6,
    'high_voltage': 7,
    'secondary_open_circuit': 8,
    'bushing_partial_discharge': 9
}

label_dict_inv = {v: k for k, v in label_dict.items()}

# 获得数据集
ef = ExtractAudioFeatures(label_dict, root_dir, max_file=500)

# 获取特征feature以及类别的label
index = ef.extract_audio_features()
index = np.array(index)
data = index.transpose()

# 获取特征
X = np.vstack(data[:, 0])
# 获取标签
Y = np.array(data[:, 1])

print(Y)

# 在Keras库中：to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
Y = np_utils.to_categorical(Y)

'''最终数据'''
print(X.shape)
print(Y.shape)

# 划分数据集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=1, stratify=Y)
print('训练集的大小', len(X_train))
print('测试集的大小', len(X_val))

# reshape，重新设置形状
X_train = X_train.reshape(-1, 8, 16, 1)
X_val = X_val.reshape(-1, 8, 16, 1)

# 得到模型
train_model = deepergooglenetModel.create_model()

# 模型训练，得到训练过程中的精度和loss曲线
H = train_model.fit(X_train, Y_train, epochs=15, batch_size=8, validation_data=(X_val, Y_val))

X_test = ef.extract_audio_test_features('.\\audio_dataset\\test\\')
X_test = np.vstack(X_test)

# 获得预测结果
predictions = train_model.predict(X_test.reshape(-1, 8, 16, 1))

# 结果后处理
preds = np.argmax(predictions, axis=1)
preds = [label_dict_inv[x] for x in preds]

path = glob.glob('.\\audio_dataset\\test\\*.wav')
result = pd.DataFrame({'name': path, 'label': preds})

result['name'] = result['name'].apply(lambda x: x.split('/')[-1])
result.to_csv('deepergooglenet_submit.csv', index=None)

# 构建命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--savepath", required=True,  help="path to output model")
args = vars(ap.parse_args())


train_model.save(args["savepath"])

# 绘制loss和acc曲线变化
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy for DeeperGoogLeNet model.")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("DeeperGoogLeNet_train_monitor.png")
plt.close()



