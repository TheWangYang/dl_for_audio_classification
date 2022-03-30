import keras
from keras.models import load_model
from tqdm import tqdm
import os
import glob
import librosa
import numpy as np


model = load_model("./weights/20220328_myself_model.h5")

print(model.summary())


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

# 加载test数据集
test_dir = '.\\audio_dataset\\test\\'
file_text = "*.wav"
feature = []

for fn in tqdm(glob.glob(os.path.join(test_dir, file_text))[:]):  # 遍历数据集的所有文件
    X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
    predict_result = model.predict(mels.reshape(-1, 16, 8, 1))
    preds = np.argmax(predict_result)
    preds = label_dict_inv[preds]
    print(preds)








