from tqdm import tqdm
import librosa
import numpy as np
import glob
import os


class ExtractFeatures:
    def __init__(self, label_dict, parent_dir, sub_dirs, max_file):
        self.label_dict = label_dict
        self.parent_dir = parent_dir
        self.sub_dirs = sub_dirs
        self.max_file = max_file

    # 设置的特征提取方法
    def extract_features(self, file_ext="*.wav"):
        c = 0
        label, feature = [], []
        for sub_dir in self.sub_dirs:
            for fn in tqdm(glob.glob(os.path.join(self.parent_dir, sub_dir, file_ext))[:self.max_file]):  # 遍历数据集的所有文件

                # segment_log_specgrams, segment_labels = [], []
                # sound_clip,sr = librosa.load(fn)
                # print(fn)
                print(fn)
                label_name = fn.split('\\')[-2]
                label.extend([self.label_dict[label_name]])
                X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
                mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)   # 计算梅尔频谱(mel spectrogram),并把它作为特征
                feature.extend([mels])

        return [feature, label]

    def extract_test_features(self, test_dir, file_ext="*.wav"):
        feature = []
        for fn in tqdm(glob.glob(os.path.join(test_dir, file_ext))[:]):  # 遍历数据集的所有文件
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])
        return feature




