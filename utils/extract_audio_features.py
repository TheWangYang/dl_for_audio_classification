from tqdm import tqdm
import librosa
import numpy as np
import glob
import os


class ExtractAudioFeatures:
    def __init__(self, label_dict, root_dir, max_file):
        self.label_dict = label_dict
        self.root_dir = root_dir
        self.max_file = max_file

    # 设置的特征提取方法
    def extract_audio_features(self, file_ext="*.wav"):
        label, feature = [], []
        # 由于不再需要子目录，所以注视掉第一个for循环
        # for sub_dir in self.sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(self.root_dir, file_ext))[:self.max_file]):  # 遍历数据集的所有文件
            # 首先得到文件名，包含.wav后缀名
            file_name = fn.split('\\')[-1]
            # 得到当前文件的类别
            file_name_list = file_name.split('_')
            label_name = ""
            for i in file_name_list[:-2]:
                label_name += i + "_"
            label_name += file_name_list[-2]
            label.extend([self.label_dict[label_name]])
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)   # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])

        return [feature, label]

    def extract_audio_test_features(self, test_dir, file_text="*.wav"):
        feature = []
        for fn in tqdm(glob.glob(os.path.join(test_dir, file_text))[:]):  # 遍历数据集的所有文件
            X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
            mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)  # 计算梅尔频谱(mel spectrogram),并把它作为特征
            feature.extend([mels])
        return feature




