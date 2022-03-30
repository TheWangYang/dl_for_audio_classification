import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import wave
from scipy.io.wavfile import write
import scipy

# 设置的数据集路径
data_path = "..\\audio_dataset\\train\\"


# 通过噪声因子来增加噪声
def noise_augment_by_noise_factor(data_path, output_path, w, number):
    '''
        噪声增强函数
        这里的output_path:..\\audio_dataset\\augument_train\\noise_augment_by_noise_factor\\
        :param data_path:数据存放根目录
        :param output_path:数据输出目录
        :param w:设置的噪声因子
        :param number:表示第几次使用数据增强，不同数据增强方法需要按照number值进行+1运行
    '''

    origin_output_path = output_path

    for fn in tqdm(glob.glob(os.path.join(data_path, "*.wav"))[:]):  # 遍历数据集的所有文件
        rate, X = scipy.io.wavfile.read(fn)

        # 数据增强后的噪音
        noise_augment_audio = X + w * np.random.normal(loc=0, scale=1, size=len(X))

        noise_augment_audio = np.asarray(noise_augment_audio, dtype=np.int16)

        # 得到音频文件的名字
        file_name = fn.split('\\')[-1]
        # 得到当前文件的类别
        label_name = file_name[:-6]
        # 生成新的文件名
        new_label_name = label_name + "_" + number
        # 组合成.wav文件名
        new_wav_name = new_label_name + ".wav"
        # 得到音频输出路径
        output_path = output_path + new_wav_name

        write(output_path, rate, noise_augment_audio)

        output_path = origin_output_path


# 通过控制信噪比来增加噪声
def noise_augment_by_signal_to_noise_ratio(data_path, output_path, snr, number):
    '''
        噪声增强函数
        这里的output_path:..\\audio_dataset\\augument_train\\noise_augment_by_noise_factor\\
        :param data_path:数据存放根目录
        :param output_path:数据输出目录
        :param snr:设置的信噪比
        :param number:表示第几次使用数据增强，不同数据增强方法需要按照number值进行+1运行
    '''
    origin_output_path = output_path

    for fn in tqdm(glob.glob(os.path.join(data_path, "*.wav"))[:]):  # 遍历数据集的所有文件
        rate, X = scipy.io.wavfile.read(fn)
        P_signal = np.mean(X ** 2)  # 信号功率
        k = np.sqrt(P_signal / 10 ** (snr / 10.0))  # 噪声系数 k
        noise_augment_audio = X + np.random.randn(len(X)) * k
        noise_augment_audio = np.asarray(noise_augment_audio, dtype=np.int16)
        # 得到音频文件的名字
        file_name = fn.split('\\')[-1]
        # 得到当前文件的类别
        label_name = file_name[:-6]
        # 生成新的文件名
        new_label_name = label_name + "_" + number
        # 组合成.wav文件名
        new_wav_name = new_label_name + ".wav"
        # 得到音频输出路径
        output_path = output_path + new_wav_name
        write(output_path, rate, noise_augment_audio)
        output_path = origin_output_path


# 设置的波形位移函数用来进行数据增强
def waveform_displacement(data_path, output_path, shift, number):
    '''
        波形位移增强函数
        这里的output_path:..\\audio_dataset\\augument_train\\waveform_displacement_augment\\
        :param data_path:数据存放根目录
        :param output_path:数据输出目录
        :param shift:设置的偏移量
        :param number:表示第几次使用数据增强，不同数据增强方法需要按照number值进行+1运行
    '''

    origin_output_path = output_path

    for fn in tqdm(glob.glob(os.path.join(data_path, "*.wav"))[:]):  # 遍历数据集的所有文件
        rate, X = scipy.io.wavfile.read(fn)
        noise_augment_audio = np.roll(X, int(shift))
        noise_augment_audio = np.asarray(noise_augment_audio, dtype=np.int16)
        # 得到音频文件的名字
        file_name = fn.split('\\')[-1]
        # 得到当前文件的类别
        label_name = file_name[:-6]
        # 生成新的文件名
        new_label_name = label_name + "_" + number
        # 组合成.wav文件名
        new_wav_name = new_label_name + ".wav"
        # 得到音频输出路径
        output_path = output_path + new_wav_name
        write(output_path, rate, noise_augment_audio)
        output_path = origin_output_path


# 设置的重采样数据增强
def resampling_data_augmentation(data_path, output_path, sr, number):
    '''
        重采样增强函数
        这里的output_path:..\\audio_dataset\\augument_train\\resampling_data_augmentation\\
        :param data_path:数据存放根目录
        :param output_path:数据输出目录
        :param sr:设置的偏移量
        :param number:表示第几次使用数据增强，不同数据增强方法需要按照number值进行+1运行
    '''

    origin_output_path = output_path
    resample_sr = np.random.uniform(sr)  # 从一个均匀分布中随机采样
    print(resample_sr)
    for fn in tqdm(glob.glob(os.path.join(data_path, "*.wav"))[:]):  # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        resample = librosa.resample(X, orig_sr=sample_rate, target_sr=resample_sr)
        resample = librosa.resample(resample, orig_sr=resample_sr, target_sr=sample_rate)
        # resample = np.asarray(resample, dtype=np.int16)
        # print(resample)

        # 得到音频文件的名字
        file_name = fn.split('\\')[-1]
        # 得到当前文件的类别
        label_name = file_name[:-6]
        # 生成新的文件名
        new_label_name = label_name + "_" + number
        # 组合成.wav文件名
        new_wav_name = new_label_name + ".wav"
        # 得到音频输出路径
        output_path = output_path + new_wav_name
        write(output_path, int(sample_rate), resample)
        output_path = origin_output_path


# 变速变调函数用来对声音进行增强
def variable_pitch(data_path, output_path, rate, number):
    '''
        设置的变速变调函数用来对声音进行增强
    :param rate:表示对音频的处理，放慢速度或降低速度
    :param number:表示第几次使用数据增强，不同数据增强方法需要按照number值进行+1运行
    '''

    origin_output_path = output_path
    for fn in tqdm(glob.glob(os.path.join(data_path, "*.wav"))[:]):  # 遍历数据集的所有文件
        X, sample_rate = librosa.load(fn, res_type='kaiser_fast')
        X = librosa.effects.time_stretch(X, rate)

        # 得到音频文件的名字
        file_name = fn.split('\\')[-1]
        # 得到当前文件的类别
        label_name = file_name[:-6]
        # 生成新的文件名
        new_label_name = label_name + "_" + number
        # 组合成.wav文件名
        new_wav_name = new_label_name + ".wav"
        # 得到音频输出路径
        output_path = output_path + new_wav_name
        write(output_path, int(sample_rate), X)
        output_path = origin_output_path


if __name__ == "__main__":
    # output_path = "..\\audio_dataset\\augument_train\\waveform_displacement_augment\\"
    # noise_augment_by_noise_factor(data_path, noise_ouput_path, 0.004, "2")
    # noise_augment_by_signal_to_noise_ratio(data_path, noise_ouput_path, 50, "3")
    # waveform_displacement(data_path, noise_ouput_path, 16000//2, "4")
    # resampling_data_augmentation(data_path, output_path, 16000, "5")
    # variable_pitch(data_path, output_path, 2, "6")

    label_dict = {
        "noise_augment_by_noise_factor": "..\\audio_dataset\\augument_train\\noise_augment_by_noise_factor\\",
                  "noise_augment_by_signal_to_noise_ratio": "..\\audio_dataset\\augument_train\\noise_augment_by_signal_to_noise_ratio\\",
                  "resampling_data_augmentation": "..\\audio_dataset\\augument_train\\resampling_data_augmentation\\",
                  "variable_pitch": "..\\audio_dataset\\augument_train\\variable_pitch\\",
                  "waveform_displacement_augment": "..\\audio_dataset\\augument_train\\waveform_displacement_augment\\"
                  }

    # 表示对每个噪声的类别一共生成50个训练集
    for i in range(7, 51, 5):
        # 调用每个数据增强方法生成对应的数据集
        noise_augment_by_noise_factor(data_path, label_dict["noise_augment_by_noise_factor"], 0.004, str(i))
        noise_augment_by_signal_to_noise_ratio(data_path, label_dict["noise_augment_by_signal_to_noise_ratio"], 50, str(i + 1))
        waveform_displacement(data_path, label_dict["waveform_displacement_augment"], 16000//2, str(i + 2))
        resampling_data_augmentation(data_path, label_dict["resampling_data_augmentation"], 16000, str(i + 3))
        variable_pitch(data_path, label_dict["variable_pitch"], 2, str(i + 4))




