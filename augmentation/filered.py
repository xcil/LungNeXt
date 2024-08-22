import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from librosa.display import waveshow, specshow

# 读取和处理音频
files = librosa.util.find_files('./wav/test/', ext=['wav'], recurse=True)
file = files[1160]
audio, sr = librosa.load(file, sr=None)
# 设计滤波器和滤波
lowcut = 100.0
highcut = 2000.0
nyq = 0.5 * sr
low = lowcut / nyq
high = highcut / nyq
sos = butter(10, [low, high], analog = False, btype = 'band', output = 'sos')

# 滤波后的音频
filtered_audio = sosfiltfilt(sos, audio)
# 归一化音频
normalized_audio = filtered_audio / np.max(np.abs(filtered_audio))

# 定义STFT的变量
n_fft = 2048
hop_length = n_fft // 4

# 计算频谱
D_original = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
D_filtered = librosa.amplitude_to_db(np.abs(librosa.stft(normalized_audio)), ref=np.max)

# 创建一个画布
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# 原始音频的波形图
librosa.display.waveshow(audio, sr=sr, ax=axs[0])
axs[0].set(title='Original Audio Waveform', ylabel='Amplitude')

# 滤波后的音频的波形图
librosa.display.waveshow(normalized_audio, sr=sr, ax=axs[1])
axs[1].set(title='Filtered and Normalized Audio Waveform', ylabel='Amplitude')

# 原始音频的频谱图
librosa.display.specshow(D_original, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[2])
axs[2].set(title='Original Audio Spectrogram')
axs[2].label_outer()  # 隐藏x轴标签，因为下面的图会显示

# 滤波后的音频的频谱图
librosa.display.specshow(D_filtered, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=axs[3])
axs[3].set(title='Filtered Audio Spectrogram')
axs[3].label_outer()  # 隐藏x轴标签

# 设置整个图形的标签和标题
fig.suptitle('Audio Processing Visualization')
plt.xlabel('Time (s)')

# 设置y轴的范围限制
axs[0].set_ylim([-1, 1])
axs[1].set_ylim([-1, 1])

# 显示图例
fig.legend(['Original', 'Filtered'], loc='upper right')

# 自动调整子图间距
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# 显示整个图
plt.show()