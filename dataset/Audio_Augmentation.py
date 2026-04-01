import numpy as np
import random
import librosa


class TimeStretchProcessor:
    """
    时间拉伸处理器
    作用于带通滤波后的音频
    """

    def __init__(self, min_rate=0.8, max_rate=1.2, p_random=0.5):
        """
        参数:
            min_rate: 最小拉伸率
            max_rate: 最大拉伸率
            p_random: 应用增强的概率
        """
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p_random = p_random

    def __call__(self, audio):
        """
        参数:
            audio: 输入音频信号
        返回:
            增强后的音频
        """
        if random.random() > self.p_random:
            return audio

        rate = random.uniform(self.min_rate, self.max_rate)
        return librosa.effects.time_stretch(y=audio.astype(np.float32), rate=rate)


class NoiseInjectionProcessor:
    """
    噪声注入处理器
    作用于归一化后的音频
    """

    def __init__(self, noise_level_min=0.001, noise_level_max=0.015, p_random=0.5):
        """
        参数:
            noise_level_min: 最小噪声级别
            noise_level_max: 最大噪声级别
            p_random: 应用增强的概率
        """
        self.noise_level_min = noise_level_min
        self.noise_level_max = noise_level_max
        self.p_random = p_random

    def __call__(self, audio):
        """
        参数:
            audio: 输入音频信号
        返回:
            增强后的音频
        """
        if random.random() > self.p_random:
            return audio

        noise_level = random.uniform(self.noise_level_min, self.noise_level_max)
        noise = np.random.normal(0, noise_level, len(audio))
        augmented = audio + noise
        return np.clip(augmented, -1, 1)


class PitchShiftProcessor:
    """
    音高偏移处理器
    作用于重采样后的音频
    """

    def __init__(self, n_steps_min=-4, n_steps_max=4, p_random=0.5, sr=16000):
        """
        参数:
            n_steps_min: 最小音高偏移半音数
            n_steps_max: 最大音高偏移半音数
            p_random: 应用增强的概率
            sr: 采样率
        """
        self.n_steps_min = n_steps_min
        self.n_steps_max = n_steps_max
        self.p_random = p_random
        self.sr = sr

    def __call__(self, audio):
        """
        参数:
            audio: 输入音频信号
        返回:
            增强后的音频
        """
        if random.random() > self.p_random:
            return audio

        n_steps = random.uniform(self.n_steps_min, self.n_steps_max)
        return librosa.effects.pitch_shift(y=audio.astype(np.float32), sr=self.sr, n_steps=n_steps)