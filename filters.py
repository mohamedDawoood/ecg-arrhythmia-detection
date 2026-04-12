import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class ECGFilter:
    def __init__(self, fs=360):
        # fs = sampling rate = 360 Hz في MIT-BIH
        self.fs = fs

    def highpass(self, signal, cutoff=0.5, order=4):
        # يشيل موجة التنفس البطيئة (أقل من 0.5 Hz)
        nyquist = self.fs / 2
        b, a = butter(order, cutoff / nyquist, btype='high')
        return filtfilt(b, a, signal)

    def notch(self, signal, freq=50, quality=30):
        # يشيل ضوضاء الكهرباء عند 50 Hz بالظبط
        nyquist = self.fs / 2
        b, a = iirnotch(freq / nyquist, quality)
        return filtfilt(b, a, signal)

    def lowpass(self, signal, cutoff=40, order=4):
        # يشيل الضوضاء العالية التردد (أكتر من 40 Hz)
        nyquist = self.fs / 2
        b, a = butter(order, cutoff / nyquist, btype='low')
        return filtfilt(b, a, signal)

    def apply_all(self, signal):
        # بيطبق الـ 3 فلاتر بالترتيب الصح
        signal = self.highpass(signal)
        signal = self.notch(signal)
        signal = self.lowpass(signal)
        return signal