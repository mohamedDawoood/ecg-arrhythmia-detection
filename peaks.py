import numpy as np
from scipy.signal import butter, filtfilt


class PanTompkins:
    def __init__(self, fs=360):
        # fs = sampling rate
        self.fs = fs

    def bandpass(self, signal):
        # بيبرز الـ QRS complex بين 5-15 Hz
        # الـ R-Peak معظم طاقته في النطاق ده
        nyquist = self.fs / 2
        b, a = butter(4, [5 / nyquist, 15 / nyquist], btype='band')
        return filtfilt(b, a, signal)

    def derivative(self, signal):
        # بيحسب معدل التغيير — الـ R-Peak عنده أعلى معدل تغيير
        return np.diff(signal, prepend=signal[0])

    def squaring(self, signal):
        # بيربع الإشارة — يكبّر الفروق ويخلي كل القيم موجبة
        return signal ** 2

    def moving_average(self, signal, window=None):
        # بيعمل smoothing للإشارة
        # window = 150ms بالـ samples
        if window is None:
            window = int(0.15 * self.fs)
        return np.convolve(signal, np.ones(window) / window, mode='same')

    def detect_peaks(self, signal):
        # بيطبق كل الخطوات ويرجع مواقع الـ R-Peaks

        # الخطوة 1-4: معالجة الإشارة
        processed = self.bandpass(signal)
        processed = self.derivative(processed)
        processed = self.squaring(processed)
        processed = self.moving_average(processed)

        # الخطوة 5: Thresholding
        # الـ threshold = 50% من أعلى قيمة في الإشارة
        threshold = 0.5 * np.max(processed)

        # بندور على النقاط اللي فوق الـ threshold
        # refractory_period = أقل مسافة بين نبضتين = 200ms
        refractory_period = int(0.2 * self.fs)

        peaks = []
        last_peak = -refractory_period

        for i in range(1, len(processed) - 1):
            # الشرط: فوق الـ threshold + أعلى من جيرانه + بعيد عن آخر peak
            if (processed[i] > threshold and
                processed[i] > processed[i-1] and
                processed[i] > processed[i+1] and
                    i - last_peak > refractory_period):
                peaks.append(i)
                last_peak = i

        return np.array(peaks)