import numpy as np


class HRVFeatures:
    def __init__(self, fs=360):
        self.fs = fs

    def extract(self, r_peaks):
        if len(r_peaks) < 2:
            return None

        rr_intervals = np.diff(r_peaks) / self.fs
        mean_rr = np.mean(rr_intervals)
        heart_rate = 60 / mean_rr
        sdnn = np.std(rr_intervals)
        diff_rr = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diff_rr ** 2))
        pnn50 = np.sum(np.abs(diff_rr) > 0.05) / len(diff_rr) * 100

        return {
            'mean_rr': round(mean_rr, 4),
            'heart_rate': round(heart_rate, 2),
            'sdnn': round(sdnn, 4),
            'rmssd': round(rmssd, 4),
            'pnn50': round(pnn50, 2)
        }