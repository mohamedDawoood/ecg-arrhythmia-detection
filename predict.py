import wfdb
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from filters import ECGFilter
from train import extract_beat_features, WINDOW, NORMAL_SYMBOL, ABNORMAL_SYMBOLS


def predict_ecg(record_name: str):
    """
    بتاخد رقم تسجيل من MIT-BIH
    وبترجع التشخيص + إحصائيات الإشارة كاملة
    """

    # تحميل الإشارة
    record = wfdb.rdrecord(record_name, pn_dir='mitdb')
    annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
    signal = record.p_signal[:, 0]
    fs = record.fs

    # تنظيف الإشارة
    cleaner = ECGFilter(fs=fs)
    clean_signal = cleaner.apply_all(signal)

    # تحميل الموديل
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # استخراج features لكل نبضة وتصنيفها
    total_beats = 0
    arrhythmia_beats = 0
    normal_beats = 0

    for i in range(len(annotation.sample)):
        idx = annotation.sample[i]
        symbol = annotation.symbol[i]

        if symbol != NORMAL_SYMBOL and symbol not in ABNORMAL_SYMBOLS:
            continue

        if idx <= WINDOW or idx >= len(clean_signal) - WINDOW:
            continue

        beat = clean_signal[idx - WINDOW: idx + WINDOW]
        features = extract_beat_features(beat, fs)
        feature_values = np.array(list(features.values())).reshape(1, -1)

        prediction = model.predict(feature_values)[0]
        total_beats += 1

        if prediction == 1:
            arrhythmia_beats += 1
        else:
            normal_beats += 1

    if total_beats == 0:
        return {"error": "No beats detected in this record"}

    # نسبة النبضات الغير طبيعية
    arrhythmia_ratio = arrhythmia_beats / total_beats

    # التشخيص النهائي — لو أكتر من 10% من النبضات غير طبيعية = Arrhythmia
    diagnosis = "Arrhythmia" if arrhythmia_ratio > 0.1 else "Normal"
    confidence = round(arrhythmia_ratio if diagnosis == "Arrhythmia" else 1 - arrhythmia_ratio, 2)

    return {
        "record": record_name,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "total_beats": total_beats,
        "normal_beats": normal_beats,
        "arrhythmia_beats": arrhythmia_beats,
        "arrhythmia_ratio": round(arrhythmia_ratio * 100, 2)
    }


# تجربة مباشرة
if __name__ == "__main__":
    result = predict_ecg('100')
    print("\nPrediction Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")