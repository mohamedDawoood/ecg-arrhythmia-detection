import wfdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

from filters import ECGFilter
from peaks import PanTompkins
from features import HRVFeatures

RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# الرموز الطبية للنبضات الطبيعية والغير طبيعية
NORMAL_SYMBOL = 'N'
ABNORMAL_SYMBOLS = ['V', 'A', 'L', 'R', 'E', 'j', 'F']

# حجم النافذة حول كل نبضة = 90 sample قبل + 90 بعد = 180 sample
WINDOW = 90


def extract_beat_features(beat, fs=360):
    """
    بتاخد نبضة واحدة (180 sample)
    وبترجع features بسيطة منها
    """
    # الـ features اللي هنستخرجها من كل نبضة
    mean_val = np.mean(beat)
    std_val = np.std(beat)
    max_val = np.max(beat)
    min_val = np.min(beat)
    # amplitude = الفرق بين أعلى وأدنى نقطة
    amplitude = max_val - min_val
    # مكان أعلى نقطة في النبضة
    peak_idx = np.argmax(beat)
    # متوسط النصف الأول والثاني
    first_half_mean = np.mean(beat[:WINDOW])
    second_half_mean = np.mean(beat[WINDOW:])

    return {
        'mean': round(mean_val, 4),
        'std': round(std_val, 4),
        'max': round(max_val, 4),
        'min': round(min_val, 4),
        'amplitude': round(amplitude, 4),
        'peak_idx': peak_idx,
        'first_half_mean': round(first_half_mean, 4),
        'second_half_mean': round(second_half_mean, 4),
    }


def process_record(record_name):
    """
    بتاخد رقم تسجيل وبترجع
    قائمة features لكل نبضة + label لكل نبضة
    """
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb')
        annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
        signal = record.p_signal[:, 0]
        fs = record.fs

        # تنظيف الإشارة
        cleaner = ECGFilter(fs=fs)
        clean_signal = cleaner.apply_all(signal)

        features_list = []
        labels_list = []

        for i in range(len(annotation.sample)):
            idx = annotation.sample[i]
            symbol = annotation.symbol[i]

            # نتجاهل النبضات اللي مش Normal أو Abnormal
            if symbol != NORMAL_SYMBOL and symbol not in ABNORMAL_SYMBOLS:
                continue

            # نتأكد إن النبضة مش على حافة الإشارة
            if idx <= WINDOW or idx >= len(clean_signal) - WINDOW:
                continue

            # نقطع النبضة — 90 sample قبل الـ peak و 90 بعده
            beat = clean_signal[idx - WINDOW: idx + WINDOW]

            # نستخرج الـ features
            features = extract_beat_features(beat, fs)

            # نحدد الـ label
            label = 0 if symbol == NORMAL_SYMBOL else 1

            features_list.append(features)
            labels_list.append(label)

        return features_list, labels_list

    except Exception as e:
        print(f"Error in record {record_name}: {e}")
        return [], []


if __name__ == "__main__":
    all_features = []
    all_labels = []

    print(f"Processing {len(RECORDS)} records...")

    for record in RECORDS:
        features_list, labels_list = process_record(record)
        if len(features_list) > 0:
            all_features.extend(features_list)
            all_labels.extend(labels_list)
            print(f"Record {record}: {len(features_list)} beats extracted")

    # تحويل لـ DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = all_labels

    print(f"\nDataset Summary before balancing:")
    print(f"Total beats: {len(df)}")
    print(f"Normal (0): {sum(df['label'] == 0)}")
    print(f"Arrhythmia (1): {sum(df['label'] == 1)}")

    # موازنة الداتا — بناخد نفس عدد النبضات من كل class
    df_normal = df[df['label'] == 0]
    df_arrhythmia = df[df['label'] == 1]

    # بناخد أصغر عدد بين الاتنين عشان يتوازنوا
    min_count = min(len(df_normal), len(df_arrhythmia))

    df_normal_balanced = resample(df_normal, n_samples=min_count, random_state=42)
    df_arrhythmia_balanced = resample(df_arrhythmia, n_samples=min_count, random_state=42)

    df_balanced = pd.concat([df_normal_balanced, df_arrhythmia_balanced])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nDataset Summary after balancing:")
    print(f"Total beats: {len(df_balanced)}")
    print(f"Normal (0): {sum(df_balanced['label'] == 0)}")
    print(f"Arrhythmia (1): {sum(df_balanced['label'] == 1)}")

    # تقسيم الداتا
    X = df_balanced.drop('label', axis=1)
    y = df_balanced['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # تدريب الموديل
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # تقييم الموديل
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Normal', 'Arrhythmia']
    ))

    # حفظ الموديل
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model saved as model.pkl")