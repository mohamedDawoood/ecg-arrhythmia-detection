import streamlit as st
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import requests

from filters import ECGFilter
from peaks import PanTompkins
from train import extract_beat_features, WINDOW, NORMAL_SYMBOL, ABNORMAL_SYMBOLS

# ============================
# إعداد الصفحة
# ============================
st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ ECG Arrhythmia Detection System")
st.markdown("نظام كشف اضطرابات القلب باستخدام DSP + Machine Learning")
st.divider()

# ============================
# الـ Sidebar — اختيار التسجيل
# ============================
st.sidebar.header("⚙️ Settings")
record_name = st.sidebar.text_input("Record Number", value="115")
analyze_btn = st.sidebar.button("🔍 Analyze", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown("**Normal Records:** 115, 122, 113")
st.sidebar.markdown("**Arrhythmia Records:** 208, 200, 109")

# ============================
# التحليل
# ============================
if analyze_btn:
    with st.spinner("Loading and analyzing ECG signal..."):
        try:
            # تحميل الإشارة
            record = wfdb.rdrecord(record_name, pn_dir='mitdb')
            annotation = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')
            signal = record.p_signal[:, 0]
            fs = record.fs

            # تنظيف الإشارة
            cleaner = ECGFilter(fs=fs)
            clean_signal = cleaner.apply_all(signal)

            # كشف الـ R-Peaks
            detector = PanTompkins(fs=fs)
            r_peaks = detector.detect_peaks(clean_signal)

            # استدعاء الـ API للتشخيص
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"record_name": record_name}
            )
            result = response.json()

            # ============================
            # عرض النتيجة
            # ============================
            st.subheader("📊 Diagnosis Result")

            col1, col2, col3, col4 = st.columns(4)

            diagnosis = result['diagnosis']
            color = "🟢" if diagnosis == "Normal" else "🔴"

            col1.metric("Diagnosis", f"{color} {diagnosis}")
            col2.metric("Total Beats", result['total_beats'])
            col3.metric("Arrhythmia Beats", result['arrhythmia_beats'])
            col4.metric("Arrhythmia Ratio", f"{result['arrhythmia_ratio']}%")

            st.divider()

            # ============================
            # رسم الإشارات
            # ============================
            st.subheader("📈 ECG Signal Analysis")

            # بناخد أول 10 ثواني بس للرسم
            samples_10s = fs * 10
            raw_10s = signal[:samples_10s]
            clean_10s = clean_signal[:samples_10s]

            # R-Peaks في أول 10 ثواني بس
            peaks_10s = r_peaks[r_peaks < samples_10s]

            fig, axes = plt.subplots(2, 1, figsize=(14, 6))

            # الإشارة الخام
            axes[0].plot(raw_10s, color='red', linewidth=0.8, label='Raw Signal')
            axes[0].set_title("Raw ECG Signal (First 10 seconds)")
            axes[0].set_ylabel("Amplitude (mV)")
            axes[0].legend()
            axes[0].grid(True)

            # الإشارة النظيفة + R-Peaks
            axes[1].plot(clean_10s, color='green', linewidth=0.8, label='Clean Signal')
            axes[1].scatter(peaks_10s, clean_10s[peaks_10s],
                          color='red', s=50, zorder=5, label='R-Peaks')
            axes[1].set_title("Clean ECG Signal + R-Peaks Detected")
            axes[1].set_ylabel("Amplitude (mV)")
            axes[1].set_xlabel("Samples")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure the API is running: uvicorn main:app --reload")