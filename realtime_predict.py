import serial
import numpy as np
from collections import deque
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

from filters import ECGFilter
from peaks import PanTompkins
from train import extract_beat_features, WINDOW

# ============================
# إعداد الصفحة
# ============================
st.set_page_config(page_title="Real-Time ECG", page_icon="❤️", layout="wide")
st.title("❤️ Real-Time ECG Arrhythmia Detection System")
st.markdown("نظام كشف اضطرابات القلب في الوقت الفعلي")
st.divider()

# ============================
# الإعدادات
# ============================
FS = 100
BUFFER_SIZE = 500
PORT = 'COM3'
THRESHOLD = 0.1

# ============================
# تحميل الموديل
# ============================
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

cleaner = ECGFilter(fs=FS)
detector = PanTompkins(fs=FS)

# ============================
# معلومات المريض
# ============================
st.subheader("👤 Patient Information")
col1, col2, col3 = st.columns(3)
patient_name = col1.text_input("Patient Name", placeholder="Enter name...")
patient_age = col2.number_input("Age", min_value=1, max_value=120, value=25)
patient_gender = col3.selectbox("Gender", ["Male", "Female"])
st.divider()

# ============================
# زرار البدء
# ============================
start = st.button("🚀 Start ECG Recording", use_container_width=True, type="primary")

# ============================
# الـ UI placeholders
# ============================
status_placeholder = st.empty()

if start:
    if not patient_name:
        st.warning("Please enter patient name first.")
        st.stop()

    status_placeholder.info("⏳ Connecting to Arduino and collecting signal... Please wait 5 seconds.")

    # ============================
    # القراءة من Arduino
    # ============================
    try:
        ser = serial.Serial(PORT, 9600, timeout=1)
        buffer = deque(maxlen=BUFFER_SIZE)

        # نملى الـ buffer الأول
        while len(buffer) < BUFFER_SIZE:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    buffer.append(int(line))
                except ValueError:
                    pass

        ser.close()
        status_placeholder.success("✅ Signal collected successfully!")

        # ============================
        # DSP Pipeline
        # ============================
        raw_signal = np.array(buffer, dtype=float)

        # ① تنظيف الإشارة
        clean_signal = cleaner.apply_all(raw_signal)

        # ② كشف الـ R-Peaks
        r_peaks = detector.detect_peaks(clean_signal)

        # ③ حساب معدل القلب
        heart_rate = 0
        if len(r_peaks) >= 2:
            rr_intervals = np.diff(r_peaks) / FS
            heart_rate = int(60 / np.mean(rr_intervals))

        # ④ تصنيف كل نبضة
        total = 0
        arrhythmia = 0

        for idx in r_peaks:
            if idx <= WINDOW or idx >= len(clean_signal) - WINDOW:
                continue
            beat = clean_signal[idx - WINDOW: idx + WINDOW]
            features = extract_beat_features(beat, FS)
            feature_values = np.array(list(features.values())).reshape(1, -1)
            prediction = model.predict(feature_values)[0]
            total += 1
            if prediction == 1:
                arrhythmia += 1

        # ⑤ التشخيص النهائي
        ratio = (arrhythmia / total * 100) if total > 0 else 0
        diagnosis = "Arrhythmia" if (arrhythmia / total if total > 0 else 0) > THRESHOLD else "Normal"

        # ============================
        # عرض النتائج
        # ============================
        st.divider()
        st.subheader(f"📋 Diagnosis Report — {patient_name}")

        # بطاقة التشخيص
        if diagnosis == "Normal":
            st.success(f"✅ **{patient_name}** — Heart rhythm is **Normal**")
        else:
            st.error(f"⚠️ **{patient_name}** — **Arrhythmia Detected** — Please consult a doctor")

        # الأرقام التفصيلية
        st.subheader("📊 Detailed Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Diagnosis", diagnosis)
        c2.metric("Heart Rate", f"{heart_rate} BPM")
        c3.metric("Total Beats", total)
        c4.metric("Abnormal Beats", arrhythmia)
        c5.metric("Arrhythmia Ratio", f"{round(ratio, 1)}%")

        # تفسير النتيجة
        st.subheader("🔍 Result Interpretation")
        if diagnosis == "Normal":
            st.info(f"""
            **Patient:** {patient_name} | **Age:** {patient_age} | **Gender:** {patient_gender}

            ✅ Heart rhythm appears **normal** based on the ECG analysis.

            - Heart rate: **{heart_rate} BPM** — {'Normal range ✅' if 60 <= heart_rate <= 100 else 'Outside normal range ⚠️'}
            - Only **{round(ratio, 1)}%** of beats showed irregular patterns
            - No significant arrhythmia detected
            """)
        else:
            st.warning(f"""
            **Patient:** {patient_name} | **Age:** {patient_age} | **Gender:** {patient_gender}

            ⚠️ Irregular heartbeat patterns detected in the ECG signal.

            - Heart rate: **{heart_rate} BPM** — {'Normal range ✅' if 60 <= heart_rate <= 100 else 'Outside normal range ⚠️'}
            - **{round(ratio, 1)}%** of beats showed irregular patterns
            - **{arrhythmia}** out of **{total}** beats classified as abnormal
            - **Recommendation:** Consult a cardiologist for further evaluation
            """)

        # رسم الإشارات
        st.subheader("📈 Raw ECG Signal")
        st.line_chart(raw_signal)

        st.subheader("✅ Clean ECG Signal + R-Peaks")
        clean_df = {"Clean Signal": clean_signal}
        st.line_chart(clean_df)

        st.subheader("🔴 R-Peaks Locations")
        peaks_signal = np.zeros(len(clean_signal))
        peaks_signal[r_peaks] = np.max(clean_signal)
        st.line_chart({"Clean Signal": clean_signal, "R-Peaks": peaks_signal})

    except serial.SerialException:
        st.error("❌ Cannot connect to Arduino — Make sure it's connected and Serial Monitor is closed")
    except Exception as e:
        st.error(f"❌ Error: {e}")