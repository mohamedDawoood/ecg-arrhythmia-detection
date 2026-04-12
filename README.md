# ❤️ ECG Arrhythmia Detection System

An ECG signal processing system that applies DSP techniques — including Butterworth filtering and Notch filtering — to remove noise, then detects heartbeat peaks using the Pan-Tompkins algorithm. Extracted HRV features are classified by a Random Forest model trained on MIT-BIH dataset to detect cardiac arrhythmias.

---

## 🗂️ Project Structure

```
ecg_project/
├── filters.py       # DSP filters — Butterworth, Notch
├── peaks.py         # Pan-Tompkins R-Peak detection
├── features.py      # Beat feature extraction
├── train.py         # Model training on MIT-BIH dataset
├── predict.py       # Prediction pipeline
├── main.py          # FastAPI backend
├── dashboard.py     # Streamlit dashboard
└── model.pkl        # Trained Random Forest model
```

---

## ⚙️ Pipeline

```
Raw ECG Signal
      ↓
High-Pass Filter  →  removes baseline wander (breathing noise)
      ↓
Notch Filter      →  removes powerline interference (50 Hz)
      ↓
Low-Pass Filter   →  removes high-frequency noise
      ↓
Pan-Tompkins      →  detects R-Peaks (heartbeat locations)
      ↓
Feature Extraction →  amplitude, std, mean, peak index...
      ↓
Random Forest     →  Normal / Arrhythmia
```

---

## 📊 Model Performance

| Metric | Normal | Arrhythmia |
|--------|--------|------------|
| Precision | 94% | 97% |
| Recall | 97% | 93% |
| F1-Score | 95% | 95% |
| **Accuracy** | **95%** | |

Trained on **101,175 beats** from **MIT-BIH Arrhythmia Database** (48 records).

---

## 🚀 How to Run

**1. Install dependencies**
```bash
pip install numpy scipy matplotlib pandas scikit-learn wfdb fastapi uvicorn streamlit
```

**2. Train the model**
```bash
python train.py
```

**3. Run the API**
```bash
uvicorn main:app --reload
```

**4. Run the Dashboard**
```bash
streamlit run dashboard.py
```

---

## 🔬 Dataset

- **MIT-BIH Arrhythmia Database** — PhysioNet
- 48 records, each 30 minutes long
- 360 Hz sampling rate
- Annotated by cardiologists

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Signal Processing | NumPy, SciPy |
| Machine Learning | Scikit-learn (Random Forest) |
| Backend | FastAPI |
| Dashboard | Streamlit |
| Dataset | WFDB, MIT-BIH |

---

## 📡 API Endpoint

**POST** `/predict`

Request:
```json
{
  "record_name": "208"
}
```

Response:
```json
{
  "record": "208",
  "diagnosis": "Arrhythmia",
  "confidence": 0.45,
  "total_beats": 2949,
  "normal_beats": 1615,
  "arrhythmia_beats": 1334,
  "arrhythmia_ratio": 45.24
}
```
