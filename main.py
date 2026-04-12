from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict_ecg

app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="نظام كشف اضطرابات القلب باستخدام DSP + Machine Learning",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    # رقم التسجيل من MIT-BIH مثلاً '100' أو '234'
    record_name: str


@app.get("/")
def root():
    return {
        "message": "ECG Arrhythmia Detection API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    بتاخد رقم تسجيل ECG وبترجع التشخيص كامل
    """
    result = predict_ecg(request.record_name)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result