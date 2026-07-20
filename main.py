import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="DNSight ML Prediction Service")

# 1. Load the ML Artifacts when the server starts
try:
    scaler = joblib.load("preprocessing_pipeline.pkl")
    gnb_model = joblib.load("gnb_model.pkl")
    knn_model = joblib.load("knn_model.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")

# 2. Define the expected incoming data structure
class ClinicalData(BaseModel):
    AGE: float
    SEX: int
    BMI: float
    SP: float
    BP: float
    HbA1c: float
    FPS: float
    PPS: float
    FAMILY_HO: int  # Sent as FAMILY H/O in the dataframe
    SMOKING: int
    ONSET_AGE: float

@app.post("/predict")
def predict_risk(data: ClinicalData):
    try:
        # Calculate engineered feature: Diabetes_Duration = AGE - ONSET AGE
        duration = data.AGE - data.ONSET_AGE
        if duration < 0:
            duration = 0.0

        # Construct raw feature DataFrame in the exact sequence expected by the scaler
        raw_features = pd.DataFrame([{
            'AGE': data.AGE,
            'SEX': data.SEX,
            'BMI': data.BMI,
            'SP': data.SP,
            'BP': data.BP,
            'HbA1c': data.HbA1c,
            'FPS': data.FPS,
            'PPS': data.PPS,
            'FAMILY H/O': data.FAMILY_HO,
            'SMOKING': data.SMOKING,
            'ONSET AGE': data.ONSET_AGE,
            'Diabetes_Duration': duration
        }])

        # Process data using the pre-trained scaler
        scaled_features = scaler.transform(raw_features)

        # --- Individual Model Predictions ---
        # GNB: Gaussian Naive Bayes (index 1 = class 1: Neuropathy)
        gnb_prob = float(gnb_model.predict_proba(scaled_features)[0][1])
        gnb_risk_score = gnb_prob * 100

        # KNN: K-Nearest Neighbors (index 1 = class 1: Neuropathy)
        knn_prob = float(knn_model.predict_proba(scaled_features)[0][1])
        knn_risk_score = knn_prob * 100

        # --- Fusion: Weighted Average (equal weights) ---
        fusion_prob = 0.5 * knn_prob + 0.5 * gnb_prob
        fusion_risk_score = fusion_prob * 100

        return {
            "status": "success",
            "knn_score": round(knn_risk_score, 2),
            "knn_probability": round(knn_prob, 4),
            "gnb_score": round(gnb_risk_score, 2),
            "gnb_probability": round(gnb_prob, 4),
            "risk_score": round(fusion_risk_score, 2),
            "probability": round(fusion_prob, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "DNSight ML API"}