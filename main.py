import os
import pickle
import __main__
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Define custom transformer for pickle deserialization compatibility
class InvalidNegativeTransformer(BaseEstimator, TransformerMixin):
    """Convert negative clinical measurements to NaN before imputation."""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for column in self.columns:
            if column in X.columns:
                X[column] = pd.to_numeric(X[column], errors="coerce")
                X.loc[X[column] < 0, column] = np.nan
        return X

__main__.InvalidNegativeTransformer = InvalidNegativeTransformer

app = FastAPI(
    title="DNSight ML Prediction Service",
    description="Clinical Decision Support backend for diabetic neuropathy risk prediction using GNB, KNN, Logistic Regression, and AUC-weighted soft voting.",
    version="2.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
SCALER_PATH = BASE_DIR / "preprocessing_pipeline.pkl"
GNB_PATH = BASE_DIR / "gnb_model.pkl"
KNN_PATH = BASE_DIR / "knn_model.pkl"
LR_PATH = BASE_DIR / "lr_model.pkl"
FUSION_CONFIG_PATH = BASE_DIR / "fusion_config.pkl"

try:
    scaler = joblib.load(SCALER_PATH)
    gnb_model = joblib.load(GNB_PATH)
    knn_model = joblib.load(KNN_PATH)
    lr_model = joblib.load(LR_PATH)
    fusion_config = joblib.load(FUSION_CONFIG_PATH)
    print("Successfully loaded all ML artifacts (LR, GNB, KNN, Scaler, Fusion Config).")
except Exception as e:
    print(f"Error loading model files: {e}")
    scaler = None
    gnb_model = None
    knn_model = None
    lr_model = None
    fusion_config = None


class ClinicalData(BaseModel):
    AGE: float = Field(..., description="Patient age in years")
    SEX: int = Field(..., description="Encoded sex: 0=Female, 1=Male")
    BMI: float = Field(..., description="Body Mass Index")
    SP: float = Field(..., description="Systolic Blood Pressure (mmHg)")
    BP: float = Field(..., description="Diastolic Blood Pressure (mmHg)")
    HbA1c: float = Field(..., description="Glycated haemoglobin (%)")
    FPS: float = Field(..., description="Fasting plasma sugar (mg/dL)")
    PPS: float = Field(..., description="Postprandial sugar (mg/dL)")
    FAMILY_HO: int = Field(..., description="Family history of diabetes: 0=No, 1=Yes")
    SMOKING: int = Field(..., description="Smoking status: 0=Never, 1=Former, 2=Current, 3=Heavy")
    ONSET_AGE: float = Field(..., description="Age at diabetes onset (years)")


def classify_risk_tier(fusion_probability: float) -> str:
    """
    Standard three-tier clinical risk stratification:
    - LOW: < 30% predicted probability (Routine annual monitoring)
    - MEDIUM: 30% - 70% predicted probability (Increased surveillance, 3-month follow-up)
    - HIGH: >= 70% predicted probability (Immediate clinical attention & confirmatory exams)
    """
    score = fusion_probability * 100.0
    if score < 30.0:
        return "LOW"
    elif score < 70.0:
        return "MEDIUM"
    else:
        return "HIGH"


@app.post("/predict")
def predict_risk(data: ClinicalData):
    if any(m is None for m in [scaler, gnb_model, knn_model, lr_model, fusion_config]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models or preprocessing pipeline not loaded.",
        )

    # 1. Cross-field physiological validation
    if data.ONSET_AGE > data.AGE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid clinical record: Age at diabetes onset cannot exceed current age.",
        )

    if data.AGE <= 0 or data.ONSET_AGE < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Age and Onset Age must be positive numbers.",
        )

    try:
        # 2. Derive Diabetes_Duration = AGE - ONSET_AGE
        duration = data.AGE - data.ONSET_AGE

        # 3. Construct DataFrame matching the exact feature order of the preprocessing pipeline
        raw_features = pd.DataFrame([{
            "AGE": data.AGE,
            "SEX": data.SEX,
            "BMI": data.BMI,
            "SP": data.SP,
            "BP": data.BP,
            "HbA1c": data.HbA1c,
            "FPS": data.FPS,
            "PPS": data.PPS,
            "FAMILY H/O": data.FAMILY_HO,
            "SMOKING": data.SMOKING,
            "ONSET AGE": data.ONSET_AGE,
            "Diabetes_Duration": duration,
        }])

        # 4. Preprocess features
        scaled_features = scaler.transform(raw_features)

        # 5. Parallel inference across all 3 models (Positive class index 1 = Neuropathy)
        gnb_prob = float(gnb_model.predict_proba(scaled_features)[0][1])
        knn_prob = float(knn_model.predict_proba(scaled_features)[0][1])
        lr_prob = float(lr_model.predict_proba(scaled_features)[0][1])

        # Convert to individual 0-100 percentage scores
        gnb_score = round(gnb_prob * 100, 2)
        knn_score = round(knn_prob * 100, 2)
        lr_score = round(lr_prob * 100, 2)

        # 6. Retrieve normalized AUC weights from fusion configuration
        w_gnb = float(fusion_config.get("gnb_weight", 0.3077))
        w_knn = float(fusion_config.get("knn_weight", 0.3711))
        w_lr = float(fusion_config.get("logistic_weight", 0.3212))

        weight_sum = w_gnb + w_knn + w_lr
        w_gnb /= weight_sum
        w_knn /= weight_sum
        w_lr /= weight_sum

        # 7. Calculate AUC-ROC Weighted Soft Voting Fusion
        fusion_prob = (w_gnb * gnb_prob) + (w_knn * knn_prob) + (w_lr * lr_prob)
        fusion_risk_score = round(fusion_prob * 100, 2)

        # 8. Apply Youden's J decision threshold
        threshold = float(fusion_config.get("threshold", 0.50))
        prediction = 1 if fusion_prob >= threshold else 0
        risk_category = classify_risk_tier(fusion_prob)

        return {
            "status": "success",
            "knn_score": knn_score,
            "knn_probability": round(knn_prob, 4),
            "gnb_score": gnb_score,
            "gnb_probability": round(gnb_prob, 4),
            "lr_score": lr_score,
            "lr_probability": round(lr_prob, 4),
            "risk_score": fusion_risk_score,
            "probability": round(fusion_prob, 4),
            "prediction": prediction,
            "prediction_label": (
                "Diabetic neuropathy risk detected"
                if prediction == 1
                else "No diabetic neuropathy risk detected"
            ),
            "risk_category": risk_category,
            "deployment_threshold": threshold,
            "threshold_method": fusion_config.get("threshold_method", "Youden's J"),
            "fusion_weights": {
                "GNB": round(w_gnb, 6),
                "KNN": round(w_knn, 6),
                "Logistic Regression": round(w_lr, 6),
            },
            "engineered_features": {
                "Diabetes_Duration": round(duration, 1),
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "DNSight ML API",
        "models": ["Logistic Regression", "Gaussian Naive Bayes", "K-Nearest Neighbors"],
        "fusion": "AUC-Weighted Soft Voting",
        "threshold_method": "Youden's J",
    }
