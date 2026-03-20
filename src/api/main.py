from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Churn Prediction API",
    description="API para predecir si un cliente de telecomunicaciones va a cancelar su servicio",
    version="1.0.0"
)

MODEL_PATH = "models/churn_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

model = None
encoders = None

@app.on_event("startup")
def load_model():
    global model, encoders
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")
    if not os.path.exists(ENCODERS_PATH):
        raise RuntimeError(f"Encoders no encontrados en {ENCODERS_PATH}")
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print("Modelo y encoders cargados correctamente")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 845.50
            }]
        }
    }


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    message: str


@app.get("/")
def root():
    return {"message": "Churn Prediction API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        data = customer.model_dump()
        df = pd.DataFrame([data])

        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            if col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col])
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Valor inválido en columna '{col}': {df[col].values[0]}"
                    )

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        if probability >= 0.7:
            risk_level = "HIGH"
            message = "Cliente con alto riesgo de cancelar. Acción inmediata recomendada."
        elif probability >= 0.4:
            risk_level = "MEDIUM"
            message = "Cliente con riesgo moderado. Monitorear y considerar retención."
        else:
            risk_level = "LOW"
            message = "Cliente estable. Bajo riesgo de cancelación."

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=round(float(probability), 4),
            risk_level=risk_level,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))