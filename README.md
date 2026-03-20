# churn-prediction-mlops

# Churn Prediction MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)

End-to-end MLOps pipeline para predecir churn de clientes en una empresa de telecomunicaciones. El proyecto cubre el ciclo completo: ingesta de datos, entrenamiento con tracking de experimentos, exposición como API REST y deploy en producción.

## 🔗 Demo en vivo
**API:** https://churn-prediction-mlops-production.up.railway.app/docs

## 📊 Resultados del modelo

| Métrica | Valor |
|--------|-------|
| AUC-ROC | 0.8346 |
| F1 Score | 0.6136 |
| Recall (churn) | 0.78 |
| Precisión (churn) | 0.50 |

## 🏗️ Arquitectura
```
data/                   <- Dataset raw y procesado
src/
├── pipeline/
│   └── preprocess.py   <- Limpieza y encoding de features
├── model/
│   └── train.py        <- Entrenamiento XGBoost + MLflow tracking
└── api/
    └── main.py         <- FastAPI REST API
models/                 <- Modelo entrenado y encoders
notebooks/
└── 01_eda.ipynb        <- Análisis exploratorio
Dockerfile              <- Containerización
docker-compose.yml      <- Orquestación local
```

## 🚀 Correr localmente
```bash
git clone https://github.com/JOSE2002-13/churn-prediction-mlops.git
cd churn-prediction-mlops
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

API disponible en http://localhost:8000/docs

## 🐳 Correr con Docker
```bash
docker-compose up --build
```

## 📡 Ejemplo de predicción
```bash
curl -X POST "https://churn-prediction-mlops-production.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Respuesta:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.6894,
  "risk_level": "MEDIUM",
  "message": "Cliente con riesgo moderado. Monitorear y considerar retención."
}
```

## 🛠️ Stack tecnológico

- **Modelo:** XGBoost con scale_pos_weight para manejo de desbalance
- **Tracking:** MLflow para versionado de experimentos y métricas
- **API:** FastAPI con validación Pydantic y documentación Swagger automática
- **Container:** Docker + docker-compose
- **Deploy:** Railway (Cloud Run ready)

## 📁 Dataset

[Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 clientes
- 20 features
- 26.5% tasa de churn