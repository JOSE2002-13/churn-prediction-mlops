import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix
)
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import os

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y

def get_scale_pos_weight(y):
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return round(neg / pos, 2)

def train(data_path: str = 'data/churn_processed.csv'):
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = get_scale_pos_weight(y_train)
    print(f"scale_pos_weight: {scale_pos_weight}")

    params = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'auc',
    }

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc   = roc_auc_score(y_test, y_proba)
        f1    = f1_score(y_test, y_pred)

        print(f"\n=== Resultados ===")
        print(f"AUC-ROC : {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        mlflow.log_params(params)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.xgboost.log_model(model, "model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/churn_model.pkl")
        print("Modelo guardado en models/churn_model.pkl")

if __name__ == '__main__':
    train()