import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    
    # Corregir TotalCharges y eliminar nulos
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    
    # Eliminar columna ID (no aporta al modelo)
    df = df.drop(columns=['customerID'])
    
    # Target a binario
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    return df


def encode_features(df: pd.DataFrame, fit: bool = True, encoders: dict = None):
    df = df.copy()
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col])
    
    return df, encoders


def save_encoders(encoders: dict, path: str = 'models/encoders.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoders, path)
    print(f"Encoders guardados en {path}")


def load_encoders(path: str = 'models/encoders.pkl') -> dict:
    return joblib.load(path)


if __name__ == '__main__':
    df = load_and_clean('data/churn_raw.csv')
    df_encoded, encoders = encode_features(df, fit=True)
    save_encoders(encoders)
    
    df_encoded.to_csv('data/churn_processed.csv', index=False)
    print(f"Dataset procesado guardado: {df_encoded.shape}")
    print(f"Distribución de churn:\n{df_encoded['Churn'].value_counts()}")