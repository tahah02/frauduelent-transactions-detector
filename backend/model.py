import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


DATA_PATH = 'data/featured_dataset.csv'
MODEL_PATH = 'models/isolation_forest.pkl'
SCALER_PATH = 'models/isolation_forest_scaler.pkl'

FEATURES = [
    'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
    'channel_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
    'deviation_from_avg', 'amount_to_max_ratio', 'intl_ratio',
    'user_high_risk_txn_ratio', 'user_multiple_accounts_flag', 'cross_account_transfer_ratio',
    'rolling_std', 'transaction_velocity', 'is_new_beneficiary',
    'beneficiary_txn_count_30d', 'beneficiary_risk_score', 'geo_anomaly_flag', 'recent_burst'
]


def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


# def delete_old_models():
#     for path in [MODEL_PATH, SCALER_PATH]:
#         if os.path.exists(path):
#             os.remove(path)


def train_model():
    print("=" * 50)
    print("ISOLATION FOREST MODEL TRAINING")
    print("=" * 50)
    
    df = load_data()
    if df is None:
        print(f"Error: {DATA_PATH} not found!")
        return None
    
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"Error: Missing features: {missing}")
        return None
    
    X = df[FEATURES].fillna(0)
    print(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X_scaled)
    
    delete_old_models()
    
    joblib.dump({'model': clf, 'features': FEATURES}, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")
    print("=" * 50)
    
    return clf


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
            if isinstance(data, dict):
                return data['model'], data['features'], scaler
            return data, FEATURES, scaler
        except Exception as e:
            print(f"Error loading model: {e}")
            train_model()
            return load_model()
    else:
        train_model()
        return load_model()


if __name__ == "__main__":
    train_model()
