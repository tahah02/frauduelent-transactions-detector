import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = 'data/engineered_transaction_features.csv' 
MODEL_PATH = 'models/isolation_forest.pkl'
SCALER_PATH = 'models/isolation_forest_scaler.pkl'  # New scaler file

FEATURES = [
    # Transaction features
    'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
    'channel_encoded', 'deviation_from_avg', 'amount_to_max_ratio', 'rolling_std',
    
    # Time features
    'hour', 'day_of_week', 'is_weekend', 'is_night',
    
    # User behavior features
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
    'intl_ratio',
    
    # Velocity features
    'time_since_last', 'recent_burst', 
    'txn_count_30s', 'txn_count_10min', 'txn_count_1hour',
    
    # Aggregate features
    'hourly_total', 'hourly_count', 'daily_total', 'daily_count'
]

def load_data():
    paths_to_try = [
        DATA_PATH,
        'engineered_transaction_features.csv',
        'data/engineered_transaction_features.csv'
    ]   
    for path in paths_to_try:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)          
    print("Error: engineered_transaction_features.csv not found! Please upload it to the 'data' folder.")
    return None

def train_model():
    print("Initializing Model Training...")
    df = load_data() 
    if df is None:
        return
    X = df[FEATURES].fillna(0)

    print(f"Training on {len(FEATURES)} features.")
    print(f"Sample Features: {FEATURES[:5]}...")

    # -------------------
    # Fit Scaler
    # -------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print(f"SUCCESS: Scaler saved to {SCALER_PATH}")

    # Train Isolation Forest
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X_scaled)

    # Save model + feature list
    model_data = {
        'model': clf,
        'features': FEATURES
    } 
    joblib.dump(model_data, MODEL_PATH)
    print(f"SUCCESS: Model saved to {MODEL_PATH}")
    print("Training Complete!")

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model_data = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            clf = model_data['model'] if isinstance(model_data, dict) else model_data
            features = model_data['features'] if isinstance(model_data, dict) else FEATURES
            return clf, features, scaler
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            print("Retraining model...")
            train_model()
            return load_model()
    else:
        print("Model or scaler file not found. Training new model...")
        train_model()
        return load_model()

if __name__ == "__main__":
    train_model()
