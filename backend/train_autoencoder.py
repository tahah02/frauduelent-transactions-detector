import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from backend.autoencoder import TransactionAutoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    
    DATA_PATH = 'data/featured_dataset.csv'
    MODEL_PATH = 'models/autoencoder.h5'
    SCALER_PATH = 'models/autoencoder_scaler.pkl'
    THRESHOLD_PATH = 'models/autoencoder_threshold.json'
    
    FEATURES = [
        'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
        'channel_encoded', 'deviation_from_avg', 'amount_to_max_ratio', 'rolling_std',
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
        'intl_ratio', 'user_high_risk_txn_ratio', 'user_multiple_accounts_flag',
        'cross_account_transfer_ratio', 'time_since_last', 'recent_burst',
        'txn_count_30s', 'txn_count_10min', 'txn_count_1hour',
        'transaction_velocity', 'is_new_beneficiary', 'beneficiary_txn_count_30d',
        'beneficiary_risk_score', 'geo_anomaly_flag', 'current_month_spending'
    ]
    
    def __init__(self, k=3.0):
        self.k = k
        self.scaler = None
        self.autoencoder = None
    
    def load_data(self):
        if os.path.exists(self.DATA_PATH):
            return pd.read_csv(self.DATA_PATH)
        return None
    
    def train(self, epochs=100, batch_size=64):
        logger.info("=" * 50)
        logger.info("Starting Autoencoder Training")
        logger.info("=" * 50)
        
        df = self.load_data()
        if df is None:
            raise FileNotFoundError("Training data not found")
        
        X = df[self.FEATURES].fillna(0).values
        n_samples, n_features = X.shape
        logger.info(f"Training on {n_samples} samples, {n_features} features")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.SCALER_PATH)
        
        encoding_dim = max(7, n_features // 2)
        self.autoencoder = TransactionAutoencoder(
            input_dim=n_features, encoding_dim=encoding_dim, hidden_layers=[64, 32]
        )
        
        self.autoencoder.fit(X_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
        self.autoencoder.save(self.MODEL_PATH)
        
        errors = self.autoencoder.compute_reconstruction_error(X_scaled)
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        threshold = mean_error + self.k * std_error
        
        with open(self.THRESHOLD_PATH, 'w') as f:
            json.dump({
                'threshold': threshold,
                'mean': mean_error,
                'std': std_error,
                'k': self.k,
                'n_samples': n_samples,
                'n_features': n_features
            }, f, indent=2)
        
        logger.info("=" * 50)
        logger.info(f"Training Complete! Threshold: {threshold:.4f}")
        logger.info("=" * 50)
        
        return {'threshold': threshold, 'mean_error': mean_error, 'std_error': std_error}


def train_autoencoder():
    trainer = AutoencoderTrainer(k=3.0)
    return trainer.train(epochs=100, batch_size=64)


if __name__ == "__main__":
    train_autoencoder()
