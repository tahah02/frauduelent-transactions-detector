# backend/train_autoencoder.py
"""
Training pipeline for the Transaction Autoencoder.
Handles data loading, scaler fitting, model training, threshold computation, and validation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from backend.autoencoder import TransactionAutoencoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoencoderTrainer:
    """
    Training pipeline for the Transaction Autoencoder.
    
    Responsibilities:
    - Load and prepare training data
    - Fit and save scaler
    - Train Autoencoder model
    - Compute and save threshold
    - Validate saved artifacts
    """
    
    # Configuration paths
    DATA_PATH = 'data/engineered_transaction_features.csv'
    MODEL_PATH = 'models/autoencoder.h5'
    SCALER_PATH = 'models/autoencoder_scaler.pkl'
    THRESHOLD_PATH = 'models/autoencoder_threshold.json'
    
    # Feature list (same as Isolation Forest)
    FEATURES = [
        'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
        'channel_encoded', 'deviation_from_avg', 'amount_to_max_ratio', 'rolling_std',
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
        'intl_ratio', 'time_since_last', 'recent_burst',
        'txn_count_30s', 'txn_count_10min', 'txn_count_1hour',
        'hourly_total', 'hourly_count', 'daily_total', 'daily_count'
    ]
    
    def __init__(self, k: float = 3.0):
        """
        Initialize trainer with threshold multiplier.
        
        Args:
            k: Standard deviation multiplier for threshold (default: 3.0)
        """
        self.k = k
        self.scaler: Optional[StandardScaler] = None
        self.autoencoder: Optional[TransactionAutoencoder] = None

    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load engineered features from CSV."""
        paths_to_try = [
            self.DATA_PATH,
            'engineered_transaction_features.csv',
            'data/engineered_transaction_features.csv'
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                logger.info(f"Loading data from: {path}")
                return pd.read_csv(path)
        
        logger.error("engineered_transaction_features.csv not found!")
        return None
    
    def fit_scaler(self, X: np.ndarray) -> StandardScaler:
        """
        Fit and save StandardScaler.
        
        Args:
            X: Feature array to fit scaler on
            
        Returns:
            Fitted StandardScaler
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.SCALER_PATH) if os.path.dirname(self.SCALER_PATH) else '.', exist_ok=True)
        joblib.dump(self.scaler, self.SCALER_PATH)
        logger.info(f"Scaler saved to {self.SCALER_PATH}")
        
        return self.scaler
    
    def compute_threshold(self, errors: np.ndarray) -> Dict[str, Any]:
        """
        Compute threshold from reconstruction errors.
        
        Formula: threshold = mean(errors) + k * std(errors)
        
        Args:
            errors: Array of reconstruction errors
            
        Returns:
            dict: {'threshold': float, 'mean': float, 'std': float, 'k': float}
        """
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        threshold = mean_error + self.k * std_error
        
        return {
            'threshold': threshold,
            'mean': mean_error,
            'std': std_error,
            'k': self.k
        }
    
    def save_threshold(self, threshold_config: Dict[str, Any], n_samples: int, n_features: int) -> None:
        """
        Save threshold configuration to JSON.
        
        Args:
            threshold_config: Threshold configuration dict
            n_samples: Number of training samples
            n_features: Number of features
        """
        config = {
            **threshold_config,
            'computed_at': datetime.now().isoformat(),
            'n_samples': n_samples,
            'n_features': n_features
        }
        
        os.makedirs(os.path.dirname(self.THRESHOLD_PATH) if os.path.dirname(self.THRESHOLD_PATH) else '.', exist_ok=True)
        with open(self.THRESHOLD_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Threshold config saved to {self.THRESHOLD_PATH}")

    
    def train(self, epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Execute full training pipeline.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            dict with training metrics
        """
        logger.info("=" * 50)
        logger.info("Starting Autoencoder Training Pipeline")
        logger.info("=" * 50)
        
        # Load data
        df = self.load_data()
        if df is None:
            raise FileNotFoundError("Training data not found")
        
        # Extract features
        X = df[self.FEATURES].fillna(0).values
        n_samples, n_features = X.shape
        
        logger.info(f"Training on {n_samples} samples with {n_features} features")
        logger.info(f"Features: {self.FEATURES[:5]}...")
        
        # Fit and save scaler
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)
        
        # Create and train autoencoder
        encoding_dim = max(7, n_features // 2)
        self.autoencoder = TransactionAutoencoder(
            input_dim=n_features,
            encoding_dim=encoding_dim,
            hidden_layers=[64, 32]
        )
        
        logger.info(f"Training Autoencoder (encoding_dim={encoding_dim})...")
        self.autoencoder.fit(X_scaled, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Save model
        os.makedirs(os.path.dirname(self.MODEL_PATH) if os.path.dirname(self.MODEL_PATH) else '.', exist_ok=True)
        self.autoencoder.save(self.MODEL_PATH)
        
        # Compute reconstruction errors and threshold
        errors = self.autoencoder.compute_reconstruction_error(X_scaled)
        threshold_config = self.compute_threshold(errors)
        self.save_threshold(threshold_config, n_samples, n_features)
        
        # Log results
        logger.info("=" * 50)
        logger.info("Training Complete!")
        logger.info(f"  Samples: {n_samples}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Threshold: {threshold_config['threshold']:.6f}")
        logger.info(f"  Mean Error: {threshold_config['mean']:.6f}")
        logger.info(f"  Std Error: {threshold_config['std']:.6f}")
        logger.info(f"  k: {threshold_config['k']}")
        logger.info("=" * 50)
        
        return {
            'threshold': threshold_config['threshold'],
            'mean_error': threshold_config['mean'],
            'std_error': threshold_config['std'],
            'k': self.k,
            'n_samples': n_samples,
            'n_features': n_features
        }
    
    def validate_saved_model(self, X_sample: np.ndarray, expected_errors: np.ndarray, 
                             tolerance: float = 0.01) -> bool:
        """
        Validate that saved model reproduces training metrics.
        
        Args:
            X_sample: Sample of scaled features to test
            expected_errors: Expected reconstruction errors
            tolerance: Maximum allowed relative difference (default 1%)
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating saved model...")
        
        # Load saved model
        loaded_ae = TransactionAutoencoder.load(self.MODEL_PATH)
        
        # Verify feature shape
        if loaded_ae.input_dim != X_sample.shape[1]:
            raise ValueError(
                f"Feature shape mismatch: expected {loaded_ae.input_dim}, got {X_sample.shape[1]}"
            )
        
        # Compute errors with loaded model
        loaded_errors = loaded_ae.compute_reconstruction_error(X_sample)
        
        # Compare metrics
        mean_diff = abs(np.mean(loaded_errors) - np.mean(expected_errors)) / (np.mean(expected_errors) + 1e-10)
        
        if mean_diff > tolerance:
            raise ValueError(
                f"Metrics mismatch: mean error differs by {mean_diff*100:.2f}% (tolerance: {tolerance*100}%)"
            )
        
        logger.info("Model validation PASSED")
        return True


def train_autoencoder():
    """Main entry point for training."""
    trainer = AutoencoderTrainer(k=3.0)
    metrics = trainer.train(epochs=100, batch_size=64)
    
    # Validate saved model
    df = trainer.load_data()
    X = df[trainer.FEATURES].fillna(0).values
    X_scaled = trainer.scaler.transform(X)
    
    # Use subset for validation
    sample_size = min(1000, len(X_scaled))
    X_sample = X_scaled[:sample_size]
    expected_errors = trainer.autoencoder.compute_reconstruction_error(X_sample)
    
    trainer.validate_saved_model(X_sample, expected_errors)
    
    return metrics


if __name__ == "__main__":
    train_autoencoder()
