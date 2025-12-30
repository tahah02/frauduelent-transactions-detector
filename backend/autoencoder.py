# backend/autoencoder.py
"""
Autoencoder-based anomaly detection for banking transactions.
Provides TransactionAutoencoder for training and AutoencoderInference for scoring.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class TransactionAutoencoder:
    """
    Keras-based Autoencoder for transaction anomaly detection.
    
    Architecture:
    - Input Layer: n_features (matches engineered features)
    - Encoder: Dense layers with decreasing dimensions
    - Bottleneck: Compressed representation
    - Decoder: Dense layers with increasing dimensions
    - Output Layer: n_features (reconstruction)
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 14, 
                 hidden_layers: Optional[List[int]] = None):
        """
        Initialize Autoencoder architecture.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Size of the bottleneck layer (default: 14)
            hidden_layers: Optional custom layer dimensions
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.model: Optional[Model] = None
        self._build_model()
    
    def _build_model(self) -> None:
        """Build and compile the Keras model with MSE loss."""
        # Encoder
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(x)
        
        # Decoder (mirror of encoder)
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(self.input_dim, activation='linear')(x)
        
        self.model = Model(inputs, outputs, name='transaction_autoencoder')
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
            validation_split: float = 0.1, verbose: int = 1) -> keras.callbacks.History:
        """
        Train the Autoencoder on scaled features.
        
        Args:
            X: Scaled feature array of shape (n_samples, n_features)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history object
        """
        if self.model is None:
            raise ValueError("Model not built. Call _build_model() first.")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        
        history = self.model.fit(
            X, X,  # Autoencoder reconstructs input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input features.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            Reconstructed features of same shape
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        return self.model.predict(X, verbose=0)
    
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute MSE between input and reconstruction for each sample.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            
        Returns:
            Array of reconstruction errors of shape (n_samples,)
        """
        reconstructed = self.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        return mse
    
    def save(self, path: str) -> None:
        """
        Save model to .h5 format.
        
        Args:
            path: File path for saving (should end with .h5)
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TransactionAutoencoder':
        """
        Load model from .h5 format.
        
        Args:
            path: File path to load from
            
        Returns:
            TransactionAutoencoder instance with loaded model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        loaded_model = keras.models.load_model(path)
        input_dim = loaded_model.input_shape[1]
        
        # Create instance without building new model
        instance = cls.__new__(cls)
        instance.input_dim = input_dim
        instance.model = loaded_model
        instance.encoding_dim = None  # Unknown from loaded model
        instance.hidden_layers = None
        
        logger.info(f"Model loaded from {path}")
        return instance



class AutoencoderInference:
    """
    Inference module for Autoencoder-based anomaly detection.
    
    Handles:
    - Loading model, scaler, and threshold
    - Scaling input features
    - Computing reconstruction error
    - Generating human-readable anomaly reasons
    """
    
    # Configuration paths
    MODEL_PATH = 'models/autoencoder.h5'
    SCALER_PATH = 'models/autoencoder_scaler.pkl'
    THRESHOLD_PATH = 'models/autoencoder_threshold.json'
    
    # Feature list (same as training)
    FEATURES = [
        'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
        'channel_encoded', 'deviation_from_avg', 'amount_to_max_ratio', 'rolling_std',
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
        'intl_ratio', 'time_since_last', 'recent_burst',
        'txn_count_30s', 'txn_count_10min', 'txn_count_1hour',
        'hourly_total', 'hourly_count', 'daily_total', 'daily_count'
    ]
    
    def __init__(self):
        """Initialize inference module (lazy loading of artifacts)."""
        self.model: Optional[TransactionAutoencoder] = None
        self.scaler = None
        self.threshold_config: Optional[Dict[str, Any]] = None
        self._loaded = False
    
    def load_artifacts(self) -> bool:
        """
        Load model, scaler, and threshold.
        
        Returns:
            True if all artifacts loaded successfully, False otherwise
        """
        try:
            # Check all files exist
            if not os.path.exists(self.MODEL_PATH):
                logger.warning(f"Autoencoder model not found: {self.MODEL_PATH}")
                return False
            
            if not os.path.exists(self.SCALER_PATH):
                logger.warning(f"Autoencoder scaler not found: {self.SCALER_PATH}")
                return False
            
            if not os.path.exists(self.THRESHOLD_PATH):
                logger.warning(f"Autoencoder threshold not found: {self.THRESHOLD_PATH}")
                return False
            
            # Load model
            self.model = TransactionAutoencoder.load(self.MODEL_PATH)
            logger.info(f"Loaded Autoencoder model from {self.MODEL_PATH}")
            
            # Load scaler
            self.scaler = joblib.load(self.SCALER_PATH)
            logger.info(f"Loaded scaler from {self.SCALER_PATH}")
            
            # Load threshold config
            with open(self.THRESHOLD_PATH, 'r') as f:
                self.threshold_config = json.load(f)
            logger.info(f"Loaded threshold config from {self.THRESHOLD_PATH}")
            
            self._loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading Autoencoder artifacts: {e}")
            self._loaded = False
            return False
    
    def is_available(self) -> bool:
        """Check if Autoencoder inference is available."""
        return self._loaded and self.model is not None

    
    def _validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate that input features match expected schema.
        
        Args:
            features: dict of feature name -> value
            
        Returns:
            True if valid, False otherwise
        """
        missing = [f for f in self.FEATURES if f not in features]
        if missing:
            logger.warning(f"Missing features for Autoencoder: {missing[:5]}...")
            return False
        return True
    
    def _handle_invalid_error(self, error: float) -> tuple:
        """
        Handle NaN/inf reconstruction errors.
        
        Args:
            error: The reconstruction error value
            
        Returns:
            tuple: (clipped_error, reason_string)
        """
        if np.isnan(error) or np.isinf(error):
            logger.error(f"Invalid reconstruction error detected: {error}")
            # Clip to a high value and flag as anomaly
            clipped_error = 999.0
            reason = f"Autoencoder anomaly: invalid reconstruction error (clipped to {clipped_error})"
            return clipped_error, reason
        return error, None
    
    def score_transaction(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Score a single transaction.
        
        Args:
            features: dict of feature name -> value
            
        Returns:
            dict: {
                'reconstruction_error': float,
                'threshold': float,
                'is_anomaly': bool,
                'reason': str or None
            }
            Returns None if inference is not available or features are invalid.
        """
        # Check if loaded
        if not self._loaded:
            if not self.load_artifacts():
                return None
        
        # Validate features
        if not self._validate_features(features):
            logger.warning("Feature validation failed, skipping Autoencoder scoring")
            return None
        
        try:
            # Extract feature vector
            feature_vector = np.array([[features.get(f, 0) for f in self.FEATURES]], dtype=np.float32)
            
            # Scale features
            scaled_features = self.scaler.transform(feature_vector)
            
            # Compute reconstruction error
            error = float(self.model.compute_reconstruction_error(scaled_features)[0])
            
            # Handle invalid errors
            error, invalid_reason = self._handle_invalid_error(error)
            
            # Get threshold
            threshold = self.threshold_config['threshold']
            
            # Determine if anomaly
            is_anomaly = error > threshold
            
            # Generate reason
            if invalid_reason:
                reason = invalid_reason
            elif is_anomaly:
                reason = f"Autoencoder anomaly: reconstruction error {error:.4f} exceeds threshold {threshold:.4f}"
            else:
                reason = None
            
            return {
                'reconstruction_error': error,
                'threshold': threshold,
                'is_anomaly': is_anomaly,
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error scoring transaction with Autoencoder: {e}")
            return None
