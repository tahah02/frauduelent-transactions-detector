import os
import json
import logging
import numpy as np
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from backend.features import get_autoencoder_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionAutoencoder:
    
    def __init__(self, input_dim, encoding_dim=14, hidden_layers=None):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.model = None
        self._build_model()
    
    def _build_model(self):
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(x)
        
        x = encoded
        for units in reversed(self.hidden_layers):
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
        
        outputs = layers.Dense(self.input_dim, activation='linear')(x)
        
        self.model = Model(inputs, outputs, name='transaction_autoencoder')
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def fit(self, X, epochs=100, batch_size=32, validation_split=0.1, verbose=1):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        return self.model.fit(
            X, X, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, callbacks=[early_stopping], verbose=verbose
        )
    
    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def compute_reconstruction_error(self, X):
        reconstructed = self.predict(X)
        return np.mean(np.square(X - reconstructed), axis=1)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        loaded_model = keras.models.load_model(path)
        instance = cls.__new__(cls)
        instance.input_dim = loaded_model.input_shape[1]
        instance.model = loaded_model
        instance.encoding_dim = None
        instance.hidden_layers = None
        logger.info(f"Model loaded from {path}")
        return instance


class AutoencoderInference:
    
    MODEL_PATH = 'models/autoencoder.h5'
    SCALER_PATH = 'models/autoencoder_scaler.pkl'
    THRESHOLD_PATH = 'models/autoencoder_threshold.json'
    
    FEATURES = get_autoencoder_features()
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold_config = None
        self._loaded = False
    
    def load_artifacts(self):
        try:
            if not all(os.path.exists(p) for p in [self.MODEL_PATH, self.SCALER_PATH, self.THRESHOLD_PATH]):
                return False
            
            self.model = TransactionAutoencoder.load(self.MODEL_PATH)
            self.scaler = joblib.load(self.SCALER_PATH)
            with open(self.THRESHOLD_PATH, 'r') as f:
                self.threshold_config = json.load(f)
            
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            return False
    
    def is_available(self):
        return self._loaded and self.model is not None
    
    def score_transaction(self, features):
        if not self._loaded and not self.load_artifacts():
            return None
        
        missing = [f for f in self.FEATURES if f not in features]
        if missing:
            logger.warning(f"Missing features: {missing[:5]}...")
            return None
        
        try:
            feature_vector = np.array([[features.get(f, 0) for f in self.FEATURES]], dtype=np.float32)
            scaled_features = self.scaler.transform(feature_vector)
            error = float(self.model.compute_reconstruction_error(scaled_features)[0])
            
            if np.isnan(error) or np.isinf(error):
                error = 999.0
            
            threshold = self.threshold_config['threshold']
            is_anomaly = error > threshold
            
            reason = None
            if is_anomaly:
                reason = f"Autoencoder anomaly: reconstruction error {error:.4f} exceeds threshold {threshold:.4f}"
            
            return {
                'reconstruction_error': error,
                'threshold': threshold,
                'is_anomaly': is_anomaly,
                'reason': reason
            }
        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            return None
