import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from backend.config import get_config
from backend.features import get_ml_features
from backend.exceptions import ModelLoadError, PredictionError, FileNotFoundError
from backend.logging_config import get_logger

logger = get_logger('model')
config = get_config()

FEATURES = get_ml_features()


def load_data():
    """Load feature data with error handling"""
    try:
        if not os.path.exists(config.FEATURED_CSV_PATH):
            raise FileNotFoundError(
                f"Featured dataset not found: {config.FEATURED_CSV_PATH}",
                error_code="FILE_NOT_FOUND"
            )
        
        df = pd.read_csv(config.FEATURED_CSV_PATH)
        logger.info(f"Successfully loaded {len(df)} rows from {config.FEATURED_CSV_PATH}")
        return df
        
    except pd.errors.EmptyDataError:
        raise FileNotFoundError(
            f"Featured dataset is empty: {config.FEATURED_CSV_PATH}",
            error_code="FILE_NOT_FOUND"
        )
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise FileNotFoundError(
            f"Failed to load featured dataset: {str(e)}",
            error_code="FILE_NOT_FOUND"
        )
    
def train_model():
    """Train Isolation Forest model with comprehensive error handling"""
    try:
        logger.info("=" * 50)
        logger.info("ISOLATION FOREST MODEL TRAINING")
        logger.info("=" * 50)
        
        df = load_data()
        
        # Validate features
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise PredictionError(
                f"Missing required features: {missing[:5]}{'...' if len(missing) > 5 else ''}",
                error_code="FEATURE_MISMATCH",
                context={'missing_features': missing}
            )
        
        X = df[FEATURES].fillna(0)
        logger.info(f"Training data: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Validate data quality
        if X.shape[0] < 100:
            logger.warning(f"Small dataset: only {X.shape[0]} rows")
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Model training
        clf = IsolationForest(
            n_estimators=config.ISOLATION_FOREST_N_ESTIMATORS, 
            contamination=config.ISOLATION_FOREST_CONTAMINATION, 
            random_state=config.ISOLATION_FOREST_RANDOM_STATE, 
            n_jobs=-1
        )
        clf.fit(X_scaled)
        
        # Save models
        os.makedirs(os.path.dirname(config.ISOLATION_FOREST_MODEL_PATH), exist_ok=True)
        joblib.dump({'model': clf, 'features': FEATURES}, config.ISOLATION_FOREST_MODEL_PATH)
        joblib.dump(scaler, config.ISOLATION_FOREST_SCALER_PATH)
        
        logger.info(f"Model saved: {config.ISOLATION_FOREST_MODEL_PATH}")
        logger.info(f"Scaler saved: {config.ISOLATION_FOREST_SCALER_PATH}")
        logger.info("=" * 50)
        
        return clf
        
    except (FileNotFoundError, PredictionError) as e:
        # Re-raise known errors
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise ModelLoadError(
            f"Model training failed: {str(e)}",
            error_code="MODEL_LOAD_FAILED",
            context={'error_type': type(e).__name__}
        )


def load_model():
    """Load trained model with comprehensive error handling and fallback"""
    try:
        if not os.path.exists(config.ISOLATION_FOREST_MODEL_PATH):
            logger.warning("Model file not found, training new model...")
            train_model()
            return load_model()  # Recursive call after training
        
        # Load model
        try:
            data = joblib.load(config.ISOLATION_FOREST_MODEL_PATH)
            logger.info("Successfully loaded isolation forest model")
        except Exception as e:
            logger.error(f"Error loading model file: {e}")
            raise ModelLoadError(
                f"Failed to load model from {config.ISOLATION_FOREST_MODEL_PATH}: {str(e)}",
                error_code="MODEL_LOAD_FAILED"
            )
        
        # Load scaler
        scaler = None
        if os.path.exists(config.ISOLATION_FOREST_SCALER_PATH):
            try:
                scaler = joblib.load(config.ISOLATION_FOREST_SCALER_PATH)
                logger.info("Successfully loaded feature scaler")
            except Exception as e:
                logger.warning(f"Error loading scaler, will use default: {e}")
                scaler = None
        else:
            logger.warning("Scaler file not found, predictions may be inaccurate")
        
        # Extract model and features
        if isinstance(data, dict):
            model = data.get('model')
            features = data.get('features', FEATURES)
        else:
            model = data
            features = FEATURES
            
        if model is None:
            raise ModelLoadError(
                "Model object is None after loading",
                error_code="MODEL_LOAD_FAILED"
            )
            
        logger.info(f"Model loaded with {len(features)} features")
        return model, features, scaler
        
    except (ModelLoadError, FileNotFoundError) as e:
        # Re-raise known errors
        raise e
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        # Try to train new model as fallback
        try:
            logger.info("Attempting to train new model as fallback...")
            train_model()
            return load_model()
        except Exception as train_error:
            logger.error(f"Fallback training also failed: {train_error}")
            raise ModelLoadError(
                f"Model loading and fallback training both failed: {str(e)}",
                error_code="MODEL_LOAD_FAILED",
                context={'original_error': str(e), 'training_error': str(train_error)}
            )


if __name__ == "__main__":
    train_model()
