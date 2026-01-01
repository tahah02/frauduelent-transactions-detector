"""
Configuration settings for the fraud detection system.
Centralizes all constants and settings in one place.
"""
import os
from typing import Dict, Any


class Config:
    """Main configuration class for fraud detection system."""
    
    # Transfer Type Settings
    TRANSFER_TYPE_ENCODED: Dict[str, int] = {
        'S': 4,  # Overseas
        'I': 1,  # Ajman Local
        'L': 2,  # UAE Local
        'Q': 3,  # Quick Transfer
        'O': 0   # Own Account
    }
    
    TRANSFER_TYPE_RISK: Dict[str, float] = {
        'S': 0.9,  # HIGH risk
        'I': 0.1,  # LOW risk
        'L': 0.2,  # LOW risk
        'Q': 0.5,  # MEDIUM risk
        'O': 0.0   # LOWEST risk
    }
    
    # Rule Engine Settings
    TRANSFER_MULTIPLIERS: Dict[str, float] = {
        'S': 2.0,  # Overseas - most restrictive
        'Q': 2.5,  # Quick Transfer
        'L': 3.0,  # UAE Local
        'I': 3.5,  # Ajman Local
        'O': 4.0   # Own Account - most lenient
    }
    
    TRANSFER_MIN_FLOORS: Dict[str, float] = {
        'S': 5000,  # Overseas
        'Q': 3000,  # Quick Transfer
        'L': 2000,  # UAE Local
        'I': 1500,  # Ajman Local
        'O': 1000   # Own Account
    }
    
    # Velocity Limits
    MAX_VELOCITY_10MIN: int = 5
    MAX_VELOCITY_1HOUR: int = 15
    MAX_VELOCITY_30SEC: int = 2
    
    # Session Management
    SESSION_CLEANUP_HOURS: int = 24
    PENDING_TRANSACTION_TIMEOUT_HOURS: int = 2
    
    # File Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    CLEAN_CSV_PATH: str = os.path.join(DATA_DIR, "Clean.csv")
    FEATURED_CSV_PATH: str = os.path.join(DATA_DIR, "featured_dataset.csv")
    TRANSACTION_HISTORY_PATH: str = "transaction_history.csv"
    
    # Model Paths
    ISOLATION_FOREST_MODEL_PATH: str = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    ISOLATION_FOREST_SCALER_PATH: str = os.path.join(MODELS_DIR, "isolation_forest_scaler.pkl")
    AUTOENCODER_MODEL_PATH: str = os.path.join(MODELS_DIR, "autoencoder.h5")
    AUTOENCODER_SCALER_PATH: str = os.path.join(MODELS_DIR, "autoencoder_scaler.pkl")
    AUTOENCODER_THRESHOLD_PATH: str = os.path.join(MODELS_DIR, "autoencoder_threshold.json")
    
    # ML Model Settings
    ISOLATION_FOREST_CONTAMINATION: float = 0.1
    ISOLATION_FOREST_N_ESTIMATORS: int = 100
    ISOLATION_FOREST_RANDOM_STATE: int = 42
    
    # Autoencoder Settings
    AUTOENCODER_ENCODING_DIM: int = 14
    AUTOENCODER_HIDDEN_LAYERS: list = [64, 32]
    AUTOENCODER_EPOCHS: int = 100
    AUTOENCODER_BATCH_SIZE: int = 32
    AUTOENCODER_VALIDATION_SPLIT: float = 0.1
    
    # Geographic Settings
    LOCAL_COUNTRIES: list = ['UAE', 'AE', 'United Arab Emirates', 'AJMAN']
    
    # API Settings
    API_TITLE: str = "Transaction Fraud Detection API"
    API_VERSION: str = "1.0.0"
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance Settings
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    MAX_CONCURRENT_REQUESTS: int = 100
    
    @classmethod
    def get_transfer_type_info(cls, transfer_type: str) -> Dict[str, Any]:
        """Get all information for a transfer type."""
        transfer_type = transfer_type.upper()
        return {
            'encoded': cls.TRANSFER_TYPE_ENCODED.get(transfer_type, 0),
            'risk': cls.TRANSFER_TYPE_RISK.get(transfer_type, 0.5),
            'multiplier': cls.TRANSFER_MULTIPLIERS.get(transfer_type, 3.0),
            'min_floor': cls.TRANSFER_MIN_FLOORS.get(transfer_type, 2000)
        }
    
    @classmethod
    def is_local_country(cls, country: str) -> bool:
        """Check if country is considered local."""
        return country.upper() in [c.upper() for c in cls.LOCAL_COUNTRIES]


# Environment-specific overrides
class DevelopmentConfig(Config):
    """Development environment configuration."""
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production environment configuration."""
    LOG_LEVEL = "WARNING"
    CACHE_TTL_SECONDS = 600  # 10 minutes in production


class TestConfig(Config):
    """Test environment configuration."""
    LOG_LEVEL = "DEBUG"
    TRANSACTION_HISTORY_PATH = "test_transaction_history.csv"
    CACHE_TTL_SECONDS = 1  # Short cache for testing


# Configuration factory
def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()