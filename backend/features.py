from typing import List, Dict, Any
from pydantic import BaseModel


class FeatureSchema(BaseModel):
    name: str
    description: str
    data_type: str
    required: bool = True
    default_value: Any = 0


# Isolation Forest Features (26 features)
ML_FEATURES: List[str] = [
    'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
    'channel_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_night',
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
    'deviation_from_avg', 'amount_to_max_ratio', 'intl_ratio',
    'user_high_risk_txn_ratio', 'user_multiple_accounts_flag', 'cross_account_transfer_ratio',
    'rolling_std', 'transaction_velocity', 'is_new_beneficiary',
    'beneficiary_txn_count_30d', 'beneficiary_risk_score', 'geo_anomaly_flag', 'recent_burst'
]

# Autoencoder Features (31 features)
AUTOENCODER_FEATURES: List[str] = [
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

# Feature Engineering - All Features Generated
ALL_ENGINEERED_FEATURES: List[str] = [
    # Basic transaction features
    'transaction_amount', 'flag_amount', 'transfer_type_encoded', 'transfer_type_risk',
    'channel_encoded',
    
    # Time features
    'hour', 'day_of_week', 'is_weekend', 'is_night',
    
    # Account-level statistics
    'user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency',
    'deviation_from_avg', 'amount_to_max_ratio', 'intl_ratio',
    'user_high_risk_txn_ratio', 'user_multiple_accounts_flag', 'cross_account_transfer_ratio',
    'rolling_std', 'transaction_velocity',
    
    # Velocity features
    'time_since_last', 'recent_burst', 'txn_count_30s', 'txn_count_10min', 'txn_count_1hour',
    'hourly_total', 'hourly_count', 'daily_total', 'daily_count', 'current_month_spending',
    
    # Beneficiary features
    'is_new_beneficiary', 'beneficiary_txn_count_30d', 'beneficiary_risk_score',
    
    # Geographic features
    'geo_anomaly_flag',
    
    # Placeholder features
    'applied_window', 'threshold_used', 'flag_reason'
]

# Feature Schemas with descriptions
FEATURE_SCHEMAS: Dict[str, FeatureSchema] = {
    # Transaction Features
    'transaction_amount': FeatureSchema(
        name='transaction_amount',
        description='Transaction amount in AED',
        data_type='float'
    ),
    'flag_amount': FeatureSchema(
        name='flag_amount',
        description='1 if overseas transfer (S), else 0',
        data_type='int'
    ),
    'transfer_type_encoded': FeatureSchema(
        name='transfer_type_encoded',
        description='Encoded transfer type (S=4, Q=3, L=2, I=1, O=0)',
        data_type='int'
    ),
    'transfer_type_risk': FeatureSchema(
        name='transfer_type_risk',
        description='Risk score for transfer type (0.0-1.0)',
        data_type='float'
    ),
    'channel_encoded': FeatureSchema(
        name='channel_encoded',
        description='Encoded channel ID',
        data_type='int'
    ),
    
    # Time Features
    'hour': FeatureSchema(
        name='hour',
        description='Transaction hour (0-23)',
        data_type='int',
        default_value=12
    ),
    'day_of_week': FeatureSchema(
        name='day_of_week',
        description='Day of week (0=Monday, 6=Sunday)',
        data_type='int',
        default_value=0
    ),
    'is_weekend': FeatureSchema(
        name='is_weekend',
        description='1 if weekend (Sat/Sun), else 0',
        data_type='int'
    ),
    'is_night': FeatureSchema(
        name='is_night',
        description='1 if night time (< 6 AM or >= 10 PM), else 0',
        data_type='int'
    ),
    
    # Account Statistics
    'user_avg_amount': FeatureSchema(
        name='user_avg_amount',
        description='Average transaction amount for this account',
        data_type='float'
    ),
    'user_std_amount': FeatureSchema(
        name='user_std_amount',
        description='Standard deviation of transaction amounts',
        data_type='float'
    ),
    'user_max_amount': FeatureSchema(
        name='user_max_amount',
        description='Maximum transaction amount for this account',
        data_type='float'
    ),
    'user_txn_frequency': FeatureSchema(
        name='user_txn_frequency',
        description='Total number of transactions for this account',
        data_type='int'
    ),
    
    # Velocity Features
    'txn_count_30s': FeatureSchema(
        name='txn_count_30s',
        description='Number of transactions in last 30 seconds',
        data_type='int',
        default_value=1
    ),
    'txn_count_10min': FeatureSchema(
        name='txn_count_10min',
        description='Number of transactions in last 10 minutes',
        data_type='int',
        default_value=1
    ),
    'txn_count_1hour': FeatureSchema(
        name='txn_count_1hour',
        description='Number of transactions in last 1 hour',
        data_type='int',
        default_value=1
    ),
    
    # Beneficiary Features
    'is_new_beneficiary': FeatureSchema(
        name='is_new_beneficiary',
        description='1 if first transaction to this beneficiary, else 0',
        data_type='int'
    ),
    'beneficiary_txn_count_30d': FeatureSchema(
        name='beneficiary_txn_count_30d',
        description='Total transactions to this beneficiary in 30 days',
        data_type='int'
    ),
    'beneficiary_risk_score': FeatureSchema(
        name='beneficiary_risk_score',
        description='Risk score for this beneficiary (0.0-1.0)',
        data_type='float',
        default_value=0.5
    ),
    
    # Geographic Features
    'geo_anomaly_flag': FeatureSchema(
        name='geo_anomaly_flag',
        description='1 if foreign country, 0 if local (UAE)',
        data_type='int'
    )
}


def get_ml_features() -> List[str]:
    return ML_FEATURES.copy()


def get_autoencoder_features() -> List[str]:
    return AUTOENCODER_FEATURES.copy()


def get_all_features() -> List[str]:
    return ALL_ENGINEERED_FEATURES.copy()


def validate_features(feature_dict: Dict[str, Any], feature_list: List[str]) -> Dict[str, Any]:
    validated = {}
    
    for feature_name in feature_list:
        if feature_name in feature_dict:
            validated[feature_name] = feature_dict[feature_name]
        elif feature_name in FEATURE_SCHEMAS:
            validated[feature_name] = FEATURE_SCHEMAS[feature_name].default_value
        else:
            validated[feature_name] = 0  # Fallback default
            
    return validated


def get_feature_vector(feature_dict: Dict[str, Any], feature_list: List[str]) -> List[float]:
    validated = validate_features(feature_dict, feature_list)
    return [float(validated.get(f, 0)) for f in feature_list]


def get_feature_info(feature_name: str) -> FeatureSchema:
    """Get information about a specific feature"""
    return FEATURE_SCHEMAS.get(feature_name, FeatureSchema(
        name=feature_name,
        description="Unknown feature",
        data_type="float"
    ))