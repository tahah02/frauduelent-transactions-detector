import pandas as pd
import numpy as np
from backend.rule_engine import check_rule_violation


TRANSFER_TYPE_ENCODED = {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}
TRANSFER_TYPE_RISK = {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}


def make_decision(txn, user_stats, model, features, autoencoder=None, scaler=None):
    result = {
        "is_fraud": False,
        "reasons": [],
        "risk_score": 0.0,
        "threshold": 0.0,
        "ml_flag": False,
        "ae_flag": False,
        "ae_reconstruction_error": None,
        "ae_threshold": None,
    }

    # Rule Engine
    violated, rule_reasons, threshold = check_rule_violation(
        amount=txn.get("amount", 0),
        user_avg=user_stats.get("user_avg_amount", 0),
        user_std=user_stats.get("user_std_amount", 0),
        transfer_type=txn.get("transfer_type", "O"),
        txn_count_10min=txn.get("txn_count_10min", 1),
        txn_count_1hour=txn.get("txn_count_1hour", 1),
        monthly_spending=user_stats.get("current_month_spending", 0),
    )

    result["threshold"] = threshold
    if violated:
        result["is_fraud"] = True
        result["reasons"].extend(rule_reasons)

    # Isolation Forest
    if model is not None and features is not None:
        ml_features = build_ml_features(txn, user_stats, features)
        vec = np.array([ml_features])
        
        if scaler is not None:
            vec = scaler.transform(vec)
        
        pred = model.predict(vec)[0]
        score = -model.decision_function(vec)[0]
        result["risk_score"] = float(score)

        if pred == -1:
            result["ml_flag"] = True
            result["is_fraud"] = True
            result["reasons"].append(f"ML anomaly detected (risk score {score:.4f})")

    # Autoencoder
    if autoencoder is not None and autoencoder.is_available():
        ae_features = build_ae_features(txn, user_stats)
        ae_result = autoencoder.score_transaction(ae_features)
        
        if ae_result is not None:
            result["ae_reconstruction_error"] = ae_result['reconstruction_error']
            result["ae_threshold"] = ae_result['threshold']
            
            if ae_result['is_anomaly']:
                result["ae_flag"] = True
                result["is_fraud"] = True
                result["reasons"].append(ae_result['reason'])

    return result


def build_ml_features(txn, user_stats, features):
    transfer_type = txn.get('transfer_type', 'O')
    amount = txn.get('amount', 0)
    user_avg = user_stats.get('user_avg_amount', 0)
    user_max = user_stats.get('user_max_amount', 1)
    
    feature_values = {
        'transaction_amount': amount,
        'flag_amount': 1 if transfer_type == 'S' else 0,
        'transfer_type_encoded': TRANSFER_TYPE_ENCODED.get(transfer_type, 0),
        'transfer_type_risk': TRANSFER_TYPE_RISK.get(transfer_type, 0.5),
        'channel_encoded': txn.get('channel_encoded', 0),
        'hour': txn.get('hour', 12),
        'day_of_week': txn.get('day_of_week', 0),
        'is_weekend': txn.get('is_weekend', 0),
        'is_night': txn.get('is_night', 0),
        'user_avg_amount': user_avg,
        'user_std_amount': user_stats.get('user_std_amount', 0),
        'user_max_amount': user_max,
        'user_txn_frequency': user_stats.get('user_txn_frequency', 0),
        'deviation_from_avg': abs(amount - user_avg),
        'amount_to_max_ratio': amount / max(user_max, 1),
        'intl_ratio': user_stats.get('intl_ratio', 0),
        'user_high_risk_txn_ratio': user_stats.get('user_high_risk_txn_ratio', 0),
        'user_multiple_accounts_flag': user_stats.get('user_multiple_accounts_flag', 0),
        'cross_account_transfer_ratio': user_stats.get('cross_account_transfer_ratio', 0),
        'rolling_std': user_stats.get('rolling_std', 0),
        'transaction_velocity': user_stats.get('transaction_velocity', 1),
        'is_new_beneficiary': txn.get('is_new_beneficiary', 0),
        'beneficiary_txn_count_30d': txn.get('beneficiary_txn_count_30d', 0),
        'beneficiary_risk_score': txn.get('beneficiary_risk_score', 0.5),
        'geo_anomaly_flag': txn.get('geo_anomaly_flag', 0),
        'recent_burst': 1 if txn.get('time_since_last', 3600) < 300 else 0,
    }
    
    return [feature_values.get(f, 0) for f in features]


def build_ae_features(txn, user_stats):
    transfer_type = txn.get('transfer_type', 'O')
    amount = txn.get('amount', 0)
    user_avg = user_stats.get('user_avg_amount', 0)
    user_max = user_stats.get('user_max_amount', 1)
    time_since_last = txn.get('time_since_last', 3600)
    
    return {
        'transaction_amount': amount,
        'flag_amount': 1 if transfer_type == 'S' else 0,
        'transfer_type_encoded': TRANSFER_TYPE_ENCODED.get(transfer_type, 0),
        'transfer_type_risk': TRANSFER_TYPE_RISK.get(transfer_type, 0.5),
        'channel_encoded': txn.get('channel_encoded', 0),
        'deviation_from_avg': abs(amount - user_avg),
        'amount_to_max_ratio': amount / max(user_max, 1),
        'rolling_std': user_stats.get('rolling_std', 0),
        'hour': txn.get('hour', 12),
        'day_of_week': txn.get('day_of_week', 0),
        'is_weekend': txn.get('is_weekend', 0),
        'is_night': txn.get('is_night', 0),
        'user_avg_amount': user_avg,
        'user_std_amount': user_stats.get('user_std_amount', 0),
        'user_max_amount': user_max,
        'user_txn_frequency': user_stats.get('user_txn_frequency', 0),
        'intl_ratio': user_stats.get('intl_ratio', 0),
        'user_high_risk_txn_ratio': user_stats.get('user_high_risk_txn_ratio', 0),
        'user_multiple_accounts_flag': user_stats.get('user_multiple_accounts_flag', 0),
        'cross_account_transfer_ratio': user_stats.get('cross_account_transfer_ratio', 0),
        'time_since_last': time_since_last,
        'recent_burst': 1 if time_since_last < 300 else 0,
        'txn_count_30s': txn.get('txn_count_30s', 1),
        'txn_count_10min': txn.get('txn_count_10min', 1),
        'txn_count_1hour': txn.get('txn_count_1hour', 1),
        'transaction_velocity': user_stats.get('transaction_velocity', 1),
        'is_new_beneficiary': txn.get('is_new_beneficiary', 0),
        'beneficiary_txn_count_30d': txn.get('beneficiary_txn_count_30d', 0),
        'beneficiary_risk_score': txn.get('beneficiary_risk_score', 0.5),
        'geo_anomaly_flag': txn.get('geo_anomaly_flag', 0),
        'current_month_spending': user_stats.get('current_month_spending', 0),
    }
