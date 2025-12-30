# backend/hybrid_decision.py

import numpy as np
from backend.rule_engine import check_rule_violation


def make_decision(txn, user_stats, model, features, autoencoder=None):
    """
    Make fraud decision using Rule Engine, Isolation Forest, and Autoencoder.
    
    Decision Priority:
    1. Rule Engine - Hard blocks (velocity, monthly limits)
    2. Isolation Forest - ML anomaly detection
    3. Autoencoder - Behavioral anomaly detection
    
    Args:
        txn: Transaction data
        user_stats: User statistics
        model: Isolation Forest model
        features: Feature list for Isolation Forest
        autoencoder: Optional AutoencoderInference instance
        
    Returns:
        dict: {
            'is_fraud': bool,
            'reasons': list[str],
            'risk_score': float,
            'threshold': float,
            'ml_flag': bool,
            'ae_flag': bool,
            'ae_reconstruction_error': float or None
        }
    """
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

    # -------- RULE ENGINE (FIRST, HARD BLOCK) --------
    violated, rule_reasons, threshold = check_rule_violation(
        amount=txn["amount"],
        user_avg=user_stats["user_avg_amount"],
        user_std=user_stats["user_std_amount"],
        transfer_type=txn["transfer_type"],
        txn_count_10min=txn["txn_count_10min"],
        txn_count_1hour=txn["txn_count_1hour"],
        monthly_spending=user_stats["current_month_spending"],
    )

    result["threshold"] = threshold
    if violated:
        result["is_fraud"] = True
        result["reasons"].extend(rule_reasons)

    # -------- ML ENGINE (SECOND, EXPLANATORY) --------
    if model is not None:
        vec = np.array([[txn.get(f, 0) for f in features]])
        pred = model.predict(vec)[0]
        score = -model.decision_function(vec)[0]

        result["risk_score"] = score

        if pred == -1:
            result["ml_flag"] = True
            result["is_fraud"] = True
            result["reasons"].append(
                f"ML anomaly detected: abnormal behavior pattern (risk score {score:.4f})"
            )

    # -------- AUTOENCODER ENGINE (THIRD, BEHAVIORAL) --------
    if autoencoder is not None and autoencoder.is_available():
        # Build feature dict for autoencoder
        ae_features = {
            'transaction_amount': txn.get('amount', 0),
            'flag_amount': 1 if txn.get('transfer_type') == 'S' else 0,
            'transfer_type_encoded': {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}.get(txn.get('transfer_type', 'O'), 0),
            'transfer_type_risk': {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}.get(txn.get('transfer_type', 'O'), 0.5),
            'channel_encoded': 0,
            'deviation_from_avg': abs(txn.get('amount', 0) - user_stats.get('user_avg_amount', 0)),
            'amount_to_max_ratio': txn.get('amount', 0) / max(user_stats.get('user_max_amount', 1), 1),
            'rolling_std': user_stats.get('user_std_amount', 0),
            'hour': 12,  # Default, could be extracted from transaction time
            'day_of_week': 0,
            'is_weekend': 0,
            'is_night': 0,
            'user_avg_amount': user_stats.get('user_avg_amount', 0),
            'user_std_amount': user_stats.get('user_std_amount', 0),
            'user_max_amount': user_stats.get('user_max_amount', 0),
            'user_txn_frequency': user_stats.get('user_txn_frequency', 0),
            'intl_ratio': user_stats.get('user_international_ratio', 0),
            'time_since_last': txn.get('time_since_last_txn', 3600),
            'recent_burst': 1 if txn.get('time_since_last_txn', 3600) < 300 else 0,
            'txn_count_30s': 1,
            'txn_count_10min': txn.get('txn_count_10min', 1),
            'txn_count_1hour': txn.get('txn_count_1hour', 1),
            'hourly_total': txn.get('amount', 0),
            'hourly_count': 1,
            'daily_total': txn.get('amount', 0),
            'daily_count': 1,
        }
        
        ae_result = autoencoder.score_transaction(ae_features)
        
        if ae_result is not None:
            result["ae_reconstruction_error"] = ae_result['reconstruction_error']
            result["ae_threshold"] = ae_result['threshold']
            
            if ae_result['is_anomaly']:
                result["ae_flag"] = True
                result["is_fraud"] = True
                result["reasons"].append(ae_result['reason'])

    return result
