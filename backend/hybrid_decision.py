import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from backend.rule_engine import check_rule_violation
from backend.config import get_config
from backend.features import get_feature_vector, get_ml_features, validate_features
from backend.logging_config import get_logger, log_model_performance
import time

logger = get_logger('decision')
config = get_config()


def make_decision(transaction_data: Dict[str, Any], user_statistics: Dict[str, Any], 
                 ml_model: Any, feature_names: List[str], autoencoder_model: Any = None, 
                 feature_scaler: Any = None) -> Dict[str, Any]:
    """Make fraud decision using triple-layer approach with performance logging"""
    decision_result = {
        "is_fraud": False,
        "reasons": [],
        "risk_score": 0.0,
        "threshold": 0.0,
        "ml_flag": False,
        "ae_flag": False,
        "ae_reconstruction_error": None,
        "ae_threshold": None,
    }

    logger.debug(f"Starting fraud decision for amount: {transaction_data.get('amount', 0)}")

    # Rule Engine
    try:
        rule_violated, rule_violation_reasons, rule_threshold = check_rule_violation(
            amount=transaction_data.get("amount", 0),
            user_avg=user_statistics.get("user_avg_amount", 0),
            user_std=user_statistics.get("user_std_amount", 0),
            transfer_type=transaction_data.get("transfer_type", "O"),
            txn_count_10min=transaction_data.get("txn_count_10min", 1),
            txn_count_1hour=transaction_data.get("txn_count_1hour", 1),
            monthly_spending=user_statistics.get("current_month_spending", 0),
        )

        decision_result["threshold"] = rule_threshold
        if rule_violated:
            decision_result["is_fraud"] = True
            decision_result["reasons"].extend(rule_violation_reasons)
            logger.info(f"Rule engine flagged transaction: {len(rule_violation_reasons)} violations")
        else:
            logger.debug("Rule engine passed")
            
    except Exception as e:
        logger.error(f"Rule engine error: {e}")
        # Continue without rule engine

    # Isolation Forest
    if ml_model is not None and feature_names is not None:
        try:
            ml_prediction_start_time = time.time()
            ml_feature_dictionary = build_ml_features(transaction_data, user_statistics)
            ml_feature_vector = get_feature_vector(ml_feature_dictionary, feature_names)
            model_input_vector = np.array([ml_feature_vector])
            
            if feature_scaler is not None:
                model_input_vector = feature_scaler.transform(model_input_vector)
            
            model_prediction = ml_model.predict(model_input_vector)[0]
            anomaly_score = -ml_model.decision_function(model_input_vector)[0]
            decision_result["risk_score"] = float(anomaly_score)
            
            ml_prediction_time = (time.time() - ml_prediction_start_time) * 1000
            log_model_performance("isolation_forest", ml_prediction_time, True)

            if model_prediction == -1:
                decision_result["ml_flag"] = True
                decision_result["is_fraud"] = True
                decision_result["reasons"].append(f"ML anomaly detected (risk score {anomaly_score:.4f})")
                logger.info(f"ML model flagged transaction: risk score {anomaly_score:.4f}")
            else:
                logger.debug(f"ML model passed: risk score {anomaly_score:.4f}")
                
        except Exception as e:
            logger.error(f"ML model prediction error: {e}")
            log_model_performance("isolation_forest", 0, False)
    else:
        logger.warning("ML model not available")

    # Autoencoder
    if autoencoder_model is not None and autoencoder_model.is_available():
        try:
            ae_prediction_start_time = time.time()
            autoencoder_features = build_ae_features(transaction_data, user_statistics)
            autoencoder_result = autoencoder_model.score_transaction(autoencoder_features)
            
            ae_prediction_time = (time.time() - ae_prediction_start_time) * 1000
            
            if autoencoder_result is not None:
                decision_result["ae_reconstruction_error"] = autoencoder_result['reconstruction_error']
                decision_result["ae_threshold"] = autoencoder_result['threshold']
                
                log_model_performance("autoencoder", ae_prediction_time, True)
                
                if autoencoder_result['is_anomaly']:
                    decision_result["ae_flag"] = True
                    decision_result["is_fraud"] = True
                    decision_result["reasons"].append(autoencoder_result['reason'])
                    logger.info(f"Autoencoder flagged transaction: error {autoencoder_result['reconstruction_error']:.4f}")
                else:
                    logger.debug(f"Autoencoder passed: error {autoencoder_result['reconstruction_error']:.4f}")
            else:
                log_model_performance("autoencoder", ae_prediction_time, False)
                
        except Exception as e:
            logger.error(f"Autoencoder prediction error: {e}")
            log_model_performance("autoencoder", 0, False)
    else:
        logger.debug("Autoencoder not available")

    logger.debug(f"Decision complete: fraud={decision_result['is_fraud']}, reasons={len(decision_result['reasons'])}")
    return decision_result


def build_ml_features(txn, user_stats):
    transfer_type = txn.get('transfer_type', 'O')
    amount = txn.get('amount', 0)
    user_avg = user_stats.get('user_avg_amount', 0)
    user_max = user_stats.get('user_max_amount', 1)
    
    feature_values = {
        'transaction_amount': amount,
        'flag_amount': 1 if transfer_type == 'S' else 0,
        'transfer_type_encoded': config.TRANSFER_TYPE_ENCODED.get(transfer_type, 0),
        'transfer_type_risk': config.TRANSFER_TYPE_RISK.get(transfer_type, 0.5),
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
    
    return feature_values


def build_ae_features(txn, user_stats):
    transfer_type = txn.get('transfer_type', 'O')
    amount = txn.get('amount', 0)
    user_avg = user_stats.get('user_avg_amount', 0)
    user_max = user_stats.get('user_max_amount', 1)
    time_since_last = txn.get('time_since_last', 3600)
    
    return {
        'transaction_amount': amount,
        'flag_amount': 1 if transfer_type == 'S' else 0,
        'transfer_type_encoded': config.TRANSFER_TYPE_ENCODED.get(transfer_type, 0),
        'transfer_type_risk': config.TRANSFER_TYPE_RISK.get(transfer_type, 0.5),
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
