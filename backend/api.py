import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
import time
import asyncio

from backend.model import load_model
from backend.autoencoder import AutoencoderInference
from backend.hybrid_decision import make_decision
from backend.config import get_config
from backend.exceptions import (
    ModelLoadError, AccountNotFoundError, InvalidTransactionError,
    DataValidationError, create_error_response
)
from backend.logging_config import (
    get_logger, set_correlation_id, log_transaction_start, 
    log_transaction_result, log_system_health
)
from backend.cache import (
    get_cached_account_stats, set_cached_account_stats,
    get_cached_beneficiary_stats, set_cached_beneficiary_stats,
    velocity_tracker, get_cache_stats, cleanup_all_caches
)

logger = get_logger('api')
config = get_config()
app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)

# Add middleware for correlation ID
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Add correlation ID to each request"""
    correlation_id = set_correlation_id()
    
    # Log request start
    logger.info(f"Request started: {request.method} {request.url.path}")
    
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # Log request completion
    logger.info(f"Request completed: {response.status_code} in {process_time:.2f}ms")
    
    # Add correlation ID to response headers
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response


# Global variables
model, features, scaler = None, None, None
autoencoder = None
df_features = None
session_transactions = {}
pending_transactions = {}
TRANSACTION_HISTORY_FILE = config.TRANSACTION_HISTORY_PATH


class TransactionRequest(BaseModel):
    customer_id: float
    account_no: float
    amount: float
    transfer_type: str
    ben_id: Optional[float] = None
    bank_country: Optional[str] = "UAE"


class TransactionResponse(BaseModel):
    status: str
    message: str
    txn_id: Optional[str] = None  
    risk_score: float
    risk_interpretation: str  
    threshold: float
    transfer_type: str  
    applied_limit: float  
    reasons: list
    flags: dict


@app.on_event("startup")
async def startup_event():
    """Startup tasks including model loading and background tasks"""
    load_models()
    
    # Start background cache cleanup task
    asyncio.create_task(periodic_cache_cleanup())


async def periodic_cache_cleanup():
    """Background task for periodic cache cleanup"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            cleanup_stats = cleanup_all_caches()
            total_cleaned = sum(cleanup_stats.values())
            if total_cleaned > 0:
                logger.info(f"Periodic cleanup: {total_cleaned} entries removed")
        except Exception as e:
            logger.error(f"Error in periodic cache cleanup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutting down...")
    cleanup_all_caches()


def load_models():
    """Load models with comprehensive error handling"""
    global model, features, scaler, autoencoder, df_features
    
    logger.info("=== FRAUD DETECTION SYSTEM STARTUP ===")
    
    try:
        # Load ML models
        try:
            model, features, scaler = load_model()
            log_system_health("isolation_forest", "loaded", {"features": len(features)})
        except ModelLoadError as e:
            logger.error(f"Failed to load Isolation Forest: {e}")
            model, features, scaler = None, None, None
            log_system_health("isolation_forest", "failed", {"error": str(e)})
        
        # Load autoencoder
        try:
            autoencoder = AutoencoderInference()
            if autoencoder.load_artifacts():
                log_system_health("autoencoder", "loaded")
            else:
                logger.warning("Autoencoder artifacts not available")
                autoencoder = None
                log_system_health("autoencoder", "unavailable")
        except Exception as e:
            logger.error(f"Failed to load Autoencoder: {e}")
            autoencoder = None
            log_system_health("autoencoder", "failed", {"error": str(e)})
        
        # Load feature data
        try:
            df_features = pd.read_csv(config.FEATURED_CSV_PATH)
            log_system_health("feature_data", "loaded", {"records": len(df_features)})
        except Exception as e:
            logger.error(f"Failed to load feature data: {e}")
            df_features = pd.DataFrame()
            log_system_health("feature_data", "failed", {"error": str(e)})
        
        # System health summary
        models_loaded = sum([
            model is not None,
            autoencoder is not None,
            not df_features.empty
        ])
        
        if models_loaded == 0:
            logger.critical("No models loaded successfully - system will have limited functionality")
            log_system_health("system", "degraded", {"components_loaded": f"{models_loaded}/3"})
        else:
            logger.info(f"System startup complete: {models_loaded}/3 components loaded")
            log_system_health("system", "operational", {"components_loaded": f"{models_loaded}/3"})
            
    except Exception as e:
        logger.critical(f"Critical error during startup: {e}")
        log_system_health("system", "critical_error", {"error": str(e)})


def get_account_stats(customer_id: float, account_no: float) -> Dict[str, Any]:
    """Get account statistics with caching and error handling"""
    try:
        # Validate inputs
        if customer_id <= 0:
            raise InvalidTransactionError(
                "Invalid customer ID",
                error_code="INVALID_TRANSACTION",
                context={'customer_id': customer_id}
            )
        
        if account_no <= 0:
            raise InvalidTransactionError(
                "Invalid account number",
                error_code="INVALID_TRANSACTION", 
                context={'account_no': account_no}
            )
        
        # Try cache first
        cached_account_stats = get_cached_account_stats(customer_id, account_no)
        if cached_account_stats is not None:
            logger.debug(f"Account stats cache hit for {customer_id}/{account_no}")
            return cached_account_stats
        
        # Check if data is available
        if df_features.empty:
            logger.warning("Feature data not available, using defaults")
            return get_default_account_stats()
        
        # Query account data
        account_data_df = df_features[
            (df_features['CustomerId'] == customer_id) & 
            (df_features['FromAccountNo'] == account_no)
        ]
        
        if account_data_df.empty:
            raise AccountNotFoundError(
                f"Account not found: Customer {customer_id}, Account {account_no}",
                error_code="ACCOUNT_NOT_FOUND",
                context={'customer_id': customer_id, 'account_no': account_no}
            )
        
        latest_record = account_data_df.iloc[-1]
        
        account_statistics = {
            'user_avg_amount': latest_record.get('user_avg_amount', 0),
            'user_std_amount': latest_record.get('user_std_amount', 0),
            'user_max_amount': latest_record.get('user_max_amount', 0),
            'user_txn_frequency': latest_record.get('user_txn_frequency', 0),
            'intl_ratio': latest_record.get('intl_ratio', 0),
            'user_high_risk_txn_ratio': latest_record.get('user_high_risk_txn_ratio', 0),
            'user_multiple_accounts_flag': latest_record.get('user_multiple_accounts_flag', 0),
            'cross_account_transfer_ratio': latest_record.get('cross_account_transfer_ratio', 0),
            'rolling_std': latest_record.get('rolling_std', 0),
            'transaction_velocity': latest_record.get('transaction_velocity', 1),
            'current_month_spending': latest_record.get('current_month_spending', 0),
        }
        
        # Cache the result
        set_cached_account_stats(customer_id, account_no, account_statistics)
        logger.debug(f"Account stats cached for {customer_id}/{account_no}")
        
        return account_statistics
        
    except (InvalidTransactionError, AccountNotFoundError) as e:
        # Re-raise known errors
        raise e
    except Exception as e:
        logger.error(f"Unexpected error getting account stats: {e}")
        # Return default stats as fallback
        return get_default_account_stats()


def get_default_account_stats() -> Dict[str, Any]:
    """Return default account statistics when data is unavailable"""
    return {
        'user_avg_amount': 1000,
        'user_std_amount': 500,
        'user_max_amount': 5000,
        'user_txn_frequency': 10,
        'intl_ratio': 0.1,
        'user_high_risk_txn_ratio': 0.2,
        'user_multiple_accounts_flag': 0,
        'cross_account_transfer_ratio': 0.1,
        'rolling_std': 300,
        'transaction_velocity': 1,
        'current_month_spending': 2000,
    }


def add_to_session(customer_id: float, account_no: float, amount: float) -> None:
    """Add transaction to session using efficient velocity tracker"""
    velocity_tracker.add_transaction(customer_id, account_no, amount)
    logger.debug(f"Added transaction to velocity tracker: {customer_id}/{account_no}, amount: {amount}")


def get_velocity_stats(customer_id: float, account_no: float) -> Dict[str, Any]:
    """Get velocity statistics using efficient tracker"""
    # Get historical data from CSV (if available)
    historical_velocity_data = {'txn_count_10min': 0, 'txn_count_1hour': 0, 'txn_count_30s': 0}
    
    if not df_features.empty:
        account_historical_data = df_features[
            (df_features['CustomerId'] == customer_id) & 
            (df_features['FromAccountNo'] == account_no)
        ]
        
        if not account_historical_data.empty:
            latest_historical_record = account_historical_data.iloc[-1]
            historical_velocity_data = {
                'txn_count_10min': latest_historical_record.get('txn_count_10min', 0),
                'txn_count_1hour': latest_historical_record.get('txn_count_1hour', 0),
                'txn_count_30s': latest_historical_record.get('txn_count_30s', 0),
            }

    # Get real-time session data from velocity tracker
    session_velocity_stats = velocity_tracker.get_velocity_stats(customer_id, account_no)
    
    # Combine historical and session data
    return {
        'txn_count_30s': session_velocity_stats['txn_count_30s'],
        'txn_count_10min': session_velocity_stats['txn_count_10min'],
        'txn_count_1hour': session_velocity_stats['txn_count_1hour'],
        'time_since_last': session_velocity_stats['time_since_last'],
        'session_spending': session_velocity_stats['session_spending'],
    }


def get_beneficiary_stats(ben_id: float) -> Dict[str, Any]:
    """Get beneficiary statistics with caching"""
    if ben_id is None or ben_id <= 0:
        return {'is_new_beneficiary': 1, 'beneficiary_txn_count_30d': 0, 'beneficiary_risk_score': 0.5}
    
    # Try cache first
    cached_beneficiary_stats = get_cached_beneficiary_stats(ben_id)
    if cached_beneficiary_stats is not None:
        logger.debug(f"Beneficiary stats cache hit for {ben_id}")
        return cached_beneficiary_stats
    
    # Query from data
    if df_features.empty:
        beneficiary_statistics = {'is_new_beneficiary': 1, 'beneficiary_txn_count_30d': 0, 'beneficiary_risk_score': 0.5}
    else:
        beneficiary_data_df = df_features[df_features['BenId'] == ben_id]
        
        if beneficiary_data_df.empty:
            beneficiary_statistics = {'is_new_beneficiary': 1, 'beneficiary_txn_count_30d': 0, 'beneficiary_risk_score': 0.5}
        else:
            latest_beneficiary_record = beneficiary_data_df.iloc[-1]
            beneficiary_statistics = {
                'is_new_beneficiary': 0,
                'beneficiary_txn_count_30d': latest_beneficiary_record.get('beneficiary_txn_count_30d', 0),
                'beneficiary_risk_score': latest_beneficiary_record.get('beneficiary_risk_score', 0.5),
            }
    
    # Cache the result
    set_cached_beneficiary_stats(ben_id, beneficiary_statistics)
    logger.debug(f"Beneficiary stats cached for {ben_id}")
    
    return beneficiary_statistics


def add_to_pending(customer_id: float, account_no: float, amount: float, 
                   transfer_type: str, ben_id: float, bank_country: str, reasons: list):
    key = (customer_id, account_no)
    now = datetime.now()
    
    if key not in pending_transactions:
        pending_transactions[key] = []
    
    txn_id = f"{customer_id}_{account_no}_{now.strftime('%Y%m%d%H%M%S%f')}"
    
    pending_transactions[key].append({
        'txn_id': txn_id,
        'amount': amount,
        'transfer_type': transfer_type,
        'ben_id': ben_id,
        'bank_country': bank_country,
        'reasons': reasons,
        'timestamp': now
    })
    
    return txn_id


def save_to_history(customer_id: float, account_no: float, amount: float, 
                    transfer_type: str, status: str):
    now = datetime.now()

    if not os.path.exists(TRANSACTION_HISTORY_FILE):
        with open(TRANSACTION_HISTORY_FILE, 'w') as f:
            f.write('CustomerID,AccountNo,Amount,Type,Status,Timestamp\n')
    
    # Append transaction
    with open(TRANSACTION_HISTORY_FILE, 'a') as f:
        f.write(f'{customer_id},{account_no},{amount},{transfer_type},{status},{now}\n')



@app.post("/api/v1/transaction/analyze", response_model=TransactionResponse)
def analyze_transaction(request: TransactionRequest):
    """Analyze transaction with comprehensive error handling"""
    try:
        # Log transaction start
        log_transaction_start(
            request.customer_id, request.account_no, 
            request.amount, request.transfer_type
        )
        
        # Input validation
        if request.amount <= config.MIN_TRANSACTION_AMOUNT:
            logger.warning(f"Invalid amount: {request.amount}")
            raise InvalidTransactionError(
                "Transaction amount must be positive",
                error_code="INVALID_TRANSACTION",
                context={'amount': request.amount}
            )
        
        if request.amount > config.MAX_TRANSACTION_AMOUNT:
            logger.warning(f"Amount exceeds limit: {request.amount}")
            raise InvalidTransactionError(
                "Transaction amount exceeds maximum limit",
                error_code="INVALID_TRANSACTION",
                context={'amount': request.amount, 'limit': config.MAX_TRANSACTION_AMOUNT}
            )
        
        # Get account statistics
        try:
            user_account_stats = get_account_stats(request.customer_id, request.account_no)
            logger.debug(f"Account stats retrieved for {request.customer_id}/{request.account_no}")
        except AccountNotFoundError:
            logger.warning(f"Unknown account: {request.customer_id}/{request.account_no}, using defaults")
            user_account_stats = get_default_account_stats()
        
        # Get velocity statistics
        velocity_statistics = get_velocity_stats(request.customer_id, request.account_no)
        logger.debug(f"Velocity stats: {velocity_statistics['txn_count_10min']} txns in 10min")
        
        user_account_stats['current_month_spending'] = (
            user_account_stats.get('current_month_spending', 0) + velocity_statistics['session_spending']
        )
        
        # Get beneficiary statistics
        beneficiary_statistics = get_beneficiary_stats(request.ben_id)
        
        # Geographic analysis
        is_geo_anomaly = 0 if config.is_local_country(request.bank_country) else 1
        if is_geo_anomaly:
            logger.info(f"Foreign country detected: {request.bank_country}")
        
        # Build transaction features
        current_datetime = datetime.now()
        transaction_features = {
            'amount': request.amount,
            'transfer_type': request.transfer_type.upper(),
            'txn_count_10min': velocity_statistics['txn_count_10min'] + 1,
            'txn_count_1hour': velocity_statistics['txn_count_1hour'] + 1,
            'txn_count_30s': velocity_statistics.get('txn_count_30s', 0) + 1,
            'time_since_last': velocity_statistics['time_since_last'],
            'hour': current_datetime.hour,
            'day_of_week': current_datetime.weekday(),
            'is_weekend': 1 if current_datetime.weekday() >= 5 else 0,
            'is_night': 1 if current_datetime.hour < 6 or current_datetime.hour >= 22 else 0,
            'channel_encoded': 0,
            'geo_anomaly_flag': is_geo_anomaly,
            **beneficiary_statistics,
        }
        
        # Make fraud decision with error handling
        try:
            decision_start_time = time.time()
            fraud_decision_result = make_decision(transaction_features, user_account_stats, model, features, autoencoder, scaler)
            decision_processing_time = (time.time() - decision_start_time) * 1000
            logger.debug(f"Fraud decision completed in {decision_processing_time:.2f}ms")
        except Exception as e:
            logger.error(f"Error in fraud decision: {e}")
            # Fallback to rule-based decision only
            logger.info("Falling back to rule-engine only")
            from backend.rule_engine import check_rule_violation
            rule_violated, rule_violation_reasons, rule_threshold = check_rule_violation(
                amount=request.amount,
                user_avg=user_account_stats.get('user_avg_amount', 0),
                user_std=user_account_stats.get('user_std_amount', 0),
                transfer_type=request.transfer_type.upper(),
                txn_count_10min=transaction_features['txn_count_10min'],
                txn_count_1hour=transaction_features['txn_count_1hour'],
                monthly_spending=user_account_stats.get('current_month_spending', 0)
            )
            
            fraud_decision_result = {
                'is_fraud': rule_violated,
                'reasons': rule_violation_reasons,
                'risk_score': 0.5 if rule_violated else 0.0,
                'threshold': rule_threshold,
                'ml_flag': False,
                'ae_flag': False
            }
        
        # Process result and determine status
        if fraud_decision_result['is_fraud']:
            transaction_status = "AWAITING_USER_CONFIRMATION"
            response_message = "Unusual activity detected. Please confirm this transaction."
            transaction_id = add_to_pending(
                request.customer_id, request.account_no, request.amount,
                request.transfer_type.upper(), request.ben_id or 0, 
                request.bank_country, fraud_decision_result['reasons']
            )
            save_to_history(request.customer_id, request.account_no, request.amount,
                           request.transfer_type.upper(), "Awaiting Confirmation")
            logger.info(f"Transaction flagged for confirmation: {len(fraud_decision_result['reasons'])} reasons")
        else:
            transaction_status = "APPROVED"
            response_message = "Transaction is safe to process"
            transaction_id = None 
            add_to_session(request.customer_id, request.account_no, request.amount)
            save_to_history(request.customer_id, request.account_no, request.amount,
                           request.transfer_type.upper(), "Approved")
            logger.info("Transaction approved")
        
        # Calculate applied limit
        from backend.rule_engine import calculate_threshold
        calculated_limit = calculate_threshold(
            user_account_stats['user_avg_amount'], 
            user_account_stats['user_std_amount'], 
            request.transfer_type.upper()
        )
        
        # Generate risk interpretation
        calculated_risk_score = fraud_decision_result.get('risk_score', 0.0)
        if calculated_risk_score < 0:
            risk_interpretation_text = "Normal behavior - transaction pattern is consistent with user history"
        elif calculated_risk_score < 0.5:
            risk_interpretation_text = "Slightly unusual - minor deviation from typical behavior"
        elif calculated_risk_score < 1.0:
            risk_interpretation_text = "Moderately unusual - noticeable deviation from typical behavior"
        else:
            risk_interpretation_text = "Highly unusual - significant anomaly detected in transaction pattern"
        
        # Log transaction result
        log_transaction_result(transaction_status, calculated_risk_score, fraud_decision_result.get('reasons', []))
        
        return TransactionResponse(
            status=transaction_status,
            message=response_message,
            txn_id=transaction_id,
            risk_score=calculated_risk_score,
            risk_interpretation=risk_interpretation_text,
            threshold=fraud_decision_result.get('threshold', calculated_limit),
            transfer_type=request.transfer_type.upper(),
            applied_limit=round(calculated_limit, 2),
            reasons=fraud_decision_result.get('reasons', []),
            flags={
                'rule_flag': len([r for r in fraud_decision_result.get('reasons', []) if 'Velocity' in r or 'spending' in r]) > 0,
                'ml_flag': fraud_decision_result.get('ml_flag', False),
                'ae_flag': fraud_decision_result.get('ae_flag', False),
            }
        )
        
    except (InvalidTransactionError, AccountNotFoundError) as e:
        logger.error(f"Transaction analysis error: {e}")
        raise HTTPException(status_code=400, detail=create_error_response(e))
    except Exception as e:
        logger.error(f"Unexpected error in transaction analysis: {e}")
        raise HTTPException(status_code=500, detail={
            'error': True,
            'message': 'Internal server error occurred',
            'technical_message': str(e)
        })


@app.get("/api/v1/account/limits/{customer_id}/{account_no}")
def get_account_limits(customer_id: float, account_no: float):
    from backend.rule_engine import calculate_all_limits
    
    user_stats = get_account_stats(customer_id, account_no)
    
    if user_stats is None:
        raise HTTPException(status_code=404, detail="Account not found")
    

    session = get_session_velocity(customer_id, account_no)
    
    limits = calculate_all_limits(
        user_stats['user_avg_amount'],
        user_stats['user_std_amount']
    )
    

    current_spending = user_stats['current_month_spending'] + session['session_spending']
    
    remaining = {}
    for transfer_type, limit in limits.items():
        remaining[transfer_type] = {
            "limit": round(limit, 2),
            "remaining": round(max(0, limit - current_spending), 2)
        }
    
    return {
        "customer_id": customer_id,
        "account_no": account_no,
        "current_month_spending": round(current_spending, 2),
        "csv_spending": round(user_stats['current_month_spending'], 2),
        "session_spending": round(session['session_spending'], 2),
        "user_avg_amount": round(user_stats['user_avg_amount'], 2),
        "user_std_amount": round(user_stats['user_std_amount'], 2),
        "limits_by_transfer_type": remaining
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": model is not None}


@app.get("/api/v1/system/cache/stats")
def get_cache_statistics():
    """Get cache performance statistics"""
    stats = get_cache_stats()
    return {
        "cache_statistics": stats,
        "message": "Cache statistics retrieved successfully"
    }


@app.post("/api/v1/system/cache/cleanup")
def cleanup_caches():
    """Cleanup expired cache entries"""
    cleanup_stats = cleanup_all_caches()
    return {
        "cleanup_results": cleanup_stats,
        "message": "Cache cleanup completed"
    }


@app.post("/api/v1/session/clear")
def clear_session():
    """Clear session data and caches"""
    global session_transactions, pending_transactions
    
    # Clear old session storage (if any remains)
    session_transactions = {}
    pending_transactions = {}
    
    # Cleanup caches
    cleanup_stats = cleanup_all_caches()
    
    logger.info("Session and cache data cleared")
    return {
        "status": "cleared", 
        "message": "Session and cache data cleared",
        "cleanup_stats": cleanup_stats
    }


@app.get("/api/v1/pending/{customer_id}/{account_no}")
def get_pending_transactions(customer_id: float, account_no: float):
    key = (customer_id, account_no)
    pending = pending_transactions.get(key, [])
    
    return {
        "customer_id": customer_id,
        "account_no": account_no,
        "pending_count": len(pending),
        "pending_transactions": [
            {
                "txn_id": t['txn_id'],
                "amount": t['amount'],
                "transfer_type": t['transfer_type'],
                "reasons": t['reasons'],
                "timestamp": t['timestamp'].isoformat()
            }
            for t in pending
        ]
    }


@app.get("/api/v1/pending/all")
def get_all_pending_transactions():
    all_pending = []
    
    for (customer_id, account_no), txns in pending_transactions.items():
        for t in txns:
            all_pending.append({
                "customer_id": customer_id,
                "account_no": account_no,
                "txn_id": t['txn_id'],
                "amount": t['amount'],
                "transfer_type": t['transfer_type'],
                "bank_country": t.get('bank_country', 'Unknown'),
                "reasons": t['reasons'],
                "timestamp": t['timestamp'].isoformat()
            })
    all_pending.sort(key=lambda x: x['timestamp'])
    
    return {
        "total_pending": len(all_pending),
        "message": "Transactions awaiting user confirmation. Users confirm via /confirm or /cancel endpoints.",
        "pending_transactions": all_pending
    }


@app.post("/api/v1/pending/confirm/{customer_id}/{account_no}/{txn_id}")
def confirm_pending_transaction(customer_id: float, account_no: float, txn_id: str):
    key = (customer_id, account_no)
    
    if key not in pending_transactions:
        raise HTTPException(status_code=404, detail="No pending transactions for this account")
    
    # Find the transaction
    txn = None
    for t in pending_transactions[key]:
        if t['txn_id'] == txn_id:
            txn = t
            break
    
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    

    pending_transactions[key] = [t for t in pending_transactions[key] if t['txn_id'] != txn_id]
    
    add_to_session(customer_id, account_no, txn['amount'])
    

    save_to_history(customer_id, account_no, txn['amount'], txn['transfer_type'], "User Confirmed")
    
    return {
        "status": "confirmed",
        "message": f"Transaction {txn_id} confirmed and processed",
        "amount": txn['amount'],
        "transfer_type": txn['transfer_type']
    }


@app.post("/api/v1/pending/cancel/{customer_id}/{account_no}/{txn_id}")
def cancel_pending_transaction(customer_id: float, account_no: float, txn_id: str):
    key = (customer_id, account_no)
    
    if key not in pending_transactions:
        raise HTTPException(status_code=404, detail="No pending transactions for this account")
    
    txn = None
    for t in pending_transactions[key]:
        if t['txn_id'] == txn_id:
            txn = t
            break
    
    if txn is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    pending_transactions[key] = [t for t in pending_transactions[key] if t['txn_id'] != txn_id]
    
    save_to_history(customer_id, account_no, txn['amount'], txn['transfer_type'], "User Cancelled")
    
    return {
        "status": "cancelled",
        "message": f"Transaction {txn_id} has been cancelled",
        "amount": txn['amount'],
        "transfer_type": txn['transfer_type'],
        "warning": "If you did not initiate this transaction, please secure your account immediately."
    }


@app.get("/api/v1/session/stats/{customer_id}/{account_no}")
def get_session_stats(customer_id: float, account_no: float):
    """Get session statistics using efficient velocity tracker"""
    velocity = velocity_tracker.get_velocity_stats(customer_id, account_no)
    
    return {
        "customer_id": customer_id,
        "account_no": account_no,
        "total_transactions_in_session": velocity['txn_count_1hour'],
        "txn_count_30s": velocity['txn_count_30s'],
        "txn_count_10min": velocity['txn_count_10min'],
        "txn_count_1hour": velocity['txn_count_1hour'],
        "session_spending": velocity['session_spending'],
        "time_since_last": velocity['time_since_last']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
