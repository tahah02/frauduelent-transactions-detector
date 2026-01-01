import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.model import load_model
from backend.autoencoder import AutoencoderInference
from backend.hybrid_decision import make_decision


app = FastAPI(title="Transaction Fraud Detection API", version="1.0.0")

# Load models on startup
model, features, scaler = None, None, None
autoencoder = None
df_features = None

# In-memory session store for real-time tracking (only APPROVED transactions)
# Key: (customer_id, account_no), Value: list of {amount, timestamp}
session_transactions = {}

# Pending transactions store (waiting for approval)
# Key: (customer_id, account_no), Value: list of {amount, timestamp, transfer_type, ben_id, bank_country}
pending_transactions = {}

# Transaction history file path
TRANSACTION_HISTORY_FILE = 'transaction_history.csv'


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
    txn_id: Optional[str] = None  # Transaction ID for confirm/cancel (only when AWAITING_USER_CONFIRMATION)
    risk_score: float
    risk_interpretation: str  # Explains what the risk score means
    threshold: float
    transfer_type: str  # Which transfer type was evaluated
    applied_limit: float  # Customer-specific limit for this transfer type
    reasons: list
    flags: dict


@app.on_event("startup")
def load_models():
    global model, features, scaler, autoencoder, df_features
    
    print("Loading models...")
    model, features, scaler = load_model()
    
    autoencoder = AutoencoderInference()
    autoencoder.load_artifacts()
    
    df_features = pd.read_csv('data/featured_dataset.csv')
    print(f"Loaded {len(df_features)} records from featured_dataset.csv")
    print("Models loaded successfully!")


def get_account_stats(customer_id: float, account_no: float) -> dict:
    account_data = df_features[
        (df_features['CustomerId'] == customer_id) & 
        (df_features['FromAccountNo'] == account_no)
    ]
    
    if account_data.empty:
        return None
    
    latest = account_data.iloc[-1]
    
    return {
        'user_avg_amount': latest.get('user_avg_amount', 0),
        'user_std_amount': latest.get('user_std_amount', 0),
        'user_max_amount': latest.get('user_max_amount', 0),
        'user_txn_frequency': latest.get('user_txn_frequency', 0),
        'intl_ratio': latest.get('intl_ratio', 0),
        'user_high_risk_txn_ratio': latest.get('user_high_risk_txn_ratio', 0),
        'user_multiple_accounts_flag': latest.get('user_multiple_accounts_flag', 0),
        'cross_account_transfer_ratio': latest.get('cross_account_transfer_ratio', 0),
        'rolling_std': latest.get('rolling_std', 0),
        'transaction_velocity': latest.get('transaction_velocity', 1),
        'current_month_spending': latest.get('current_month_spending', 0),
    }


def add_to_session(customer_id: float, account_no: float, amount: float):
    key = (customer_id, account_no)
    now = datetime.now()
    
    if key not in session_transactions:
        session_transactions[key] = []
    
    session_transactions[key].append({
        'amount': amount,
        'timestamp': now
    })
    
    # Clean old transactions (older than 1 hour)
    cutoff = now - timedelta(hours=1)
    session_transactions[key] = [
        t for t in session_transactions[key] if t['timestamp'] > cutoff
    ]


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
    
    # Create file if not exists
    if not os.path.exists(TRANSACTION_HISTORY_FILE):
        with open(TRANSACTION_HISTORY_FILE, 'w') as f:
            f.write('CustomerID,AccountNo,Amount,Type,Status,Timestamp\n')
    
    # Append transaction
    with open(TRANSACTION_HISTORY_FILE, 'a') as f:
        f.write(f'{customer_id},{account_no},{amount},{transfer_type},{status},{now}\n')


def get_session_velocity(customer_id: float, account_no: float) -> dict:
    key = (customer_id, account_no)
    now = datetime.now()
    
    if key not in session_transactions or len(session_transactions[key]) == 0:
        return {
            'session_txn_count_30s': 0,
            'session_txn_count_10min': 0,
            'session_txn_count_1hour': 0,
            'session_spending': 0,
            'time_since_last': 3600
        }
    
    txns = session_transactions[key]
    
    # Calculate counts for different windows
    count_30s = sum(1 for t in txns if (now - t['timestamp']).total_seconds() <= 30)
    count_10min = sum(1 for t in txns if (now - t['timestamp']).total_seconds() <= 600)
    count_1hour = sum(1 for t in txns if (now - t['timestamp']).total_seconds() <= 3600)
    
    # Total session spending
    session_spending = sum(t['amount'] for t in txns)
    
    # Time since last transaction
    if len(txns) > 1:
        sorted_txns = sorted(txns, key=lambda x: x['timestamp'], reverse=True)
        time_since_last = (now - sorted_txns[1]['timestamp']).total_seconds()
    else:
        time_since_last = 3600
    
    return {
        'session_txn_count_30s': count_30s,
        'session_txn_count_10min': count_10min,
        'session_txn_count_1hour': count_1hour,
        'session_spending': session_spending,
        'time_since_last': time_since_last
    }


def get_velocity_stats(customer_id: float, account_no: float) -> dict:
    # Get historical from CSV
    account_data = df_features[
        (df_features['CustomerId'] == customer_id) & 
        (df_features['FromAccountNo'] == account_no)
    ]
    
    historical = {'txn_count_10min': 0, 'txn_count_1hour': 0, 'txn_count_30s': 0}
    if not account_data.empty:
        latest = account_data.iloc[-1]
        historical = {
            'txn_count_10min': latest.get('txn_count_10min', 0),
            'txn_count_1hour': latest.get('txn_count_1hour', 0),
            'txn_count_30s': latest.get('txn_count_30s', 0),
        }
    
    # Get real-time from session
    session = get_session_velocity(customer_id, account_no)
    
    # Combine: use session values for real-time detection
    return {
        'txn_count_30s': session['session_txn_count_30s'],
        'txn_count_10min': session['session_txn_count_10min'],
        'txn_count_1hour': session['session_txn_count_1hour'],
        'time_since_last': session['time_since_last'],
        'session_spending': session['session_spending'],
    }


def get_beneficiary_stats(ben_id: float) -> dict:
    if ben_id is None or ben_id <= 0:
        return {'is_new_beneficiary': 1, 'beneficiary_txn_count_30d': 0, 'beneficiary_risk_score': 0.5}
    
    ben_data = df_features[df_features['BenId'] == ben_id]
    
    if ben_data.empty:
        return {'is_new_beneficiary': 1, 'beneficiary_txn_count_30d': 0, 'beneficiary_risk_score': 0.5}
    
    latest = ben_data.iloc[-1]
    
    return {
        'is_new_beneficiary': 0,
        'beneficiary_txn_count_30d': latest.get('beneficiary_txn_count_30d', 0),
        'beneficiary_risk_score': latest.get('beneficiary_risk_score', 0.5),
    }


@app.post("/api/v1/transaction/analyze", response_model=TransactionResponse)
def analyze_transaction(request: TransactionRequest):
    """
    Analyze a transaction and return APPROVED or AWAITING_USER_CONFIRMATION.
    
    **User Flow:** This endpoint is called when a user initiates a transfer.
    If the transaction looks unusual, the user is asked to confirm it's really them.
    
    **Statuses:**
    - APPROVED: Transaction is safe, processed immediately
    - AWAITING_USER_CONFIRMATION: Unusual activity detected, user must confirm
    
    **Risk Score:** Anomaly signal from ML models (negative = normal, positive = unusual)
    
    **Frontend Integration:** When status is AWAITING_USER_CONFIRMATION, show the 
    reasons to the user and ask them to confirm or cancel the transaction.
    """
    # Get account stats
    user_stats = get_account_stats(request.customer_id, request.account_no)
    
    if user_stats is None:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Get velocity stats (from session - only approved transactions)
    velocity = get_velocity_stats(request.customer_id, request.account_no)
    
    # Add session spending to monthly spending for limit check
    user_stats['current_month_spending'] = (
        user_stats.get('current_month_spending', 0) + velocity['session_spending']
    )
    
    # Get beneficiary stats
    ben_stats = get_beneficiary_stats(request.ben_id)
    
    # Determine geo anomaly
    local_countries = ['UAE', 'AE', 'United Arab Emirates', 'AJMAN']
    geo_anomaly = 0 if request.bank_country.upper() in [c.upper() for c in local_countries] else 1
    
    # Build transaction dict
    now = datetime.now()
    txn = {
        'amount': request.amount,
        'transfer_type': request.transfer_type.upper(),
        'txn_count_10min': velocity['txn_count_10min'] + 1,
        'txn_count_1hour': velocity['txn_count_1hour'] + 1,
        'txn_count_30s': velocity.get('txn_count_30s', 0) + 1,
        'time_since_last': velocity['time_since_last'],
        'hour': now.hour,
        'day_of_week': now.weekday(),
        'is_weekend': 1 if now.weekday() >= 5 else 0,
        'is_night': 1 if now.hour < 6 or now.hour >= 22 else 0,
        'channel_encoded': 0,
        'geo_anomaly_flag': geo_anomaly,
        **ben_stats,
    }
    
    # Make decision
    result = make_decision(txn, user_stats, model, features, autoencoder, scaler)
    
    # Determine status and handle accordingly
    if result['is_fraud']:
        status = "AWAITING_USER_CONFIRMATION"
        message = "Unusual activity detected. Please confirm this transaction."
        # Add to pending (NOT to session)
        txn_id = add_to_pending(
            request.customer_id, request.account_no, request.amount,
            request.transfer_type.upper(), request.ben_id or 0, 
            request.bank_country, result['reasons']
        )
        # Save to history as Awaiting Confirmation
        save_to_history(request.customer_id, request.account_no, request.amount,
                       request.transfer_type.upper(), "Awaiting Confirmation")
    else:
        status = "APPROVED"
        message = "Transaction is safe to process"
        txn_id = None  # No txn_id needed for approved transactions
        # Add to session (for velocity tracking)
        add_to_session(request.customer_id, request.account_no, request.amount)
        # Save to history as Approved
        save_to_history(request.customer_id, request.account_no, request.amount,
                       request.transfer_type.upper(), "Approved")
    
    # Calculate customer-specific limit for this transfer type
    from backend.rule_engine import calculate_threshold
    applied_limit = calculate_threshold(
        user_stats['user_avg_amount'], 
        user_stats['user_std_amount'], 
        request.transfer_type.upper()
    )
    
    # Generate risk interpretation
    if result['risk_score'] < 0:
        risk_interpretation = "Normal behavior - transaction pattern is consistent with user history"
    elif result['risk_score'] < 0.5:
        risk_interpretation = "Slightly unusual - minor deviation from typical behavior"
    elif result['risk_score'] < 1.0:
        risk_interpretation = "Moderately unusual - noticeable deviation from typical behavior"
    else:
        risk_interpretation = "Highly unusual - significant anomaly detected in transaction pattern"
    
    return TransactionResponse(
        status=status,
        message=message,
        txn_id=txn_id,
        risk_score=result['risk_score'],
        risk_interpretation=risk_interpretation,
        threshold=result['threshold'],
        transfer_type=request.transfer_type.upper(),
        applied_limit=round(applied_limit, 2),
        reasons=result['reasons'],
        flags={
            'rule_flag': len([r for r in result['reasons'] if 'Velocity' in r or 'spending' in r]) > 0,
            'ml_flag': result['ml_flag'],
            'ae_flag': result['ae_flag'],
        }
    )


@app.get("/api/v1/account/limits/{customer_id}/{account_no}")
def get_account_limits(customer_id: float, account_no: float):
    from backend.rule_engine import calculate_all_limits
    
    user_stats = get_account_stats(customer_id, account_no)
    
    if user_stats is None:
        raise HTTPException(status_code=404, detail="Account not found")
    
    # Get session spending
    session = get_session_velocity(customer_id, account_no)
    
    # Calculate limits for all transfer types
    limits = calculate_all_limits(
        user_stats['user_avg_amount'],
        user_stats['user_std_amount']
    )
    
    # Total spending = CSV + Session
    current_spending = user_stats['current_month_spending'] + session['session_spending']
    
    # Calculate remaining for each transfer type
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


@app.post("/api/v1/session/clear")
def clear_session():
    global session_transactions, pending_transactions
    session_transactions = {}
    pending_transactions = {}
    return {"status": "cleared", "message": "Session and pending transactions cleared"}


@app.get("/api/v1/pending/{customer_id}/{account_no}")
def get_pending_transactions(customer_id: float, account_no: float):
    """
    Get all transactions awaiting user confirmation for an account.
    
    **User Flow:** Call this to show the user their pending transactions
    that need confirmation before processing.
    """
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
    """
    Get ALL transactions awaiting confirmation across all accounts.
    
    **For Audit/Monitoring:** This endpoint returns all transactions
    currently awaiting user confirmation. Useful for monitoring and analytics.
    """
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
    
    # Sort by timestamp (oldest first)
    all_pending.sort(key=lambda x: x['timestamp'])
    
    return {
        "total_pending": len(all_pending),
        "message": "Transactions awaiting user confirmation. Users confirm via /confirm or /cancel endpoints.",
        "pending_transactions": all_pending
    }


@app.post("/api/v1/pending/confirm/{customer_id}/{account_no}/{txn_id}")
def confirm_pending_transaction(customer_id: float, account_no: float, txn_id: str):
    """
    User confirms a flagged transaction.
    
    **User Flow:** When a transaction is flagged as unusual, the user sees
    a confirmation screen with the reasons. If they confirm "Yes, it's me", 
    call this endpoint to proceed with the transaction.
    
    Note: Auth verification should be handled by your company's auth layer
    before this endpoint is called.
    """
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
    
    # Remove from pending
    pending_transactions[key] = [t for t in pending_transactions[key] if t['txn_id'] != txn_id]
    
    # Add to session (now it counts towards spending/velocity)
    add_to_session(customer_id, account_no, txn['amount'])
    
    # Update history status
    save_to_history(customer_id, account_no, txn['amount'], txn['transfer_type'], "User Confirmed")
    
    return {
        "status": "confirmed",
        "message": f"Transaction {txn_id} confirmed and processed",
        "amount": txn['amount'],
        "transfer_type": txn['transfer_type']
    }


@app.post("/api/v1/pending/cancel/{customer_id}/{account_no}/{txn_id}")
def cancel_pending_transaction(customer_id: float, account_no: float, txn_id: str):
    """
    User cancels a flagged transaction.
    
    **User Flow:** When a transaction is flagged as unusual, if the user
    doesn't recognize it or wants to cancel, they call this endpoint.
    This may indicate unauthorized access to their account.
    """
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
    
    # Remove from pending
    pending_transactions[key] = [t for t in pending_transactions[key] if t['txn_id'] != txn_id]
    
    # Update history status
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
    velocity = get_session_velocity(customer_id, account_no)
    key = (customer_id, account_no)
    txn_count = len(session_transactions.get(key, []))
    
    return {
        "customer_id": customer_id,
        "account_no": account_no,
        "total_transactions_in_session": txn_count,
        "txn_count_30s": velocity['session_txn_count_30s'],
        "txn_count_10min": velocity['session_txn_count_10min'],
        "txn_count_1hour": velocity['session_txn_count_1hour'],
        "session_spending": velocity['session_spending'],
        "time_since_last": velocity['time_since_last']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
