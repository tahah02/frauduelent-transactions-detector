# Fraud Detection API Documentation

Base URL: `http://127.0.0.1:8000`

---

## User Flow Overview

This API implements a **user self-confirmation flow** for fraud detection:

```
User initiates transfer in app
            │
            ▼
   POST /api/v1/transaction/analyze
            │
    ┌───────┴───────┐
    │               │
APPROVED    AWAITING_USER_CONFIRMATION
    │               │
    ▼               ▼
Transaction     Show warning to user:
completes       "This looks unusual. Is this you?"
                [Reasons displayed]
                
                [Confirm]     [Cancel]
                    │             │
                    ▼             ▼
               /confirm       /cancel
                    │             │
                    ▼             ▼
              Transaction    Transaction
              processed      cancelled
```

---

## 1. Process Transaction

Analyzes a transaction for fraud risk. Returns immediately if safe, or asks user to confirm if unusual.

**Endpoint:** `POST /api/v1/transaction/analyze`

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| customer_id | number | Yes | Customer ID |
| account_no | number | Yes | Account number |
| amount | number | Yes | Transaction amount (AED) |
| transfer_type | string | Yes | S (Overseas), Q (Quick), L (UAE), I (Ajman), O (Own) |
| ben_id | number | No | Beneficiary ID |
| bank_country | string | No | Beneficiary bank country (default: UAE) |

### Example Request
```json
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 5000,
    "transfer_type": "S",
    "ben_id": 2584644,
    "bank_country": "Germany"
}
```

### Response

| Field | Type | Description |
|-------|------|-------------|
| status | string | APPROVED or AWAITING_USER_CONFIRMATION |
| message | string | Status description |
| risk_score | number | ML risk score (positive = unusual) |
| risk_interpretation | string | Human-readable explanation of risk |
| threshold | number | Monthly spending limit |
| transfer_type | string | Transfer type evaluated |
| applied_limit | number | Customer-specific limit for this transfer type |
| reasons | array | List of reasons if flagged (show to user) |
| flags | object | Which detection systems triggered |

### Response Flags

| Flag | Description |
|------|-------------|
| rule_flag | Rule Engine triggered (velocity/spending limits) |
| ml_flag | Isolation Forest ML model detected anomaly |
| ae_flag | Autoencoder detected behavioral anomaly |

---

## Detection Examples

### Case 1: APPROVED (Transaction Proceeds)
```json
// Request
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 1000,
    "transfer_type": "L",
    "bank_country": "UAE"
}

// Response - Transaction completes immediately
{
    "status": "APPROVED",
    "message": "Transaction is safe to process",
    "risk_score": -0.05,
    "risk_interpretation": "Normal behavior - transaction pattern is consistent with user history",
    "threshold": 95544.61,
    "transfer_type": "L",
    "applied_limit": 95544.61,
    "reasons": [],
    "flags": {
        "rule_flag": false,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

### Case 2: AWAITING_USER_CONFIRMATION (User Must Confirm)
```json
// Request - Unusual overseas transfer
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 15000,
    "transfer_type": "S",
    "bank_country": "Germany"
}

// Response - Show confirmation screen to user
{
    "status": "AWAITING_USER_CONFIRMATION",
    "message": "Unusual activity detected. Please confirm this transaction.",
    "risk_score": -0.03,
    "risk_interpretation": "Normal behavior - transaction pattern is consistent with user history",
    "threshold": 65255.49,
    "transfer_type": "S",
    "applied_limit": 65255.49,
    "reasons": [
        "Monthly spending AED 66,464.77 exceeds limit AED 65,255.49"
    ],
    "flags": {
        "rule_flag": true,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

**Frontend Action:** Display the `reasons` array to the user with "Confirm" and "Cancel" buttons.

---

## 2. User Confirms Transaction

User confirms they initiated the flagged transaction.

**Endpoint:** `POST /api/v1/pending/confirm/{customer_id}/{account_no}/{txn_id}`

### Example
```
POST /api/v1/pending/confirm/4424492/14424492014/4424492_14424492014_20260101120000123456
```

### Response
```json
{
    "status": "confirmed",
    "message": "Transaction 4424492_14424492014_20260101120000123456 confirmed and processed",
    "amount": 15000,
    "transfer_type": "S"
}
```

---

## 3. User Cancels Transaction

User cancels a flagged transaction (doesn't recognize it or changed their mind).

**Endpoint:** `POST /api/v1/pending/cancel/{customer_id}/{account_no}/{txn_id}`

### Example
```
POST /api/v1/pending/cancel/4424492/14424492014/4424492_14424492014_20260101120000123456
```

### Response
```json
{
    "status": "cancelled",
    "message": "Transaction 4424492_14424492014_20260101120000123456 has been cancelled",
    "amount": 15000,
    "transfer_type": "S",
    "warning": "If you did not initiate this transaction, please secure your account immediately."
}
```

---

## 4. Get Pending Transactions (User's Account)

Get all transactions awaiting user confirmation for a specific account.

**Endpoint:** `GET /api/v1/pending/{customer_id}/{account_no}`

### Response
```json
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "pending_count": 1,
    "pending_transactions": [
        {
            "txn_id": "4424492_14424492014_20260101120000123456",
            "amount": 15000,
            "transfer_type": "S",
            "reasons": ["Monthly spending exceeds limit"],
            "timestamp": "2026-01-01T12:00:00.123456"
        }
    ]
}
```

---

## 5. Account Limits

Get account spending and remaining limits for all transfer types.

**Endpoint:** `GET /api/v1/account/limits/{customer_id}/{account_no}`

### Response
```json
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "current_month_spending": 61464.77,
    "csv_spending": 51464.77,
    "session_spending": 10000,
    "user_avg_amount": 4677.25,
    "user_std_amount": 30289.12,
    "limits_by_transfer_type": {
        "S": {"limit": 65255.49, "remaining": 3790.72},
        "I": {"limit": 110689.17, "remaining": 49224.4},
        "L": {"limit": 95544.61, "remaining": 34079.84},
        "Q": {"limit": 80400.05, "remaining": 18935.28},
        "O": {"limit": 125833.73, "remaining": 64368.96}
    }
}
```

---

## 6. Session Stats

Get current session statistics for debugging.

**Endpoint:** `GET /api/v1/session/stats/{customer_id}/{account_no}`

---

## 7. Clear Session

Clear all session data (for testing).

**Endpoint:** `POST /api/v1/session/clear`

---

## 8. Health Check

**Endpoint:** `GET /health`

### Response
```json
{
    "status": "healthy",
    "models_loaded": true
}
```

---

## 9. Get All Pending (Audit/Monitoring)

Get all transactions awaiting confirmation across all accounts.

**Endpoint:** `GET /api/v1/pending/all`

**Note:** This is for monitoring/audit purposes. Users confirm their own transactions.

---

## Transfer Types & Risk Levels

| Code | Type | Risk Level | Limit Multiplier |
|------|------|------------|------------------|
| S | Overseas | HIGH | 2.0x |
| Q | Quick | MEDIUM | 2.5x |
| L | UAE | LOW | 3.0x |
| I | Ajman | LOW | 3.5x |
| O | Own Account | LOWEST | 4.0x |

---

## Transaction Statuses

| Status | Description | History Record |
|--------|-------------|----------------|
| APPROVED | Safe, processed immediately | "Approved" |
| AWAITING_USER_CONFIRMATION | Unusual, needs user confirmation | "Awaiting Confirmation" |
| (after confirm) | User confirmed | "User Confirmed" |
| (after cancel) | User cancelled | "User Cancelled" |

---

## Running the API

```bash
cd frauduelent-transactions-detector
uvicorn backend.api:app --reload
```

API: `http://127.0.0.1:8000`
Swagger Docs: `http://127.0.0.1:8000/docs`

---

## Frontend Integration Guide

### When `status === "APPROVED"`:
- Show success message
- Transaction is complete

### When `status === "AWAITING_USER_CONFIRMATION"`:
1. Show confirmation dialog with:
   - Warning message: "We detected unusual activity"
   - Display `reasons` array as bullet points
   - Transaction details (amount, transfer_type, etc.)
2. Two buttons:
   - "Yes, it's me" → Call `/confirm` endpoint
   - "Cancel" → Call `/cancel` endpoint
3. After user action, show appropriate result

### Example UI Flow:
```
┌─────────────────────────────────────────┐
│  ⚠️ Unusual Activity Detected           │
│                                         │
│  We noticed something different about   │
│  this transaction:                      │
│                                         │
│  • Monthly spending exceeds your limit  │
│  • Transfer to new country (Germany)    │
│                                         │
│  Amount: AED 15,000                     │
│  To: Germany                            │
│                                         │
│  Is this you?                           │
│                                         │
│  [Yes, proceed]     [No, cancel]        │
└─────────────────────────────────────────┘
```
