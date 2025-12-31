# Fraud Detection API Documentation

Base URL: `http://127.0.0.1:8000`

---

## 1. Process Transaction

Analyzes a transaction for fraud risk and determines whether it can proceed automatically or requires manual review.

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
| status | string | APPROVED or PENDING_REVIEW |
| message | string | Status description |
| risk_score | number | ML risk score (positive = risky) |
| threshold | number | Monthly spending limit for this transfer type |
| reasons | array | List of reasons if flagged |
| flags | object | Which checks triggered |

### Response Flags

| Flag | Description |
|------|-------------|
| rule_flag | Rule Engine triggered (velocity/spending limits) |
| ml_flag | Isolation Forest ML model detected anomaly |
| ae_flag | Autoencoder detected behavioral anomaly |

---

## Detection Examples

### Case 1: APPROVED (All Checks Pass)
```json
// Request
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 1000,
    "transfer_type": "L",
    "bank_country": "UAE"
}

// Response
{
    "status": "APPROVED",
    "message": "Transaction is safe to process",
    "risk_score": -0.05,
    "threshold": 95544.61,
    "reasons": [],
    "flags": {
        "rule_flag": false,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

### Case 2: PENDING_REVIEW - Rule Engine Flag (Limit Exceeded)
```json
// Request - Amount exceeds monthly limit
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 15000,
    "transfer_type": "S",
    "bank_country": "Germany"
}

// Response
{
    "status": "PENDING_REVIEW",
    "message": "Transaction requires manual approval",
    "risk_score": -0.03,
    "threshold": 65255.49,
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

### Case 3: PENDING_REVIEW - ML Flag (Anomaly Detected)
```json
// Request - Unusual pattern for this customer
{
    "customer_id": 1000016,
    "account_no": 11000016019,
    "amount": 50000,
    "transfer_type": "S",
    "ben_id": 999999,
    "bank_country": "Nigeria"
}

// Response
{
    "status": "PENDING_REVIEW",
    "message": "Transaction requires manual approval",
    "risk_score": 0.0347,
    "threshold": 63772.26,
    "reasons": [
        "Monthly spending AED 411,740.07 exceeds limit AED 63,772.26",
        "ML anomaly detected (risk score 0.0347)"
    ],
    "flags": {
        "rule_flag": true,
        "ml_flag": true,
        "ae_flag": false
    }
}
```

### Case 4: PENDING_REVIEW - Velocity Flag (Burst Transactions)
```json
// Request - 6th transaction within 10 minutes
{
    "customer_id": 1000016,
    "account_no": 11000016019,
    "amount": 1000,
    "transfer_type": "O",
    "bank_country": "UAE"
}

// Response
{
    "status": "PENDING_REVIEW",
    "message": "Transaction requires manual approval",
    "risk_score": -0.02,
    "threshold": 125833.73,
    "reasons": [
        "Velocity limit exceeded: 6 transactions in last 10 minutes (max allowed 5)"
    ],
    "flags": {
        "rule_flag": true,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

---

## 2. Account Limits

Get account spending and remaining limits for all transfer types.

**Endpoint:** `GET /api/v1/account/limits/{customer_id}/{account_no}`

### Example
```
GET /api/v1/account/limits/4424492/14424492014
```

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

## 3. Pending Transactions

### Get Pending Transactions
**Endpoint:** `GET /api/v1/pending/{customer_id}/{account_no}`

### Approve Pending Transaction
**Endpoint:** `POST /api/v1/pending/approve/{customer_id}/{account_no}/{txn_id}`

### Reject Pending Transaction
**Endpoint:** `POST /api/v1/pending/reject/{customer_id}/{account_no}/{txn_id}`

---

## 4. Session Management

### Get Session Stats
**Endpoint:** `GET /api/v1/session/stats/{customer_id}/{account_no}`

### Clear Session
**Endpoint:** `POST /api/v1/session/clear`

---

## 5. Health Check

**Endpoint:** `GET /health`

### Response
```json
{
    "status": "healthy",
    "models_loaded": true
}
```

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

## Fraud Detection Flow

```
Transaction Request
       ↓
┌──────────────────┐
│   Rule Engine    │ → Velocity limits (5/10min, 15/hour)
│                  │ → Monthly spending limits
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Isolation Forest │ → 26 features analysis
│      (ML)        │ → Anomaly detection
└────────┬─────────┘
         ↓
┌──────────────────┐
│   Autoencoder    │ → 31 features analysis
│      (AE)        │ → Behavioral pattern check
└────────┬─────────┘
         ↓
    Any Flag?
    ↓       ↓
   Yes      No
    ↓       ↓
PENDING   APPROVED
REVIEW
```

---

## Running the API

```bash
cd frauduelent-transactions-detector
uvicorn backend.api:app --reload
```

API will be available at: `http://127.0.0.1:8000`

Swagger Docs: `http://127.0.0.1:8000/docs`

---

## Postman Collection Setup

Import `postman/Fraud_Detection_API.postman_collection.json` into Postman.

### Setting Variables

1. Click on "Fraud Detection API" collection name
2. Go to "Variables" tab on the right side
3. Set values for:
   - `customer_id` - Customer ID to test
   - `account_no` - Account number to test  
   - `txn_id` - Transaction ID (for approve/reject pending)

### Available Variables

| Variable | Description | Example |
|----------|-------------|---------|
| baseURL | API base URL | http://127.0.0.1:8000 |
| customer_id | Customer ID | 4424492 |
| account_no | Account Number | 14424492014 |
| txn_id | Pending Transaction ID | 4424492_14424492014_20251231... |

You can also directly edit the URL in each request instead of using variables.
