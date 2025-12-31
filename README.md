# Fraudulent Transaction Detector

A real-time fraud detection system using triple-layer protection: Rule Engine, Isolation Forest (ML), and Autoencoder (Deep Learning).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn backend.api:app --reload
```

API available at: `http://127.0.0.1:8000`

Swagger Docs: `http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/transaction/analyze` | Analyze transaction for fraud |
| GET | `/api/v1/account/limits/{customer_id}/{account_no}` | Get account spending limits |
| GET | `/api/v1/pending/{customer_id}/{account_no}` | Get pending transactions |
| POST | `/api/v1/pending/approve/{customer_id}/{account_no}/{txn_id}` | Approve pending transaction |
| POST | `/api/v1/pending/reject/{customer_id}/{account_no}/{txn_id}` | Reject pending transaction |
| GET | `/api/v1/session/stats/{customer_id}/{account_no}` | Get session velocity stats |
| POST | `/api/v1/session/clear` | Clear session data |
| GET | `/health` | Health check |

## Process Transaction

```bash
POST /api/v1/transaction/analyze

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
```json
{
    "status": "APPROVED",
    "message": "Transaction is safe to process",
    "risk_score": -0.05,
    "threshold": 65255.49,
    "reasons": [],
    "flags": {
        "rule_flag": false,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

### Status Values
- `APPROVED` - Transaction is safe
- `PENDING_REVIEW` - Requires manual approval

### Flags
- `rule_flag` - Rule Engine triggered (velocity/spending limits)
- `ml_flag` - Isolation Forest detected anomaly
- `ae_flag` - Autoencoder detected behavioral anomaly

## Transfer Types

| Code | Type | Risk Level |
|------|------|------------|
| S | Overseas | HIGH |
| Q | Quick | MEDIUM |
| L | UAE | LOW |
| I | Ajman | LOW |
| O | Own Account | LOWEST |

## Detection Flow

```
Transaction → Rule Engine → Isolation Forest → Autoencoder → Decision
                  ↓              ↓                 ↓
            Velocity/Limits   26 Features      31 Features
                  ↓              ↓                 ↓
              Any Flag? ────────────────────────────┘
                  ↓
         APPROVED / PENDING_REVIEW
```

## Project Structure

```
├── backend/
│   ├── api.py                 # FastAPI endpoints
│   ├── hybrid_decision.py     # 3-layer decision logic
│   ├── rule_engine.py         # Velocity & limit checks
│   ├── model.py               # Isolation Forest
│   ├── autoencoder.py         # Autoencoder model
│   ├── feature_engineering.py # Feature generation
│   └── utils.py               # Helper functions
├── data/
│   ├── Clean.csv              # Raw transaction data
│   └── featured_dataset.csv   # Engineered features
├── models/
│   ├── isolation_forest.pkl   # Trained IF model
│   └── autoencoder.h5         # Trained AE model
├── Docs/
│   ├── API_DOCUMENTATION.md   # Full API docs
│   ├── FEATURES_DOCUMENTATION.md
│   └── CUSTOMER_ACCOUNTS.md
└── postman/
    └── Fraud_Detection_API.postman_collection.json
```

## Documentation

- [API Documentation](Docs/API_DOCUMENTATION.md)
- [Features Documentation](Docs/FEATURES_DOCUMENTATION.md)
- [Customer Accounts](Docs/CUSTOMER_ACCOUNTS.md)

## Postman Collection

Import `postman/Fraud_Detection_API.postman_collection.json` into Postman for testing.
