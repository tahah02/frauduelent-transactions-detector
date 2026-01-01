# Fraudulent Transaction Detector

A real-time fraud detection system using triple-layer protection: Rule Engine, Isolation Forest (ML), and Autoencoder (Deep Learning) with **user self-confirmation flow**.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn backend.api:app --reload
```

API available at: `http://127.0.0.1:8000`

Swagger Docs: `http://127.0.0.1:8000/docs`

## 🔄 User Flow

This system implements a **user self-confirmation flow** for fraud detection:

```
User initiates transfer
        ↓
   Fraud Analysis
        ↓
┌───────┴───────┐
│               │
APPROVED    AWAITING_USER_CONFIRMATION
│               │
▼               ▼
Transaction     Show warning:
completes       "This looks unusual. Is this you?"
                
                [Confirm]     [Cancel]
                    │             │
                    ▼             ▼
              Transaction    Transaction
              processed      cancelled
```

## 📡 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/transaction/analyze` | Analyze transaction for fraud |
| POST | `/api/v1/pending/confirm/{customer_id}/{account_no}/{txn_id}` | User confirms flagged transaction |
| POST | `/api/v1/pending/cancel/{customer_id}/{account_no}/{txn_id}` | User cancels flagged transaction |
| GET | `/api/v1/pending/{customer_id}/{account_no}` | Get user's pending transactions |

### Account & Session Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/account/limits/{customer_id}/{account_no}` | Get account spending limits |
| GET | `/api/v1/session/stats/{customer_id}/{account_no}` | Get session velocity stats |
| POST | `/api/v1/session/clear` | Clear session data (testing) |
| GET | `/health` | Health check |

### Monitoring (Optional)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/pending/all` | Get all pending transactions (audit) |

## 💡 Example Usage

### 1. Process Transaction

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

### 2. Response - Normal Transaction
```json
{
    "status": "APPROVED",
    "message": "Transaction is safe to process",
    "txn_id": null,
    "risk_score": -0.05,
    "risk_interpretation": "Normal behavior - consistent with user history",
    "threshold": 65255.49,
    "transfer_type": "S",
    "applied_limit": 65255.49,
    "reasons": [],
    "flags": {
        "rule_flag": false,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

### 3. Response - Unusual Transaction
```json
{
    "status": "AWAITING_USER_CONFIRMATION",
    "message": "Unusual activity detected. Please confirm this transaction.",
    "txn_id": "4424492_14424492014_20260101120000123456",
    "risk_score": 0.0048,
    "risk_interpretation": "Slightly unusual - minor deviation from typical behavior",
    "reasons": [
        "Monthly spending AED 66,464.77 exceeds limit AED 65,255.49",
        "Transfer to new country (Germany)"
    ],
    "flags": {
        "rule_flag": true,
        "ml_flag": false,
        "ae_flag": false
    }
}
```

### 4. User Confirms Transaction
```bash
POST /api/v1/pending/confirm/4424492/14424492014/4424492_14424492014_20260101120000123456
```

## 🏷️ Status Values

| Status | Description | User Action |
|--------|-------------|-------------|
| `APPROVED` | Transaction is safe, processed immediately | None - transaction completes |
| `AWAITING_USER_CONFIRMATION` | Unusual activity detected | User must confirm or cancel |

## 🚩 Detection Flags

| Flag | Description | Trigger |
|------|-------------|---------|
| `rule_flag` | Rule Engine triggered | Velocity/spending limits exceeded |
| `ml_flag` | Isolation Forest detected anomaly | 26-feature ML analysis |
| `ae_flag` | Autoencoder detected behavioral anomaly | 31-feature deep learning |

## 💳 Transfer Types

| Code | Type | Risk Level | Limit Multiplier |
|------|------|------------|------------------|
| S | Overseas | HIGH | 2.0x |
| Q | Quick | MEDIUM | 2.5x |
| L | UAE | LOW | 3.0x |
| I | Ajman | LOW | 3.5x |
| O | Own Account | LOWEST | 4.0x |

## 🔍 Detection Flow

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
AWAITING  APPROVED
USER_CONF
```

## 📁 Project Structure

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
│   ├── autoencoder.h5         # Trained AE model
│   └── *_scaler.pkl          # Feature scalers
├── Docs/
│   ├── API_DOCUMENTATION.md   # Complete API guide
│   ├── PROJECT_COMPLETE_DOCUMENTATION.md
│   ├── FEATURES_DOCUMENTATION.md
│   └── CUSTOMER_ACCOUNTS.md
└── postman/
    └── Fraud_Detection_API.postman_collection.json
```

## 📚 Documentation

- [**API Documentation**](Docs/API_DOCUMENTATION.md) - Complete API guide with examples
- [**Project Documentation**](Docs/PROJECT_COMPLETE_DOCUMENTATION.md) - Full system overview
- [**Features Documentation**](Docs/FEATURES_DOCUMENTATION.md) - ML feature explanations
- [**Customer Accounts**](Docs/CUSTOMER_ACCOUNTS.md) - Test account details

## 🧪 Testing with Postman

1. Import `postman/Fraud_Detection_API.postman_collection.json`
2. Set collection variables:
   - `customer_id`: 4424492
   - `account_no`: 14424492014
   - `txn_id`: (from analyze response)
3. Test the complete user flow

## 🚀 Production Deployment

For production deployment as a microservice:

### Required Changes:
- [ ] Replace CSV data source with SQL Server connection
- [ ] Replace in-memory session store with Redis
- [ ] Replace in-memory pending store with database
- [ ] Add authentication middleware
- [ ] Configure Docker containerization

### Current Status:
- ✅ API endpoints ready
- ✅ ML models trained and working
- ✅ User self-confirmation flow implemented
- ✅ Comprehensive documentation
- ⏳ Database integration pending

## 🔧 Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000

# Test endpoints
curl http://localhost:8000/health
```

## 📊 Key Features

- **Real-time Analysis**: Sub-second fraud detection
- **Triple-layer Protection**: Rule Engine + ML + Deep Learning
- **User-friendly**: Self-confirmation instead of blocking
- **Scalable**: FastAPI microservice architecture
- **Comprehensive**: Full documentation and testing suite
- **Production-ready**: Clear deployment roadmap