# Fraudulent Transaction Detector - Complete Project Documentation

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Business Overview](#2-business-overview)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Detection Layers](#6-detection-layers)
7. [API Documentation](#7-api-documentation)
8. [Database & Storage](#8-database--storage)
9. [Project Structure](#9-project-structure)
10. [Deployment Guide](#10-deployment-guide)

---

## 1. Executive Summary

### 1.1 Project Purpose
A real-time fraud detection system for banking transactions that analyzes each transaction through a triple-layer protection mechanism and returns either APPROVED or PENDING_REVIEW status.

### 1.2 Key Highlights
- **Triple-Layer Detection**: Rule Engine + Isolation Forest (ML) + Autoencoder (Deep Learning)
- **Real-Time Processing**: Sub-second transaction analysis
- **Account-Level Tracking**: Per CustomerId + AccountNo statistics
- **Session Management**: In-memory tracking for velocity and spending
- **Manual Review Flow**: Pending transactions can be approved/rejected

### 1.3 Business Logic (4 Lines)
1. Transaction aati hai → Customer ID + Account No ke saath
2. 3 checks hote hain → Rule Engine (velocity/limits) + Isolation Forest (ML) + Autoencoder (behavioral)
3. Koi bhi check fail → PENDING_REVIEW with reasons, user approval required
4. Sab pass → APPROVED, transaction safe

---

## 2. Business Overview

### 2.1 Problem Statement
Banks need to detect fraudulent transactions in real-time while minimizing false positives that inconvenience legitimate customers.

### 2.2 Solution Approach
Instead of directly rejecting suspicious transactions, the system flags them for manual review, allowing human oversight while maintaining security.

### 2.3 Transaction Flow
```
Customer initiates transaction
         ↓
    API receives request
         ↓
┌─────────────────────────┐
│   Triple-Layer Check    │
├─────────────────────────┤
│ 1. Rule Engine          │
│ 2. Isolation Forest     │
│ 3. Autoencoder          │
└─────────────────────────┘
         ↓
    Any flag raised?
    ↓           ↓
   Yes          No
    ↓           ↓
PENDING      APPROVED
REVIEW       (Auto-process)
    ↓
Manual Review
    ↓
Approve/Reject
```

### 2.4 Transfer Types & Risk Levels

| Code | Type | Risk Level | Limit Multiplier |
|------|------|------------|------------------|
| S | Overseas | HIGH | 2.0x |
| Q | Quick Transfer | MEDIUM | 2.5x |
| L | UAE Local | LOW | 3.0x |
| I | Ajman Local | LOW | 3.5x |
| O | Own Account | LOWEST | 4.0x |

---

## 3. System Architecture

### 3.1 High-Level Architecture
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Client     │────▶│   FastAPI    │────▶│   Decision   │
│  (Postman)   │     │   Server     │     │   Engine     │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Session    │     │   ML Models  │
                     │   Store      │     │  (IF + AE)   │
                     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  CSV Files   │     │   PKL/H5     │
                     │  (History)   │     │   (Models)   │
                     └──────────────┘     └──────────────┘
```

### 3.2 Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| API Layer | `backend/api.py` | FastAPI endpoints, session management |
| Decision Engine | `backend/hybrid_decision.py` | Combines all 3 detection layers |
| Rule Engine | `backend/rule_engine.py` | Velocity & spending limit checks |
| ML Model | `backend/model.py` | Isolation Forest anomaly detection |
| Deep Learning | `backend/autoencoder.py` | Behavioral pattern analysis |
| Feature Engineering | `backend/feature_engineering.py` | Generate 30+ features from raw data |

### 3.3 Technology Stack
- **Backend**: Python 3.13, FastAPI, Uvicorn
- **ML**: Scikit-learn (Isolation Forest)
- **Deep Learning**: TensorFlow/Keras (Autoencoder)
- **Data Processing**: Pandas, NumPy
- **Storage**: CSV files, Pickle, HDF5

---

## 4. Data Pipeline

### 4.1 Data Sources

| File | Purpose | Records |
|------|---------|---------|
| `data/Clean.csv` | Raw transaction data | 3,502 |
| `data/featured_dataset.csv` | Engineered features | 3,502 |
| `transaction_history.csv` | API transaction log | Dynamic |

### 4.2 Data Flow
```
Clean.csv (Raw Data)
       ↓
Feature Engineering
       ↓
featured_dataset.csv (30+ features)
       ↓
Model Training (IF + AE)
       ↓
models/*.pkl, models/*.h5
       ↓
API Inference
       ↓
transaction_history.csv (Logs)
```

### 4.3 Key Columns in Clean.csv

| Column | Type | Description |
|--------|------|-------------|
| CustomerId | float | Unique customer identifier |
| FromAccountNo | float | Source account number |
| Amount | float | Transaction amount (AED) |
| TransferType | string | S, Q, L, I, O |
| BenId | float | Beneficiary identifier |
| CreateDate | datetime | Transaction timestamp |
| ChannelId | string | Transaction channel |
| BankCountry | string | Beneficiary bank country |

---

## 5. Feature Engineering

### 5.1 Feature Categories (30 Features)

#### Transaction Features
| Feature | Description |
|---------|-------------|
| transaction_amount | Transaction amount |
| flag_amount | 1 if overseas (S), else 0 |
| transfer_type_encoded | S=4, Q=3, L=2, I=1, O=0 |
| transfer_type_risk | S=0.9, Q=0.5, L=0.2, I=0.1, O=0.0 |
| channel_encoded | Encoded channel ID |

#### Time Features
| Feature | Description |
|---------|-------------|
| hour | Transaction hour (0-23) |
| day_of_week | Day (0=Mon, 6=Sun) |
| is_weekend | 1 if Sat/Sun |
| is_night | 1 if hour < 6 or >= 22 |

#### Account-Level Stats (Per CustomerId + AccountNo)
| Feature | Description |
|---------|-------------|
| user_avg_amount | Average transaction amount |
| user_std_amount | Standard deviation |
| user_max_amount | Maximum transaction |
| user_txn_frequency | Total transaction count |
| deviation_from_avg | abs(amount - avg) |
| amount_to_max_ratio | amount / max_amount |
| intl_ratio | % of overseas transfers |
| user_high_risk_txn_ratio | % of S or Q transfers |
| rolling_std | Std of last 5 transactions |

#### Velocity Features
| Feature | Description |
|---------|-------------|
| time_since_last | Seconds since last transaction |
| recent_burst | 1 if time_since_last < 300s |
| txn_count_30s | Transactions in last 30 seconds |
| txn_count_10min | Transactions in last 10 minutes |
| txn_count_1hour | Transactions in last 1 hour |
| transaction_velocity | Total transaction count |

#### Beneficiary Features
| Feature | Description |
|---------|-------------|
| is_new_beneficiary | 1 if first transfer to this BenId |
| beneficiary_txn_count_30d | Total transfers to this BenId |
| beneficiary_risk_score | amount / beneficiary_avg (0-1) |

#### Geographic Features
| Feature | Description |
|---------|-------------|
| geo_anomaly_flag | 0 if UAE/local, 1 if foreign |

### 5.2 Null Handling Rules
- Categorical columns → 'Unknown'
- Numerical columns → 0
- No rows dropped, no mean/median imputation

---

## 6. Detection Layers

### 6.1 Layer 1: Rule Engine

**Purpose**: Hard business rule enforcement

**Checks**:
1. **Velocity Limits**
   - Max 5 transactions in 10 minutes
   - Max 15 transactions in 1 hour

2. **Monthly Spending Limits**
   - Dynamic threshold = user_avg + (multiplier × user_std)
   - Minimum floors per transfer type

**Threshold Calculation**:
```python
threshold = max(user_avg + multiplier * user_std, floor)

# Multipliers: S=2.0, Q=2.5, L=3.0, I=3.5, O=4.0
# Floors: S=5000, Q=3000, L=2000, I=1500, O=1000
```

### 6.2 Layer 2: Isolation Forest (ML)

**Purpose**: Statistical anomaly detection

**Configuration**:
- Trees: 100
- Contamination: 5%
- Features: 26

**Features Used**:
```
transaction_amount, flag_amount, transfer_type_encoded, transfer_type_risk,
channel_encoded, hour, day_of_week, is_weekend, is_night, user_avg_amount,
user_std_amount, user_max_amount, user_txn_frequency, deviation_from_avg,
amount_to_max_ratio, intl_ratio, user_high_risk_txn_ratio,
user_multiple_accounts_flag, cross_account_transfer_ratio, rolling_std,
transaction_velocity, is_new_beneficiary, beneficiary_txn_count_30d,
beneficiary_risk_score, geo_anomaly_flag, recent_burst
```

**Output**: Anomaly score (positive = risky)

### 6.3 Layer 3: Autoencoder (Deep Learning)

**Purpose**: Behavioral pattern analysis

**Architecture**:
```
Input (31) → Dense(64) → Dense(32) → Dense(16) → Dense(32) → Dense(64) → Output (31)
```

**Configuration**:
- Epochs: 100 (early stopping)
- Batch Size: 64
- Loss: MSE
- Threshold: 95th percentile of reconstruction error

**Features Used**: 31 features (26 from IF + velocity + spending features)

**Output**: Reconstruction error (high = anomaly)

---

## 7. API Documentation

### 7.1 Base URL
```
http://127.0.0.1:8000
```

### 7.2 Endpoints

#### Process Transaction
```
POST /api/v1/transaction/analyze

Request:
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 5000,
    "transfer_type": "S",
    "ben_id": 2584644,
    "bank_country": "Germany"
}

Response:
{
    "status": "APPROVED" | "PENDING_REVIEW",
    "message": "...",
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

#### Account Limits
```
GET /api/v1/account/limits/{customer_id}/{account_no}

Response:
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "current_month_spending": 61464.77,
    "csv_spending": 51464.77,
    "session_spending": 10000,
    "limits_by_transfer_type": {
        "S": {"limit": 65255.49, "remaining": 3790.72},
        ...
    }
}
```

#### Pending Transactions
```
GET /api/v1/pending/{customer_id}/{account_no}
POST /api/v1/pending/approve/{customer_id}/{account_no}/{txn_id}
POST /api/v1/pending/reject/{customer_id}/{account_no}/{txn_id}
```

#### Session Management
```
GET /api/v1/session/stats/{customer_id}/{account_no}
POST /api/v1/session/clear
```

#### Health Check
```
GET /health
```

### 7.3 Response Status Values
- `APPROVED`: Transaction is safe, auto-processed
- `PENDING_REVIEW`: Requires manual approval

### 7.4 Flag Meanings
- `rule_flag`: Rule Engine triggered (velocity/spending)
- `ml_flag`: Isolation Forest detected anomaly
- `ae_flag`: Autoencoder detected behavioral anomaly

---

## 8. Database & Storage

### 8.1 File-Based Storage

| File | Purpose | Format |
|------|---------|--------|
| `data/Clean.csv` | Raw transaction data | CSV |
| `data/featured_dataset.csv` | Engineered features | CSV |
| `transaction_history.csv` | API transaction log | CSV |
| `models/isolation_forest.pkl` | Trained IF model | Pickle |
| `models/isolation_forest_scaler.pkl` | IF feature scaler | Pickle |
| `models/autoencoder.h5` | Trained AE model | HDF5 |
| `models/autoencoder_scaler.pkl` | AE feature scaler | Pickle |
| `models/autoencoder_threshold.json` | AE threshold config | JSON |

### 8.2 In-Memory Storage

| Store | Purpose | Structure |
|-------|---------|-----------|
| session_transactions | Track approved transactions | Dict[(customer_id, account_no)] → List[{amount, timestamp}] |
| pending_transactions | Track pending transactions | Dict[(customer_id, account_no)] → List[{txn_id, amount, ...}] |

### 8.3 Customer Data
- **Total Customers**: 41
- **Total Accounts**: 142
- See `Docs/CUSTOMER_ACCOUNTS.md` for full list

---

## 9. Project Structure

```
frauduelent-transactions-detector/
├── backend/
│   ├── api.py                    # FastAPI endpoints
│   ├── hybrid_decision.py        # 3-layer decision logic
│   ├── rule_engine.py            # Velocity & limit checks
│   ├── model.py                  # Isolation Forest
│   ├── autoencoder.py            # Autoencoder inference
│   ├── train_autoencoder.py      # Autoencoder training
│   ├── feature_engineering.py    # Feature generation
│   ├── utils.py                  # Helper functions
│   └── __init__.py
├── data/
│   ├── Clean.csv                 # Raw transaction data
│   └── featured_dataset.csv      # Engineered features
├── models/
│   ├── isolation_forest.pkl      # Trained IF model
│   ├── isolation_forest_scaler.pkl
│   ├── autoencoder.h5            # Trained AE model
│   ├── autoencoder_scaler.pkl
│   └── autoencoder_threshold.json
├── Docs/
│   ├── API_DOCUMENTATION.md
│   ├── FEATURES_DOCUMENTATION.md
│   ├── CUSTOMER_ACCOUNTS.md
│   └── PROJECT_COMPLETE_DOCUMENTATION.md
├── postman/
│   └── Fraud_Detection_API.postman_collection.json
├── transaction_history.csv       # API transaction log
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 10. Deployment Guide

### 10.1 Prerequisites
- Python 3.8+
- pip package manager

### 10.2 Installation
```bash
# Clone repository
git clone https://github.com/tahah02/frauduelent-transactions-detector.git
cd frauduelent-transactions-detector

# Install dependencies
pip install -r requirements.txt
```

### 10.3 Running the API
```bash
uvicorn backend.api:app --reload
```

API available at: `http://127.0.0.1:8000`

Swagger Docs: `http://127.0.0.1:8000/docs`

### 10.4 Testing with Postman
1. Import `postman/Fraud_Detection_API.postman_collection.json`
2. Set collection variables (customer_id, account_no)
3. Test endpoints

### 10.5 Retraining Models

**Isolation Forest**:
```bash
python -m backend.model
```

**Autoencoder**:
```bash
python -m backend.train_autoencoder
```

### 10.6 Regenerating Features
```bash
python -m backend.feature_engineering
```

---

## Appendix

### A. Sample Test Cases

**Case 1: Normal Transaction (APPROVED)**
```json
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 1000,
    "transfer_type": "L",
    "bank_country": "UAE"
}
```

**Case 2: Limit Exceeded (PENDING_REVIEW - Rule Flag)**
```json
{
    "customer_id": 4424492,
    "account_no": 14424492014,
    "amount": 15000,
    "transfer_type": "S",
    "bank_country": "Germany"
}
```

**Case 3: Anomaly Detected (PENDING_REVIEW - ML Flag)**
```json
{
    "customer_id": 1000016,
    "account_no": 11000016019,
    "amount": 50000,
    "transfer_type": "S",
    "ben_id": 999999,
    "bank_country": "Nigeria"
}
```

### B. Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Account not found |
| 422 | Validation error |
| 500 | Internal server error |

### C. Related Documentation
- [API Documentation](API_DOCUMENTATION.md)
- [Features Documentation](FEATURES_DOCUMENTATION.md)
- [Customer Accounts](CUSTOMER_ACCOUNTS.md)
