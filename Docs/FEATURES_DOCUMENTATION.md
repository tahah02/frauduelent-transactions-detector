# Feature Engineering Documentation

## What We Did (Latest Update)

### Key Changes:
1. **Account-Level Stats** - Purani approach mein stats sirf CustomerId pe the, ab CustomerId + FromAccountNo pe hain
2. **BenId Features Added** - Beneficiary related features jo pehle missing the
3. **Velocity Per Account** - Burst detection ab per account hai, per customer nahi
4. **Missing Features Added** - 6 new features jo model expect karta tha
5. **Consistent Null Handling** - Numerical → 0, Categorical → 'Unknown'

---

## Feature List (30 Features)

### 1. Transaction Features

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `transaction_amount` | Transaction ki amount | Direct from `Amount` column |
| `flag_amount` | Overseas transfer flag | 1 if TransferType='S', else 0 |
| `transfer_type_encoded` | Transfer type as number | S=4, Q=3, L=2, I=1, O=0 |
| `transfer_type_risk` | Risk score per transfer type | S=0.9, Q=0.5, L=0.2, I=0.1, O=0.0 |
| `channel_encoded` | Channel ID encoded | Unique mapping per ChannelId |

### 2. Time Features

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `hour` | Transaction hour (0-23) | Extracted from CreateDate |
| `day_of_week` | Day of week (0=Mon, 6=Sun) | Extracted from CreateDate |
| `is_weekend` | Weekend flag | 1 if Saturday/Sunday, else 0 |
| `is_night` | Night transaction flag | 1 if hour < 6 or hour >= 22 |

### 3. Account-Level User Stats (NEW: Per CustomerId + FromAccountNo)

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `user_avg_amount` | Average transaction amount | Mean of all transactions for this account |
| `user_std_amount` | Standard deviation | Std of all transactions for this account |
| `user_max_amount` | Maximum transaction | Max of all transactions for this account |
| `user_txn_frequency` | Total transaction count | Count of all transactions for this account |
| `deviation_from_avg` | How different from average | abs(transaction_amount - user_avg_amount) |
| `amount_to_max_ratio` | Ratio to max amount | transaction_amount / user_max_amount |
| `intl_ratio` | International transfer ratio | % of 'S' type transfers for this account |
| `user_high_risk_txn_ratio` | High risk transfer ratio | % of 'S' or 'Q' transfers for this account |
| `rolling_std` | Rolling standard deviation | Std of last 5 transactions for this account |

### 4. Customer-Level Features

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `user_multiple_accounts_flag` | Multiple accounts flag | 1 if customer has > 1 account |
| `cross_account_transfer_ratio` | Cross account transfers | % of transfers where BenId = -1 |

### 5. Velocity Features (NEW: Per Account)

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `time_since_last` | Seconds since last transaction | Diff between current and previous CreateDate (per account) |
| `recent_burst` | Burst activity flag | 1 if time_since_last < 300 seconds |
| `txn_count_30s` | Transactions in 30 seconds | Count of transactions in last 30s window (per account) |
| `txn_count_10min` | Transactions in 10 minutes | Count of transactions in last 10min window (per account) |
| `txn_count_1hour` | Transactions in 1 hour | Count of transactions in last 1hr window (per account) |
| `transaction_velocity` | Total transaction count | Same as user_txn_frequency |

### 6. Aggregate Features (Per Account)

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `hourly_total` | Total spent this hour | Sum of amounts in current hour (per account) |
| `hourly_count` | Transactions this hour | Count of transactions in current hour (per account) |
| `daily_total` | Total spent today | Sum of amounts today (per account) |
| `daily_count` | Transactions today | Count of transactions today (per account) |
| `current_month_spending` | Monthly spending | Sum of amounts this month (per account) |

### 7. Beneficiary (BenId) Features (NEW)

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `is_new_beneficiary` | First time transfer to this beneficiary | 1 if this is the first transaction to this BenId |
| `beneficiary_txn_count_30d` | Total transfers to this beneficiary | Count of all transactions to this BenId |
| `beneficiary_risk_score` | Beneficiary risk score | transaction_amount / beneficiary_avg_amount (clipped 0-1) |

### 8. Geographic Features

| Feature | Description | How It's Calculated |
|---------|-------------|---------------------|
| `geo_anomaly_flag` | Foreign country flag | 0 if UAE/local, 1 if foreign country |

---

## Data Source

- **Input:** `data/Clean.csv`
- **Output:** `data/featured_dataset.csv`
- **Rows:** 3502
- **Columns:** 63 (30 ML features + original columns)

---

## Null Handling Rules

| Column Type | Fill Value |
|-------------|------------|
| Categorical (TransferType, Currency, etc.) | 'Unknown' |
| Text (ReceipentName, etc.) | 'Unknown' |
| Identifiers (CustomerId, BenId, etc.) | 0 |
| Numerical (Amount, etc.) | 0 |

**Note:** No rows are dropped. All nulls are filled with 0 or 'Unknown'.

---

## Key Difference: Account-Level vs Customer-Level

### Old Approach (Customer-Level):
```
Customer 1000016 → user_avg = 8038 (same for ALL accounts)
```

### New Approach (Account-Level):
```
Customer 1000016, Account 11000016019 → user_avg = 8038
Customer 1000016, Account 11000016033 → user_avg = 1898
Customer 1000016, Account 11000016084 → user_avg = 100
```

This is more accurate because each account has its own spending pattern!

---

## Usage

```python
from backend.feature_engineering import engineer_features

# Generate featured_dataset.csv
df = engineer_features()
```

Or run directly:
```bash
python -m backend.feature_engineering
```
