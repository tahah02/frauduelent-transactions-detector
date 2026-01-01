import pandas as pd
import numpy as np
from backend.utils import ensure_data_dir, get_clean_csv_path
from backend.config import get_config

config = get_config()


def engineer_features(output_path='data/featured_dataset.csv'):
    ensure_data_dir()
    df = pd.read_csv(get_clean_csv_path())
    print(f"Loaded {len(df)} rows from Clean.csv")
    
    # Handle nulls - categorical to 'Unknown', numerical to 0
    categorical_cols = ['TransferType', 'FromAccountCurrency', 'SwiftCode', 'Currency', 
                        'PurposeCode', 'Charges', 'BankName', 'PurposeDetails', 
                        'AccountType', 'BankCountry', 'FlagCurrency']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    text_cols = ['ReceipentAccount', 'ReceipentName']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    
    id_cols = ['CustomerId', 'FromAccountNo', 'BenId', 'ChannelId']
    for col in id_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    numerical_cols = ['Amount', 'FlagAmount', 'AmountInAed', 'ChargesAmount', 'Status', 'BankStatus']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Parse date
    if 'CreateDate' in df.columns:
        df['CreateDate'] = pd.to_datetime(df['CreateDate'], errors='coerce')
    
    # Basic features
    df['transaction_amount'] = df['Amount'].fillna(0)
    
    if 'TransferType' in df.columns:
        df['flag_amount'] = df['TransferType'].apply(lambda x: 1 if str(x).upper() == 'S' else 0)
        df['transfer_type_encoded'] = df['TransferType'].apply(lambda x: config.TRANSFER_TYPE_ENCODED.get(str(x).upper(), 0))
        df['transfer_type_risk'] = df['TransferType'].apply(lambda x: config.TRANSFER_TYPE_RISK.get(str(x).upper(), 0.5))
    else:
        df['flag_amount'], df['transfer_type_encoded'], df['transfer_type_risk'] = 0, 0, 0.5
    
    # Channel encoding
    df['channel_encoded'] = 0
    if 'ChannelId' in df.columns:
        mapping = {v: i for i, v in enumerate(df['ChannelId'].dropna().unique())}
        df['channel_encoded'] = df['ChannelId'].map(mapping).fillna(0).astype(int)
    
    # Time features
    if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
        df['hour'] = df['CreateDate'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['CreateDate'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    else:
        df['hour'], df['day_of_week'], df['is_weekend'], df['is_night'] = 12, 0, 0, 0
    
    # Account-level stats
    if 'CustomerId' in df.columns and 'FromAccountNo' in df.columns:
        print("Calculating account-level statistics...")
        
        account_stats = df.groupby(['CustomerId', 'FromAccountNo'])['transaction_amount'].agg(
            ['mean', 'std', 'max', 'count']
        ).reset_index()
        account_stats.columns = ['CustomerId', 'FromAccountNo', 'user_avg_amount', 
                                 'user_std_amount', 'user_max_amount', 'user_txn_frequency']
        account_stats['user_std_amount'] = account_stats['user_std_amount'].fillna(0)
        
        df = df.merge(account_stats, on=['CustomerId', 'FromAccountNo'], how='left')
        df['user_avg_amount'] = df['user_avg_amount'].fillna(0)
        df['user_std_amount'] = df['user_std_amount'].fillna(0)
        df['user_max_amount'] = df['user_max_amount'].fillna(0)
        df['user_txn_frequency'] = df['user_txn_frequency'].fillna(0)
        
        df['deviation_from_avg'] = abs(df['transaction_amount'] - df['user_avg_amount'])
        df['amount_to_max_ratio'] = df['transaction_amount'] / df['user_max_amount'].replace(0, 1)
        
        # Ratios per account
        if 'TransferType' in df.columns:
            intl_ratio = df.groupby(['CustomerId', 'FromAccountNo']).apply(
                lambda x: (x['TransferType'].str.upper() == 'S').sum() / len(x) if len(x) > 0 else 0,
                include_groups=False
            ).reset_index(name='intl_ratio')
            df = df.merge(intl_ratio, on=['CustomerId', 'FromAccountNo'], how='left')
            df['intl_ratio'] = df['intl_ratio'].fillna(0)
            
            high_risk = df.groupby(['CustomerId', 'FromAccountNo']).apply(
                lambda x: (x['TransferType'].str.upper().isin(['S', 'Q'])).sum() / len(x) if len(x) > 0 else 0,
                include_groups=False
            ).reset_index(name='user_high_risk_txn_ratio')
            df = df.merge(high_risk, on=['CustomerId', 'FromAccountNo'], how='left')
            df['user_high_risk_txn_ratio'] = df['user_high_risk_txn_ratio'].fillna(0)
        else:
            df['intl_ratio'] = 0
            df['user_high_risk_txn_ratio'] = 0
        
        # Multiple accounts flag
        account_counts = df.groupby('CustomerId')['FromAccountNo'].nunique().reset_index()
        account_counts.columns = ['CustomerId', 'num_accounts']
        df = df.merge(account_counts, on='CustomerId', how='left')
        df['user_multiple_accounts_flag'] = (df['num_accounts'] > 1).astype(int)
        df.drop(columns=['num_accounts'], inplace=True)
        
        # Cross account transfer ratio
        if 'BenId' in df.columns:
            cross_transfer = df.groupby('CustomerId').apply(
                lambda x: (x['BenId'] == -1).sum() / len(x) if len(x) > 0 else 0,
                include_groups=False
            ).reset_index(name='cross_account_transfer_ratio')
            df = df.merge(cross_transfer, on='CustomerId', how='left')
            df['cross_account_transfer_ratio'] = df['cross_account_transfer_ratio'].fillna(0)
        else:
            df['cross_account_transfer_ratio'] = 0
        
        # Velocity features
        if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
            df = df.sort_values(['CustomerId', 'FromAccountNo', 'CreateDate']).reset_index(drop=True)
            
            df['time_since_last'] = df.groupby(['CustomerId', 'FromAccountNo'])['CreateDate'].diff().dt.total_seconds().fillna(3600)
            df['recent_burst'] = (df['time_since_last'] < 300).astype(int)
            df['last_txn_time'] = df.groupby(['CustomerId', 'FromAccountNo'])['CreateDate'].shift(1)
            
            def count_txns_in_window(group, seconds):
                counts = []
                timestamps = group['CreateDate'].values
                for i, ts in enumerate(timestamps):
                    if pd.isna(ts):
                        counts.append(1)
                        continue
                    window_start = ts - np.timedelta64(seconds, 's')
                    count = np.sum((timestamps[:i] >= window_start) & (timestamps[:i] <= ts)) + 1
                    counts.append(count)
                return pd.Series(counts, index=group.index)
            
            df['txn_count_30s'] = df.groupby(['CustomerId', 'FromAccountNo'], group_keys=False).apply(
                lambda g: count_txns_in_window(g, 30), include_groups=False
            ).fillna(1)
            df['txn_count_10min'] = df.groupby(['CustomerId', 'FromAccountNo'], group_keys=False).apply(
                lambda g: count_txns_in_window(g, 600), include_groups=False
            ).fillna(1)
            df['txn_count_1hour'] = df.groupby(['CustomerId', 'FromAccountNo'], group_keys=False).apply(
                lambda g: count_txns_in_window(g, 3600), include_groups=False
            ).fillna(1)
            
            # Aggregates
            df['hour_key'] = df['CreateDate'].dt.floor('h')
            hourly_stats = df.groupby(['CustomerId', 'FromAccountNo', 'hour_key'])['transaction_amount'].agg(['sum', 'count'])
            hourly_stats.columns = ['hourly_total', 'hourly_count']
            df = df.merge(hourly_stats.reset_index(), on=['CustomerId', 'FromAccountNo', 'hour_key'], how='left')
            df.drop(columns=['hour_key'], inplace=True)
            
            df['day_key'] = df['CreateDate'].dt.floor('D')
            daily_stats = df.groupby(['CustomerId', 'FromAccountNo', 'day_key'])['transaction_amount'].agg(['sum', 'count'])
            daily_stats.columns = ['daily_total', 'daily_count']
            df = df.merge(daily_stats.reset_index(), on=['CustomerId', 'FromAccountNo', 'day_key'], how='left')
            df.drop(columns=['day_key'], inplace=True)
            
            df['month_key'] = df['CreateDate'].dt.to_period('M')
            monthly_stats = df.groupby(['CustomerId', 'FromAccountNo', 'month_key'])['transaction_amount'].sum().reset_index()
            monthly_stats.columns = ['CustomerId', 'FromAccountNo', 'month_key', 'current_month_spending']
            df = df.merge(monthly_stats, on=['CustomerId', 'FromAccountNo', 'month_key'], how='left')
            df['current_month_spending'] = df['current_month_spending'].fillna(0)
            df.drop(columns=['month_key'], inplace=True)
        else:
            df['time_since_last'], df['recent_burst'] = 3600, 0
            df['last_txn_time'] = None
            df['txn_count_30s'], df['txn_count_10min'], df['txn_count_1hour'] = 1, 1, 1
            df['hourly_total'], df['hourly_count'] = df['transaction_amount'], 1
            df['daily_total'], df['daily_count'] = df['transaction_amount'], 1
            df['current_month_spending'] = df['transaction_amount']
        
        df['rolling_std'] = df.groupby(['CustomerId', 'FromAccountNo'])['transaction_amount'].transform(
            lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std()
        ).fillna(0)
        
        df['transaction_velocity'] = df.groupby(['CustomerId', 'FromAccountNo'])['transaction_amount'].transform('count')
    else:
        df['user_avg_amount'] = df['user_std_amount'] = df['user_max_amount'] = df['user_txn_frequency'] = 0
        df['deviation_from_avg'] = df['amount_to_max_ratio'] = df['intl_ratio'] = 0
        df['user_high_risk_txn_ratio'] = df['user_multiple_accounts_flag'] = df['cross_account_transfer_ratio'] = 0
        df['time_since_last'], df['recent_burst'] = 3600, 0
        df['last_txn_time'] = None
        df['txn_count_30s'], df['txn_count_10min'], df['txn_count_1hour'] = 1, 1, 1
        df['hourly_total'], df['hourly_count'] = df['transaction_amount'], 1
        df['daily_total'], df['daily_count'] = df['transaction_amount'], 1
        df['current_month_spending'] = df['transaction_amount']
        df['rolling_std'] = df['transaction_velocity'] = 0
    
    # BenId features
    print("Calculating BenId features...")
    if 'BenId' in df.columns:
        if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
            ben_first_txn = df.groupby('BenId')['CreateDate'].transform('min')
            df['is_new_beneficiary'] = (df['CreateDate'] == ben_first_txn).astype(int)
        else:
            df['is_new_beneficiary'] = 0
        
        ben_counts = df.groupby('BenId').size().reset_index(name='beneficiary_txn_count_30d')
        df = df.merge(ben_counts, on='BenId', how='left')
        df['beneficiary_txn_count_30d'] = df['beneficiary_txn_count_30d'].fillna(0)
        
        ben_avg = df.groupby('BenId')['transaction_amount'].mean().reset_index()
        ben_avg.columns = ['BenId', 'ben_avg_amount']
        df = df.merge(ben_avg, on='BenId', how='left')
        df['beneficiary_risk_score'] = (df['transaction_amount'] / df['ben_avg_amount'].replace(0, 1)).clip(0, 1)
        df.drop(columns=['ben_avg_amount'], inplace=True)
    else:
        df['is_new_beneficiary'] = 0
        df['beneficiary_txn_count_30d'] = 0
        df['beneficiary_risk_score'] = 0.5
    
    # Geo anomaly flag
    if 'BankCountry' in df.columns:
        df['geo_anomaly_flag'] = df['BankCountry'].apply(
            lambda x: 0 if str(x).upper() in [c.upper() for c in config.LOCAL_COUNTRIES] else 1
        )
    else:
        df['geo_anomaly_flag'] = df['flag_amount']
    
    # Placeholders
    df['applied_window'] = ''
    df['threshold_used'] = 0.0
    df['flag_reason'] = ''
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    engineer_features()
