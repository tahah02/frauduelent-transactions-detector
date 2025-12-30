import pandas as pd
import numpy as np
from backend.utils import ensure_data_dir, get_clean_csv_path, get_feature_engineered_path, TRANSFER_TYPE_ENCODED, TRANSFER_TYPE_RISK

def engineer_features():
    ensure_data_dir()
    df = pd.read_csv(get_clean_csv_path())
    
    # Parse CreateDate
    if 'CreateDate' in df.columns:
        df['CreateDate'] = pd.to_datetime(df['CreateDate'], errors='coerce')
    
    # Basic transaction amount
    df['transaction_amount'] = pd.to_numeric(df.get('Amount', 0), errors='coerce').fillna(0)
    
    # Transfer type features
    if 'TransferType' in df.columns:
        df['flag_amount'] = df['TransferType'].apply(lambda x: 1 if str(x).upper() == 'S' else 0)
        df['transfer_type_encoded'] = df['TransferType'].apply(lambda x: TRANSFER_TYPE_ENCODED.get(str(x).upper(), 0))
        df['transfer_type_risk'] = df['TransferType'].apply(lambda x: TRANSFER_TYPE_RISK.get(str(x).upper(), 0.5))
    else:
        df['flag_amount'], df['transfer_type_encoded'], df['transfer_type_risk'] = 0, 0, 0.5
    
    # Channel encoding
    df['channel_encoded'] = 0
    if 'ChannelId' in df.columns:
        mapping = {v: i for i, v in enumerate(df['ChannelId'].dropna().unique())}
        df['channel_encoded'] = df['ChannelId'].map(mapping).fillna(0).astype(int)
    
    # Time-based features
    if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
        df['hour'] = df['CreateDate'].dt.hour.fillna(12).astype(int)
        df['day_of_week'] = df['CreateDate'].dt.dayofweek.fillna(0).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    else:
        df['hour'], df['day_of_week'], df['is_weekend'], df['is_night'] = 12, 0, 0, 0
    
    # User behavior features
    if 'CustomerId' in df.columns:
        stats = df.groupby('CustomerId')['transaction_amount'].agg(['mean', 'std', 'max', 'count'])
        stats.columns = ['user_avg_amount', 'user_std_amount', 'user_max_amount', 'user_txn_frequency']
        stats['user_std_amount'] = stats['user_std_amount'].fillna(0)
        df = df.merge(stats.reset_index(), on='CustomerId', how='left')
        
        df['deviation_from_avg'] = abs(df['transaction_amount'] - df['user_avg_amount'])
        df['amount_to_max_ratio'] = df['transaction_amount'] / df['user_max_amount'].replace(0, 1)
        
        # International ratio per user
        if 'TransferType' in df.columns:
            df['intl_ratio'] = df.groupby('CustomerId')['flag_amount'].transform('mean')
        else:
            df['intl_ratio'] = 0
        
        # Sort by customer and time for velocity calculations
        if 'CreateDate' in df.columns and df['CreateDate'].notna().any():
            df = df.sort_values(['CustomerId', 'CreateDate']).reset_index(drop=True)
            
            # Time since last transaction
            df['time_since_last'] = df.groupby('CustomerId')['CreateDate'].diff().dt.total_seconds().fillna(3600)
            df['recent_burst'] = (df['time_since_last'] < 300).astype(int)
            
            # Last transaction time (for reference)
            df['last_txn_time'] = df.groupby('CustomerId')['CreateDate'].shift(1)

            # Velocity features - transactions in time windows
            # txn_count_30s: transactions within 30 seconds
            # txn_count_10min: transactions within 10 minutes
            # txn_count_1hour: transactions within 1 hour
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
            
            df['txn_count_30s'] = df.groupby('CustomerId', group_keys=False).apply(lambda g: count_txns_in_window(g, 30))
            df['txn_count_10min'] = df.groupby('CustomerId', group_keys=False).apply(lambda g: count_txns_in_window(g, 600))
            df['txn_count_1hour'] = df.groupby('CustomerId', group_keys=False).apply(lambda g: count_txns_in_window(g, 3600))
            
            # Hourly aggregates
            df['hour_key'] = df['CreateDate'].dt.floor('H')
            hourly_stats = df.groupby(['CustomerId', 'hour_key'])['transaction_amount'].agg(['sum', 'count'])
            hourly_stats.columns = ['hourly_total', 'hourly_count']
            df = df.merge(hourly_stats.reset_index(), on=['CustomerId', 'hour_key'], how='left')
            df.drop(columns=['hour_key'], inplace=True)
            
            # Daily aggregates
            df['day_key'] = df['CreateDate'].dt.floor('D')
            daily_stats = df.groupby(['CustomerId', 'day_key'])['transaction_amount'].agg(['sum', 'count'])
            daily_stats.columns = ['daily_total', 'daily_count']
            df = df.merge(daily_stats.reset_index(), on=['CustomerId', 'day_key'], how='left')
            df.drop(columns=['day_key'], inplace=True)
            
        else:
            df['time_since_last'], df['recent_burst'] = 3600, 0
            df['last_txn_time'] = None
            df['txn_count_30s'], df['txn_count_10min'], df['txn_count_1hour'] = 1, 1, 1
            df['hourly_total'], df['hourly_count'] = df['transaction_amount'], 1
            df['daily_total'], df['daily_count'] = df['transaction_amount'], 1
        
        # Rolling standard deviation (last 5 transactions)
        df['rolling_std'] = df.groupby('CustomerId')['transaction_amount'].transform(
            lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std()
        ).fillna(0)
        
    else:
        # Fallback if no CustomerId
        df['user_avg_amount'] = df['transaction_amount'].mean()
        df['user_std_amount'] = df['transaction_amount'].std()
        df['user_max_amount'] = df['transaction_amount'].max()
        df['user_txn_frequency'] = len(df)
        df['deviation_from_avg'], df['amount_to_max_ratio'] = 0, 0
        df['intl_ratio'] = 0
        df['time_since_last'], df['recent_burst'] = 3600, 0
        df['last_txn_time'] = None
        df['txn_count_30s'], df['txn_count_10min'], df['txn_count_1hour'] = 1, 1, 1
        df['hourly_total'], df['hourly_count'] = df['transaction_amount'], 1
        df['daily_total'], df['daily_count'] = df['transaction_amount'], 1
        df['rolling_std'] = 0
    
    # Placeholder columns for rule engine output (populated during analysis)
    df['applied_window'] = ''
    df['threshold_used'] = 0.0
    df['flag_reason'] = ''
    
    # Save to feature_engineered_data.csv
    df.to_csv(get_feature_engineered_path(), index=False)
    print(f"Features saved to {get_feature_engineered_path()}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    engineer_features()
