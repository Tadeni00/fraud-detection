# src/feature_engineering.py
import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame, target: str | None = None):
    """
    Lightweight feature construction:
     - expand timestamp into hour/day/month/weekend
     - per-user time delta (seconds) since previous tx
     - per-user aggregated counts & mean amount
     - log amount + amount per tenure + high-amount flag
    Returns: (X_df, y_series) if target given, else (df_enriched, None)
    """
    df = df.copy()
    # timestamp features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(['user_id', 'timestamp'])
        df['ts_hour'] = df['timestamp'].dt.hour
        df['ts_dayofweek'] = df['timestamp'].dt.dayofweek
        df['ts_month'] = df['timestamp'].dt.month
        df['ts_is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        # time diff in seconds per user
        df['prev_ts'] = df.groupby('user_id')['timestamp'].shift(1)
        df['time_diff_s'] = (df['timestamp'] - df['prev_ts']).dt.total_seconds().fillna(-1)
        df = df.drop(columns=['prev_ts'])
    else:
        df['time_diff_s'] = -1

    # per-user aggregates (fast)
    if 'user_id' in df.columns:
        agg = df.groupby('user_id')['amount'].agg(['count','mean']).rename(columns={'count':'user_tx_count','mean':'user_amt_mean'})
        df = df.join(agg, on='user_id')
    else:
        df['user_tx_count'] = 1
        df['user_amt_mean'] = df['amount'].fillna(0)

    # derived numeric
    if 'amount' in df.columns:
        df['amount_log1p'] = np.log1p(df['amount'].fillna(0))
        if 'account_tenure_months' in df.columns:
            df['amount_per_tenure'] = df['amount'] / df['account_tenure_months'].replace(0, np.nan)
            df['amount_per_tenure'] = df['amount_per_tenure'].fillna(df['amount'])
        else:
            df['amount_per_tenure'] = df['amount']
        df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    else:
        df['amount_log1p'] = 0.0
        df['amount_per_tenure'] = 0.0
        df['is_high_amount'] = 0

    # simple categorical normalisation (lowercase)
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({'': pd.NA, 'nan': pd.NA})

    # drop raw text timestamp/ids in X (keep user_id if you want user-level signals)
    X = df.copy()
    if target is not None and target in X.columns:
        y = X[target].astype(int)
        # keep target out of X
        X = X.drop(columns=[target])
    else:
        y = None

    return X, y