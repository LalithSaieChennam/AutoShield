# src/anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, contamination=0.01):
    """
    Applies IsolationForest to detect anomalies in the time-series.
    
    Args:
        df: DataFrame with a 'value_scaled' column
        contamination: expected proportion of outliers

    Returns:
        df: DataFrame with an added 'anomaly' column (1=anomaly, 0=normal)
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df[['value_scaled']])
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    return df
