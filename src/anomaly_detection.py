# src/anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, contamination=0.01, return_model=False):
    """
    Applies IsolationForest to detect anomalies in the time-series.
    Returns dataframe with 'anomaly' column and optionally the model.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df[['value_scaled']])
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    
    if return_model:
        return df, model
    return df
