# src/anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest

# LOF-based anomaly detection
from sklearn.neighbors import LocalOutlierFactor

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

def detect_anomalies_lof(df, contamination=0.01):
    """
    Applies Local Outlier Factor to detect anomalies in the time-series.

    LOF compares the local density of each point to its neighbors — 
    lower-density points are flagged as anomalies.

    Parameters:
    - df (DataFrame): must include 'value_scaled'
    - contamination (float): % of expected anomalies

    Returns:
    - df (DataFrame): with a new column 'anomaly' (1 = anomaly, 0 = normal)
    """
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)

    # LOF uses fit_predict, but doesn't support predict after fitting
    preds = model.fit_predict(df[['value_scaled']])
    
    # LOF returns -1 for anomaly, 1 for normal
    df['anomaly'] = [1 if p == -1 else 0 for p in preds]
    
    return df