# preprocessing.py
# This module handles all preprocessing steps needed before running anomaly detection.
# It's designed to work with time-series sensor or system logs.

import pandas as pd
import os

def load_and_process(filepath, save_to=None):
    """
    Loads and preprocesses raw time-series CSV data.

    - Assumes the CSV has two columns: timestamp and value
    - Converts timestamp strings into pandas datetime format
    - Sorts data by timestamp to ensure chronological order
    - Normalizes the 'value' column using z-score normalization
      to ensure scale-invariance for ML models

    Parameters:
    - filepath (str): path to the raw CSV file
    - save_to (str): optional path to save the cleaned version

    Returns:
    - df (DataFrame): cleaned and normalized data
    """
    df = pd.read_csv(filepath)

    # Rename columns to standard format if needed
    df.columns = ['timestamp', 'value']

    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by time just in case the raw data is unordered
    df = df.sort_values(by='timestamp')

    # Normalize values: (x - mean) / std
    df['value_scaled'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Save the processed file if requested
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        df.to_csv(save_to, index=False)
        print(f"Processed data saved to: {save_to}")

    return df
