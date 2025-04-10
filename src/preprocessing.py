# src/preprocessing.py

import pandas as pd
import os

def load_and_process(filepath, save_to=None):
    """
    Load and preprocess the time-series dataset:
    - Parses timestamp
    - Sorts by time
    - Normalizes the 'value' column
    - Optionally saves processed data
    """

    # Load
    df = pd.read_csv(filepath)
    df.columns = ['timestamp', 'value']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    # Normalize the value
    df['value_scaled'] = (df['value'] - df['value'].mean()) / df['value'].std()

    # Save to processed/ folder if needed
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        df.to_csv(save_to, index=False)
        print(f"✅ Processed data saved to: {save_to}")

    return df
