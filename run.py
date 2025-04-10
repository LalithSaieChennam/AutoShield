# run.py
# This script runs the full AutoShield pipeline in CLI mode.
# It performs data loading, preprocessing, anomaly detection, and saves the final output.

from src.preprocessing import load_and_process
from src.anomaly_detection import detect_anomalies

if __name__ == "__main__":
    # Step 1: Load and preprocess the raw CSV log file.
    # This function handles timestamp parsing, sorting, and value normalization.
    df = load_and_process(
        filepath="data/raw/ambient_temperature_system_failure.csv",
        save_to="data/processed/ambient_temperature_processed.csv"
    )

    # Step 2: Detect anomalies using an unsupervised ML model (Isolation Forest).
    # The 'contamination' parameter controls how aggressive the model is in flagging outliers.
    df = detect_anomalies(df, contamination=0.01)

    # Step 3: Save the processed + labeled results into the 'outputs' directory.
    # The output will include a binary 'anomaly' column: 1 = anomaly, 0 = normal.
    df.to_csv("outputs/ambient_temperature_labeled.csv", index=False)

    # Final log to confirm processing and print sample output.
    print("Anomaly detection complete. Labeled file saved to outputs/")
    print(df.head())
