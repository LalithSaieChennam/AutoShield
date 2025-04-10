# run.py

from src.preprocessing import load_and_process
from src.anomaly_detection import detect_anomalies

if __name__ == "__main__":
    df = load_and_process(
        filepath="data/raw/ambient_temperature_system_failure.csv",
        save_to="data/processed/ambient_temperature_processed.csv"
    )

    df = detect_anomalies(df, contamination=0.01)

    df.to_csv("outputs/ambient_temperature_labeled.csv", index=False)
    print("✅ Anomaly detection complete. Labeled file saved to outputs/")
    print(df.head())
