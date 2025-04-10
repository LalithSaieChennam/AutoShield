# dashboard.py
# Streamlit dashboard with dynamic column detection for any uploaded CSV

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.anomaly_detection import detect_anomalies, detect_anomalies_lof

st.set_page_config(page_title="AutoShield Dashboard", layout="wide")
st.title("🛡️ AutoShield – Anomaly Detection Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV (must include timestamp + value)", type=["csv"])
default_path = "data/processed/ambient_temperature_processed.csv"

try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # --- Auto-detect timestamp + numeric value ---
        time_col, value_col = None, None

        # Try to detect timestamp column
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col])
                if parsed.notnull().sum() > 0:
                    df[col] = parsed
                    time_col = col
                    break
            except:
                continue

        # Detect first numeric column
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]

        # Safety check
        if not time_col or not value_col:
            st.error("❌ Could not find timestamp and numeric column.")
            st.stop()

        # Rename for internal consistency
        df.rename(columns={time_col: 'timestamp', value_col: 'value'}, inplace=True)
        df = df.sort_values('timestamp')
        df['value_scaled'] = (df['value'] - df['value'].mean()) / df['value'].std()

    else:
        st.info("No file uploaded. Loading default dataset.")
        df = pd.read_csv(default_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sidebar controls
    st.sidebar.title("⚙️ Model Settings")
    contamination = st.sidebar.slider("Contamination", 0.001, 0.1, 0.01, 0.001)
    model_choice = st.sidebar.selectbox("Model", ["Isolation Forest", "Local Outlier Factor (LOF)"])

    if st.sidebar.button("🚀 Run Anomaly Detection"):
        if model_choice == "Isolation Forest":
            df = detect_anomalies(df, contamination=contamination)
        elif model_choice == "Local Outlier Factor (LOF)":
            df = detect_anomalies_lof(df, contamination=contamination)

        # Plot
        fig = px.line(df, x="timestamp", y="value", title="Sensor Readings with Anomalies")
        anomalies = df[df['anomaly'] == 1]
        fig.add_scatter(
            x=anomalies['timestamp'], y=anomalies['value'],
            mode='markers', marker=dict(color='red', size=8),
            name='Anomalies'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Points", len(df))
        col2.metric("Anomalies Found", len(anomalies))
        col3.metric("Anomaly %", f"{(len(anomalies) / len(df)) * 100:.2f}%")

        # Table
        st.subheader("📋 Anomaly Details")
        st.dataframe(anomalies[['timestamp', 'value', 'value_scaled']].reset_index(drop=True), use_container_width=True)

        # Download
        csv = anomalies.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Anomalies as CSV", csv, "detected_anomalies.csv", "text/csv")

    else:
        st.warning("Adjust settings and click 'Run Anomaly Detection' to begin.")

except Exception as e:
    st.error(f"❌ Error: {e}")
