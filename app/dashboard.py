# dashboard.py
# Streamlit dashboard for running and visualizing anomaly detection
# Users can choose between Isolation Forest and LOF, tune contamination, and download results.

import sys
import os

# Add project root to path for clean src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.anomaly_detection import detect_anomalies

# Set page config
st.set_page_config(page_title="AutoShield Dashboard", layout="wide")

st.title("🛡️ AutoShield – Anomaly Detection Dashboard")

# Upload CSV or fallback to default processed data
uploaded_file = st.file_uploader("Upload a processed CSV", type=["csv"])
default_path = "data/processed/ambient_temperature_processed.csv"

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Loading default dataset.")
    df = pd.read_csv(default_path)

# Timestamp format handling
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ========== SIDEBAR ==========
st.sidebar.title("⚙️ Model Tuning")

# Contamination slider (sensitivity to anomalies)
contamination = st.sidebar.slider(
    "Contamination (Anomaly Proportion)",
    min_value=0.001, max_value=0.1, value=0.01, step=0.001,
    help="Adjust how sensitive the model is to anomalies."
)

# Model selection dropdown
model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["Isolation Forest", "Local Outlier Factor (LOF)"],
    index=0
)

# Run detection button
if st.sidebar.button("Run Anomaly Detection"):

    # Apply selected model
    if model_choice == "Isolation Forest":
        df = detect_anomalies(df, contamination=contamination)
    elif model_choice == "Local Outlier Factor (LOF)":
        from src.anomaly_detection import detect_anomalies_lof
        df = detect_anomalies_lof(df, contamination=contamination)

    # ========== CHART ==========
    fig = px.line(df, x="timestamp", y="value", title=f"Sensor Readings with Anomalies ({model_choice})")

    # Highlight anomalies
    anomalies = df[df['anomaly'] == 1]
    fig.add_scatter(
        x=anomalies['timestamp'], y=anomalies['value'],
        mode='markers', marker=dict(color='red', size=8),
        name='Anomalies'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ========== METRICS ==========
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Points", len(df))
    col2.metric("Anomalies Found", len(anomalies))
    col3.metric("Anomaly %", f"{(len(anomalies)/len(df)) * 100:.2f}%")

    # ========== TABLE ==========
    st.subheader("📋 Anomaly Details")
    st.dataframe(anomalies[['timestamp', 'value', 'value_scaled']].reset_index(drop=True), use_container_width=True)

    # ========== CSV EXPORT ==========
    csv = anomalies.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Anomalies as CSV",
        data=csv,
        file_name='detected_anomalies.csv',
        mime='text/csv'
    )

else:
    st.warning("Adjust slider & click 'Run Anomaly Detection' to see results.")
