# dashboard.py
# This Streamlit dashboard provides an interactive interface for running and visualizing
# anomaly detection on time-series sensor data using Isolation Forest.
# Users can upload datasets, adjust model sensitivity (contamination),
# and explore anomalies via charts, stats, and a downloadable CSV report.

import sys
import os

# Add project root directory to Python path so we can import from /src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# UI, plotting, and data libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from src.anomaly_detection import detect_anomalies

# Set app-wide configuration: title + layout
st.set_page_config(page_title="AutoShield Dashboard", layout="wide")

# App Header
st.title("🛡️ AutoShield – Anomaly Detection Dashboard")

# File uploader in the sidebar
uploaded_file = st.file_uploader("Upload a processed CSV", type=["csv"])

# Load default dataset if no file is uploaded
default_path = "data/processed/ambient_temperature_processed.csv"
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Loading default dataset.")
    df = pd.read_csv(default_path)

# Convert timestamps to datetime format for plotting
df['timestamp'] = pd.to_datetime(df['timestamp'])

# ====================
# Sidebar: Model Tuning
# ====================
st.sidebar.title("⚙️ Model Tuning")

# Contamination slider: controls % of expected anomalies
contamination = st.sidebar.slider(
    "Contamination (Anomaly Proportion)",
    min_value=0.001, max_value=0.1, value=0.01, step=0.001,
    help="Adjust how sensitive the model is to anomalies."
)

# Button to trigger anomaly detection
if st.sidebar.button("Run Anomaly Detection"):

    # Run Isolation Forest with selected contamination
    df = detect_anomalies(df, contamination=contamination)

    # ====================
    # Line Chart with Anomalies
    # ====================
    fig = px.line(df, x="timestamp", y="value", title="Sensor Readings with Anomalies")

    # Highlight anomaly points in red
    anomalies = df[df['anomaly'] == 1]
    fig.add_scatter(
        x=anomalies['timestamp'], y=anomalies['value'],
        mode='markers', marker=dict(color='red', size=8),
        name='Anomalies'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ====================
    # KPI Metrics (top summary stats)
    # ====================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Points", len(df))
    col2.metric("Anomalies Found", len(anomalies))
    col3.metric("Anomaly %", f"{(len(anomalies)/len(df)) * 100:.2f}%")

    # ====================
    # Table View of Anomalies
    # ====================
    st.subheader("📋 Anomaly Details")
    st.dataframe(anomalies[['timestamp', 'value', 'value_scaled']].reset_index(drop=True), use_container_width=True)

    # ====================
    # CSV Export Button
    # ====================
    csv = anomalies.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Anomalies as CSV",
        data=csv,
        file_name='detected_anomalies.csv',
        mime='text/csv'
    )

# If button is not clicked
else:
    st.warning("Adjust slider & click 'Run Anomaly Detection' to see results.")
