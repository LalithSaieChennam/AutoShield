# app/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="AutoShield Dashboard", layout="wide")

st.title("🛡️ AutoShield – Anomaly Detection Dashboard")

# Load CSV
uploaded_file = st.file_uploader("Upload a processed CSV", type=["csv"])
default_path = "outputs/ambient_temperature_labeled.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Loading default dataset.")
    df = pd.read_csv(default_path)

# Timestamp parsing
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Plot time series with anomalies
fig = px.line(df, x="timestamp", y="value", title="Sensor Reading Over Time")

# Overlay anomalies in red
anomalies = df[df['anomaly'] == 1]
fig.add_scatter(x=anomalies['timestamp'], y=anomalies['value'],
                mode='markers', marker=dict(color='red', size=8),
                name='Anomalies')

st.plotly_chart(fig, use_container_width=True)

# Summary Stats
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Anomalies Detected", anomalies.shape[0])
col3.metric("Anomaly Rate", f"{(anomalies.shape[0]/len(df)) * 100:.2f}%")
