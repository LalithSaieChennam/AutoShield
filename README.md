# 🛡️ AutoShield

AutoShield is a real-time anomaly detection system for time-series sensor data.  
It leverages unsupervised machine learning (Isolation Forest) to identify unusual patterns in logs or metrics — and includes both a **Streamlit dashboard** and **CLI pipeline** to run, explore, and export results.

<p align="center">
  <img src="https://img.shields.io/badge/ML-IsolationForest-blue" />
  <img src="https://img.shields.io/badge/UI-Streamlit-orange" />
  <img src="https://img.shields.io/badge/Status-MVP-brightgreen" />
</p>

---

## 🚀 Features

- 🔍 Detects anomalies in time-series CSV data
- 🧠 ML-powered backend (Isolation Forest)
- 📊 Interactive Streamlit dashboard
- ⚙️ Slider to control model sensitivity
- 📋 View detailed anomaly logs in a table
- 📥 One-click export to CSV
- 🧪 CLI mode for batch processing

---

## 🧰 Tech Stack

- **Python 3.9+**
- **scikit-learn** (Isolation Forest)
- **pandas**, **numpy**
- **plotly** + **Streamlit** (for dashboard)
- **CSV-based input/output**

---

## 📂 Project Structure

AutoShield/
│
├── data/
│ ├── raw/ # Original datasets (NAB CSVs)
│ └── processed/ # Cleaned & transformed data
│
├── src/
│ ├── preprocessing.py # Data cleaning & feature extraction
│ ├── utils.py # Common helper functions(if)
│
├── outputs/ # Output CSV reports with anomaly tags
│
├── app/
│ └── dashboard.py # Streamlit dashboard
│
├── models/ # Saved model files
│
├── notebooks/ # Jupyter experiments (optional)
│
├── README.md
├── requirements.txt
└── run.py # Main script interface

## ⚙️ How to Run

### 🔁 1. Install Dependencies

```bash
pip install -r requirements.txt

### 🧪 2. Run from CLI

python run.py

## Input: data/raw/ambient_temperature_system_failure.csv
## Output: outputs/ambient_temperature_labeled.csv

🧠 Future Roadmap (AutoShield v2)
 Add Local Outlier Factor (LOF) model

 Implement AutoEncoder-based detection

 Deploy dashboard on Streamlit Cloud

 Add evaluation metrics (Precision, Recall)

 Enable multi-dataset selector

 Add Auto-email alert on detection
```
