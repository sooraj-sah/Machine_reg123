# 🏥 Medical Cost & Predictive Valuation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](#)

An end-to-end Machine Learning regression pipeline designed to forecast individual annual medical insurance costs based on demographic and lifestyle indicators. The project leverages Scikit-Learn pipelines to eliminate data leakage and serves real-time predictions via an interactive Streamlit web dashboard.

---

## 📌 Key Highlights

- **Data Leakage Prevention:** Built with Scikit-Learn `ColumnTransformer` and `Pipeline` abstractions, ensuring all scaling and categorical encodings are fitted strictly on training folds.
- **Model Benchmarking:** Evaluated regularized linear models and non-linear ensemble algorithms (Ridge, Random Forest, Gradient Boosting).
- **Production Artifacts:** Serialized composite estimators with `joblib` for zero-latency pipeline deployment.
- **Interactive Cloud Inference:** Deployed on Streamlit Cloud with sub-150ms prediction latency.

---

## 🏗️ Architecture Overview

```text
Raw Tabular Data (CSV)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Preprocessing Pipeline               │
│  ├─ Numerical Features (Age, BMI, Dependents) ➔ StandardScaler│
│  └─ Categorical Features (Sex, Smoker, Region) ➔ OneHotEncoder │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Model Benchmark & Evaluation                 │
│  ├─ Ridge Regression (L2 Baseline)                          │
│  ├─ Random Forest Regressor                                 │
│  └─ Gradient Boosting Regressor (Top Performer: R² ~ 0.87)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Artifact Serialization & Serving            │
│  ├─ Serialized Pipeline: best_regression_pipeline.pkl        │
│  └─ Streamlit UI: Real-Time User Input & Ingestion          │
└─────────────────────────────────────────────────────────────┘
