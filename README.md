<![CDATA[<div align="center">

# 🏧 ATM Cash Flow Forecasting & Analysis

### *Predicting Daily ATM Withdrawals Using Ensemble Machine Learning & Time Series Models*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boost-02569B?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

<img src="regression_diagnostics.png" alt="Regression Diagnostics — Actual vs Predicted & Error Scatter" width="90%"/>

<br/>

*End-to-end ML pipeline forecasting cash demand across 111 UK ATMs — from raw time series data to production-ready ensemble predictions.*

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Models Implemented](#-models-implemented)
- [Evaluation Metrics](#-evaluation-metrics)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Results Deep Dive](#-results-deep-dive)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

Cash management at ATMs is a critical challenge in banking operations. **Overstocking** ties up capital unnecessarily, while **understocking** leads to cash-outs and customer dissatisfaction. This project builds an **end-to-end machine learning pipeline** to forecast daily cash withdrawals across a network of **111 ATMs** in the United Kingdom.

### Problem Statement
> *Given historical daily withdrawal data for 111 ATMs over ~2 years, forecast the next 56 days of withdrawals per ATM to optimize cash replenishment schedules.*

### Approach

The project combines **classical time series models** (ARIMA, SARIMAX) with **global machine learning models** (XGBoost, Random Forest, LightGBM, MLP) trained on panel data, culminating in a **weighted ensemble** that leverages the strengths of each approach.

### Highlights

- 🔬 **Comprehensive EDA** — Distribution analysis, weekly seasonality decomposition, stationarity testing (ADF)
- ⚙️ **Rich Feature Engineering** — 30+ features including calendar, lag, rolling statistics, and domain-specific (UK holidays, payday)
- 🤖 **6 Models Compared** — From classical ARIMA to gradient-boosted trees and neural networks
- 🧬 **Hyperparameter Optimization** — Bayesian optimization via Optuna with TimeSeriesSplit cross-validation
- 📊 **K-Means Clustering** — Behavioral segmentation of ATMs into 4 distinct usage clusters
- 🏆 **Weighted Ensemble** — Optimal blending of top models for robust production forecasts
- 📉 **Concept Drift Monitoring** — Kolmogorov-Smirnov tests to detect feature distribution shifts

---

## 🏆 Key Results

<div align="center">

| Metric | Value |
|:-------|:------|
| **ATMs Forecasted** | 111 |
| **Forecast Horizon** | 56 days |
| **Average MAE** | **3.32** (cash units) |
| **Median MAE** | **3.09** (cash units) |
| **Best ATM MAE** | 1.41 (ATM T103) |
| **ATMs with MAE < 3.0** | 50 / 111 (45.0%) |
| **ATMs with MAE < 4.0** | 86 / 111 (77.5%) |

</div>

### Top 5 Best Predicted ATMs

| ATM | Avg Actual | Avg Predicted | MAE | MAPE |
|-----|-----------|--------------|-----|------|
| T103 | 8.83 | 8.77 | **1.41** | 16.81% |
| T27 | 7.76 | 8.00 | **1.44** | 25.35% |
| T55 | 10.89 | 10.99 | **1.49** | 19.93% |
| T53 | 11.66 | 11.65 | **1.66** | 15.99% |
| T22 | 18.99 | 18.65 | **1.70** | 9.34% |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      ATM Cash Flow Forecasting Pipeline               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Raw Data     │───▶│  Data Parser  │───▶│  Long-Format DataFrame   │  │
│  │  (.tsf file)  │    │  (Custom)     │    │  (Date, ATM_ID, Amount)  │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────────┘  │
│                                                      │                  │
│                                         ┌────────────▼────────────┐    │
│                                         │   Exploratory Analysis  │    │
│                                         │  • Distribution plots   │    │
│                                         │  • Seasonal decompose   │    │
│                                         │  • ADF stationarity     │    │
│                                         │  • K-Means clustering   │    │
│                                         └────────────┬────────────┘    │
│                                                      │                  │
│                                         ┌────────────▼────────────┐    │
│                                         │  Feature Engineering    │    │
│                                         │  • Calendar features    │    │
│                                         │  • Lag features (1–28d) │    │
│                                         │  • Rolling statistics   │    │
│                                         │  • UK holidays/payday   │    │
│                                         └────────────┬────────────┘    │
│                                                      │                  │
│                          ┌───────────────────────────┼──────────┐      │
│                          │                           │          │      │
│                ┌─────────▼──────┐  ┌─────────────────▼──┐  ┌───▼───┐  │
│                │  Time Series   │  │  ML Models (Global) │  │  MLP  │  │
│                │  ARIMA/SARIMAX │  │  XGBoost │ RF │ LGBM│  │  (NN) │  │
│                └─────────┬──────┘  └─────────────────┬──┘  └───┬───┘  │
│                          │                           │          │      │
│                          └───────────────┬───────────┘──────────┘      │
│                                          │                             │
│                              ┌───────────▼───────────┐                 │
│                              │  Weighted Ensemble     │                 │
│                              │  (Optimized Weights)   │                 │
│                              └───────────┬───────────┘                 │
│                                          │                             │
│                              ┌───────────▼───────────┐                 │
│                              │  Evaluation & Output   │                 │
│                              │  MAE │ RMSE │ MAPE │R² │                │
│                              └───────────────────────┘                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

This project uses the **NN5 Forecasting Competition Dataset**, a well-known benchmark in the time series forecasting community.

| Property | Details |
|:---------|:--------|
| **Source** | [NN5 Competition](http://www.neural-forecasting-competition.com/NN5/) / [Monash Time Series Repository](https://forecastingdata.org/) |
| **Domain** | Banking — UK ATM Cash Withdrawals |
| **Format** | `.tsf` (Monash Time Series Forecasting format) |
| **Time Range** | March 18, 1996 — May 17, 1998 |
| **Frequency** | Daily |
| **Number of Series** | 111 ATMs (`T1` – `T111`) |
| **Observations per ATM** | 791 daily records |
| **Total Records** | 87,801 (panel format) |
| **Forecast Horizon** | 56 days |
| **Missing Values** | Pre-imputed using same-day-of-week median |

### Data Characteristics
- **Strong weekly seasonality** — Fridays consistently show the highest withdrawals; Sundays the lowest
- **Variable volatility** — Some ATMs (e.g., T68) exhibit high variance while others (e.g., T103) are highly predictable
- **Trend patterns** — Subtle upward trends observed in several high-traffic ATMs

---

## ⚙️ Feature Engineering

A total of **30+ features** are engineered from the raw withdrawal time series:

### Calendar & Temporal Features
| Feature | Description |
|:--------|:------------|
| `DayOfWeek` | Day of the week (0–6) |
| `Is_Weekend` | Binary flag for Saturday/Sunday |
| `Month` | Month of year (1–12) |
| `DayOfMonth` | Day of the month (1–31) |
| `Quarter` | Calendar quarter (1–4) |
| `DayOfYear` | Ordinal day of the year (1–366) |
| `Is_UK_Holiday` | Binary flag for UK bank holidays (via `holidays` library) |
| `PayDay` | Binary flag for end-of-month / common payday window (25th–31st) |

### Lag Features
| Feature | Description |
|:--------|:------------|
| `Withdrawal_Lag_1` | Previous day's withdrawal |
| `Withdrawal_Lag_7` | Same day last week |
| `Withdrawal_Lag_14` | Same day 2 weeks ago |
| `Withdrawal_Lag_21` | Same day 3 weeks ago |
| `Withdrawal_Lag_28` | Same day 4 weeks ago |
| `Same_Day_Last_Week` | Alias for lag-7 |
| `Week_Over_Week_Change` | Difference between current and lag-7 |

### Rolling Statistics
| Feature | Description |
|:--------|:------------|
| `Rolling_Mean_7d / 14d / 28d` | Moving average over 7, 14, 28 days |
| `Rolling_Std_7d / 14d / 28d` | Moving standard deviation |
| `Rolling_Min_7d / 14d / 28d` | Moving minimum |
| `Rolling_Max_7d / 14d / 28d` | Moving maximum |

### Target Engineering
| Feature | Description |
|:--------|:------------|
| `Withdrawal_Delta` | First difference: `W(t) - W(t-1)` for stationarity |

---

## 🤖 Models Implemented

### 1. Classical Time Series

#### ARIMA (AutoRegressive Integrated Moving Average)
- **Configuration**: ARIMA(1, 1, 1) fitted per individual ATM
- **Strength**: Captures autocorrelation in stationary series
- **Limitation**: No cross-ATM learning; sensitive to parameter selection

#### SARIMAX (Seasonal ARIMA with Exogenous Variables)
- **Configuration**: `(p,d,q) × (P,D,Q,7)` — weekly seasonal component
- **Strength**: Explicitly models 7-day weekly seasonality
- **Limitation**: Computationally expensive to fit across all 111 ATMs

### 2. Machine Learning Models (Global / Panel-Data)

#### XGBoost Regressor
- **Hyperparameter Tuning**: Optuna Bayesian optimization (50 trials)
- **Cross-Validation**: 5-fold `TimeSeriesSplit`
- **Target**: Predicts `Withdrawal_Delta` (differenced target)
- **Key Strength**: Handles nonlinear interactions and feature importance ranking

#### Random Forest Regressor
- **Configuration**: 200 estimators
- **Approach**: Bagged decision tree ensemble on panel features
- **Key Strength**: Robust to outliers and overfitting

#### LightGBM Regressor
- **Approach**: Histogram-based gradient boosting
- **Key Strength**: Fast training on large panel datasets with categorical support

#### MLP Neural Network
- **Implementation**: `sklearn.neural_network.MLPRegressor`
- **Approach**: Multi-layer perceptron for nonlinear function approximation
- **Key Strength**: Captures complex nonlinear withdrawal patterns

### 3. Ensemble Model
- **Strategy**: Weighted average of XGBoost, Random Forest, LightGBM, and MLP predictions
- **Weight Optimization**: Optimal weights stored in `ensemble_config.pkl`
- **Key Strength**: Reduces variance and leverages complementary model strengths

### 4. ATM Clustering (Data Mining)
- **Algorithm**: K-Means Clustering (K=4)
- **Features Used**: Mean withdrawal, std deviation, weekend usage ratio, volatility
- **Purpose**: Behavioral segmentation of ATMs for cluster-specific analysis

---

## 📏 Evaluation Metrics

| Metric | Formula | Purpose |
|:-------|:--------|:--------|
| **MAE** | `mean(\|actual - predicted\|)` | Primary metric — average absolute forecast error |
| **RMSE** | `sqrt(mean((actual - predicted)²))` | Penalizes large errors more heavily |
| **MAPE** | `mean(\|actual - predicted\| / actual) × 100` | Percentage error (with safe division threshold) |
| **R² Score** | `1 - SS_res / SS_tot` | Proportion of variance explained by the model |

### Concept Drift Detection
- **Kolmogorov-Smirnov Test** (`scipy.stats.ks_2samp`): Monitors distribution shifts between training and test periods to detect potential model degradation

---

## 📁 Project Structure

```
ATM_CashFlow_Analysis/
│
├── 📓 Atm_CashFlow_Analysis.ipynb          # Main notebook — full ML pipeline
├── 📓 Atm_CashFlow_Analysis_executed.ipynb  # Pre-executed version with outputs
├── 📓 Data_Mining_Project.ipynb             # K-Means clustering & data mining
│
├── 📊 nn5_daily_dataset_without_missing_values.tsf   # Raw dataset (NN5 competition)
│
├── 📈 atm_forecast_results.csv              # Daily forecasts — 6,771 rows
│                                             #   (Date, ATM_ID, Actual, Predicted, Error)
├── 📈 atm_performance_summary.csv           # Per-ATM aggregated metrics — 111 ATMs
│                                             #   (ATM_ID, Avg_Actual, Avg_Predicted, MAE, MAPE)
├── 🖼️ regression_diagnostics.png            # Actual vs Predicted & Error scatter plots
│
├── models/                                   # Serialized trained models
│   ├── xgboost_model.pkl                     #   XGBoost regressor (~1.6 MB)
│   ├── random_forest_model.pkl               #   Random Forest regressor (~503 MB)
│   ├── lightgbm_model.pkl                    #   LightGBM regressor (~3 MB)
│   ├── mlp_model.pkl                         #   MLP neural network (~339 KB)
│   ├── feature_scaler.pkl                    #   StandardScaler (fitted)
│   ├── label_encoder.pkl                     #   LabelEncoder for ATM IDs
│   └── ensemble_config.pkl                   #   Ensemble model weights
│
└── README.md                                 # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Jupyter Notebook** or **JupyterLab**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/ATM_CashFlow_Analysis.git
   cd ATM_CashFlow_Analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```

### Dependencies

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `pandas` | ≥ 1.3 | Data manipulation & analysis |
| `numpy` | ≥ 1.21 | Numerical computing |
| `scikit-learn` | ≥ 1.0 | ML models, preprocessing, evaluation |
| `xgboost` | ≥ 1.5 | XGBoost gradient boosting |
| `lightgbm` | ≥ 3.3 | LightGBM gradient boosting |
| `optuna` | ≥ 3.0 | Bayesian hyperparameter optimization |
| `statsmodels` | ≥ 0.13 | ARIMA, SARIMAX, seasonal decomposition |
| `matplotlib` | ≥ 3.5 | Static visualizations |
| `seaborn` | ≥ 0.11 | Statistical plots |
| `scipy` | ≥ 1.7 | Statistical tests (KS test) |
| `holidays` | ≥ 0.14 | UK bank holiday calendar |
| `joblib` | ≥ 1.1 | Model serialization |

---

## 💡 Usage

### Run the Full Pipeline
Open `Atm_CashFlow_Analysis.ipynb` and execute all cells sequentially. The notebook will:

1. **Parse** the `.tsf` dataset into a clean panel DataFrame
2. **Explore** distributions, seasonality, and stationarity
3. **Engineer** 30+ features (calendar, lag, rolling)
4. **Train** XGBoost (with Optuna), Random Forest, LightGBM, and MLP models
5. **Evaluate** across all 111 ATMs and export results
6. **Save** trained models to the `models/` directory

### Data Mining & Clustering
Open `Data_Mining_Project.ipynb` for:
- K-Means clustering of ATMs into 4 behavioral segments
- Cluster profiling and visualization

### Load Pre-trained Models
```python
import joblib

# Load models
xgb_model = joblib.load('models/xgboost_model.pkl')
rf_model = joblib.load('models/random_forest_model.pkl')
lgbm_model = joblib.load('models/lightgbm_model.pkl')
mlp_model = joblib.load('models/mlp_model.pkl')

# Load preprocessing artifacts
scaler = joblib.load('models/feature_scaler.pkl')
encoder = joblib.load('models/label_encoder.pkl')
ensemble_weights = joblib.load('models/ensemble_config.pkl')
```

---

## 🔍 Results Deep Dive

### Performance Distribution Across 111 ATMs

```
MAE Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MAE < 2.0  │██░░░░░░░░░░░░░░│  11 ATMs ( 9.9%)
  MAE < 3.0  │████████░░░░░░░░│  50 ATMs (45.0%)
  MAE < 4.0  │█████████████░░░│  86 ATMs (77.5%)
  MAE < 5.0  │██████████████░░│  ~96 ATMs
  All ATMs   │████████████████│ 111 ATMs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall:  Avg MAE = 3.32  |  Median MAE = 3.09
```

### Feature Importance (Top Predictors)

The global tree-based models consistently rank these features as the most important:

1. 🥇 `Withdrawal_Lag_7` — Same day last week (strongest predictor due to weekly seasonality)
2. 🥈 `Withdrawal_Lag_1` — Previous day's withdrawal
3. 🥉 `Rolling_Mean_7d` — 7-day moving average
4. `Rolling_Mean_14d` — 14-day moving average
5. `DayOfWeek` — Day-of-week effect
6. `Withdrawal_Lag_28` — Same day 4 weeks ago
7. `Is_Weekend` — Weekend binary indicator

### Key Insights

- **Weekly seasonality dominates** — Lag-7 is the single strongest predictor across all tree models, confirming the strong day-of-week withdrawal pattern
- **Global models outperform univariate** — Panel-data ML models learn shared patterns across ATMs, outperforming per-ATM ARIMA
- **Ensemble smooths errors** — The weighted ensemble reduces extreme forecast errors compared to any single model
- **High-volume ATMs are harder** — ATMs like T68 (avg ~43 units) show higher absolute error but proportionally reasonable MAPE
- **Concept drift detected** — KS tests flagged distribution shifts in some features between training and test windows, informing retraining needs

---

## 🔮 Future Improvements

- [ ] **Deep Learning** — Implement LSTM / Temporal Fusion Transformer for sequence modeling
- [ ] **External Data** — Integrate weather data, local events, and foot traffic
- [ ] **Online Learning** — Incremental model updates as new data streams in
- [ ] **Anomaly Detection** — Flag unusual withdrawal patterns (potential fraud or events)
- [ ] **Deployment** — REST API with FastAPI/Flask for real-time inference
- [ ] **Dashboard** — Interactive Streamlit/Dash dashboard for monitoring forecasts

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NN5 Competition** — For providing the benchmark ATM withdrawal dataset
- **Monash Time Series Repository** — For hosting the curated `.tsf` format dataset
- **Optuna** — For Bayesian hyperparameter optimization framework

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Made with ❤️ for the financial forecasting community

</div>
]]>
