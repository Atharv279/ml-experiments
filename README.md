<div align="center">

# 🧠 ML Experiments

[![Daily Training](https://github.com/Atharv279/ml-experiments/actions/workflows/daily_run.yml/badge.svg)](https://github.com/Atharv279/ml-experiments/actions/workflows/daily_run.yml)
![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Automated](https://img.shields.io/badge/Runs-Daily%20via%20CI-blue)

**Automated LightGBM hyperparameter tuning with loss curve visualization and day-over-day performance tracking.**

</div>

---

## Architecture

```mermaid
graph TD
    A[📦 Mock Dataset Generator] --> B[🎲 Hyperparameter Sampler]
    B --> C[🏋️ Training Simulator]
    C -->|Loss Curves| D[📉 Convergence Analyzer]
    C -->|Feature Importance| E[🔍 Feature Ranker]
    D --> F[📊 Dashboard Generator]
    E --> F
    C -->|Load Previous| G[🔄 Delta Engine]
    G --> H[📋 Report]
    F --> H
    H -->|Git Push| I[🚀 GitHub]
```

## Experiments

| Name | Type | Metric | Features |
|------|------|--------|----------|
| **Churn Prediction** | Binary Classification | AUC | 12 |
| **Price Regression** | Continuous | RMSE | 20 |
| **Fraud Detection** | Binary Classification | AUC | 15 |
| **Demand Forecast** | Continuous | MAE | 18 |

## Live Dashboard Preview

> Loss curves + feature importance for each experiment

![Dashboard](logs/2026-03-07_dashboard.png)

## Output Structure

```
logs/
├── YYYY-MM-DD.json          # Full trial data + loss curves
├── YYYY-MM-DD.md            # Markdown report with delta
├── YYYY-MM-DD_dashboard.png # Loss curves + feature importance
└── YYYY-MM-DD_trend.png     # 14-day score trend
```

## Quick Start

```bash
pip install -r dev-requirements.txt
python main.py
```
