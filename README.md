
# BTC AI Agent: Quantitative Research & Execution Pipeline

An end-to-end algorithmic trading framework designed for Bitcoin (BTC), featuring an AI-driven alpha signal generator, rigorous backtesting engine, and risk-managed execution layer.

---

## 🔬 Project Objectives

This project demonstrates the application of machine learning to high-frequency financial time-series data. The goal is to isolate predictive alpha while maintaining a neutral exposure during high-volatility regimes.

### Core Quantitative Components:

* **Feature Engineering:** Implementation of multi-scale technical indicators, order flow imbalance (OFI), and Fourier transforms for seasonality detection.
* **Predictive Modeling:** Architecture optimized for minimizing Mean Squared Error (MSE) on forward-looking returns.
* **Backtesting Framework:** A custom-built, event-driven backtester that accounts for **slippage, exchange fees, and latency.**
* **Risk Engine:** Real-time calculation of Value at Risk (VaR), Maximum Drawdown (MDD) constraints, and dynamic position sizing using Kelly Criterion principles.

---

## 🛠 Tech Stack

* **Languages:** Python (Core Logic), SQL (Market Data Storage)
* **Quantitative Libraries:** `NumPy`, `Pandas`, `Scipy`
* **Machine Learning:** `Scikit-learn`, `XGBoost` / `PyTorch`
* **Data API:** Alpaca API
---

## 📊 Methodology & Performance Metrics

The agent evaluates trade signals based on a probabilistic threshold. Performance is tracked using institutional-grade metrics:

| Metric | Description |
| --- | --- |
| **Sharpe Ratio** | Risk-adjusted return performance. |
| **Sortino Ratio** | Focus on downside volatility protection. |
| **Information Ratio** | Active return relative to BTC benchmark. |
| **Calmar Ratio** | Relationship between CAGR and Max Drawdown. |

---

## 📁 Repository Structure


├── Ichi.CloudAI_NN/            # Serialized weights and hyperparameter logs
│   ├── .../          
├── Ichi.Cloud_NN/          # Exploratory Data Analysis (EDA) & Research
│   ├── .../          
├── SM_NN/
│   ├── .../          


---

## 🚀 Future Roadmap

* **Sentiment Analysis:** Integrating NLP to process Twitter/X and News headlines for "black swan" event detection.
* **Cross-Exchange Arbitrage:** Expanding the execution layer to capture spreads between liquidity providers.
* **Reinforcement Learning (PPO):** Moving from supervised learning to an agent that optimizes for long-term cumulative reward.

---

### 💼 Contact Information

**John Alyn** – [Your LinkedIn Profile] – [Your Portfolio Website]

> **Disclaimer:** This repository is a research project. All strategies are tested in simulated environments. Past performance is not indicative of future results.
