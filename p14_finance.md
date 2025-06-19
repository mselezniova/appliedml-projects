## Project 14: *Predicting Market Direction from Technical Indicators*

### 🧠 Motivation

Short-term forecasting in financial markets is notoriously challenging due to noise, regime shifts, and the nonstationary nature of price movements. While raw returns are difficult to interpret or model directly, **hand-crafted features** based on price patterns — such as moving averages or momentum — are widely used in technical trading strategies.

This project investigates the usefulness of **technical indicators** as features for predicting daily market direction. By contrasting these engineered features with baseline models trained on raw returns, students will explore the value of domain knowledge, feature design, and model interpretability in a high-noise setting.

### 🎯 Objectives

- Compute and analyze a variety of **technical indicators** from daily market data  
- Train classifiers to predict short-term market direction based on these features  
- Compare performance to models using raw lagged returns  
- Evaluate which features provide the most predictive signal  

### 📊 Dataset

Use historical daily data from **SPY** (S&P 500 ETF). Data can be loaded using the [`yfinance` Python package](https://ranaroussi.github.io/yfinance/), e.g.:

```python
import yfinance as yf
data = yf.download("SPY", start="2015-01-01", end="2023-12-31")
```

Each row includes open, high, low, close, and volume (OHLCV) data. You may optionally work with multiple assets (e.g., QQQ, IWM, AAPL), but single-asset analysis is sufficient.

> **Target variable**: Predict whether the **next day's return** is positive or negative.  
> For each day $t$, define the label as:
> 
> $$
> y_t = \begin{cases}
> 1 & \text{if } r_{t+1} > 0 \\
> 0 & \text{if } r_{t+1} \leq 0
> \end{cases}
> $$
> where $r_{t+1} = \frac{P_{t+1}^{\text{close}} - P_t^{\text{close}}}{P_t^{\text{close}}}$

### 🛠️ Suggested Models / Tools

#### 1. Feature engineering

Compute several technical indicators such as:

- Simple Moving Average (SMA), Exponential Moving Average (EMA)  
- Relative Strength Index (RSI)  
- Bollinger Bands  
- Momentum, MACD  
- Daily return, rolling volatility (e.g., 5-day std)  
- Lagged close prices or returns (e.g., $r_{t-1}, r_{t-2}, \dots$)

Each day’s feature vector should summarize recent price behavior.

#### 2. Models

Train and evaluate classification models, e.g.:

- **Logistic Regression** (with regularization)
- **Linear SVM**
- **Shallow Neural Network** (optional comparison)

Use 70–80% of the time series for training, and evaluate generalization on a held-out recent period. Use cross-validation that respects time ordering (e.g., rolling or expanding window CV).

### ✅ Minimum Requirements

- Download and preprocess daily OHLCV data  
- Implement or extract at least **5 distinct technical indicators**  
- Define a binary classification task (next-day direction)  
- Train at least **two classical classifiers** on engineered features  
- Compare performance to a baseline using raw lagged returns (i.e., linear autoregressive model)
- Report and visualize results, e.g.:
  - Accuracy on test data  
  - Time series of predicted vs. true directions  
  - Feature importance (e.g., via model coefficients or ablation)  

### 🚀 Optional Extensions

- Test your model on a different asset without retraining (transferability)  
- Add PCA to reduce feature dimensionality and assess its effect  
- Evaluate performance during high-volatility vs. low-volatility regimes  
- Build a naive strategy that trades on your predictions and plot cumulative return (no transaction costs)

### 📚 References

- Murphy (1999) - *Technical Analysis of the Financial Markets*  
- `pandas-ta` Technical Analysis Library (as a reference for a lot of indicators): [https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas](https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas)
- `yfinance` documentation: [https://ranaroussi.github.io/yfinance/](https://ranaroussi.github.io/yfinance/)
