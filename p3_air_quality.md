## Project 3: *Forecasting Air Quality with Linear Autoregressive Models*

### 🧠 Motivation

Air quality monitoring is critical for environmental safety and public health. Many cities operate dense sensor networks to track pollutants such as CO, NO₂, and O₃. These measurements vary across hours, days, and seasons, and their fluctuations are influenced by human activity and weather conditions.

Modeling this data is essential for short-term forecasting, anomaly detection, and policy planning — and it also offers an ideal introduction to **time series regression**. Unlike i.i.d. datasets, time series exhibit **temporal dependencies**, where past values directly influence future ones.

This project introduces the **supervised framing of time series forecasting**, using linear models on lag-structured features. Students will explore how **autoregressive structure**, **regularization**, and **stationarity** affect performance and interpretability.

The central idea is to convert a univariate time series into an input–target regression problem:

$$
\text{Given past values } [y_{t-1}, y_{t-2}, \dots, y_{t-k}], \text{ predict } y_t.
$$

### 🎯 Objectives

- Construct autoregressive feature matrices from raw time series  
- Train and evaluate linear models for one-step-ahead forecasting  
- Analyze autocorrelation and choose appropriate lag depth  
- Compare regularized and unregularized regression models  
- Visualize temporal prediction quality and error patterns

### 📊 Suggested Dataset

- **UCI Air Quality Dataset**: Hourly measurements of pollutants (e.g., CO, NO₂, O₃) and meteorological data from an Italian city  
- Available at the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

Focus on:
- A single pollutant (e.g., NO₂) as the target series $\{y_t\}_{t=1}^T$  
- A continuous subset of data (e.g., 30–90 days of hourly values)  
- Optional: include weather variables as exogenous covariates

### 🛠️ Suggested Models

- **Autoregressive linear regression**:

$$
y_t = \beta_1 y_{t-1} + \beta_2 y_{t-2} + \dots + \beta_k y_{t-k} + \varepsilon_t
$$

- **Ridge regression** (for high-dimensional lag settings)  
- **Naive baseline**:

$$
\hat{y}_t = y_{t-1}
$$

### ✅ Minimum Requirements

- Clean and preprocess the dataset (e.g., handle missing values, normalize if needed)  
- Construct a **lagged feature matrix** $\mathbf{X} \in \mathbb{R}^{n \times k}$ and target vector $\mathbf{y} \in \mathbb{R}^{n}$ using:

$$
\mathbf{X} = 
\begin{bmatrix}
y_{k} & y_{k-1} & \dots & y_{1} \\
y_{k+1} & y_{k} & \dots & y_{2} \\
\vdots & \vdots & \ddots & \vdots \\
y_{T-1} & y_{T-2} & \dots & y_{T-k}
\end{bmatrix},
\quad
\mathbf{y} =
\begin{bmatrix}
y_{k+1} \\
y_{k+2} \\
\vdots \\
y_T
\end{bmatrix}
$$

- Train at least two models and compare using RMSE and/or MAE  
- Plot predictions vs. actual values over a test period  
- Plot autocorrelation (ACF) and partial autocorrelation (PACF) to justify choice of lag $k$

### 🚀 Optional Extensions

- Use multiple input features: past values of other pollutants or weather variables  
- Analyze how increasing lag depth $k$ affects overfitting and stability   
- Analyze performance by hour-of-day or weekday/weekend split  
- Visualize learned coefficients as functions of lag index $i$

### 📚 References

- Hyndman & Athanasopoulos (2021) – [*Forecasting: Principles and Practice* (Ch. 7: Time series regression models)](https://otexts.com/fpp3/regression.html)  
- Samad et al. (2023) - [Air pollution prediction using machine learning techniques – An approach to replace existing monitoring stations with virtual monitoring stations](https://www.sciencedirect.com/science/article/pii/S1352231023004132)