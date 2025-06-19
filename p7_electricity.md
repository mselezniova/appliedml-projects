## **Project 7: *Forecasting Electricity Load with Time-Based Fourier Features***

### 🧠 Motivation

Electricity consumption patterns exhibit strong periodic structure — shaped by daily routines, weekly work schedules, and seasonal weather cycles. Modeling these patterns is essential for energy planning, load balancing, and predictive maintenance in modern power systems.

This project introduces a **time-based regression approach** to short-term electricity forecasting. Rather than using past consumption as input, you will explore how well electricity demand can be predicted directly from **calendar time**, transformed using **Fourier features**. These features encode periodic cycles using sine and cosine functions, enabling linear and nonlinear models to capture repeating structure across hours, days, and months.

The central idea is to predict consumption at time $t$ using time-derived inputs:

$$
\text{Given timestamp features (e.g., hour, day, season), predict } y_t = \text{electricity usage at time } t.
$$

### 🎯 Objectives

- Parse and clean timestamped power usage data  
- Construct Fourier features from time variables (e.g., hour of day, day of year)  
- Train and compare regression models using periodic encodings  
- Vary the number of frequencies and study underfitting/overfitting tradeoffs  
- Visualize basis functions, predicted load curves, and residual error patterns

### 📊 Suggested Dataset

- **UCI Individual Household Electric Power Consumption**:  
  Minute-level electricity usage from 2006–2010 in a single household  
  Available at the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

Focus on:
- A continuous subset of **hourly data** (e.g., several months to a year)  
- The `Global_active_power` column (in kilowatts) as the target series $\{y_t\}_{t=1}^T$  
- Timestamps converted to time-of-day, day-of-week, and day-of-year features

### 🛠️ Suggested Models

- **Linear regression** using Fourier-encoded time features  
- **Ridge regression** to control for high-frequency overfitting  
- **Shallow feedforward neural network** (optional comparison)

Fourier feature construction:

$$
\phi_{\text{hour}}(t) = 
\begin{bmatrix}
\sin\left(2\pi \cdot \text{hour}(t)/24\right) \\
\cos\left(2\pi \cdot \text{hour}(t)/24\right) \\
\vdots \\
\sin\left(2\pi K \cdot \text{hour}(t)/24\right) \\
\cos\left(2\pi K \cdot \text{hour}(t)/24\right)
\end{bmatrix}
$$

Repeat similarly for weekly or yearly cycles as needed.

### ✅ Minimum Requirements

- Clean and preprocess the dataset (e.g., convert to hourly resolution, handle missing data)  
- Construct Fourier features using at least one periodic time variable (e.g., hour of day)  
- Train at least two models:
  - A linear model (OLS or Ridge)
  - A nonlinear model (e.g., shallow neural net)  
- Evaluate performance on a held-out time period using RMSE or MAE  
- Run experiments for different numbers of frequency components (e.g., $K = 1, 3, 5, 10$)  
- Plot:
  - Fourier basis functions (sinusoids over time)  
  - Predictions vs. actual values over time  
  - Residual errors and trends (e.g., by time of day)

### 🚀 Optional Extensions

- Compare Fourier features to alternative encodings (e.g., one-hot day-of-week)  
- Consider other time series models (e.g., autoregressive models, RNNs, etc.)
- Try combining MLPs with Fourier features 

### 📚 References

- Mallat (1999) - [*A Wavelet Tour of Signal Processing*](https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-2-3.pdf)
- Tancik et al. (2020) - [*Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*](https://arxiv.org/pdf/2006.10739)
