## Project 15: *Double Descent in Linear Models*

### 🧠 Motivation

Classical bias–variance trade-off theory predicts that test error should decrease with model complexity up to a point, then increase as overfitting sets in. However, recent theoretical and empirical work has revealed a more nuanced picture: in many settings, **test error drops again after the interpolation threshold**, leading to a **double descent** curve.

This phenomenon occurs even in **simple linear regression models**, particularly when the number of parameters approaches or exceeds the number of training samples. This project explores double descent from empirical perspective, with a focus on understanding how the generalization error behaves as a function of model complexity in high-dimensional linear systems.

### 🎯 Objectives

- Construct overparameterized linear regression problems with synthetic data  
- Plot training and test error as a function of model dimension $d$  
- Reproduce the classical U-shaped curve and observe the **second descent** after interpolation  
- Analyze how noise level, sample size, and input dimension affect the curve  
- Interpret results through the lens of linear algebra and conditioning of the design matrix  

### 📊 Dataset

Use **synthetic data** generated with known ground truth:

- Choose $n$ training samples and vary the number of features $d$
- Generate a true linear model $y = X \beta^\ast + \varepsilon$, where:
  - $X \in \mathbb{R}^{n \times d}$ with entries sampled i.i.d. from $\mathcal{N}(0,1)$
  - $\beta^\ast \in \mathbb{R}^d$ (nonzero in a small subset or fully dense)
  - $\varepsilon \sim \mathcal{N}(0, \sigma^2)$

Create a separate test set $(X_{\text{test}}, y_{\text{test}})$ to evaluate generalization error.


### ✅ Minimum Requirements

- Simulate linear regression problems with $n = 100$ and vary $d$, e.g. from 10 to 300  
- For each $d$, generate training data, fit the model, and compute:
  - Training error: $\|X \hat{\beta} - y\|_2^2 / n$  
  - Test error: $\|X_{\text{test}} \hat{\beta} - y_{\text{test}}\|_2^2 / n_{\text{test}}$  
- Plot both errors against model dimension $d$  
- Clearly mark the interpolation threshold ($d = n$) and explain observed behavior  
- Repeat for different noise levels $\sigma$ and discuss trends  

### 🚀 Optional Extensions

- Compare standard least-squares to **Ridge regression** and show how regularization affects double descent  
- Examine test error vs. **effective rank** of $X^\top X$ instead of raw $d$  
- Use real data (e.g., MNIST PCA projections) to illustrate whether double descent appears in practice  
- Include derivations or interpretations based on the spectrum or nullspace of $X$  

### 📚 References

- Belkin et al. (2018) – [Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118) 
- Hastie et al. (2019) – [Surprises in High-Dimensional Ridgeless Least Squares Interpolation](https://arxiv.org/abs/1903.08560)  

