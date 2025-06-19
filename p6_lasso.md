## **Project 6: *Efficient Algorithms for Lasso Regression***

### 🧠 Motivation

In class, we introduced **Lasso regression** as a convex objective that induces sparsity via $\ell_1$ regularization:

$$
\min_{\mathbf{w} \in \mathbb{R}^d} \left\\{ \frac{1}{2n} \|\mathbf{Xw} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1 \right\\}
$$

This problem is nonsmooth due to the $\|\mathbf{w}\|_1$ term, which complicates optimization. We implemented a **basic subgradient descent method**, which often converges slowly and is highly sensitive to learning rate.

This project explores more **efficient first-order optimization algorithms** for solving the Lasso objective, with particular attention to their **convergence properties**, **sparsity-inducing behavior**, and **sensitivity to initialization**. Your goal is to analyze and implement alternatives such as **coordinate descent** and **proximal gradient descent** (ISTA), and to understand how algorithmic structure reflects the geometry of the underlying problem.

### 🎯 Objectives

- Understand why Lasso is convex but nonsmooth, and how this impacts optimization
- Implement and compare at least two solvers:
  - **Subgradient descent** (baseline, implemented in the course)
  - **Coordinate Descent** (cyclic or randomized)
  - **Proximal Gradient Descent** (ISTA)
- Analyze:
  - Convergence speed (loss vs. iteration)
  - Final sparsity of solutions
  - Robustness to initialization and regularization strength

### 🛠 Suggested Models

Implement Lasso solvers for:
- **Subgradient descent** (baseline)
- **Coordinate descent**, with closed-form soft-thresholding update
- **ISTA**, using the proximal operator of $\ell_1$-norm

Let $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{y} \in \mathbb{R}^n$, and define  $g(\mathbf{w}) = \frac{1}{2n} \|\mathbf{Xw} - \mathbf{y}\|_2^2$ (smooth part of the loss). 

Then **ISTA iterates** are given by:

$$
\mathbf{w}^{(t+1)} = \text{prox}_{\eta \lambda} \left( \mathbf{w}^{(t)} - \eta \nabla g(\mathbf{w}^{(t)}) \right)
\quad \text{with} \quad
\text{prox}_{\eta \lambda}(z_j) = \text{sign}(z_j) \cdot \max(|z_j| - \eta \lambda, 0)
$$

**Coordinate descent** updates a single coordinate $w_j$ using:

$$
w_j^{(t+1)} \leftarrow \text{prox}_{\lambda}\left( \frac{1}{n} \sum_{i=1}^n x_{ij} \left(y_i - \sum_{k \ne j} x_{ik} w_k^{(t)} \right) \right),
$$
where the data is assumed to be standardized.
### 📊 Suggested Datasets

This project emphasizes convergence and sparsity behavior over test accuracy. Use:

#### 🔬 Synthetic Data

Generate:

$$
\mathbf{y} = \mathbf{Xw}^* + \boldsymbol{\varepsilon}, \quad \text{with} \quad \|\mathbf{w}^*\|_0 \ll d, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

- Choose $n = 100$, $d = 200$, with $\mathbf{w}^*$ sparse (e.g. 10 nonzeros)
- Vary $\sigma$ to test noise robustness

#### 🧪 Real Data

- Boston Housing or Diabetes (standardized)

### ✅ Minimum Requirements

- Implement at least two Lasso solvers: baseline + one advanced (CD or ISTA)
- Run experiments for **a rnage of $\lambda$ values**
- Track and visualize:
  - Objective vs. iteration count
  - Sparsity ($\|\mathbf{w}^{(t)}\|_0$) vs. $\lambda$
  - Final prediction error or distance from ground truth (for synthetic data)
- Analyze:
  - Convergence behavior
  - Sensitivity to initialization
  - Pros/cons of each method in terms of efficiency and accuracy

### 🚀 Optional Extensions

- Implement **FISTA** (accelerated proximal gradient)
- Study the **Lasso path** as $\lambda \to 0$
- Compare to **Elastic Net** (combined $\ell_1 + \ell_2$ penalty)
- Compare with sklearn’s `Lasso`

### 📚 References

- Hastie, Tibshirani, Friedman (2009) – [*The Elements of Statistical Learning* (Ch. 3)](https://hastie.su.domains/ElemStatLearn/)
- Beck & Teboulle (2009) – [*A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)*](https://www.tau.ac.il/~becka/FISTA.pdf) 
- Friedman et al. (2009) – [*Regularization Paths for Generalized Linear Models via Coordinate Descent*](https://hastie.su.domains/Papers/glmnet.pdf)
