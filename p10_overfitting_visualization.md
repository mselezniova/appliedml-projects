## Project 10: *Visualizing Overfitting and Model Complexity*

### 🧠 Motivation  

This project explores how increasing model complexity influences decision boundaries and generalization. Using 2D datasets, you'll examine how different models separate data in input space, and how this reflects underfitting, overfitting, and the bias–variance tradeoff.

The task in this project is to **first choose a model of interest**, then **identify an appropriate notion of complexity for that model**, and finally **visualize how generalization behavior changes as complexity increases**.

### 🎯 Objectives

- Choose 2–3 model types and define a model-specific complexity parameter  
- Visualize how decision boundaries evolve with increasing complexity  
- Compare training and test accuracy across settings  
- Identify and explain signs of underfitting and overfitting  

### 🧠 Suggested Models

- Kernel SVM (e.g., RBF kernel width or margin scaling)  
- Multi-layer perceptrons (e.g., number of layers or hidden units)  
- Optionally: decision trees ensembles for comparison  

### 📊 Suggested Datasets

- 2D synthetic datasets with nonlinear class boundaries, such as:
  - Interleaving half-moons  
  - Concentric circles  
  - Separated Gaussian clusters  
- Custom patterns like XOR or spirals  
- Optional: 2D PCA projection of MNIST for more complex structure  

### ✅ Minimum Requirements

For each selected model:

- Choose and justify a relevant complexity parameter  
- Vary that parameter across multiple settings  
- Plot decision boundaries on at least two datasets  
- Report training and test accuracy  
- Discuss how generalization changes with complexity  


### 🚀 Optional Extensions

- Introduce label noise or domain shifts to evaluate robustness  
- Compare different forms of regularization (e.g., L2, dropout)  
- Apply models to higher-dimensional data reduced via PCA  
- Analyze optimization dynamics (e.g., training stability, convergence rate)  
- Visualize margin width or boundary curvature as measures of complexity  

### 📚 References

- Mickisch et al. (2020) – [Understanding the Decision Boundary of Deep Neural Networks: An Empirical Study](https://arxiv.org/abs/2002.01810)  
- Belkin et al. (2018) – [Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)  
- Hastie et al. (2019) – [Surprises in High-Dimensional Ridgeless Least Squares Interpolation](https://arxiv.org/abs/1903.08560)  
