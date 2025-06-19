## Project 9: *Initialization Strategies for MLPs*

### 🧠 Motivation

Weight initialization plays a central role in training deep neural networks. Poor initialization can lead to vanishing or exploding activations and gradients, making learning unstable or impossible. This project focuses on analyzing the statistical behavior of MLPs under different initialization schemes, with particular attention to signal propagation and the onset of instability.

In the course, we used He initialization for ReLU networks:
$$
W^\ell_{ij}\sim \mathcal{N}\left(0, \frac{2}{\text{fan-out}}\right)
$$

Explore why this initialization is effective, and derive or identify suitable alternatives for other activation functions (e.g., tanh, sigmoid).

The key task in this project is to **select an initialization strategy**, **define a notion of signal stability or trainability**, and **empirically assess the consequences of initialization choices** through both statistical and visual analysis.

### 🎯 Objectives

- Investigate how different initialization schemes affect activation and gradient dynamics in deep MLPs
- Analyze layer-wise behavior of forward activations and backward gradients
- Compare at least two distinct initialization methods (e.g., Gaussian vs. orthogonal)
- Relate your findings to theoretical motivations where possible

### 🧠 Suggested Models

- Deep MLPs with varying depth, width, and activation types  
- Optionally: CNNs or residual MLPs for contrast

### 📊 Suggested Datasets

This project emphasizes statistical behavior over accuracy, so the dataset choice should not have a large impact on conclusions. For illustrative experiments or performance comparisons, you may use:

- MNIST or Fashion-MNIST (basic image classification)
- Synthetic data (e.g., 2D Gaussian mixtures or blobs)

### ✅ Minimum Requirements

- Begin with Gaussian initialization and explore the effects of the variance scaling constant (e.g., 2 in He initializaion). Observe how this affects:
  - Activation statistics (mean, variance across layers)
  - Gradient norms (vanishing or exploding behavior)
- Compare at least one other initialization distribution (e.g., uniform, orthogonal, truncated normal)
- Track and visualize layer-wise statistics at initialization and during training:
  - Activation mean and variance
  - Gradient norms
  - Signal preservation across depth
- (Optional) Report model performance, but this is **not the primary focus**

### 🚀 Optional Extensions

- Explore how initialization influences convergence rate or generalization performance
- Connect empirical results to theory (e.g., mean-field dynamics, dynamical isometry)
- Visualize how activation and weight distributions evolve during training
- Study how different activation functions interact with initialization choices

### 📚 References

- Glorot & Bengio (2010) – [Understanding the Difficulty of Training Deep Feedforward Neural Networks](http://proceedings.mlr.press/v9/glorot10a.html)  
- He et al. (2015) – [Delving Deep into Rectifiers](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)  
- Poole et al. (2016) – [Exponential Expressivity in Deep Neural Networks Through Transient Chaos](https://arxiv.org/abs/1606.05340)  
- Xiao et al. (2019) – [Disentangling Trainability and Generalization in Deep Neural Networks](https://arxiv.org/abs/1912.13053)  
