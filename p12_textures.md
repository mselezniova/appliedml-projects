## Project 12: *Texture Classification with Wavelet Features*

### 🧠 Motivation

Modern image classification often relies on deep neural networks applied to raw pixel values. However, some visual categories—such as fabrics, natural textures, or satellite imagery—are best understood in terms of their **spatial-frequency structure**. The **wavelet transform** is a powerful tool from harmonic analysis that allows multiscale decomposition of images, separating coarse structure from fine texture in a mathematically interpretable way.

This project focuses on using the **Discrete Wavelet Transform (DWT)** to extract compact, structured features from grayscale **textures**. You will implement a wavelet-based feature pipeline and compare it to a neural network baseline trained directly on image pixels.

### 🎯 Objectives

- Apply **2D DWT** to extract spatial-frequency features of images
- Classify grayscale textures using classical models trained on wavelet statistics  
- Compare with basic neural networks trained on raw image input  
- Analyze the relative performance and interpretability of each method  

### 📊 Dataset

**Kylberg Texture Dataset v.1.0 (grayscale)**  
Available at [https://user.it.uu.se/~gusky180/texture/](https://user.it.uu.se/~gusky180/texture/)

- 28 texture classes (e.g. canvas, corduroy, leather)  
- 160 grayscale images per class, each 576×576 pixels  
- Images are uniform and clean; you may crop or resize them (e.g., to 128×128)

> **Note:** The dataset is provided as raw images organized in folders by class. You can use `cv2` (as in the GitHub example given below) or `scikit-image` packages to load these images. In particular, `skimage.io.ImageCollection` provides tools to load images from a base folder. 


### 🛠️ Suggested Models/Tools

#### 1. Wavelet-based feature extraction

The **2D Discrete Wavelet Transform** decomposes an image into **sub-bands** that separate **frequency** and **direction**. For each decomposition level $\ell$, the image is split into:
- Approximation $A_\ell$: low-frequency structure  
- Details $D_\ell^{(H)}, D_\ell^{(V)}, D_\ell^{(D)}$: horizontal, vertical, diagonal high-frequency content

Mathematically, the wavelet sub-bands are computed via convolutions with low-pass and high-pass filters $g[n]$ and $h[n]$:

$$
A[i, j] = \sum_{m,n} I[2i + m, 2j + n] \cdot g[m] \cdot g[n], \quad
D_H[i, j] = \sum_{m,n} I[2i + m, 2j + n] \cdot h[m] \cdot g[n]
$$

(similar for $D_V$ and $D_D$)

> **Note**: You can use the `pywt` package ([PyWavelets](https://pywavelets.readthedocs.io/en/latest/)) to perform decomposition and sub-band extraction.

To build features, compute statistics (e.g. mean, variance, energy) from each sub-band:
- Decompose each image by DWT with 1–2 levels (e.g. using `pywt.dwt2` function)
- For each sub-band, compute:  
  - Mean $\mu = \frac{1}{N} \sum x_i$  
  - Variance $\sigma^2 = \frac{1}{N} \sum (x_i - \mu)^2$

    (taken over the whole sub-band image)

Collecting means and variances for all the sub-bands in a single vector gives compact and interpretable features summarizing multiscale texture content.



#### 2. Neural network baseline

As a baseline, train a shallow feedforward neural network on flattened grayscale images (e.g., 64×64). Alternatively, you may implement a small convolutional network if time permits.

### ✅ Minimum Requirements

- Load and preprocess the Kylberg Texture Dataset
- Implement wavelet-based feature extraction using at least one wavelet family (e.g., Haar, Daubechies)  
- Train and evaluate at least one classical model (e.g., logistic regression, linear SVM) on extracted features  
- Train and evaluate a shallow NN on raw image pixels  
- Report and visualize:  
  - Test accuracy and confusion matrix for each model  
  - Summary of which sub-bands or levels contribute most to class separation  
  - Visual examples of DWT sub-band decompositions for selected classes  

### 🚀 Optional Extensions

- Compare different wavelet families (e.g., Haar, db2, sym4)  
- Compute alternative sub-band features (e.g., entropy, histogram bins)  
- Analyze performance as a function of decomposition level  
- Implement a basic CNN for comparison

### 📚 References


- Arivazhagan & Ganesan (2003) – [*Texture classification using wavelet transform*](https://www.researchgate.net/profile/Arivazhagan-Selvaraj/publication/222911521_Texture_classification_using_wavelet_transform/links/586d09cd08ae329d62136f61/Texture-classification-using-wavelet-transform.pdf) 
- Mallat (1999) – [*A Wavelet Tour of Signal Processing*](https://www.di.ens.fr/~mallat/papiers/WaveletTourChap1-6.pdf) 
- PyWavelets documentation: [https://pywavelets.readthedocs.io](https://pywavelets.readthedocs.io)  


