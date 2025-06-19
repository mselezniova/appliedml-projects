## Project 1: *Classifying Music Genres from Audio Features*

### 🧠 Motivation

Music genre classification is a classic problem in audio-based machine learning. While raw audio is high-dimensional and complex, features like MFCCs, spectral contrast, and zero-crossing rate offer structured, lower-dimensional representations of timbre and rhythm.

This project focuses on using classical and neural models to predict genre from precomputed audio features. The goal is to analyze model performance across genres, evaluate feature effectiveness, and consider alternative representations (e.g., spectrograms).

### 🎯 Objectives

- Implement and compare at least two supervised models for genre classification
- Use precomputed audio features as inputs
- Evaluate accuracy and confusion patterns across genres
- Interpret genre-wise performance differences
- Optionally: apply deep models to raw or time-frequency representations (e.g., spectrograms)

### 📦 Dataset

- **GTZAN Genre Collection**: 1000 audio tracks across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- Dataset available at: [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- Use precomputed features (MFCCs, chroma, spectral contrast, etc.) extracted from short segments
- Optionally compute your own features from the available raw data

### 🛠️ Suggested Models

- Classical:
  - Support Vector Machine (linear or kernel)
  - Logistic regression
- Neural:
  - MLP on feature vectors
  - CNN on spectrograms (optional extension)

### ✅ Minimum Requirements

- Train at least two different models (e.g., SVM and MLP) on MFCC-based features
- Evaluate and compare test accuracy
- Include confusion matrix and genre-wise performance analysis
- Discuss feature limitations and genre-specific challenges


### 🚀 Optional Extensions

- Train a CNN on mel spectrograms
- Incorporate temporal structure via segment aggregation or sequence models
- Use data augmentation (pitch shift, time stretch, noise)
- Compare performance on different subsets (e.g., percussive vs. harmonic genres)
- Use model interpretation tools to identify important features (e.g., SHAP)
- Use PCA or t-SNE to visualize the feature space by genre

### 📚 References
- Sturm (2013) – [The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use](https://arxiv.org/pdf/1306.1461)
- Tzanetakis & Cook (2002) – [*Musical genre classification of audio signals*](https://www.cs.cmu.edu/~gtzan/work/pubs/tsap02gtzan.pdf)
- Choi et al. (2017) – [*Convolutional recurrent neural networks for music classification*](https://arxiv.org/pdf/1609.04243)

