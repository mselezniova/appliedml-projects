## Project 8: *Text Classification with String Kernels*

### 🧠 Motivation

Text classification tasks such as **spam detection** benefit from models that can compare strings based on **lexical similarity** and **character-level patterns**. Traditional feature representations (e.g., bag-of-words) discard sequential structure, while deep learning may be overkill for short texts.

In this project, you will explore **string kernels** — similarity functions defined directly over text — in the context of **kernel SVMs**.

The idea is to compare two kernel functions:  
- the **spectrum (n-gram) kernel**, which counts shared substrings  
- the **Levenshtein kernel**, which transforms edit distance into similarity  

Both are applied to a short-text spam classification task.

### 🎯 Objectives

- Implement and compare two string kernels:
  - Spectrum (n-gram) kernel
  - Levenshtein distance kernel
- Train kernel SVMs on the SMS Spam dataset  
- Visualize kernel matrices and evaluate model performance  
- Compare to standard baselines using bag-of-words or TF-IDF

### 📊 Dataset

**SMS Spam Collection**  
- ~5,500 text messages labeled as `spam` or `ham` (not spam)  
- Short texts (1–3 sentences)  
- Available at: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

### 🛠️ String Kernels

#### 1. Spectrum (n-gram) Kernel

The spectrum kernel counts the number of common continuous substrings of fixed length $n$ between two strings $s$ and $t$:

$$
K_n(s, t) = \sum_{u \in \Sigma^n} \phi^u(s) \cdot \phi^u(t)
$$

where $\phi^u(s)$ is the number of times substring $u$ occurs in $s$.

- Feature space: $\mathbb{R}^{|\Sigma|^n}$, implicit via kernel matrix  
- **Note**: This is the same as the *spectrum kernel* used for DNA sequence comparison earlier in the course.

#### 2. Levenshtein (Edit Distance) Kernel

The Levenshtein distance $d(s, t)$ is the minimum number of insertions, deletions, and substitutions required to convert $s$ into $t$.

To define a similarity kernel, use an RBF-like transformation:

$$
K(s, t) = \exp\left( -\frac{d(s, t)^2}{2\sigma^2} \right)
$$

- $\sigma$ controls smoothness of similarity  
- No explicit feature space; kernel is computed pairwise

### ✅ Minimum Requirements

- Preprocess the dataset (e.g., lowercase, remove punctuation, tokenize)  
- Implement both kernels and compute full kernel matrices for training and test splits  
- Train two kernel SVMs 
- Evaluate both models, e.g. with:
  - Accuracy, precision, recall
  - Confusion matrix  
  - Number of support vectors  
- Visualize, e.g.:
  - Kernel matrix heatmaps  
  - ROC curves or error analysis

### 🚀 Optional Extensions

- Compare to a baseline classifier using bag-of-words or TF-IDF + linear SVM  
- Tune hyperparameters ($n$ for n-gram, $\sigma$ for Levenshtein)  
- Combine kernels via interpolation:

$$
K_{\text{combined}} = \alpha K_{\text{ngram}} + (1 - \alpha) K_{\text{edit}}
$$

- Use kernel PCA to visualize message embeddings  
- Study how kernel similarity aligns with semantic similarity

### 📚 References

- Lodhi et al. (2002) – [*Text Classification Using String Kernels*](https://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf) 
- Ristad & Yianilos (1998) – [*Learning String Edit Distance*](https://arxiv.org/pdf/cmp-lg/9610005)
