## Project 5: *Improving Poisson Regression with Polynomial Feature Expansion*

### 🧠 Motivation

In class, we applied **Poisson regression** to the Bike Sharing dataset to model hourly rental counts using features such as temperature, humidity, and time of day. While the Poisson model is well-suited for count data and respects non-negativity, our results showed noticeable limitations — the model often underfit nonlinear trends in time and weather patterns.

This project revisits that problem from a new angle: rather than switching models, we **expand the feature space**. By applying **polynomial transformations** to the input variables, we allow the Poisson regression model to represent nonlinear interactions while retaining its probabilistic and interpretable structure.

The core idea is to lift the input space via a mapping:

$$
\log(\lambda_i) = \mathbf{w}^\top \phi(\mathbf{x}_i),
$$

where $\phi(\mathbf{x})$ includes polynomial features of increasing degree (e.g., $x_1^2$, $x_1 x_2$, etc.). This turns a simple GLM into a more expressive model without abandoning its statistical foundation.

### 🎯 Objectives

- Construct polynomial feature expansions of the original input space  
- Train Poisson regression models with these expanded features  
- Compare performance across polynomial degrees and to standard linear regression  
- Visualize overfitting and underfitting behavior as a function of feature complexity  
- Interpret which nonlinear interactions (if any) improve predictive accuracy

### 📊 Dataset

- **Bike Sharing Dataset** (UCI Machine Learning Repository)  
  - Hourly data on bike rentals, weather, seasonality, and time of day  
  - [Link](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)

Focus on the hourly subset:
- Target variable: `cnt` (total rental count per hour)  
- Input variables: `hr`, `temp`, `humidity`, `windspeed`, `weekday`, `workingday`, `season`  
- Optional: drop redundant or highly collinear features (e.g., `atemp`)

### 🛠️ Suggested Models

- **Poisson regression** (GLM with log link):
  
$$
y_i \sim \text{Poisson}(\lambda_i), \quad \text{with } \log(\lambda_i) = \mathbf{w}^\top \phi(\mathbf{x}_i)
$$

- **Feature expansion** via polynomial mappings:
  
$$
\phi_d(\mathbf{x}) = \text{all monomials of degree} \leq d
$$

- Optional comparison to:
  - Linear regression on raw and expanded features  
  - Ridge-regularized Poisson regression

### ✅ Minimum Requirements

- Preprocess the data (filter hours, normalize or scale if needed)  
- Implement polynomial feature expansion for varying degree $d$
- Train Poisson regression models with each feature expansion  
- Compare models using:
  - RMSE and MAE on held-out test data  
  - Poisson deviance or log-likelihood  
- Plot:
  - Predicted vs. true counts  
  - Training and test performance vs. degree $d$  
  - (Optional) heatmaps or curves of predicted rentals across time and temperature

### 🚀 Optional Extensions

- Experiment with the **full polynomial feature basis**, including both monomials and interactions:
  
$$
\phi_d(\mathbf{x}) =  \Bigl\\{ \prod_{j=1}^p x_j^{\alpha_j} : \sum_{j=1}^p \alpha_j \leq d  \Bigr\\}
$$

  This allows the model to learn nonlinear **combinations** of input variables — e.g., how temperature interacts with season or time of day.
- Identify which polynomial interaction terms contribute most to predictive performance  
- Apply regularization (e.g., Ridge) to stabilize high-degree expansions  
- Analyze the data statistics to determine wether Poisson distribution is an appropriate model 
- Compare to alternative count models (e.g., negative binomial regression)  


### 📚 References

- McCullagh & Nelder (1989) – [*Generalized Linear Models* (Ch. 6: Log-Linear Models)](https://www.utstat.toronto.edu/brunner/oldclass/2201s11/readings/glmbook.pdf)
- Hastie, Tibshirani, Friedman (2009) – [*The Elements of Statistical Learning* (Ch. 5)](https://hastie.su.domains/ElemStatLearn/)
