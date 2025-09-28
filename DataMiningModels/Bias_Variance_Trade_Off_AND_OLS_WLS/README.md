# Bias–Variance Trade-off, Ridge & Lasso, and Weighted Regression

This project contains a series of experiments and visualizations to explore **core regression concepts** in machine learning and statistics.  
It focuses on understanding the **bias–variance trade-off**, **ridge vs lasso regularization**, and the difference between **Ordinary Least Squares (OLS)** and **Weighted Least Squares (WLS)** under heteroskedastic noise.  
It also includes 3D geometric interpretations of orthogonalization, normalization, and constraint sets.

---

## 📂 Project Overview

### 1. Bias–Variance Trade-off on UCI Energy Efficiency Dataset
- Dataset: [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency)  
- Target: **Heating Load (Y1)**  
- Procedure:
  - Polynomial feature expansion with degrees **1–6**  
  - Ridge regression with λ ∈ {0, 1e-3, 1e-2, 1e-1, 1, 10, 100}  
  - Model selection based on validation error  
- Output: **MSE vs Polynomial Degree** plots for Train, Validation, and Test sets  

---

### 2. Weighted Linear Regression with Heteroskedastic Noise
- Setup:
  - Noise variance switches at an **unknown threshold τ** on one feature.  
  - Known σ_s (small variance) and σ_L (large variance).  
- Models:
  - **OLS (unweighted)**  
  - **WLS (weighted by estimated τ)**  
- Procedure:
  - Search over candidate τ values  
  - Select τ* with lowest validation MSE  
- Output:
  - Comparison of OLS vs best WLS test errors  
  - Plot of **Validation MSE vs τ**

---

### 3. Gradient Descent with Regularization
- Quadratic loss function with L2 penalty:  
  \( L_\lambda(w) = L(w) + \lambda \|w\|^2 \)  
- Experiments:
  - Varying step size (η ∈ {1, 0.1, 0.01})  
  - Varying regularization strength (λ ∈ {0, 1, 10})  
- Output:
  - **Convergence curves** showing effect of η and λ on optimization stability and speed  

---

### 4. Ridge vs Lasso: Geometry and Optimization
- Visualizations:
  - Constraint sets:
    - Ridge → L2 ball (sphere)  
    - Lasso → L1 ball (diamond / octahedron)  
  - Loss function surfaces intersecting with constraints  
- Key insights:
  - Ridge shrinks weights smoothly (no sparsity)  
  - Lasso induces sparsity by forcing some coefficients to 0  

---

### 5. Orthogonal and Normalized Vectors in 3D
- Visualization of:
  - Non-normalized orthogonal vectors (different lengths)  
  - Normalized orthogonal vectors (unit length, equal magnitude)  
- Demonstrates:
  - Orthogonality (dot product = 0)  
  - Normalization (‖v‖ = 1)  
  - Geometric intuition for whitening and simplifications in regularization theory  

---

## 🚀 Features

- Hands-on implementation of **OLS, WLS, Ridge, and Lasso**  
- Exploration of **bias–variance trade-off** with polynomial regression  
- Visualization of **constraint geometries** in 2D and 3D  
- Gradient descent dynamics for different hyperparameters  
- Educational plots explaining regression theory from multiple perspectives  

---

## 🔧 Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
