# 📊 Multiple Linear Regression with Regularization

This project implements **multiple linear regression** using the **50_Startups dataset**, with a focus on model evaluation and regularization techniques.

## ✅ Features

- **Data Preprocessing**  
  Utilizes `OneHotEncoder` to convert the categorical `State` feature into numerical columns.

- **Model Training**  
  Trains a `LinearRegression` model to predict startup profits based on:
  - R&D Spend
  - Administration
  - Marketing Spend
  - State (encoded)

- **Performance Evaluation**
  - Splits the dataset into training and test sets
  - Predicts on the test set and prints actual vs predicted values
  - Computes:
    - **MAE** (Mean Absolute Error)
    - **RMSE** (Root Mean Squared Error)

## 🧠 Regularization: Lasso vs Ridge

To prevent overfitting and compare model performance, both **L1 (Lasso)** and **L2 (Ridge)** regularization were implemented.

- Loops through different `alpha` values: `0.01`, `0.1`, `1`, `10`
- Trains both models for each `alpha`
- Calculates and compares **RMSE** values
- Results printed in a formatted table

## 📄 Example Output Table

Model Alpha RMSE
Lasso 0.01 9156.23
Ridge 0.01 8932.11
Lasso 0.1 9284.77
Ridge 0.1 8901.42


## 📁 Files

- `multiple_linear_regression.py` — Full implementation of preprocessing, model training, regularization, and evaluation

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `sklearn.linear_model` — `LinearRegression`, `Lasso`, `Ridge`
- `sklearn.metrics` — `mean_squared_error`
- `sklearn.preprocessing` — `OneHotEncoder`
- `sklearn.model_selection` — `train_test_split`

---

