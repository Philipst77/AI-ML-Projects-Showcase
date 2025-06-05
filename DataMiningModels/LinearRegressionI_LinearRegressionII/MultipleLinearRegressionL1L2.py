# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# -----------------------------------------------------------
# L1 and L2 Regularization Comparison (Lasso and Ridge)
# -----------------------------------------------------------

from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

alphas = [0.01, 0.1, 1, 10]
results = []

for alpha in alphas:
    # Lasso Regression (L1)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_preds = lasso.predict(X_test)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_preds))
    results.append(["Lasso", alpha, lasso_rmse])

    # Ridge Regression (L2)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    results.append(["Ridge", alpha, ridge_rmse])

# Print results table
print("\nComparison of Lasso and Ridge Regression:")
print("{:<10} {:<10} {:<10}".format("Model", "Alpha", "RMSE"))
for row in results:
    print("{:<10} {:<10} {:.2f}".format(row[0], row[1], row[2]))
