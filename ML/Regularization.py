import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Generating random data
np.random.seed(42)
X = np.random.rand(100, 1) * 100
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

# Transform inputs into polymonial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(y, X_poly, test_size = 0.2, random_state = 42)

# Ridge Model
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

# Lasso Model
lasso_model = Lasso(alpha=1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)

# Evaluate the models
ridge_mse = mean_squared_error(y_test, ridge_pred)
print("Ridge Regression MSE:", ridge_mse)

lasso_mse = mean_squared_error(y_test, lasso_pred)
print("Lasso Regression MSE:", lasso_mse)