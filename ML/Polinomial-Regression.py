import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating random data
np.random.seed(42)
X = np.random.rand(100, 1) * 100
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

# Transform inputs into polymonial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Spliting the data
# X_train, X_test, y_train, y_test = train_test_split(
#                                                     X, y, 
#                                                     test_size = 0.2, 
#                                                     random_state = 42
#                                                     )

# Fit Polynomial regression
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)


# Visualize our prediction
plt.scatter(y, y_pred, color='blue', label='Actual Data')
plt.plot(y, y_pred, color='red', label='Predicted Data')
plt.title('Polymonial Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Evaluating performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mse: \n", mse)
print("R-squared: \n", r2)


# Regularization Techinics Lasso and Ridge
"""
* What is Regularizations?
    * Techinique used to prevent overfitting by adding penalties term to the cost function of a regression model 
* Types of regularizations:
    * Ridge regression(L2 regularization)
        * Adds the sum of the squared coefficients to the cost function
    * Lasso Regularization:
        * Adds the sum of the absolute coefficient to the cost function
* Key Differences:
    * Ridge shrinks coeffients but does not eliminate them
    * Lasso can shrinks some coefficient to zero , removing irrelevant features                
"""