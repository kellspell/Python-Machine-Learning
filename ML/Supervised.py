# Overview of a supervised learning introduction to Regression analizes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating random data
np.random.seed(42)
X = np.random.rand(100, 1) * 100
y = 3 * X + np.random.randn(100, 1) * 2

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42
                                                    )

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions based on our dataset
y_pred = model.predict(X_test)

# print coeficients
print("Slope: \n", model.coef_[0][0])
print("Intercept: \n", model.intercept_[0])

# Visualize our prediction
plt.scatter(X_test, y_pred, color='purple', label='Actual')
plt.plot(X_test, y_pred, color='red', label='prediction')
plt.title('Linear Regression Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


# Evaluating performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mse: \n", mse)
print("R-squared: \n", r2)

