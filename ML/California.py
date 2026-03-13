import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score 

# Loading the dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Selectind the data input(Median income) and output(Median houses values)
x = df[['MedInc']]
y = df[['MedHouseVal']]

# Transform the data into polynomial features or inputs
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(x_poly, y, test_size = 0.2, random_state = 42 )



# Fit the Polynomial model
model = LinearRegression()
model.fit(x_poly, y)

# Make predictions 
y_pred = model.predict(x_poly)


# Plot actual values vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Actual Data', alpha=0.5)
plt.scatter(x, y_pred, color='red', label='Predicted Curve', alpha=0.5)
plt.title('Polymonial Model')
plt.xlabel('Median Income in california')
plt.ylabel('Median housing price in california')
plt.legend()
plt.show()


# Evaluating performance
mse = mean_squared_error(y, y_pred)
print("Mse: \n", mse)
