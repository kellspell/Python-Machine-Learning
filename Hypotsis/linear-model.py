from sklearn.linear_model import LinearRegression
import numpy as np

# Sample Data
x = np.array([1,2,3,4,5,6,7,8,10]).reshape(-1, 1)
y = np.array([2,4,3,6,7,4,9,10, 0.1])

# Fit Linear Regression
model = LinearRegression()
model.fit(x, y)

print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared: ", model.score(x, y))