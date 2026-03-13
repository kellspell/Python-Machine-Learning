import sympy as sp  
import numpy as np

# Define a function
x = sp.Symbol('x')
f = sp.exp(-x) 

# Compute indefinite integral
indefinite_integral = sp.integrate(f, x)
print("Indefinite-Integral", indefinite_integral)

# Compute definite integral
Definite_integral = sp.integrate(f, (x, 0, sp.oo))
print("Definite-Integral", Definite_integral)

# Implementing SGD
# Generate random dataset

np.random.seed(42)
"""
Set the random number generator's starting point to 42 
so that the same random numbers will be generated each time this code runs 
(making results reproducible).
"""
X = 2 * np.random.rand(100, 1)
"""
Create 100 random numbers between 0 and 1, arranged in a single column, 
then multiply each by 2 to get 100 random numbers between 0 and 2. 
Store these as the input feature X.
"""
y = 4 + 3 * X + np.random.rand(100, 1)
"""
For each of the 100 values in X, calculate y by taking 4, adding 3 times X, 
and then adding a small random number (between 0 and 1) to introduce noise. 
This creates the target values y.
"""

# Add bias term to x
X_b = np.c_[np.ones((100, 1)), X] 

# SGD Implementation
def stochastic_gd_descent(X, y, theta, learning_rate, n_epochs):
    m = len(y)
    for epochs in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradrients =2 * xi.T @ (xi @ theta - yi)
            theta -= learning_rate * gradrients
        return theta    

# Initialized the parameters
theta = np.random.rand(2,1)
learning_rate = 0.001
n_epochs = 50

# Perfom SGD
theta_opt = stochastic_gd_descent(X_b, y, theta, learning_rate, n_epochs)
print("Optimised paramiters: \n", theta_opt)    