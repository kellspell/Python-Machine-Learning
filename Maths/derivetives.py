"""
* Derivatives and their role in optimization
    * Measure the rate at which a function changes with respect to its input 
    * For a funtion f(X), the derivative f' (X) indicate the slope of the tangent line at a point X
* Role in Optimization
    * Common Derivatives    
"""

import sympy as sp  
import numpy as np


x = sp.Symbol('X')
f = x**2
Derivatives = sp.diff(f, x)
# print("Derivatives: \n", Derivatives)

"""
* Partial derivatives 
    * Measure how a function changes  with respect to one variable while keeping other value constant 
* Gradient
    * Vector of all partial derivatives, indication the direction of the steepest ascent     

"""

x, y = sp.symbols('x, y')
f = x**2 + y**2
grad_x = sp.diff(f, x)
grad_y = sp.diff(f, y)
# print("Partial Derivatives: \n", grad_x, grad_y)

"""
* Gradient Decent Optimization Algorithm
    * What is Gradient Decent?
        * Iterative optimization algorithm used to minimize a function
        * Updates parameters in the direction of the negative gradient to find the minimum
    * Why is gradient decent import in machine learning?


"""
# Define the function
x = sp.symbols('x')
f = x**3 - 5*x + 7

# Compute Derivatives
derivative = sp.diff(f, x)

# print("Function:", f)
# print("Derivatives:", derivative)


# Define the function
# x = sp.symbols('x, y')
# f = x**2 + 3*y**2 - 4*x*y

# Compute partial derivatives
# grad_x = sp.diff(f, x)
# grad_y = sp.diff(f, y)
# print("Partial Derivatives: \n", grad_x, grad_y)


def gradient_decent(X, y, theta, learning_rate, iterations ):
    m = len(y)
    for _ in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradients = (1/m) * np.dot(X.T, errors)
        theta -= learning_rate * gradients
    return theta

# Sample Data
X = np.array([[1,1], [1,2],[1,3]])
y = np.array([2, 2.25, 3.25])
theta = np.array([0.1, 0.1])
learning_rate = 0.1
iterations = 500

# Performing a gradient decent 
optimize_theta = gradient_decent(X, y, theta, learning_rate, iterations)
print("Optimized Parameters: \n", optimize_theta)   