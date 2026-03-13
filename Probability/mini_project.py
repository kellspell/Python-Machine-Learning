import numpy as np

# Generating random values
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


# Add bias term to feature matrix
X_b = np.c_[np.ones((100, 1)), X]

# Initialize the paramiters
theta = np.random.randn(2, 1)
learning_rate = 0.001
iterations = 1000

def predict(X, theta):
    return np.dot(X, theta)

def gradience_dec(X, y, theta, learning_rate, iterations):
    m = len(y)
    
    for _ in range(iterations):
        gradients = (1/m) * np.dot(X.T, (np.dot(X, theta) - y))
        theta -= learning_rate * gradients
        return theta 
    
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


theta_optimized = gradience_dec(X_b, y, theta, learning_rate, iterations)
        
# Predictions and  evaluation
y_pred = predict(X_b, theta_optimized)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred) 

print("Optimized Parameters (theta): ", theta_optimized)
print("MSE:", mse) 
print("R2:", r2) 
    
