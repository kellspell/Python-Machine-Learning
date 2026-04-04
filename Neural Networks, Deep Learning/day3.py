import numpy as np

# Define Mean Squared Error(MSE)
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Define Cross-Entropy(BCE) loss
def binary_cross(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Let's create some sample data
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.2, 0.8, 0.7])

# Calculation of loss
mse = mse_loss(y_true, y_pred)
bce = binary_cross(y_true, y_pred)

print(f"MSE LOSS: {mse:.4f}")
print(f"BCE LOSS: {bce:.4f}")

# Derivative of MSE loss
def mse_gradient(y_true, y_pred):
    return 2 * (y_true - y_pred) / len(y_true)

# Derivative of BCE loss
def bce_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # Prevent division by zero
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

# Calling both derivatives
grad_mse = mse_gradient(y_true, y_pred)
grad_bce = bce_gradient(y_true, y_pred)

print(f"MSE GRADIENT LOSS: {grad_mse}")
print(f"BCE GRADIENT LOSS: {grad_bce}")

