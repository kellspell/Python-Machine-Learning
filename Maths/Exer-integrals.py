# First, Let's Create a More Complex Dataset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import time

# Set random seed for reproducibility
np.random.seed(42)

# Generate a more complex dataset (non-linear with noise)
n_samples = 1000
n_features = 2

# Create features
X = np.random.randn(n_samples, n_features)

# Create non-linear target: y = 3*x1^2 - 2*x2^2 + 4*x1*x2 + noise
y = (3 * X[:, 0]**2 - 2 * X[:, 1]**2 + 4 * X[:, 0] * X[:, 1] + 
     0.5 * np.random.randn(n_samples))

# Add bias term
X_b = np.c_[np.ones((n_samples, 1)), X]

# Standardize features (important for Adam)
scaler = StandardScaler()
X_b[:, 1:] = scaler.fit_transform(X_b[:, 1:])

print(f"Dataset shape: {X_b.shape}")
print(f"Target shape: {y.shape}")

##########################* Optimizers *############################################

# Implement Different Optimizers
class Optimizers:
    @staticmethod
    def sgd(X, y, theta, learning_rate, n_epochs, batch_size=1, track_path=False):
        """Vanilla SGD"""
        m = len(y)
        path = [theta.copy()] if track_path else None
        losses = []
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, m, batch_size):
                # Get mini-batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Compute gradients
                y_pred = X_batch @ theta
                error = y_pred - y_batch.reshape(-1, 1)
                gradients = (2/len(X_batch)) * X_batch.T @ error
                
                # Update parameters
                theta -= learning_rate * gradients
                
                # Track loss
                batch_loss = np.mean(error**2)
                epoch_loss += batch_loss * len(X_batch)
                
            # Store loss and path
            avg_loss = epoch_loss / m
            losses.append(avg_loss)
            if track_path:
                path.append(theta.copy())
                
        return theta, losses, path
    
    @staticmethod
    def sgd_momentum(X, y, theta, learning_rate, n_epochs, batch_size=32, 
                     momentum=0.9, track_path=False):
        """SGD with Momentum"""
        m = len(y)
        velocity = np.zeros_like(theta)
        path = [theta.copy()] if track_path else None
        losses = []
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = X_batch @ theta
                error = y_pred - y_batch.reshape(-1, 1)
                gradients = (2/len(X_batch)) * X_batch.T @ error
                
                # Update with momentum
                velocity = momentum * velocity - learning_rate * gradients
                theta += velocity
                
                batch_loss = np.mean(error**2)
                epoch_loss += batch_loss * len(X_batch)
                
            avg_loss = epoch_loss / m
            losses.append(avg_loss)
            if track_path:
                path.append(theta.copy())
                
        return theta, losses, path
    
    @staticmethod
    def adam(X, y, theta, learning_rate=0.001, n_epochs=100, batch_size=32,
             beta1=0.9, beta2=0.999, epsilon=1e-8, track_path=False):
        """Adam Optimizer"""
        m = len(y)
        
        # Initialize Adam parameters
        m_t = np.zeros_like(theta)  # First moment
        v_t = np.zeros_like(theta)  # Second moment
        t = 0  # Time step
        
        path = [theta.copy()] if track_path else None
        losses = []
        
        for epoch in range(n_epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = X_batch @ theta
                error = y_pred - y_batch.reshape(-1, 1)
                gradients = (2/len(X_batch)) * X_batch.T @ error
                
                # Update time step
                t += 1
                
                # Update biased moments
                m_t = beta1 * m_t + (1 - beta1) * gradients
                v_t = beta2 * v_t + (1 - beta2) * (gradients**2)
                
                # Bias correction
                m_t_hat = m_t / (1 - beta1**t)
                v_t_hat = v_t / (1 - beta2**t)
                
                # Update parameters
                theta -= learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
                
                batch_loss = np.mean(error**2)
                epoch_loss += batch_loss * len(X_batch)
                
            avg_loss = epoch_loss / m
            losses.append(avg_loss)
            if track_path:
                path.append(theta.copy())
                
        return theta, losses, path

##########################* Visualization Functions *############################################    

def plot_loss_comparison(results_dict):
    """Plot loss curves for different optimizers"""
    plt.figure(figsize=(12, 6))
    
    for name, (theta, losses, _) in results_dict.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Loss Curves: Optimizer Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often helps see differences
    plt.show()

def plot_optimization_path_2d(results_dict, X, y):
    """Plot optimization path in 2D parameter space"""
    # Create a grid of parameter values
    theta0_range = np.linspace(-5, 5, 100)
    theta1_range = np.linspace(-5, 5, 100)
    T0, T1 = np.meshgrid(theta0_range, theta1_range)
    
    # Compute loss surface
    Z = np.zeros_like(T0)
    for i in range(len(theta0_range)):
        for j in range(len(theta1_range)):
            theta_test = np.array([[T0[j, i]], [T1[j, i]]])
            y_pred = X @ theta_test
            Z[j, i] = np.mean((y_pred - y.reshape(-1, 1))**2)
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    # Contour plot of loss surface
    plt.subplot(1, 2, 1)
    contour = plt.contourf(T0, T1, np.log(Z), levels=50, cmap='viridis')
    plt.colorbar(contour, label='Log Loss')
    
    # Plot optimization paths
    colors = ['red', 'blue', 'green']
    for (name, (theta, losses, path)), color in zip(results_dict.items(), colors):
        if path:
            path = np.array(path).squeeze()
            plt.plot(path[:, 0], path[:, 1], 'o-', color=color, 
                    label=name, markersize=3, linewidth=1.5)
            plt.plot(path[0, 0], path[0, 1], 's', color=color, 
                    markersize=10, label=f'{name} start')
            plt.plot(path[-1, 0], path[-1, 1], '*', color=color, 
                    markersize=15, label=f'{name} end')
    
    plt.xlabel('θ₀ (bias)')
    plt.ylabel('θ₁ (weight)')
    plt.title('Optimization Path in Parameter Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3D surface plot
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot_surface(T0, T1, np.log(Z), cmap='viridis', alpha=0.8)
    ax.set_xlabel('θ₀')
    ax.set_ylabel('θ₁')
    ax.set_zlabel('Log Loss')
    ax.set_title('Loss Surface (3D View)')
    
    plt.tight_layout()
    plt.show()

def compare_batch_sizes(X, y, theta_init):
    """Compare different mini-batch sizes"""
    batch_sizes = [1, 16, 64, 256]
    results = {}
    
    plt.figure(figsize=(12, 6))
    
    for batch_size in batch_sizes:
        theta = theta_init.copy()
        start_time = time.time()
        
        _, losses, _ = Optimizers.sgd(X, y, theta, learning_rate=0.01, 
                                      n_epochs=50, batch_size=batch_size)
        
        elapsed_time = time.time() - start_time
        results[f'Batch {batch_size}'] = losses
        
        plt.plot(losses, label=f'Batch Size {batch_size} (Time: {elapsed_time:.2f}s)', 
                linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Mini-Batch Size on Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    return results

##########################* Run the Comparison *############################################   

# Initialize parameters
theta_init = np.random.randn(X_b.shape[1], 1) * 0.1

# Dictionary to store results
results = {}

# 1. Vanilla SGD (full batch for path tracking - slower but clearer path)
print("Training Vanilla SGD...")
theta_sgd, losses_sgd, path_sgd = Optimizers.sgd(
    X_b, y, theta_init.copy(), learning_rate=0.01, 
    n_epochs=50, batch_size=32, track_path=True
)
results['SGD (batch=32)'] = (theta_sgd, losses_sgd, path_sgd)

# 2. SGD with Momentum
print("Training SGD with Momentum...")
theta_momentum, losses_momentum, path_momentum = Optimizers.sgd_momentum(
    X_b, y, theta_init.copy(), learning_rate=0.01, 
    n_epochs=50, batch_size=32, momentum=0.9, track_path=True
)
results['SGD + Momentum'] = (theta_momentum, losses_momentum, path_momentum)

# 3. Adam
print("Training Adam...")
theta_adam, losses_adam, path_adam = Optimizers.adam(
    X_b, y, theta_init.copy(), learning_rate=0.01, 
    n_epochs=50, batch_size=32, track_path=True
)
results['Adam'] = (theta_adam, losses_adam, path_adam)

# Plot results
plot_loss_comparison(results)
plot_optimization_path_2d(results, X_b[:, :2], y)  # Only first 2 params for visualization

# Compare mini-batch sizes
print("\nComparing mini-batch sizes...")
compare_batch_sizes(X_b, y, theta_init.copy()) 

##########################* Print Final Results *############################################  

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

for name, (theta, losses, _) in results.items():
    print(f"\n{name}:")
    print(f"  Final parameters: {theta.ravel()}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.2f}%") 
    
##########################* Bonus: Interactive Learning Rate Comparison *############################################ 
    
def compare_learning_rates(X, y, theta_init):
    """Compare different learning rates"""
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    plt.figure(figsize=(12, 6))
    
    for lr in learning_rates:
        theta = theta_init.copy()
        _, losses, _ = Optimizers.adam(X, y, theta, learning_rate=lr, 
                                       n_epochs=50, batch_size=32)
        plt.plot(losses, label=f'LR = {lr}', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Effect of Learning Rate on Convergence (Adam)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

# Run learning rate comparison
compare_learning_rates(X_b, y, theta_init.copy())

