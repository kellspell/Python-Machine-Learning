import numpy as np
import matplotlib.pyplot as plt

# Fix 1: Assign both mu AND sigma
mu, sigma = 0, 1  # Standard normal distribution

# Fix 2: Your formula was actually correct! Good job!
x = np.linspace(-4, 4, 100)
y = (1 / (np.sqrt(2 * np.pi * sigma ** 2))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Fix 3: plt.plot, not ply.show
plt.plot(x, y)
plt.title('Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True, alpha=0.3)
plt.show()  # Note the parentheses!
