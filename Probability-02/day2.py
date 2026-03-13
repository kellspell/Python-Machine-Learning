from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, binom, poisson

# Gaussian Distribution
X = np.linspace(-5, 5, 100)
plt.plot(norm.pdf(X, loc = 0, scale = 1 ), label = 'Gaussian (u=0, s=1)')

# Binomial Distribution
n, p = 10, 0.5
x = np.arange(0, n+1)
plt.bar(x, binom.pmf(x, n, p), alpha=0.7, label='Binomial (n=10, p=0.5)')

# Poisson Distribution
lam = 3
x = np.arange(0, 10)
plt.bar(x, poisson.pmf(x, lam), alpha=0.7, label='Poisson(L = 3)')

# Visualizing all 3 together
plt.title('Probability distribution: ')
plt.legend()
plt.show()

