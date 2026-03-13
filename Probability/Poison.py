import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters
lambda_ = 5  # average rate (e.g., 5 customers per hour)

# Possible number of events (0 to 15)
k = np.arange(0, 16)

# Calculate probabilities
probabilities = poisson.pmf(k, lambda_)

# Visualize
plt.bar(k, probabilities)
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ={lambda_})')
plt.show()

# What's the probability of exactly the average?
prob_avg = poisson.pmf(lambda_, lambda_)
print(f"Probability of exactly {lambda_} events: {prob_avg:.2%}")

# Generate random samples
samples = np.random.poisson(lambda_, 1000)
print(f"Average from 1000 samples: {np.mean(samples):.2f} (should be close to {lambda_})")