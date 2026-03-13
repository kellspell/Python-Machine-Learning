import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 20  # number of trials (e.g., 20 customers)
p = 0.3  # probability of success (30% buy something)

# Possible number of successes (0 to 20)
k = np.arange(0, n+1)

# Calculate probabilities
probabilities = binom.pmf(k, n, p)

# Visualize
plt.bar(k, probabilities)
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.show()

# What's the most likely outcome?
most_likely = k[np.argmax(probabilities)]
print(f"Most likely: {most_likely} successes")
print(f"Average (mean): {n*p} successes")