import numpy as np
import matplotlib.pyplot as plt

# Bernoulli distribution parameters
p = 0.7  # 70% chance of success

# Generate 20 random Bernoulli trials (0 or 1)
trials = np.random.binomial(1, p, 20)  # 1 means "one trial"
print("20 random outcomes:", trials)
print("Number of successes:", sum(trials))
print("Observed probability:", sum(trials)/20, "(should be close to", p, ")")

# Visualize
outcomes = [0, 1]
probs = [1-p, p]

plt.bar(outcomes, probs)
plt.xticks([0, 1], ['Failure (0)', 'Success (1)'])
plt.ylabel('Probability')
plt.title(f'Bernoulli Distribution (p={p})')
plt.ylim(0, 1)
plt.show()