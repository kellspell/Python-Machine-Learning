from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform



# Create a sample of dice roll
sample_space = list(range(1, 6))

# Probability of rolling an even number
even_numbers = [2,4,6]
P_even = len(even_numbers) / len(sample_space)
print("Probability of even num: ", P_even)


#//////////////////////////////////////////////#

# Random variable: Dice roll
outcomes = np.array([1,2,3,4,5,6])
Probabilities = np.array([1/6] * 6)

# Expectation
Expectation = np.sum(outcomes * Probabilities)
print("Expectation (Mean)", Expectation)

# Variance and Standard deviation
Variance = np.sum((outcomes - Expectation) **2 * Probabilities)
std_dev = np.sqrt(Variance)
print("Variance: ", Variance)
print("Standard Dev: ", std_dev)

#//////////////////////////////////////////////////////////////////#

# Simulating 10.000 dice rolls
rolls = np.random.randint(1, 7, size = 10000)

# Calculate probability
P_even = np.sum(rolls % 2 == 0) / len(rolls)
print("P_even: ", P_even)

# Now lets check what is the probability to get the number 4 or greater than 4
p_greater_than_4 = np.sum(rolls > 4) / len(rolls)
print("The probability to be greater than 4 is: ", p_greater_than_4)

#//////////////////////////////////////////////////////////////////#

# Discrete random variable dice roll
# outcomes = [1,2,3,4,5,6]
# Probabilities = [1/6] * 6
# plt.bar(outcomes, Probabilities, color='blue', alpha=0.7)
# plt.title('PMF of a dice roll')
# plt.xlabel('Outcomes')
# plt.ylabel('Probabilities')
# plt.show()


#//////////////////////////////////////////////////////////////////#

# Continuous random variable: Uniform distribution
X = np.linspace(0, 1, 100)
pdf = uniform.pdf(X, loc = 0, scale = 1 )
plt.plot(X, pdf, color = 'red')
plt.title('PDF uniform distribution: ')
plt.xlabel('X')
plt.ylabel('f(X)')
plt.show()
