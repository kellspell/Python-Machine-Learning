import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, binom, poisson, skew, kurtosis, t, ttest_1samp

# Sample Data 
data = [12,32,13,14,15,17,18,19]

# NullHypothesis mean = 15
population_mean = 15

# perform t-test
t_stat, p_value = ttest_1samp(data, population_mean)
print("T-Statistics", t_stat)
print("P-value", p_value)

# interprete Results
alpha = 0.05
if p_value <= alpha:
    print("Reject Null hypothesis: significant difference")
else:
    print("Fail to reject the null hypothesis: no significate difference")    