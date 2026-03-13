import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, binom, poisson, skew, kurtosis, t 

# Sample dataset
data = [1,34,7564,98762, 45, 6523,]

# Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data, ddof = 1)

# Get 95% of Confidence Interval (using t-distribution)
n = len(data)
t_value = t.ppf(0.975, df=n-1)
margin_of_error = t_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)
print("Confidence Interval", ci)