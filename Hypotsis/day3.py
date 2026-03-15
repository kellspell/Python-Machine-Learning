import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# Creating the dataset
x = np.array([1,2,3,4,5,6,7,8,10])
y = np.array([2,4,3,6,7,4,9,10, 0.1])

# Pearson Correlation
r, _ = pearsonr(x, y)
print("Pearson Correlation: ", r)

# Spearman Correlation
rh, _ = spearmanr(x, y)
print("Spearman Correlation: ", rh)