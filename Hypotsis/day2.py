import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway

# Contengecy Table
data = [[50,30], [20, 90]]

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(data)
print("Chi-Square Statistic:", chi2)
print("P-Value: ", p)
print("DOF: ", dof)
print("Expected frequency \n: ", expected)

# Create data for 3 groups 
g1 = [1,2,3,4,5,6,7,8,9]
g2 = [11,12,13,14,15,16,17,18,19]
g3 = [21,22,23,24,25,26,27,28,29]

# Performing Anova
f_stat, p_value = f_oneway(g1,g2, g3)
print("F-Statistics", f_stat)
print("P-Value", p_value)