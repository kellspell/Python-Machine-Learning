from itertools import product
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, binom, poisson, skew, kurtosis

# Load the dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
df = pd.read_csv(url)

# Analyze sepal_length
feature = df['sepal_length']
print("Skewness", skew(feature))
print("Kurtosis", kurtosis(feature))

# Visualize distribution
sns.histplot(feature, kde=True)
plt.title('Distribution os Sepal_length: ')
plt.show()