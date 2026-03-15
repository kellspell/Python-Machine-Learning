import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, binom, poisson, skew, kurtosis, t, ttest_ind

# Load the dataset
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
df = pd.read_csv(url)

# Inspect the dataset
print(df.info())
print(df.describe())

# Visualize distribution
# sns.histplot(df['total_bill'],kde=True)
# plt.title('Distribution of Total Bill')
# plt.show()

# In order to work with heatmap in this case we need to delete few columns 
del df['sex']
del df['smoker']
del df['day']
del df['time']

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()