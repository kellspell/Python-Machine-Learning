# Define features and target variable
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
 
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv'
df = pd.read_csv(url)

# Features and Target
features = df[['total_bill', 'size']]
target = df[['tip']]


# Setting the train and test dataset
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size = 0.2, random_state = 42) # there library makes easy to split the dataset
print("Training dataset: \n", X_train.shape)
print("Test dataset: \n", X_test.shape)


# Visualise the data 
sns.pairplot(df, 
             x_vars=['total_bill', 'size'], 
             y_vars=['tip'], 
             height=5, 
             aspect=0.8, 
             kind='scatter' )
plt.title('Features vs target relationships')
plt.show()


