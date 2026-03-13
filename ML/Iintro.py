"""
* Types of Machine learning 
    * Supervised Learning 
        * Model is trained of labeled data
        * The Model learns to map inputs(features) to output(target)
            * Examples: Classification | Regression
            * Key Features:
                * Requires Labeled data
                * Accuracy dependes heavily on the quality of the training data
    
    * Unsupervised Learning
        * Model works on unlabeled data to find hidden patters or structures
            * Examples Clustering | Dimentionality Rediction
            * Key Features:
                * No labeled data need it 
                * Focused on explolatory analysis and identifying paterns 
                
    * Reinforcement Learning
        * An agent interacts with an environment and learns by trial and error to maximize cumulative rewards
            * Examples: Robotics | Gaming | Dynamic System
            * Key Features:
                * Goal-oriented learning based on rewards and penalties 
                * Sustaible for sequential decision-making problems 
    * What is Features:
        * Features are the input variables(independent variables)used to train the model
        Examples: In predicting house prices, features could include the number of bedrooms, size and location
        
        * Target
            * The output variable(dependent variable) that the model predicts
            * Examples: House price is the target variable 
        
        * Training and Testing dataset
            * The data is split into two subset training set and testing set
            * A tipycal split is 80% training  and 20% testing 
        
        * Overfitting 
            * Model learns noise and details in the training data, performng poorly on new data
            * Model become to complex for the dataset
            
        * Underfitting 
            * The Model is too simple to capture the underlying patterns in the data 
                * Example: Fitting a linear model with a non-linear data
        
        * Bias-Variance Tradeoff
            * Bias: The error introduced by assuming a simplified model 
            * Variance: Error introduced by the model's sensitivity to small changes in training data 
            * Goal: Balance bias and variance to archive optimal performance                                     
                                   
"""

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


