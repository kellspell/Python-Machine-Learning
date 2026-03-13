# This is an EDA data science project a very basic one 
# Tasks 1 - Perform Data Cleaning, Aggregation and Filtering 

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

 

# Load the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# Lets add the data into DataFrame
df = pd.read_csv(url)

# Let's inspect the data
# print("Data from the DataFram", df.info())
# print(df.describe())

# Handle the missing values 
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

# Filter data: Passengers in the first class
first_class = df[df['Pclass'] == 1]
print("First class  Passengers: \n", first_class.head())


# Tasks 2 - Generate Visualisations to ilustrate the key insights 
# survival_by_cl = df.groupby("Pclass")['Survived'].mean()
# survival_by_cl.plot(kind='bar', color='skyblue')
# plt.title('Survival rate by class')
# plt.ylabel('survival rate')
# plt.show()

# Histogram: Age distribution
# sns.histplot(df['Age'], kde = True, bins = 20, color = 'Purple')
# plt.title('Age Distribution')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# plt.show()

# Scatter plot: Age vs Fare
plt.scatter(df['Age'], df['Fare'], alpha = 0.5, color = 'Green')
plt.title('Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# Summary of our findings 
"""

Titanic Dataset EDA Report
1. Overview
    - Dataset contains  891 rows and 12 columns 
    - The missing values handle for 'Age'(filled with median), Embarked(filled with mode)
2. Key insights:
    - Survival rates are highest for firt-class passengers (62%) and lowest for second-class customers (24%)
    - Majority of passengers are age between 20 - 40 years old
    - A positive correlation  exists  between fare and survival
3. Visualisations
    - Screenshots go's here         

"""

