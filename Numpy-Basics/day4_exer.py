import pandas as pd  
import numpy as np

# Create a simple dataset
data = {
    'Name': ['Alice', 'Bob', 'Marya', np.nan, 'David'],
    'Age': [29, 23, 20, np.nan, np.nan],
    'Score': [np.nan, 85, 90, 88, 45]
}

# Converting into a dictionary
df = pd.DataFrame(data)
print("Printing our data frame dictionary", df)

# Mean 
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Score'] = df['Score'].interpolate()
print("The New values are: \n", df)

# Rename columns
df = df.rename(columns={'Age':'ID', 'Name':'Student'})
print("The New Columns are: \n", df)


df1 = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Marya'],
    'Age': [29, 23, 20],
    
})

df2 = pd.DataFrame({
    'Age': [29, 23, 20],
    'Score': [90, 88, 45]
})

print("Dataset 1", df1)
print("Dataset 2", df2)

merged = pd.merge(df1, df2, how='inner', on='Age')
print("Merged dataset: \n", merged)

merged['Score_percentage'] = (merged['Score'] / 100) * 300
print("Transformed dataset: \n", merged)