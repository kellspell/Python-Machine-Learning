import pandas as pd  

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# Explore the data by printing the first 5 rows 
print('The first 5 rows: \n', df.head())
print('The last 5 rows: \n', df.tail())

# Checking informations about the dataset
print(df.info())
print(df.describe())

# Selecting columns from the dataset
sl_cl = df[['species', 'sepal_length']]
print("Selected Columns: \n", sl_cl)

# Filter rows
f_r = df[(df['sepal_length'] > 5.0) & (df['species'] == 'setosa')]
print("Filtered Rows: \n", f_r)