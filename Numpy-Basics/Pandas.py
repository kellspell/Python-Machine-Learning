import pandas as pd  

s = pd.Series([20,37,49,98], index=['a','b','c', 'd'])
print(s)

data = {'Name': ["Alice", "Bob"], 'Age': [20, 30]}
df = pd.DataFrame(data)
print(df)

# View data 
print(df.head())
print(de.tail(3))
print(df.info())
print(df.describe())

# Read from file 
df = pd.read_csv('data.csv')
df = pd.to_csv('data.csv', index = False)
df = pd.to_excel('data.xlsx', index = False)   