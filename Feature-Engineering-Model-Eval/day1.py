import pandas as pd  

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Display dataset information
# print("Dataset info: ", df.info())

# # Preview the first few rolls
# print("Dataset preview: ", df.describe())

# Let's separate the features by category
categorical_features = df.select_dtypes(include=['object', 'string']).columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
print("Categories: ", categorical_features.tolist(), numerical_features.tolist())

# Display summary of categorical features
print('\n Categorical feature summary \n')
for col in categorical_features:
    print(f"{col}:\n", df[col].value_counts(), '\n')
    
# Display summary of numerical features
print("\n Bumerial features summary \n")
print(df[numerical_features].describe())    