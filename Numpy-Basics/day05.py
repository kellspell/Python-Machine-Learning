# Grouped by 

grouped = df.grouped('column_name')

for name, group in grouped:
    print(name)
    print(group)
    
grouped.mean()
grouped.max()
grouped.sum()

df.groupby('Category_name')['numeric_column'].mean()
df.groupby('Category_name').agg({'Numeric_column':['mean', 'max', 'min']})

# Another way to group tables 
pivot = df.pivot_table(
    values='Numeric_values',
    index='category_columns',
    aggfunc='mean'
)

# Creating a function to arange tables
def arange_fun(x):
    return x.max() - x.min()

df.groupby('category_column')['numeric_value'].agg(arange_fun)
# or 
df.groupby('category_column')['numeric_value'].mean()
# or 
df.groupby('category_column')['numeric_value'].max()
# And so on

df.groupby('category_column').agg({'numeric_value':['mean','max', 'min']})    