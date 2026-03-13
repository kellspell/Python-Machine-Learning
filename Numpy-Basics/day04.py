df = df.dropna() # This function lets you to drop columns
df = df.dropna(axis = 1) # This will drop columns with missing values

df['column_name'] = df['column_name'].fillna(0) # In this case if the columns has missing values we can fill up with zeros 

df.fillna(method='ffill') # add values at the front
df.fillna(method='bfill') # add values at the back

df['column_name'] = df['column_name'].interpolate()

df = df.rename(columns ={'old_name':'new_name'})

df['column_name'] = df['column_name'].astype('float')
df['column_name'] = df.to.datetime(df['column_name'])

combined = pd.concat([df1,df2], axis=0)
combined = pd.concat([df1,df2], axis=1)

merged = pd.merged(df1, df2, on='common_column')
merged = pd.merged(df1, df2, how = 'left', on='common_column') 
merged = pd.merged(df1, df2, how = 'inner', on='common_column')   

joined = df1.join(df2, how='inner') 


