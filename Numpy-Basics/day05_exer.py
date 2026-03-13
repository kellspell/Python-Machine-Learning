import pandas as pd

dt = {
    'Team': ['Madrid','Liverpool', 'Manchester', 'Barcelona'],
    'Country_code': ['Es', 'GB', 'GB', 'ES'],
    'Salary': [1200, 2000, 1550, 400],
    'Code': [44, 33, 12, 10],
}

df = pd.DataFrame(dt) # converting into a dictionary
print('Original dataset: \n', df)

# grouped = df.groupby('Salary').mean()
# print(grouped)

stats = df.groupby('Team').agg(
    {'Salary':['mean','max','min'], 'Code':['mean','max','min']}
)
print(stats)
