from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display dataset info
# print(df.head())
# print(df.info())

# Calculate the correlation matrix
correlation_mx = df.corr()

# Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_mx, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# Select feature with high correlation to the target
correlated_features = correlation_mx['target'].sort_values(ascending=False)
print("Features must correlate with target")
print(correlated_features)

# Seperate features and target
X = df.drop(columns=['target'])
y = df['target']

# Calculate mutual info
mutual_info = mutual_info_regression(X, y)

# Create a dataframe for better visualization
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual informations': mutual_info})
mi_df = mi_df.sort_values(by= "Mutual informations", ascending=False)
print("Mutual informations Score: ")
print(mi_df)

# Taing the model 
model = RandomForestRegressor(random_state=42)
model.fit(X,y)

# Get feature importance 
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importance from Random Forest: ")
print(importance_df) 

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_xaxis()
plt.title('Feature Importance from Random Forest')
plt.show()