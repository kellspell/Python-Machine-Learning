import pandas as pd  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load our dataset
df = pd.read_csv('/home/Kellspell/Py-Dev/Feature-Engineering-Model-Eval/bike_sharing_daily.csv')

# Display dataset info
# print(df.head())
# print(df.info())
# print(df.head())

# convert dteday to datetime
df['dteday'] = pd.to_datetime(df['dteday'])

# Create a new feature
df['day_of_week'] = df['dteday'].dt.day_name()
df['month'] = df['dteday'].dt.month
df['year'] = df['dteday'].dt.year

# Display the new features
# print("\n New features derrived from Date Columns")
# print(df[['dteday', 'day_of_week', 'month', 'year']].head())


# Select features and target
X = df[['temp']]
y = df[['cnt']]


# Apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Display the transformend feature
# print("\n Original and polynomial features")
# print(pd.DataFrame(X_poly, columns=['temp', 'temp^2']).head())


# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train_poly, X_test_poly = train_test_split(X_poly, test_size=0.2, random_state=42)

# train and evaluate model with original features
model_original = LinearRegression()
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
mse_original = mean_squared_error(y_test, y_pred_original)


# Train and evaluating model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)

# compare Results
print(f"MSE Original: {mse_original:.2f}")
print(f"MSE Poly: {mse_poly:.2f}")
