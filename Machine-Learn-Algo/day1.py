import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# loading the dataset
data = load_iris()
X, y = data.data, data.target

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train individual models
log_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

# Creating a voting classifier model 
vote_model = VotingClassifier(
    estimators=[
        ('log_model', log_model),
        ('dt_model', dt_model),
        ('knn_model', knn_model)
    ],
    voting='hard'
)

# Fitting the models
log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
vote_model.fit(X_train, y_train)

# prediction of ensemble model
y_pred = vote_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble model accuracy: {accuracy:.2f}")

# Evaluate individual model 
y_pred_log = log_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)
print(f"Logistic model accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print(f"Decission model accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(f"KNN model accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")

