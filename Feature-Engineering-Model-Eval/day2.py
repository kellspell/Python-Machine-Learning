from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# Load the dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Display dataset infor
print("Dataset info")
print(X.describe())
print("\n Target Classes \n", data.target)    

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Initialize k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Let's make a prediction on the test dataset
y_pred = knn.predict(X_test)
    
# Evaluate precision
acc = accuracy_score(y_test, y_pred)
print(f"accuracy = {acc:.2f}")

# Apply Min/Max Scaler
scaler = MinMaxScaler()
X_scaler = scaler.fit_transform(X)

# Taking the scaled data and spliting into new variables 
X_train_Sc, X_test_Sc, y_train_Sc, y_test_Sc = train_test_split(X_scaler, y, test_size=0.2, random_state=42)

# Train the k-NN on the scale data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_Sc, y_train_Sc)

# Let's make a prediction on the test dataset
y_pred_Sc = knn_scaled.predict(X_test_Sc)
print("Accuracy with Min/Max Scaling: ", accuracy_score(y_test_Sc, y_pred_Sc))