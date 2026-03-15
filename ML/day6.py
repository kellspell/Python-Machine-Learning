import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report




# Load the dataset
data = load_iris()
X, y = data.data, data.target

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y, 
                                                    test_size = 0.2, 
                                                    random_state = 42
                                                    )

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Experiment with different values of k
for k in range(1, 11):
    # Initialize k-NN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Let's make a prediction on the test dataset
    y_pred = knn.predict(X_test)
    
    # Evaluate precision
    acc = accuracy_score(y_test, y_pred)
    print(f"k = {k}, accuracy = {acc:.2f}")