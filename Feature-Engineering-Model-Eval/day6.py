from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Or from sklearn.svm import SVC
# Or from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data = load_iris()
X = data.data
y = (data.target == 0).astype(int)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and train our model - Use a classifier instead of regressor
model = LogisticRegression()  # Better for classification
# model = DecisionTreeClassifier()  # Alternative
# model = SVC()  # Alternative
model.fit(X_train, y_train)

# Predict - classifiers directly output class labels
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Class 0', 'Class 0'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Classifications metrics 
print("\nClassification metrics\n")
print(classification_report(y_test, y_pred))