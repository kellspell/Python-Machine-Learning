from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# loading the dataset
data = load_breast_cancer()
X, y =  data.data, data.target

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Display the dataset informations 
# print("Features: ", data.feature_names)
# print("Classes: ", data.target_names)

# Train the model Gradient boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Model prediction
y_pred = gb_model.predict(X_test)

# Evaluate performance
accuracy_gb = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Accuracy: {accuracy_gb:.2f}")
print(f"\n Classification Report:\n {classification_report(y_test, y_pred)}")


# Testing with hyperparameters
params_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=params_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Displaying the results from grid search
print(f"Best Parameters are: {grid_search.best_params_}")
print(f"Best Cross-validation are: {grid_search.best_score_}")


# Displaying the best estimator 
best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print("Tuned Gradient Bossting Classifier accuracy:", tuned_accuracy)
print("\nTuned Classification Report:\n", classification_report(y_test, y_pred_tuned))