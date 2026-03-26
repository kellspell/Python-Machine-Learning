from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier

# loading the dataset
data = load_breast_cancer()
X, y =  data.data, data.target

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Display the dataset informations 
# print("Features: ", data.feature_names)
# print("Classes: ", data.target_names)

# Converting the dataset to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train the XGB model
params_xgb = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1
}

xgb_model = xgb.train(params_xgb, dtrain, num_boost_round=100)

# Predict
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

# Evaluate 
accuracy = accuracy_score(y_test, y_pred)
print(f"XGB model accuracy: {accuracy}")
print("\nXGB Classification Report:\n", classification_report(y_test, y_pred))

# Testing with hyperparameters
params_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize XGB Classifier
xgb_clf = XGBClassifier( eval_metric='logloss', random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=params_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Displaying the results from grid search
print(f"Best Parameters are: {grid_search.best_params_}")
print(f"Best Cross-validation are: {grid_search.best_score_}")

# Let's compare with Gradient Bosst Classifier
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Model prediction
y_pred = gb_model.predict(X_test)

# Evaluate performance
accuracy_gb = accuracy_score(y_test, y_pred)
print(f"Gradient Boosting Accuracy: {accuracy_gb}")
print(f"\n Classification Report:\n {classification_report(y_test, y_pred)}")