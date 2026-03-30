import pandas as pd 
import numpy as np
from  sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('/home/Kellspell/Py-Dev/Machine-Learn-Algo/telco_churn.csv')

# Display dataset info
print("\n Dataset info \n")
print(df.info())
print("\n Class distribution \n")
print(df['Churn'].value_counts())
print("\n Sample data: \n", df.head())

# Handling the missing values 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)


# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])
        
        
# Encode the Target variable 
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Scaler numerical feature
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Features and target variable
X = df.drop(columns=['Churn'])
y = df['Churn']
        

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest model 
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model Accuracy \n {accuracy:.4f}")

# Printing classification report
print("Classification Report:\n", classification_report(y_test, y_pred ))

# Define parameters for grid search
param_dist = {
    'n_estimators': np.arange(50, 100, 200),  # Note: This will create [50] only because 50 to 100 with step 200
    'min_samples_split': [2, 5, 10, 20],  # Fixed: 'min_samples_split' (not 'min_sample_split')
    'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Fixed: 'min_samples_leaf' (not 'min.sample_leaf')
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Perform the random search
random_search.fit(X_train, y_train)

# Displaying the results from random search
best_params = random_search.best_params_
print(f"Best Parameters are: {best_params}")

# Train best model
best_model = random_search.best_estimator_

# Predict and Evaluate
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Tuned accuracy: {accuracy_tuned:.4f}")
print("\n(Tuned) Classification Report:\n", classification_report(y_test, y_pred_tuned))