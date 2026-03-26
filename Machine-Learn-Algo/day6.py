import pandas as pd  
# import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
# from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)

# Print data info
print("Dataset info: ", df.info())
print('\n Class Distribution \n')
print(df['Class'].value_counts)

# loading the dataset
X = df.drop(columns=['Class'])
y = df['Class']

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print('\n Classification Report \n')
print(f"{classification_report(y_test, y_pred)}") 

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc}")

# Apply Smote
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Dispay new class classification 
print('\n Class Distribution after Smote \n')
print(pd.Series(y_resampled).value_counts())

# Train Random Forest on the smote resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled, y_resampled)

# Predict and evaluate random forest smote
y_pred_smote = rf_model_smote.predict(X_test)
print('\n Classification Report (SMOTE) \n')
print(f"{classification_report(y_test, y_pred_smote)}") 

roc_auc_smote = roc_auc_score(y_test, rf_model_smote.predict_proba(X_test)[:,1])
print(f"ROC-AUC (SMOTE): {roc_auc_smote}")

