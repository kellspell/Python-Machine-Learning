import pandas as pd 
from imblearn.over_sampling import SMOTE
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# Load the dataset
df = pd.read_csv('/home/Kellspell/Py-Dev/Machine-Learn-Algo/telco_churn.csv')

# Display dataset info
print("\n Dataset info \n")
print(df.info())

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

# apply Smote to balance our data
smote = SMOTE(random_state=42)
X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)

# Display class distribution after the data has been reshaped by Smote
print("\n Class distribution after smote \n")
print(pd.Series(y_train_resample).value_counts())

# Train random forest model 
rt_model = RandomForestClassifier(random_state=42)
rt_model.fit(X_train_resample, y_train_resample)
y_pred_rt = rt_model.predict(X_test)
roc_auc_rt = roc_auc_score(y_test, rt_model.predict_proba(X_test)[:, 1])

# Printing classification report
print("Random Forest Report:\n", classification_report(y_test, y_pred_rt ))