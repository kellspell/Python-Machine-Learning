import pandas as pd  
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Select only what we are going to need from the dataset
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = ['Survived']

# handle the missing values
df.fillna({'Age': df['Age'].median()}, inplace=True)
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# Spliting the data after encoding
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's print the shape of our data after encoding
print(f"Traing Data shape: {X_train.shape}")    
print(f"Test Data shape: {X_test.shape}")   

# Now let's create the LightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

# Predict and evaluate
ldb_pred = lgb_model.predict(X_test)
print(f"LightGBM accuracy: {accuracy_score(y_test, ldb_pred):.4f}") 

# Let's create a CatBoost model
cat_features = ['Pclass', 'Sex', 'Embarked']
cat_model = CatBoostClassifier(cat_features=cat_features, verbose=0)
cat_model.fit(X_train, y_train)

# Predict and evaluate
cat_pred = cat_model.predict(X_test)
print(f"Cat accuracy: {accuracy_score(y_test, cat_pred):.4f}") 


# Comparing now with XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss') 
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
print(f"XGB accuracy: {accuracy_score(y_test, xgb_pred):.4f}") 


# Train catboost without encoding categorical features we've setup above
cat_model_native = CatBoostClassifier(cat_features=['Sex', 'Embarked'], verbose=0)
cat_model_native.fit(X_train, y_train)

# Predict and evaluate
cat_pred_native = cat_model_native.predict(X_test)
print(f"Native accuracy: {accuracy_score(y_test, cat_pred_native):.4f}") 







