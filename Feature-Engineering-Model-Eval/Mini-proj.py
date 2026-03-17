import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV    


# Loading the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Selected relevants features
df = df[['Pclass','Sex', 'Age', 'Fare', 'Embarked', 'Survived']]

# Handle missing values
df.fillna({'Age':df['Age'].median()}, inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()[0]}, inplace=True)

# Define featerues and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# apply feature scaling and encoding 
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare']),
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

X_preprocessor = preprocessor.fit_transform(X)

# Train and evaluate Logistic regression
log_model = LogisticRegression()
log_score = cross_val_score(log_model, X_preprocessor, y, cv=5, scoring='accuracy')
print(f"Logistic regression Accuracy: {log_score.mean():.2f}") 

# Train and evaluate random forest
rf_model = RandomForestClassifier(random_state=42)
rf_score = cross_val_score(rf_model, X_preprocessor, y, cv=5, scoring='accuracy')
print(f"Random Forest Accuracy: {rf_score.mean():.2f}")  

# Define hyperparameter grid 
param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Performing Grid Search 
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,  
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# train the model
grid_search.fit(X_preprocessor, y)

# Display Hyperparameters score
print(f"Best Hyperparameters: {grid_search.best_params_}")  
print(f"Best Accuracy: {grid_search.best_score_:.2f}")   

