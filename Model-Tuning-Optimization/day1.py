from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import optuna

# loading the dataset
data = load_breast_cancer()
X, y =  data.data, data.target

# splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Display the dataset informations 
# print("Features: ", data.feature_names)
# print("Classes: ", data.target_names)

# Scale feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training shape: ", X_train.shape)
print("Test shape: ", X_test.shape)


# Implementing XGBoost model
bl_model = XGBClassifier(eval_metric='logloss', random_state=42)
bl_model.fit(X_train, y_train)

# Evaluate model
bl_pred = bl_model.predict(X_test)
bl_accuracy = accuracy_score(y_test, bl_pred)
print(f"Baseline XGBoost Accuracy: {bl_accuracy:.4f}")

# to work optuna is required to create a objective function
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.1, 0.7),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.6),
        'gamma': trial.suggest_int('gamma', 0, 5),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 10),
    }
    
    # Train XGBoost model with objective from above
    model = XGBClassifier(eval_metric='logloss', random_state=42, **params)
    model.fit(X_train, y_train) 

    # Evaluate model
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    return accuracy
    # print(f"Baseline XGBoost Accuracy: {accuracy:.4f}")  
    
# applying Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# best hyperparameters
print("Best Hyperparamenters: ", study.best_params)
print("Best values: ", study.best_value)    
    



