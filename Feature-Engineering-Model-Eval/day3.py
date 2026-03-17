import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression




url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Display dataset information
# print("Dataset info: ", df.info())

# # Preview the first few rolls
# print("Dataset preview: ", df.describe())

# apply one-hot encoding 
df_one_hot = pd.get_dummies(df,columns=['Sex', 'Embarked'], drop_first=True)

# Display the encoded columns 
print("\n One hot encoded dataset\n")
print(df_one_hot.head())


# Apply Label-Encoder
L_encoder = LabelEncoder()
df['Pclass_encoded'] = L_encoder.fit_transform(df['Pclass'])

# Display the label encoded columns 
print("\n Label encoded dataset\n")
print(df[['Pclass', 'Pclass_encoded']].head())


# Apply Frequency-Encoder
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())

# Display the frequency encoded columns 
print("\n Frequency encoded dataset\n")
print(df[['Ticket', 'Ticket_frequency']].head())

# Droping few columns from the dataset
X = df_one_hot.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
y = df_one_hot['Survived']

# Then check for NaN
print("NaN values:", X.isnull().sum().sum())
if X.isnull().sum().sum() > 0:
    X = X.dropna()
    y = y[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)  

# Creating our model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test) 

# Evaluating the model
print("\n Accuracy with one-hot encoding: \n", accuracy_score(y_test, y_pred)) 
