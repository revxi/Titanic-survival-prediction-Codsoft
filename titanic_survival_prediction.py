# Titanic Survival Prediction using Random Forest

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("Titanic-Dataset.csv")  # Ensure this file is in the same directory

# Show basic info
print("Dataset Information:\n")
print(data.info())
print("\nFirst 5 Rows:\n")
print(data.head())

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Cabin'].fillna("Unknown", inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop irrelevant columns
data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Encode categorical variables
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])         # male = 1, female = 0
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Cabin'] = le.fit_transform(data['Cabin'])     # Not optimal but okay for now

# Separate features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot feature importances
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
