# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('finance_management_dataset_large_expanded.csv')

# Features and target variable
X = data[['current_income', 'monthly_expenses', 'current_savings', 'loan_amount', 'credit_score']]
y = data['future_savings']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'savings_model.pkl')

print("Model training complete and saved as savings_model.pkl.")
