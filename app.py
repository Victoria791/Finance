import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Budget Planning and Investment Recommendation Tool")

# Load the dataset
@st.cache_data
def load_data():
    # Replace with your dataset path
    df = pd.read_csv("Finance_data.csv")
    return df

df = load_data()

# Preprocessing the dataset
df['Risk Tolerance'] = df['Risk Tolerance'].map({'Low (1)': 1, 'Medium (2)': 2, 'High (3)': 3})
df = pd.get_dummies(df, columns=['Financial Goal'], drop_first=True)

df['Total Expenses'] = df['Fixed Expenses'] + df['Discretionary Expenses']
df['Spending Ratio'] = df['Total Expenses'] / df['Income']
df['Savings Rate'] = df['Savings'] / df['Income']

# Define features and target
X = df[['Income', 'Fixed Expenses', 'Discretionary Expenses']]
y = df['Savings']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Define budget ratios
fixed_expense_ratio = 0.50
discretionary_ratio = 0.30
savings_ratio = 0.20

# Define investment recommendations
investment_mapping = {
    1: "Low-risk investments: Bonds, Savings Accounts",
    2: "Moderate-risk investments: Balanced ETFs, Mutual Funds",
    3: "High-risk investments: Stocks, Real Estate"
}

# Interactive user input
st.header("Input Your Financial Details")

income = st.number_input("Enter your monthly income ($)", min_value=0.0)
fixed_expenses = st.number_input("Enter your fixed expenses ($)", min_value=0.0)
discretionary_expenses = st.number_input("Enter your discretionary expenses ($)", min_value=0.0)
risk_tolerance = st.selectbox("Select your risk tolerance", options=["Low (1)", "Medium (2)", "High (3)"])

if income > 0:
    # Predict savings
    user_features = pd.DataFrame([[income, fixed_expenses, discretionary_expenses]], 
                              columns=['Income', 'Fixed Expenses', 'Discretionary Expenses'])
    predicted_savings = model.predict(user_features)[0]

    # Suggest budget
    suggested_fixed = income * fixed_expense_ratio
    suggested_discretionary = income * discretionary_ratio
    suggested_savings = income * savings_ratio

    # Check overspending
    total_expenses = fixed_expenses + discretionary_expenses
    overspending_alert = total_expenses > income

    # Get investment recommendation
    risk_tolerance_numeric = {'Low (1)': 1, 'Medium (2)': 2, 'High (3)': 3}[risk_tolerance]
    investment_recommendation = investment_mapping[risk_tolerance_numeric]

    # Display results
    st.subheader("Predicted Savings and Budget Suggestions")
    st.write(f"**Predicted Savings:** ${predicted_savings:.2f}")
    st.write(f"**Suggested Fixed Expenses:** ${suggested_fixed:.2f}")
    st.write(f"**Suggested Discretionary Expenses:** ${suggested_discretionary:.2f}")
    st.write(f"**Suggested Savings:** ${suggested_savings:.2f}")
    st.write(f"**Overspending Alert:** {'Yes' if overspending_alert else 'No'}")

    st.subheader("Investment Recommendation")
    st.write(f"**Based on your risk tolerance:** {investment_recommendation}")
