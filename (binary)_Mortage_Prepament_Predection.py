import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('C:/Users/manic/Downloads/LoanExport.csv')

# Assuming 'OrigInterestRate' is the annual interest rate in your dataset
r = df['OrigInterestRate'] / (12 * 100)
n = df['OrigLoanTerm']
P = df['OrigUPB']

# Calculate EMI
df['EMI'] = P * r * (1 + r) ** n / ((1 + r) ** n - 1)

# Total Payment
df['total_payment'] = df['EMI'] * df['OrigLoanTerm']  #EMI and monthly_payment are same

# Interest Amount
df['interest_amount'] = df['total_payment'] - df['OrigUPB']

# Monthly Income
df['monthly_income'] = df['EMI'] / (df['DTI'] / 100)

# replace infinite value into 0
df['monthly_income'] = df['monthly_income'].replace([np.inf, -np.inf], 0)

 # Monthly Interest Rate
monthly_rate = df['OrigInterestRate'] / (12 * 100)

# Number of payments made
months_paid = df['MonthsInRepayment']

# Current Principal
df['current_principal'] = P * ((1 + monthly_rate) ** n - (1 + monthly_rate) ** months_paid) / ((1 + monthly_rate) ** n - 1)

def prepay(DTI,monthly_income):
  if (DTI< 40):
    p = monthly_income/2
  else:
    p = monthly_income* 3/4
  return p

df['pre_payment'] = np.vectorize(prepay)(df['DTI'],df['monthly_income']*24)
df['pre_payment'] = df['pre_payment']-(df['EMI']*24)

# Replace values <= 0 with zero
df['pre_payment'] = df['pre_payment'].apply(lambda x: 0 if x <= 0 else x)

# Define the percentage threshold (e.g., 10% of the original loan balance)
percentage_threshold = 11 / 100  # 11% as a decimal

# Calculate the threshold value for each loan based on OrigUPB
df['pre_payment_threshold'] = df['OrigUPB'] * percentage_threshold

# Create the binary target variable based on whether pre_payment exceeds the threshold
df['pre_payment_binary'] = (df['pre_payment'] > df['pre_payment_threshold']).astype(int)

# Verify the result
binary_distribution = df['pre_payment_binary'].value_counts()
print("Binary Distribution of Pre-payment Amount:")
print(binary_distribution)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming df is already loaded with your data
# If not, you can load it using pd.read_csv or similar methods

# Prepare the data
data = df
x = data[['monthly_income', 'OrigLoanTerm', 'OrigUPB']].values
y = data["pre_payment_binary"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Streamlit user inputs
st.title('Pre-Payment Prediction Model')
st.write("Enter the following details:")

monthly_income = st.number_input('Monthly Income')
OrigLoanTerm = st.number_input('Original Loan Term')
OrigUPB = st.number_input('Original Loan Balance')

# Predict button
if st.button('Predict'):
    user_input = [[monthly_income, OrigLoanTerm, OrigUPB]]
    prediction = rf_classifier.predict(user_input)
    
    if prediction == 1:
        st.write('The model predicts: Pre-payment is likely.')
    else:
        st.write('The model predicts: Pre-payment is unlikely.')
    
    # Display accuracy, classification report, and confusion matrix
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    st.write(f"Model Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.text(report)
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

# Run the app using 'streamlit run app_name.py'
