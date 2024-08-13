import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('C:/Users/manic/Downloads/LoanExport.csv')

# Function to replace CreditScore values
def map_credit_score(score):
    if score <= 620:
        return 1
    elif 620 < score <= 700:
        return 2
    elif 700 < score <= 750:
        return 3
    elif 750 < score <= 900:
        return 3
    else:
        return None  # or handle the case if score is outside 0-900 range

# Apply the function to the CreditScore column and store in a new column Credit_range
df['Credit_range'] = df['CreditScore'].apply(map_credit_score)

# Function to replace CreditScore values
def map_credit_score(score):
    if score <= 760:
        return 0
    else:
        return 1  # or handle the case if score is outside 0-900 range

# Apply the function to the CreditScore column and store in a new column Credit_range
df['Credit_binary'] = df['CreditScore'].apply(map_credit_score)


# Function to replace CreditScore values
def map_credit_score(score):
    if score <= 25:
        return 0
    elif 25 < score <= 50:
        return 1
    elif 50 < score <= 100:
        return 2
    else:
        return 3  # or handle the case if score is outside 0-900 range

# Apply the function to the CreditScore column and store in a new column Credit_range
df['LTV_range'] = df['LTV'].apply(map_credit_score)

# Function to replace CreditScore values
def map_credit_score(score):
    if score <= 48:
        return 0
    elif 48 < score <= 96:
        return 1
    elif 96 < score <= 144:
        return 2
    elif 144 < score <= 192:
        return 3
    elif 192 < score <= 240:
        return 4
    else:
        return 5  # or handle the case if score is outside 0-900 range

# Apply the function to the CreditScore column and store in a new column Credit_range
df['repay_range'] = df['MonthsInRepayment'].apply(map_credit_score)

# Creating a new column 'first_time_buyer' with replaced values
df['first_time_buyer'] = df['FirstTimeHomebuyer'].replace({'N': 0, 'Y': 1, 'X': 1}).astype(int)

# Create the default column
df['default'] = df['EverDelinquent'].apply(lambda x: 0 if x == 0 else 1)

df['RemainingTermRatio'] = df['MonthsInRepayment'] / df['OrigLoanTerm']

df['LTV_InterestRate'] = df['LTV'] * df['OrigInterestRate']

# Assume points_and_fees is 1% of the OrigUPB for simplicity
df['points_and_fees'] = df['OrigUPB'] * 0.0001

# For simplicity, let's assume target profit is calculated as 5% of OrigUPB minus points_and_fees
df['target_profit'] = (df['OrigUPB'] * 0.0000005) - df['points_and_fees']

# Create a new column with rounded values
df['OrigInterestRate_Rounded'] = df['OrigInterestRate'].round().astype(int)

import pandas as pd

# Assuming df is your existing DataFrame

# List of columns to be one-hot encoded
columns_to_encode = ['LoanPurpose', 'PPM', 'Channel','repay_range','LTV_range','Credit_range','Occupancy']

# Define all possible categories for each column
loan_purpose_categories = ['P', 'N', 'C']
ppm_categories = ['N', 'X', 'Y']
channel_categories = ['T', 'R', 'C', 'B']
repay_range_categories = [1, 2, 0, 3, 4]
LTV_range_categories = [2, 1, 0, 3]
Credit_range_categories = [1, 2, 3]
Occupancy_categories = ['O', 'I', 'S']

# Create a dictionary with column names and their possible categories
categories = {
    'LoanPurpose': loan_purpose_categories,
    'PPM': ppm_categories,
    'Channel': channel_categories,
    'repay_range': repay_range_categories,
    'LTV_range': LTV_range_categories,
    'Credit_range': Credit_range_categories,
    'Occupancy': Occupancy_categories
}

# Perform one-hot encoding for each column, ensuring all categories are represented
encoded_dfs = []
for col, cats in categories.items():
    encoded_df = pd.get_dummies(df[col], prefix=col).reindex(columns=[f"{col}_{cat}" for cat in cats], fill_value=0)
    encoded_dfs.append(encoded_df)

# Concatenate the encoded columns with the original DataFrame
df = pd.concat([df] + encoded_dfs, axis=1)

# Display the first few rows of the updated DataFrame
#print(df.head())

# Create a new column with rounded values
df['OrigInterestRate_Rounded'] = df['OrigInterestRate'].round().astype(int)

# Function to replace CreditScore values
def map_credit_score(score):
    if score == 4 or score == 12 or score == 11 or score == 10 or score == 5 or score == 9 :
        return 1
    elif score == 8 or score == 6:
        return 1
    else:
        return 0  # or handle the case if score is outside 0-900 range

# Apply the function to the CreditScore column and store in a new column Credit_range
df['OrigInterestRate_binary'] = df['OrigInterestRate_Rounded'].apply(map_credit_score)


# Assuming df is your original DataFrame
# Perform one-hot encoding on the specified columns
df = pd.get_dummies(df, columns=['PropertyType','SellerName','NumBorrowers'], prefix=['PropertyType','SellerName','NumBorrowers'])

# Display the first few rows of the DataFrame to verify the changes
#print(df.head())

import streamlit as st
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset and split features and target
# Assuming df is your DataFrame
data = df

x = data[['Credit_range_3', 'repay_range_0', 'Occupancy_O', 'MonthsInRepayment', 'Credit_binary', 'NumBorrowers_2']].values
y = data["EverDelinquent"].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(xgb_model, 'xgboost_model.pkl')

# Load the model in the Streamlit app
xgb_model = joblib.load('xgboost_model.pkl')

# Streamlit app for making predictions
st.title('Mortgage Default Prediction')

# User input for new data
st.header('Enter the Details:')

Credit_range_3 = st.number_input('Credit Range 3', min_value=0, max_value=1, value=1)
repay_range_0 = st.number_input('Repay Range 0', min_value=0, max_value=1, value=0)
Occupancy_O = st.number_input('Occupancy O', min_value=0, max_value=1, value=1)
MonthsInRepayment = st.number_input('Months In Repayment', min_value=0, max_value=360, value=12)
Credit_binary = st.number_input('Credit Binary', min_value=0, max_value=1, value=1)
NumBorrowers_2 = st.number_input('Number of Borrowers', min_value=1, max_value=2, value=2)

# Convert user input to dataframe
new_data = pd.DataFrame({
    'Credit_range_3': [Credit_range_3],
    'repay_range_0': [repay_range_0],
    'Occupancy_O': [Occupancy_O],
    'MonthsInRepayment': [MonthsInRepayment],
    'Credit_binary': [Credit_binary],
    'NumBorrowers_2': [NumBorrowers_2]
})

# Display input dataframe
st.write('Input Data:')
st.dataframe(new_data)
'''
# Predict button
if st.button('Predict'):
    # Convert new data to the format required by the model
    X_new = new_data.values

    # Make predictions
    predictions = xgb_model.predict(X_new)
    
    # Display prediction result
    st.write('Prediction:', 'Default' if predictions[0] == 1 else 'No Default')
'''

# Predict button
if st.button('Predict'):
    # Convert new data to the format required by the model
    X_new = new_data.values

    # Make predictions
    predictions = xgb_model.predict(X_new)
    
    # Display prediction result
    st.write('Prediction:', int(predictions[0]))

