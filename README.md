### Mortgage Trading Analysis and Prediction
## Project Overview
This project focuses on analyzing and predicting various aspects of mortgage loans, including EMI calculation, total payment, interest amounts, and the risk of pre-payment or delinquency. The project uses machine learning models for both classification and regression tasks to assess the risk of delinquency and predict the likelihood of pre-payment.

## Dataset
The dataset used for this project contains information about various mortgage attributes such as loan amounts, interest rates, debt-to-income ratios (DTI), loan terms, and others.

## Key Features:
OrigInterestRate: Original interest rate of the loan.
OrigLoanTerm: Original term of the loan in months.
OrigUPB: Original unpaid principal balance.
DTI: Debt-to-income ratio.
MonthsInRepayment: Number of months the borrower has been repaying.
EMI: Monthly installment calculated based on the loan terms and interest rate.
current_principal: Remaining principal balance on the loan after a certain number of months.
Delinquent: Indicates whether a borrower is delinquent based on the payment-to-income ratio.
PrePayment_Risk: Indicator of whether the borrower is at high or low risk of pre-payment based on the remaining principal.
Project Workflow
## Data Loading and Preprocessing:

Load the mortgage dataset and calculate important features such as EMI, total payment, interest amount, monthly income, and current principal.
Calculate the monthly payment-to-income ratio and create indicators for delinquency and pre-payment risk.
Replace infinite values with 0 for robustness.
## Feature Engineering:

Calculate custom features like EMI, monthly_income, current_principal, Payment_to_Income_Ratio, and PrePayment_Risk.
Generate a binary delinquency indicator (delinquent_binary) based on the payment-to-income ratio.
## Classification and Regression:

Use a Logistic Regression model to predict the likelihood of a borrower becoming delinquent.
Train a Linear Regression model to predict the likelihood of pre-payment for borrowers classified as delinquent.
## Pipeline:

Implement a custom pipeline combining classification and regression models to streamline the prediction process. The pipeline first classifies borrowers into delinquent or non-delinquent, then applies the regression model to predict pre-payment risk for delinquent borrowers.
Key Code Components

## EMI Calculation:



    r = df['OrigInterestRate'] / (12 * 100)
    n = df['OrigLoanTerm']
    P = df['OrigUPB']
    df['EMI'] = P * r * (1 + r) ** n / ((1 + r) ** n - 1)



## Pre-payment Calculation:


    def prepay(DTI, monthly_income):
        if DTI < 50:
            p = monthly_income / 2
        else:
            p = monthly_income * 3 / 4
        return p
    df['pre_payment'] = np.vectorize(prepay)(df['DTI'], df['monthly_income'] * 24) - (df['EMI'] * 24)


## Classification Model:


    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_class_train)


## Regression Model:


    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train_reg, y_reg_train_filtered)
    
    Custom Pipeline:
    
    class CustomPipeline(BaseEstimator, TransformerMixin):
        def __init__(self, clf, reg):
            self.clf = clf
            self.reg = reg
    
        def fit(self, X, y_class, y_reg):
            self.clf.fit(X, y_class)
            X_filtered = X[y_class == 1]
            y_reg_filtered = y_reg[y_class == 1]
            self.reg.fit(X_filtered, y_reg_filtered)
            return self
    
        def predict(self, X):
            y_class_pred = self.clf.predict(X)
            X_filtered = X[y_class_pred == 1]
            y_reg_pred = self.reg.predict(X_filtered)
            return y_class_pred, y_reg_pred

        
Evaluation Metrics
Classification:

Accuracy
Precision
Recall
F1-Score
Classification Report
Regression:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
Mean Absolute Percentage Error (MAPE)
Median Absolute Error (MedAE)

pip install pandas scikit-learn numpy
Clone the repository and navigate to the project folder.

Load the dataset using pandas.read_csv() and preprocess the data as described in the code.

Train the models and evaluate their performance using the provided metrics.

## Conclusion
This project successfully predicts mortgage-related risks using a combination of classification and regression models. The custom pipeline approach helps efficiently handle both delinquency and pre-payment risk predictions in a streamlined manner.
