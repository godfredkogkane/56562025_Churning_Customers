import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier

# Loading the trained model and scaler
model = load_model('best_model.h5')
scaler = joblib.load('scaler.pkl')
le = LabelEncoder()

# Create a streamlit app to host the model and allow users to enter new data
st.title('Customer Churn Prediction')
st.write('This app predicts if a customer is likely to churn or not based on their profile and service details.')

# Create a sidebar to collect the input features
st.sidebar.header('Customer Input')
SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ('Yes', 'No'))
Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
Dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
MultipleLines = st.sidebar.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
OnlineSecurity = st.sidebar.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
OnlineBackup = st.sidebar.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
DeviceProtection = st.sidebar.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
TechSupport = st.sidebar.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
StreamingTV = st.sidebar.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
MonthlyCharges = st.sidebar.slider('Monthly Charges ($)', 0.0, 120.0, 50.0)
TotalCharges = st.sidebar.slider('Total Charges ($)', 0.0, 9000.0, 1000.0)

# Create a function to preprocess user input and make predictions
def preprocess_input(SeniorCitizen, Partner, Dependents, tenure, MultipleLines,
                     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                     TechSupport, StreamingTV, StreamingMovies, Contract,
                     PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
    data = {
        'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
        'Partner': [1 if Partner == 'Yes' else 0],
        'Dependents': [1 if Dependents == 'Yes' else 0],
        'tenure': [tenure],
        'MultipleLines': [le.fit_transform([MultipleLines])[0]],
        'InternetService': [le.fit_transform([InternetService])[0]],
        'OnlineSecurity': [le.fit_transform([OnlineSecurity])[0]],
        'OnlineBackup': [le.fit_transform([OnlineBackup])[0]],
        'DeviceProtection': [le.fit_transform([DeviceProtection])[0]],
        'TechSupport': [le.fit_transform([TechSupport])[0]],
        'StreamingTV': [le.fit_transform([StreamingTV])[0]],
        'StreamingMovies': [le.fit_transform([StreamingMovies])[0]],
        'Contract': [le.fit_transform([Contract])[0]],
        'PaperlessBilling': [1 if PaperlessBilling == 'Yes' else 0],
        'PaymentMethod': [le.fit_transform([PaymentMethod])[0]],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
    }
    return pd.DataFrame(data)

# Predict using the model
if st.sidebar.button('Predict Churn'):
    input_data = preprocess_input(SeniorCitizen, Partner, Dependents, tenure, MultipleLines,
                                  InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                                  TechSupport, StreamingTV, StreamingMovies, Contract,
                                  PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(scaled_data)

    # Extract the confidence score 
    confidence_score = prediction[0]

    # Display the prediction and confidence score
    formatted_confidence_score = float(confidence_score)  # Convert to a Python float
    st.success(f'The customer is likely to churn with a confidence score of {formatted_confidence_score:.2f}' 
    if confidence_score >= 0.5 else f'The customer is not likely to churn with a confidence score of {formatted_confidence_score:.2f}')

