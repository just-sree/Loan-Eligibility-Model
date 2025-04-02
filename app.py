# app.py

import streamlit as st
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_data
from src.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Predictor")

# Load models
@st.cache_resource
def load_models():
    try:
        logistic_model = joblib.load("models/logistic_model.pkl")
        random_forest_model = joblib.load("models/random_forest_model.pkl")
        return logistic_model, random_forest_model
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

log_model, rf_model = load_models()

# Sidebar for model choice
model_choice = st.sidebar.selectbox("Select Prediction Model", ["Logistic Regression", "Random Forest"])

# Input form
st.subheader("Enter Applicant Details")
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", value=0)
    loan_amount = st.number_input("Loan Amount", value=150)
    loan_term = st.number_input("Loan Term (in days)", value=360)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    try:
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_emp,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        processed = preprocess_data(input_df, scale_numeric=False)
        if 'Loan_Approved' in processed.columns:
            processed.drop('Loan_Approved', axis=1, inplace=True)

        # Predict
        model = log_model if model_choice == "Logistic Regression" else rf_model
        prediction = model.predict(processed)[0]

        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error("An error occurred during prediction.")
        logger.error(f"Prediction error: {e}")
