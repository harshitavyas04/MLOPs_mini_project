# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# from src.predict import predict_loan_status

# st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

# # --- Sidebar: Model Info ---
# st.sidebar.header("Model Information")
# model_path = "models/best_model.pkl"

# try:
#     model = joblib.load(model_path)
#     st.sidebar.success(f"Model loaded: {type(model).__name__}")
# except Exception as e:
#     st.sidebar.error(f"Error loading model: {e}")
#     st.stop()

# # --- App Header ---
# st.title("🏦 Loan Approval Prediction App")
# st.write("""
# This app predicts whether a loan application is **Approved** or **Rejected** based on applicant details.
# """)

# st.markdown("---")

# # --- Input Form ---
# st.header("Enter Applicant Details")

# col1, col2 = st.columns(2)

# with col1:
#     Gender = st.selectbox("Gender", ["Male", "Female"])
#     Married = st.selectbox("Married", ["Yes", "No"])
#     Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
#     Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
#     Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

# with col2:
#     ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, step=100.0)
#     CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, step=100.0)
#     LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, step=10.0)
#     Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0.0, step=10.0)
#     Credit_History = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)")
#     Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# # --- Predict Button ---
# if st.button("🔍 Predict Loan Status"):
#     try:
#         # Prepare input data as a dict
#         input_data = {
#     "Gender": Gender,
#     "Married": Married,
#     "Dependents": Dependents,
#     "Education": Education,
#     "Self_Employed": Self_Employed,
#     "ApplicantIncome": ApplicantIncome,
#     "CoapplicantIncome": CoapplicantIncome,
#     "LoanAmount": LoanAmount,
#     "Loan_Amount_Term": Loan_Amount_Term,
#     "Credit_History": Credit_History,
#     "Property_Area": Property_Area
# }

#         # Predict
#         result = predict_loan_status(input_data)
#         st.success(f"Prediction: **{result}**")

#         # Add visual feedback
#         if result == "Approved":
#             st.balloons()
#         else:
#             st.warning("Loan likely to be rejected. Check applicant details again.")

#     except Exception as e:
#         st.error(f"Error during prediction: {e}")

# # --- Footer ---
# st.markdown("---")
# st.caption("Built using Streamlit & scikit-learn")


import streamlit as st
import joblib
import pandas as pd
from src.predict import predict_loan_status

st.set_page_config(page_title="Loan Approval Prediction App (v2.0)", page_icon="🏦", layout="centered")

# --- Sidebar: Model Info ---
st.sidebar.header("Model Information")
model_path = "models/best_model.pkl"

try:
    model = joblib.load(model_path)
    st.sidebar.success(f"Model loaded: {type(model).__name__}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# --- App Header ---
st.title("🏦 Loan Approval Prediction App")
st.write("""
This app predicts whether a loan application is **Approved** or **Rejected** based on applicant details.
""")
st.markdown("---")

# --- Input Form ---
st.header("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, step=100.0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, step=100.0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, step=10.0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0.0, step=10.0)
    Credit_History = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Poor (0.0)")
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Predict Button ---
if st.button("🔍 Predict Loan Status"):
    try:
        # Prepare input data as a dict (exactly like your local test)
        input_data = {
            "Gender": Gender,
            "Married": Married,
            "Dependents": Dependents,
            "Education": Education,
            "Self_Employed": Self_Employed,
            "ApplicantIncome": ApplicantIncome,
            "CoapplicantIncome": CoapplicantIncome,
            "LoanAmount": LoanAmount,
            "Loan_Amount_Term": Loan_Amount_Term,
            "Credit_History": Credit_History,
            "Property_Area": Property_Area
        }

        # Predict
        result = predict_loan_status(input_data)
        st.success(f"Prediction: **{result}**")

        # Visual feedback
        if result == "Approved":
            st.balloons()
        else:
            st.warning("Loan likely to be rejected. Check applicant details again.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Built using Streamlit & scikit-learn")
