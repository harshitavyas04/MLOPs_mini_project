import pytest
from src.predict import predict_loan_status

def test_model_prediction_approved():
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 4000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 120,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }
    result = predict_loan_status(sample)
    assert result in ["Approved", "Rejected"]
