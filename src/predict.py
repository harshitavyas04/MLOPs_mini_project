import joblib
import pandas as pd

def predict_loan_status(input_data):
    model = joblib.load("models/random_forest_model.pkl")
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return "Approved" if prediction == 1 else "Rejected"
