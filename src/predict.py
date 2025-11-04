import joblib
import pandas as pd

def predict_loan_status(input_data: dict):
    model = joblib.load("models/best_model.pkl")

    df = pd.DataFrame([input_data])

    # --- Encode categorical variables manually ---
    categorical_mappings = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
    }

    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # --- Fill missing expected columns ---
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else df.columns
    for col in expected_features:
        if col not in df.columns:
            if col == "Loan_ID":
                df[col] = 0
            elif col == "Dependents":
                df[col] = 0
            else:
                df[col] = 0

    df = df[expected_features]

    prediction = model.predict(df)[0]
    return "Approved" if prediction == 1 else "Rejected"
