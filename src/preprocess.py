import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna(df.mode().iloc[0])

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Split features & target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoders


if __name__ == "__main__":
    # Input data file
    file_path = "data/loan-train.csv"

    # Preprocess and split
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(file_path)

    # Ensure output directory
    os.makedirs("data", exist_ok=True)

    # Save processed data for DVC tracking
    X_train.to_csv("data/processed_loan_train.csv", index=False)
    X_test.to_csv("data/processed_loan_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("Preprocessing complete. Files saved to 'data/'")
