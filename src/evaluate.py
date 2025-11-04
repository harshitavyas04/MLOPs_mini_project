import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import os

def evaluate_model(model_path, X_test_path, y_test_path):
    # Load model and data
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    # Predict
    preds = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='weighted')
    recall = recall_score(y_test, preds, average='weighted')
    f1 = f1_score(y_test, preds, average='weighted')
    cm = confusion_matrix(y_test, preds).tolist()

    # Log to MLflow
    with mlflow.start_run(run_name="Model_Evaluation"):
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1_Score", f1)
        mlflow.log_dict({"Confusion_Matrix": cm}, "confusion_matrix.json")

    # Save metrics locally for DVC tracking
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_report.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Evaluation complete. Metrics saved to 'reports/evaluation_report.json'")
    print(results)


if __name__ == "__main__":
    evaluate_model(
        "models/best_model.pkl",
        "data/processed_loan_test.csv",
        "data/y_test.csv"
    )
