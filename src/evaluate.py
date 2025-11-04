import json
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model_path, X_test_path, y_test_path):
    # --- Connect to MLflow tracking server ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Loan_Eligibility_Evaluation")

    # --- Load model and data ---
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()  # y_test stored separately

    # --- Make predictions ---
    y_pred = model.predict(X_test)

    # --- Compute metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # --- Save evaluation metrics locally ---
    os.makedirs("reports", exist_ok=True)

    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    with open("reports/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Evaluation report saved at reports/evaluation_report.json")

    # --- Log metrics to MLflow ---
    with mlflow.start_run(run_name="Evaluation_Run"):
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Recall", rec)
        mlflow.log_metric("F1_Score", f1)
        mlflow.log_dict({"Confusion_Matrix": cm}, "confusion_matrix.json")

    print("\n✅ Model Evaluation Results:")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("\nEvaluation complete and logged to MLflow UI at http://127.0.0.1:5000")

if __name__ == "__main__":
    evaluate_model(
        model_path="models/best_model.pkl",
        X_test_path="data/processed_loan_test.csv",
        y_test_path="data/y_test.csv"
    )
