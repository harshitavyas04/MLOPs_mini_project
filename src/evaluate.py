import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model_path, test_data_path):
    # --- Set MLflow tracking server ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Loan_Eligibility_Evaluation")

    # --- Load model and data ---
    model = joblib.load(model_path)
    X_test = pd.read_csv(test_data_path)
    y_test = X_test.pop("Loan_Status")

    # --- Make predictions ---
    y_pred = model.predict(X_test)

    # --- Compute metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # --- Log results in MLflow ---
    with mlflow.start_run(run_name="Evaluation_Run"):
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Recall", rec)
        mlflow.log_metric("F1_Score", f1)
        mlflow.log_dict({"Confusion_Matrix": cm}, "confusion_matrix.json")

        print("\nModel Evaluation Results:")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1 Score:  {f1:.3f}")

    print("\nEvaluation complete and logged to MLflow UI.")

if __name__ == "__main__":
    evaluate_model(
        model_path="models/best_model.pkl",
        test_data_path="data/processed_loan_test.csv"
    )
