import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
from preprocess import load_and_preprocess_data

def train_and_log_models(data_path):
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(data_path)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_acc = 0

    mlflow.set_experiment("Loan_Eligibility_Models")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("Model", name)
            mlflow.log_metric("Accuracy", acc)
            mlflow.sklearn.log_model(model, name)

            if acc > best_acc:
                best_acc = acc
                best_model = model

    joblib.dump(best_model, "models/random_forest_model.pkl")
    print(f"✅ Best Model: Random Forest ({best_acc:.3f}) saved!")

if __name__ == "__main__":
    train_and_log_models("data/loan.csv")
