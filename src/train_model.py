# # import pandas as pd
# # import mlflow
# # import mlflow.sklearn
# # from sklearn.metrics import accuracy_score
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# # from xgboost import XGBClassifier
# # import joblib
# # import os

# # def train_and_log_models(X_train_path, X_test_path, y_train_path, y_test_path):
# #     # Load data
# #     X_train = pd.read_csv(X_train_path)
# #     X_test = pd.read_csv(X_test_path)
# #     y_train = pd.read_csv(y_train_path).squeeze()
# #     y_test = pd.read_csv(y_test_path).squeeze()

# #     models = {
# #         "Logistic Regression": LogisticRegression(max_iter=500),
# #         "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
# #         "Gradient Boosting": GradientBoostingClassifier(),
# #         "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# #     }

# #     best_model = None
# #     best_acc = 0

# #     mlflow.set_experiment("Loan_Eligibility_Models")

# #     for name, model in models.items():
# #         with mlflow.start_run(run_name=name):
# #             model.fit(X_train, y_train)
# #             preds = model.predict(X_test)
# #             acc = accuracy_score(y_test, preds)

# #             mlflow.log_param("Model", name)
# #             mlflow.log_metric("Accuracy", acc)
# #             mlflow.sklearn.log_model(model, name)

# #             print(f"{name} Accuracy: {acc:.3f}")

# #             if acc > best_acc:
# #                 best_acc = acc
# #                 best_model = model

# #     # Save best model
# #     os.makedirs("models", exist_ok=True)
# #     joblib.dump(best_model, "models/best_model.pkl")
# #     print(f"Best Model saved with Accuracy: {best_acc:.3f}")


# # if __name__ == "__main__":
# #     train_and_log_models(
# #         "data/processed_loan_train.csv",
# #         "data/processed_loan_test.csv",
# #         "data/y_train.csv",
# #         "data/y_test.csv"
# #     )


# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
# import joblib
# import os
# import warnings

# warnings.filterwarnings("ignore")


# def train_and_log_models(X_train_path, X_test_path, y_train_path, y_test_path):
#     # --- Load data ---
#     X_train = pd.read_csv(X_train_path)
#     X_test = pd.read_csv(X_test_path)
#     y_train = pd.read_csv(y_train_path).squeeze()
#     y_test = pd.read_csv(y_test_path).squeeze()

#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
#         "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
#         "Gradient Boosting": GradientBoostingClassifier(),
#         "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
#     }

#     best_model = None
#     best_acc = 0.0
#     best_name = None

#     mlflow.set_tracking_uri("http://127.0.0.1:5000")
#     mlflow.set_experiment("Loan_Eligibility_Models")

#     # --- Train and log models ---
#     for name, model in models.items():
#         print(f"\n🔹 Training {name} ...")
#         with mlflow.start_run(run_name=name):
#             model.fit(X_train, y_train)
#             preds = model.predict(X_test)
#             acc = accuracy_score(y_test, preds)

#             mlflow.log_param("Model", name)
#             mlflow.log_metric("Accuracy", acc)
#             mlflow.sklearn.log_model(model, name)

#             print(f"{name} Accuracy: {acc:.3f}")

#             if acc > best_acc:
#                 best_acc = acc
#                 best_model = model
#                 best_name = name

#     # --- Validate model type before saving ---
#     if best_model is None or not hasattr(best_model, "predict"):
#         raise ValueError("❌ No valid trained model found! The model object seems invalid.")

#     # --- Save the best model ---
#     os.makedirs("models", exist_ok=True)
#     model_path = "models/best_model.pkl"

#     print(f"\n✅ Saving best model: {best_name} (Accuracy: {best_acc:.3f})")
#     print(f"✅ Model type before saving: {type(best_model)}")

#     joblib.dump(best_model, model_path)

#     # --- Double check saved model type ---
#     loaded_model = joblib.load(model_path)
#     print(f"✅ Model successfully saved. Reload check: {type(loaded_model)}")


# if __name__ == "__main__":
#     train_and_log_models(
#         "data/processed_loan_train.csv",
#         "data/processed_loan_test.csv",
#         "data/y_train.csv",
#         "data/y_test.csv"
#     )


import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

def train_and_log_models(X_train_path, X_test_path, y_train_path, y_test_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_model, best_acc, best_name = None, 0.0, None
    mlflow.set_experiment("Loan_Eligibility_Models")

    for name, model in models.items():
        print(f"\n🔹 Training {name} ...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            mlflow.log_param("Model", name)
            mlflow.log_metric("Accuracy", acc)
            mlflow.sklearn.log_model(model, name)

            print(f"{name} Accuracy: {acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name

    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n✅ Saved best model ({best_name}) with accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    train_and_log_models(
        "data/processed_loan_train.csv",
        "data/processed_loan_test.csv",
        "data/y_train.csv",
        "data/y_test.csv"
    )
