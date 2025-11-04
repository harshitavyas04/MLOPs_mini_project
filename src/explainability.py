import lime
import shap
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

def explain_prediction(sample, X_train):
    model = joblib.load("models/best_model.pkl")

    # LIME
    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        mode="classification"
    )
    explanation = explainer.explain_instance(sample, model.predict_proba)
    explanation.show_in_notebook(show_table=True)

    # SHAP
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)
