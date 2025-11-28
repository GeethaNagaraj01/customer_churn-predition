import shap
import joblib
import pandas as pd

def explain_model(sample_input):
    model = joblib.load("models/churn_model.pkl")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_input)
    shap.summary_plot(shap_values, sample_input, plot_type="bar")
