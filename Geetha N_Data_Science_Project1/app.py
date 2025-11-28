import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

st.title("💡 Smart Customer Retention System ")
st.write("Upload any dataset to predict churn, see recommendations, and explain predictions")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "src", "models", "churn_model.pkl")
model = joblib.load(model_path)


model_features = model.get_booster().feature_names


uploaded_file = st.file_uploader("📂 Upload Customer CSV", type=["csv"])


def preprocess_for_model(data):

    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))


    data = data.fillna(0)

    data = data.reindex(columns=model_features, fill_value=0)

    return data


if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 🔍 Data Preview", data.head())

    if "Churn" in data.columns:
        data = data.drop(columns=["Churn"])
        st.info("ℹ️ Dropped 'Churn' column from uploaded file (target variable).")

    if st.button("🚀 Predict Churn"):

        data_processed = preprocess_for_model(data.copy())


        churn_pred = model.predict(data_processed)
        churn_prob = model.predict_proba(data_processed)[:, 1]


        data["Churn Prediction"] = churn_pred
        data["Churn Probability"] = churn_prob



        def recommend_action(prob):
            if prob > 0.8:
                return "Offer 30% discount + priority support"
            elif prob > 0.5:
                return "Personalized offer (free month/upgrade)"
            elif prob > 0.2:
                return "Send engagement email"
            else:
                return "No action needed"


        data["Recommendation"] = data["Churn Probability"].apply(recommend_action)


        st.write("### ✅ Prediction Results with Recommendations", data)


        st.subheader("Churn Probability Distribution")
        plt.figure(figsize=(8, 4))
        sns.histplot(data['Churn Probability'], bins=10, kde=True, color='skyblue')
        plt.xlabel("Churn Probability")
        plt.ylabel("Number of Customers")
        st.pyplot(plt)


        st.subheader("Recommendations Breakdown")
        st.bar_chart(data['Recommendation'].value_counts())


        if st.checkbox("🔍 Show Explainability"):
            st.subheader("SHAP Explainability")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_processed)

            st.write("### Example: First Customer Feature Contribution")
            shap.initjs()
            force_plot = shap.force_plot(
                explainer.expected_value[1],
                shap_values[1][0, :],
                data_processed.iloc[0, :],
                matplotlib=True
            )
            st.pyplot(force_plot)
