# customer_churn-predition
Customer Churn Prediction System – End-to-End Machine Learning Project
📌 Project Overview

Customer churn is one of the biggest problems in subscription-based businesses (telecom, banking, SaaS, streaming apps, etc.).
This project predicts whether a customer is likely to churn, i.e., leave the company, using machine learning.

The project also provides:

✔ Real-time prediction using Streamlit UI
✔ Personalized recommendations based on churn probability
✔ Churn probability distribution visualization
✔ Customer insights dashboard

🎯 Problem Statement

Businesses lose a lot of revenue when customers cancel their service.
The goal is:

Predict which customers are at risk of churn using their usage patterns, demographics, and account information.

This helps companies take preventive actions such as offering discounts, improved support, or personalized retention strategies.

🧠 Solution Approach

We use a machine learning model trained on customer behavior data to classify customers as:

Churn (1)

Not Churn (0)

The model analyzes important indicators like:

Tenure (how long the customer stayed)

Monthly charges

Total charges

Payment method

Contract type

Internet service

Customer support calls

The Streamlit app then uses this model to make real-time predictions.

📂 Dataset Information

You can use any churn dataset, but the most commonly used one is:

Telco Customer Churn Dataset

🔗 Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Columns include:

CustomerID

Gender

SeniorCitizen

Partner

Dependents

Tenure

PhoneService

MultipleLines

InternetService

OnlineSecurity

OnlineBackup

TechSupport

Contract

PaymentMethod

MonthlyCharges

TotalCharges

Churn

📊 Modeling Steps
1️⃣ Data Preprocessing

Handle missing values

Convert categorical → numerical

Encode labels

Scale numerical features

Split into train/test

2️⃣ Model Selection

Used algorithms:

Logistic Regression

Random Forest

Gradient Boosting

Best performing model: Random Forest Classifier

3️⃣ Model Evaluation

Metrics used:

Accuracy

Precision

Recall

ROC-AUC Score

Confusion Matrix

🖥️ Streamlit Web App

The app has four sections:

1️⃣ Dataset Preview Screen

Heading to display:

📄 Dataset Preview

Shows the top 10 rows so users understand the input structure.

2️⃣ Prediction Result Screen

Heading to display:

🧮 Prediction Result with Recommendations

It shows:

Predicted status: Churn / Not Churn

Churn probability (0–100%)

Personalized recommendations:

Offer discount plan

Improve customer support

Provide long-term contract benefits

Upgrade to faster internet

3️⃣ Churn Probability Distribution

Heading:

📊 Churn Probability Distribution (Histogram)

This visual explains:

How the model assigns churn probability

Whether customers are generally at high risk

4️⃣ Recommendation Breakdown

Heading:

💡 Recommendations Breakdown

This section explains why the customer is predicted as churn:

Example sentences:

“High monthly charges indicate risk of churn.”

“Short tenure suggests customer may still be exploring alternatives.”

“Month-to-month contract customers have high churn probability.”

📁 Project Folder Structure (Upload to GitHub)
customer-churn-prediction/
│
├── data/
│   └── telco_churn.csv
│
├── model/
│   └── churn_model.pkl
│   └── scaler.pkl
│
├── app/
│   └── app.py (Streamlit application)
│
├── notebooks/
│   └── churn_model_training.ipynb
│
├── screenshots/
│   ├── dataset_preview.png
│   ├── prediction_output.png
│   ├── churn_distribution.png
│   └── recommendations.png
│
├── README.md
└── requirements.txt

🛠️ Installation & Running Instructions
1️⃣ Clone the Repo
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app/app.py

🧪 Code Snippet (Model Training)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("data/telco_churn.csv")

# Encode categorical columns
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include=["object"]).columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model & scaler
pickle.dump(model, open("model/churn_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

🖥️ Streamlit App Code (app.py)
import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.title("Customer Churn Prediction System")

st.sidebar.header("Customer Input Features")

def user_input():
    tenure = st.sidebar.number_input("Tenure (Months)", 0, 72)
    monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0)
    total = st.sidebar.number_input("Total Charges", 0.0, 10000.0)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract
    }

    return pd.DataFrame([data])

df = user_input()

st.subheader("Dataset Preview")
st.write(df)

df_scaled = scaler.transform(df)
prediction = model.predict(df_scaled)
proba = model.predict_proba(df_scaled)[0][1] * 100

st.subheader("Prediction Result with Recommendation")
if prediction == 1:
    st.error(f"🔴 Customer Will Churn (Probability: {proba:.2f}%)")
else:
    st.success(f"🟢 Customer Will Not Churn (Probability: {proba:.2f}%)")

st.subheader("Recommendations Breakdown")
if prediction == 1:
    st.write("- Offer discount on plans")
    st.write("- Improve customer support response time")
    st.write("- Provide long-term contract benefits")
else:
    st.write("- Maintain current engagement level")

✔ Accurate churn prediction
✔ Real-time prediction UI
✔ Visual insights
✔ Actionable business recommendations
