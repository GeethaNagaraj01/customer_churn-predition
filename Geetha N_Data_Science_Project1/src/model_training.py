import joblib
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_preprocessing import load_and_preprocess_data

def train_model():

    X_train, X_test, y_train, y_test, encoders = load_and_preprocess_data("data/Telco-Customer-Churn.csv")


    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)


    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))


    os.makedirs("models", exist_ok=True)


    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(encoders, "models/encoder.pkl")

if __name__ == "__main__":
    train_model()
