from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from utils import compute_lengths

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Required user inputs
    required_fields = [
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "telecommuting",
        "has_company_logo",
        "has_questions",
        "employment_type",
        "required_experience",
        "required_education",
        "function",
        "mean_salary",
        "industry_cleaned",
        "country_enc",
        "state_enc",
        "city_enc"
    ]

    # Check missing fields
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Derived features
    derived = compute_lengths(data)

    # Construct DataFrame row
    df = pd.DataFrame([{
        **data,  # user inputs
        **derived  # auto generated
    }])

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "fraud_prediction": int(pred),
        "fraud_probability": float(prob)
    })


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI Fraud Detection API Running"})


if __name__ == "__main__":
    app.run(debug=True)
