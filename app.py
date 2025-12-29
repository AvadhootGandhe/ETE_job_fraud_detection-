from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
numeric_cols = pickle.load(open("numeric_cols.pkl", "rb"))

tfidf_company_profile  = pickle.load(open("tfidf_company_profile.pkl", "rb"))
tfidf_description      = pickle.load(open("tfidf_description.pkl", "rb"))
tfidf_requirements     = pickle.load(open("tfidf_requirements.pkl", "rb"))
tfidf_benefits         = pickle.load(open("tfidf_benefits.pkl", "rb"))


def clean_text(text):
    if text is None:
        return ""
    return (
        str(text)
            .strip()
            .replace("\n", " ")
            .replace("\r", " ")
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    company   = clean_text(data.get("company_profile", ""))
    desc      = clean_text(data.get("description", ""))
    req       = clean_text(data.get("requirements", ""))
    benefits  = clean_text(data.get("benefits", ""))

    t1 = tfidf_company_profile.transform([company])
    t2 = tfidf_description.transform([desc])
    t3 = tfidf_requirements.transform([req])
    t4 = tfidf_benefits.transform([benefits])

    text_features = hstack([t1, t2, t3, t4])

    numeric_input = []
    for col in numeric_cols:
        numeric_input.append(data.get(col, 0))

    numeric_sparse = csr_matrix([numeric_input])

    final_vector = hstack([text_features, numeric_sparse])

    pred_class = model.predict(final_vector)[0]
    pred_prob = model.predict_proba(final_vector)[0][1]

    return jsonify({
        "fraud_prediction": int(pred_class),
        "fraud_probability": float(pred_prob)
    })


@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection Model API is running"})


if __name__ == "__main__":
    app.run(debug=True)
