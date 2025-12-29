from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# =================================================
# LOAD ALL PKL FILES
# =================================================

# Load SVM model (Pipeline: scaler + SVC)
model = pickle.load(open("model.pkl", "rb"))

# Load numeric column order
numeric_cols = pickle.load(open("numeric_cols.pkl", "rb"))

# Load TF-IDF vectorizers
tfidf_company_profile  = pickle.load(open("tfidf_company_profile.pkl", "rb"))
tfidf_description      = pickle.load(open("tfidf_description.pkl", "rb"))
tfidf_requirements     = pickle.load(open("tfidf_requirements.pkl", "rb"))
tfidf_benefits         = pickle.load(open("tfidf_benefits.pkl", "rb"))


# =================================================
# HELPER: CLEAN TEXT (SAME AS TRAINING)
# =================================================
def clean_text(text):
    if text is None:
        return ""
    return (
        str(text)
            .strip()
            .replace("\n", " ")
            .replace("\r", " ")
    )


# =================================================
# PREDICTION API
# =================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # ---------------------------------------------
    # 1. CLEAN TEXT INPUTS
    # ---------------------------------------------
    company   = clean_text(data.get("company_profile", ""))
    desc      = clean_text(data.get("description", ""))
    req       = clean_text(data.get("requirements", ""))
    benefits  = clean_text(data.get("benefits", ""))

    # ---------------------------------------------
    # 2. TF-IDF TRANSFORM FOR ALL TEXT FIELDS
    # ---------------------------------------------
    t1 = tfidf_company_profile.transform([company])
    t2 = tfidf_description.transform([desc])
    t3 = tfidf_requirements.transform([req])
    t4 = tfidf_benefits.transform([benefits])

    text_features = hstack([t1, t2, t3, t4])

    # ---------------------------------------------
    # 3. NUMERIC FEATURES (MUST FOLLOW TRAIN ORDER)
    # ---------------------------------------------
    numeric_input = []
    for col in numeric_cols:
        numeric_input.append(data.get(col, 0))

    numeric_sparse = csr_matrix([numeric_input])

    # ---------------------------------------------
    # 4. COMBINE TEXT + NUMERIC INTO FULL VECTOR
    # ---------------------------------------------
    final_vector = hstack([text_features, numeric_sparse])

    # ---------------------------------------------
    # 5. MAKE PREDICTION
    # ---------------------------------------------
    pred_class = model.predict(final_vector)[0]
    pred_prob = model.predict_proba(final_vector)[0][1]

    return jsonify({
        "fraud_prediction": int(pred_class),
        "fraud_probability": float(pred_prob)
    })

# =================================================
# ROOT ROUTE
# =================================================
@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection Model API is running"})


# =================================================
# RUN APP
# =================================================
if __name__ == "__main__":
    app.run(debug=True)
