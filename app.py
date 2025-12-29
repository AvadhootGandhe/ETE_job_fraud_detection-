from flask import Flask, request, jsonify, render_template
import pickle
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

# Load model & vectorizers
model = pickle.load(open("model.pkl", "rb"))
numeric_cols = pickle.load(open("numeric_cols.pkl", "rb"))

tfidf_company_profile  = pickle.load(open("tfidf_company_profile.pkl", "rb"))
tfidf_description      = pickle.load(open("tfidf_description.pkl", "rb"))
tfidf_requirements     = pickle.load(open("tfidf_requirements.pkl", "rb"))
tfidf_benefits         = pickle.load(open("tfidf_benefits.pkl", "rb"))


def clean_text(text):
    if text is None:
        return ""
    return (str(text).strip().replace("\n", " ").replace("\r", " "))


@app.route('/')
def home():
    return render_template('home.html', numeric_cols=numeric_cols)


# -------------------------------------------------------------
# JSON API Prediction (same style as /predict_api)
# -------------------------------------------------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']

    company   = clean_text(data.get("company_profile", ""))
    desc      = clean_text(data.get("description", ""))
    req       = clean_text(data.get("requirements", ""))
    benefits  = clean_text(data.get("benefits", ""))

    t1 = tfidf_company_profile.transform([company])
    t2 = tfidf_description.transform([desc])
    t3 = tfidf_requirements.transform([req])
    t4 = tfidf_benefits.transform([benefits])

    text_features = hstack([t1, t2, t3, t4])

    numeric_input = [float(data.get(col, 0)) for col in numeric_cols]
    numeric_sparse = csr_matrix([numeric_input])

    final_vector = hstack([text_features, numeric_sparse])

    pred = model.predict(final_vector)[0]
    prob = model.predict_proba(final_vector)[0][1]

    return jsonify({
        "fraud_prediction": int(pred),
        "fraud_probability": float(prob)
    })


# -------------------------------------------------------------
# Form-Based Prediction (same style as /predict)
# -------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()

    company   = clean_text(form_data.get("company_profile", ""))
    desc      = clean_text(form_data.get("description", ""))
    req       = clean_text(form_data.get("requirements", ""))
    benefits  = clean_text(form_data.get("benefits", ""))

    t1 = tfidf_company_profile.transform([company])
    t2 = tfidf_description.transform([desc])
    t3 = tfidf_requirements.transform([req])
    t4 = tfidf_benefits.transform([benefits])

    text_features = hstack([t1, t2, t3, t4])

    numeric_input = [float(form_data.get(col, 0)) for col in numeric_cols]
    numeric_sparse = csr_matrix([numeric_input])

    final_vector = hstack([text_features, numeric_sparse])

    pred = model.predict(final_vector)[0]
    prob = model.predict_proba(final_vector)[0][1]

    result = f"Prediction: {int(pred)}, Fraud Probability: {prob:.4f}"

    return render_template("home.html", prediction_text=result, numeric_cols=numeric_cols)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

