"""
Flask Web Application for Capstone Machine Learning Prediction Systems.
Provides interactive web interfaces and REST API endpoints for:
1. Loan Approval Prediction
2. Disease Prediction
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOAN_MODEL_DIR = os.path.join(BASE_DIR, "Project1_Loan_Approval_Prediction", "models")
DISEASE_MODEL_DIR = os.path.join(BASE_DIR, "Project2_Disease_Prediction", "models")

# Lazy model loaders
_loan_model = None
_loan_preprocessor = None
_disease_model = None
_disease_encoder = None
_symptom_features = None


def get_loan_artifacts():
    global _loan_model, _loan_preprocessor
    if _loan_model is None:
        model_path = os.path.join(LOAN_MODEL_DIR, "best_loan_model.joblib")
        prep_path = os.path.join(LOAN_MODEL_DIR, "loan_preprocessor.joblib")
        if os.path.exists(model_path) and os.path.exists(prep_path):
            _loan_model = joblib.load(model_path)
            _loan_preprocessor = joblib.load(prep_path)
    return _loan_model, _loan_preprocessor


def get_disease_artifacts():
    global _disease_model, _disease_encoder, _symptom_features
    if _disease_model is None:
        model_path = os.path.join(DISEASE_MODEL_DIR, "best_disease_model.joblib")
        enc_path = os.path.join(DISEASE_MODEL_DIR, "disease_label_encoder.joblib")
        sym_path = os.path.join(DISEASE_MODEL_DIR, "symptom_features.json")
        if os.path.exists(model_path) and os.path.exists(enc_path) and os.path.exists(sym_path):
            _disease_model = joblib.load(model_path)
            _disease_encoder = joblib.load(enc_path)
            with open(sym_path, "r") as f:
                _symptom_features = json.load(f)
    return _disease_model, _disease_encoder, _symptom_features


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/loan", methods=["GET", "POST"])
def loan_prediction():
    result = None
    if request.method == "POST":
        try:
            model, preprocessor = get_loan_artifacts()
            if model is None:
                return render_template("loan.html", error="Model not trained yet. Run train.py first.")

            # Extract form inputs
            gender = request.form.get("Gender", "Male")
            married = request.form.get("Married", "Yes")
            dependents = request.form.get("Dependents", "0")
            education = request.form.get("Education", "Graduate")
            self_employed = request.form.get("Self_Employed", "No")
            app_income = float(request.form.get("ApplicantIncome", 5000))
            coapp_income = float(request.form.get("CoapplicantIncome", 0))
            loan_amount = float(request.form.get("LoanAmount", 150))
            loan_term = float(request.form.get("Loan_Amount_Term", 360))
            credit_hist = float(request.form.get("Credit_History", 1.0))
            prop_area = request.form.get("Property_Area", "Semiurban")

            # Engineer derived features
            total_income = app_income + coapp_income
            loan_ratio = (loan_amount * 1000) / (total_income if total_income > 0 else 1)
            income_log = np.log1p(total_income)
            loan_log = np.log1p(loan_amount)

            input_dict = {
                "ApplicantIncome": [app_income],
                "CoapplicantIncome": [coapp_income],
                "LoanAmount": [loan_amount],
                "Loan_Amount_Term": [loan_term],
                "Total_Income": [total_income],
                "Loan_to_Income_Ratio": [loan_ratio],
                "Income_Log": [income_log],
                "LoanAmount_Log": [loan_log],
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [education],
                "Self_Employed": [self_employed],
                "Credit_History": [credit_hist],
                "Property_Area": [prop_area]
            }

            input_df = pd.DataFrame(input_dict)
            transformed = preprocessor.transform(input_df)

            pred = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0] if hasattr(model, "predict_proba") else [1.0 - pred, pred]

            approval_prob = round(prob[1] * 100, 1)
            verdict = "Approved" if pred == 1 else "Rejected"
            status_class = "success" if pred == 1 else "danger"

            result = {
                "verdict": verdict,
                "status_class": status_class,
                "approval_probability": approval_prob,
                "rejection_probability": round(100 - approval_prob, 1),
                "applicant_income": app_income,
                "loan_amount": loan_amount,
                "credit_history": "Good (1.0)" if credit_hist == 1.0 else "Poor / Delinquent (0.0)"
            }
        except Exception as e:
            return render_template("loan.html", error=str(e))

    return render_template("loan.html", result=result)


@app.route("/disease", methods=["GET", "POST"])
def disease_prediction():
    _, _, symptom_features = get_disease_artifacts()
    symptoms_list = symptom_features or []

    result = None
    if request.method == "POST":
        try:
            model, encoder, symptoms = get_disease_artifacts()
            if model is None:
                return render_template("disease.html", error="Model not trained yet.", symptoms=symptoms_list)

            # Get selected symptoms from multi-select form
            selected = request.form.getlist("symptoms")
            if not selected:
                return render_template("disease.html", error="Please select at least 1 symptom.", symptoms=symptoms_list)

            # Build feature vector
            vector = np.zeros((1, len(symptoms)))
            for sym in selected:
                if sym in symptoms:
                    idx = symptoms.index(sym)
                    vector[0, idx] = 1.0

            probs = model.predict_proba(vector)[0]
            top_3_idx = np.argsort(probs)[::-1][:3]

            top_predictions = []
            for idx in top_3_idx:
                disease = encoder.classes_[idx]
                confidence = round(probs[idx] * 100, 1)
                top_predictions.append({"disease": disease, "confidence": confidence})

            result = {
                "primary_disease": top_predictions[0]["disease"],
                "primary_confidence": top_predictions[0]["confidence"],
                "top_3": top_predictions,
                "symptoms_reported": [s.replace("_", " ").title() for s in selected]
            }
        except Exception as e:
            return render_template("disease.html", error=str(e), symptoms=symptoms_list)

    return render_template("disease.html", symptoms=symptoms_list, result=result)


# REST API endpoints
@app.route("/api/predict_loan", methods=["POST"])
def api_predict_loan():
    model, preprocessor = get_loan_artifacts()
    data = request.get_json(force=True)

    app_inc = float(data.get("ApplicantIncome", 5000))
    coapp_inc = float(data.get("CoapplicantIncome", 0))
    loan_amt = float(data.get("LoanAmount", 150))
    term = float(data.get("Loan_Amount_Term", 360))
    tot_inc = app_inc + coapp_inc

    input_df = pd.DataFrame({
        "ApplicantIncome": [app_inc],
        "CoapplicantIncome": [coapp_inc],
        "LoanAmount": [loan_amt],
        "Loan_Amount_Term": [term],
        "Total_Income": [tot_inc],
        "Loan_to_Income_Ratio": [(loan_amt * 1000) / tot_inc],
        "Income_Log": [np.log1p(tot_inc)],
        "LoanAmount_Log": [np.log1p(loan_amt)],
        "Gender": [data.get("Gender", "Male")],
        "Married": [data.get("Married", "Yes")],
        "Dependents": [str(data.get("Dependents", "0"))],
        "Education": [data.get("Education", "Graduate")],
        "Self_Employed": [data.get("Self_Employed", "No")],
        "Credit_History": [float(data.get("Credit_History", 1.0))],
        "Property_Area": [data.get("Property_Area", "Semiurban")]
    })

    transformed = preprocessor.transform(input_df)
    pred = int(model.predict(transformed)[0])
    prob = float(model.predict_proba(transformed)[0][1])

    return jsonify({
        "status": "success",
        "loan_approved": bool(pred == 1),
        "approval_probability": round(prob, 4),
        "verdict": "Approved" if pred == 1 else "Rejected"
    })


@app.route("/api/predict_disease", methods=["POST"])
def api_predict_disease():
    model, encoder, symptoms = get_disease_artifacts()
    data = request.get_json(force=True)
    selected = data.get("symptoms", [])

    vector = np.zeros((1, len(symptoms)))
    for sym in selected:
        if sym in symptoms:
            idx = symptoms.index(sym)
            vector[0, idx] = 1.0

    probs = model.predict_proba(vector)[0]
    top_idx = int(np.argmax(probs))
    disease = encoder.classes_[top_idx]
    confidence = float(probs[top_idx])

    return jsonify({
        "status": "success",
        "predicted_disease": disease,
        "confidence": round(confidence, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
