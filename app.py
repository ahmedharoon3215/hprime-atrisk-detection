"""
app.py - At-Risk Trainee Detection API
CSCI323 Modern Artificial Intelligence - Spring 2026
University of Wollongong in Dubai

Run: python app.py
Endpoint: POST http://localhost:5000/predict
"""
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model        = joblib.load("model.pkl")
preprocessor = joblib.load("hprime_preprocessor.pkl")

NUMERIC_COLS = [
    "patient_age_years","patient_is_female","is_emergency","days_since_last",
    "rolling_avg_3","rolling_fail_rate_3","cumulative_fail_count",
    "discussion_time_mins","prep_time_mins","feedback_length","num_competencies"
]
CATEGORICAL_COLS = ["form_type","assessor_seniority","setting"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "At-Risk Trainee Detector v1.0"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        row  = {col: data.get(col, 0) for col in NUMERIC_COLS}
        row.update({col: data.get(col, "other") for col in CATEGORICAL_COLS})
        df          = pd.DataFrame([row])
        X_processed = preprocessor.transform(df)
        risk_prob   = float(model.predict_proba(X_processed)[0][0])
        risk_score  = round(risk_prob * 100, 1)
        if risk_score >= 75:   tier = "critical"
        elif risk_score >= 50: tier = "high"
        elif risk_score >= 25: tier = "moderate"
        else:                  tier = "low"
        recommendations = {
            "critical": "Immediate supervisor review required.",
            "high":     "Schedule check-in within the next assessment cycle.",
            "moderate": "Monitor closely in upcoming assessments.",
            "low":      "Trainee is progressing well. Continue standard monitoring."
        }
        return jsonify({
            "risk_score":     risk_score,
            "risk_tier":      tier,
            "prediction":     "at_risk" if risk_prob > 0.5 else "on_track",
            "recommendation": recommendations[tier]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        records = request.get_json()
        results = []
        for rec in records:
            row = {col: rec.get(col, 0) for col in NUMERIC_COLS}
            row.update({col: rec.get(col, "other") for col in CATEGORICAL_COLS})
            df          = pd.DataFrame([row])
            X_processed = preprocessor.transform(df)
            risk_prob   = float(model.predict_proba(X_processed)[0][0])
            risk_score  = round(risk_prob * 100, 1)
            results.append({
                "candidate_id": rec.get("candidate_id","unknown"),
                "risk_score":   risk_score,
                "risk_tier":    "critical" if risk_score>=75 else
                                "high"     if risk_score>=50 else
                                "moderate" if risk_score>=25 else "low",
                "prediction":   "at_risk" if risk_prob>0.5 else "on_track"
            })
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return jsonify({"total": len(results), "predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Starting hPrime At-Risk Trainee Detection API...")
    app.run(debug=True, port=5000)
