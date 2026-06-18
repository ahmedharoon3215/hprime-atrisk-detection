# =============================================================================
# CODE COMPONENT 3: The Metrics & API Integrator
# CSCI323 Modern Artificial Intelligence — Spring 2026
# University of Wollongong in Dubai
#
# PURPOSE: Proves the model works through visualisations (Confusion Matrix,
#          ROC Curve, Feature Importances) and exposes it as a working API
#          so Imzans/hPrime can integrate predictions into their platform.
#
# HOW TO RUN: Paste this into a new Colab cell after Component 2.
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# ── Styling ───────────────────────────────────────────────────────────────────
BRAND_BLUE  = "#1F4E79"
BRAND_MID   = "#2E75B6"
FAIL_RED    = "#C00000"
PASS_GREEN  = "#538135"
WARN_ORANGE = "#E97C30"
BG          = "#F7F9FC"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#CCCCCC", "axes.labelsize": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlecolor": BRAND_BLUE, "font.family": "DejaVu Sans",
    "grid.color": "#E0E0E0", "grid.linewidth": 0.7,
})

print("=" * 60)
print("COMPONENT 3: Evaluation & API Integration")
print("=" * 60)

# ── Step 1: Load saved artifacts from Component 2 ────────────────────────────
print("\n── Step 1: Loading model artifacts ──")
try:
    best_model     = joblib.load('model.pkl')
    rf_model       = joblib.load('random_forest_model.pkl')
    lr_model       = joblib.load('logistic_regression_model.pkl')
    feature_names  = joblib.load('feature_names.pkl')
    X_test_proc    = joblib.load('X_test_processed.pkl')
    y_test         = joblib.load('y_test.pkl')
    rf_proba       = joblib.load('rf_proba.pkl')
    lr_proba       = joblib.load('lr_proba.pkl')
    preprocessor   = joblib.load('hprime_preprocessor.pkl')
    print("  All artifacts loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Missing file: {e}. Please run Component 2 first.")

# ── Step 2: Generate all predictions ─────────────────────────────────────────
rf_preds = rf_model.predict(X_test_proc)
lr_preds = lr_model.predict(X_test_proc)

# ── Step 3: Full Metrics Report ───────────────────────────────────────────────
print("\n── Step 3: Full Metrics Report ──")
print("\n  RANDOM FOREST:")
print(classification_report(y_test, rf_preds,
      target_names=["At-Risk (Fail)", "On-Track (Pass)"]))

print("  LOGISTIC REGRESSION (Baseline):")
print(classification_report(y_test, lr_preds,
      target_names=["At-Risk (Fail)", "On-Track (Pass)"]))

# Summary table
metrics = {
    "Model": ["Logistic Regression\n(Baseline)", "Random Forest\n(Main)"],
    "Accuracy":  [accuracy_score(y_test, lr_preds),  accuracy_score(y_test, rf_preds)],
    "Precision": [precision_score(y_test, lr_preds, pos_label=0, zero_division=0),
                  precision_score(y_test, rf_preds, pos_label=0, zero_division=0)],
    "Recall":    [recall_score(y_test, lr_preds, pos_label=0),
                  recall_score(y_test, rf_preds, pos_label=0)],
    "F1-Score":  [f1_score(y_test, lr_preds, pos_label=0, zero_division=0),
                  f1_score(y_test, rf_preds, pos_label=0, zero_division=0)],
    "AUC-ROC":   [auc(*roc_curve(y_test, lr_proba)[:2]),
                  auc(*roc_curve(y_test, rf_proba)[:2])],
}
metrics_df = pd.DataFrame(metrics)
print("\n  Summary Table (At-Risk / Fail class):")
print(metrics_df.to_string(index=False))

# ── Step 4: Visualisations ───────────────────────────────────────────────────
print("\n── Step 4: Generating Evaluation Plots ──")

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("hPrime At-Risk Trainee Detection — Model Evaluation",
             fontsize=16, fontweight="bold", color=BRAND_BLUE, y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 4a: Confusion Matrix — Logistic Regression ───────────────────────────────
ax = fig.add_subplot(gs[0, 0])
cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["At-Risk", "On-Track"],
            yticklabels=["At-Risk", "On-Track"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix\nLogistic Regression (Baseline)")

# ── 4b: Confusion Matrix — Random Forest ─────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
cm_rf = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="OrRd", ax=ax,
            xticklabels=["At-Risk", "On-Track"],
            yticklabels=["At-Risk", "On-Track"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix\nRandom Forest (Main Model)")

# ── 4c: ROC Curves — Both Models ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for proba, label, color in [
    (lr_proba, "Logistic Regression", BRAND_MID),
    (rf_proba, "Random Forest",       FAIL_RED),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{label} (AUC = {roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier (AUC = 0.500)")
ax.fill_between(*roc_curve(y_test, rf_proba)[:2],
                alpha=0.08, color=FAIL_RED)
ax.set_xlabel("False Positive Rate (Incorrectly flagged healthy trainees)")
ax.set_ylabel("True Positive Rate (At-risk trainees correctly caught)")
ax.set_title("ROC Curve Comparison")
ax.legend(fontsize=8); ax.grid()

# ── 4d: Precision-Recall Curve ───────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
for proba, label, color in [
    (lr_proba, "Logistic Regression", BRAND_MID),
    (rf_proba, "Random Forest",       FAIL_RED),
]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    pr_auc       = auc(rec, prec)
    ax.plot(rec, prec, color=color, lw=2,
            label=f"{label} (AUC = {pr_auc:.3f})")
ax.axhline(y_test.mean(), color="#999", ls="--", lw=1.2,
           label=f"Baseline (fail rate = {y_test.mean():.1%})")
ax.set_xlabel("Recall (At-risk trainees caught)")
ax.set_ylabel("Precision (Of flagged, how many are truly at-risk)")
ax.set_title("Precision-Recall Curve")
ax.legend(fontsize=8); ax.grid()

# ── 4e: Feature Importances ───────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1:])
importances = pd.Series(
    rf_model.feature_importances_, index=feature_names
).sort_values(ascending=True).tail(12)

colors_imp = [FAIL_RED if i >= len(importances) - 3
              else BRAND_MID if i >= len(importances) - 6
              else BRAND_LIGHT if hasattr(importances, 'values') else BRAND_MID
              for i in range(len(importances))]

BRAND_LIGHT = "#BDD7EE"
bar_colors = []
for i, v in enumerate(importances.values):
    if i >= len(importances) - 3:
        bar_colors.append(FAIL_RED)
    elif i >= len(importances) - 6:
        bar_colors.append(WARN_ORANGE)
    else:
        bar_colors.append(BRAND_MID)

bars = ax.barh(importances.index, importances.values,
               color=bar_colors, edgecolor="white")
for bar, v in zip(bars, importances.values):
    ax.text(v + 0.001, bar.get_y() + bar.get_height()/2,
            f"{v:.4f}", va="center", fontsize=8)
ax.set_xlabel("Feature Importance Score")
ax.set_title("Top 12 Feature Importances — Random Forest")
ax.grid(axis="x")

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=FAIL_RED,    label="Top 3 predictors"),
    Patch(color=WARN_ORANGE, label="Strong predictors"),
    Patch(color=BRAND_MID,   label="Supporting features"),
], loc="lower right", fontsize=8)

plt.savefig("evaluation_plots.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
print("  Saved: evaluation_plots.png")

# ── Step 5: Risk Score Output ─────────────────────────────────────────────────
# The model outputs a risk score (0–100) per assessment.
# This is what would actually appear in the hPrime dashboard.
print("\n── Step 5: Sample Risk Score Output ──")
print("  (This is what supervisors would see in the hPrime dashboard)\n")

risk_scores = ((1 - rf_model.predict_proba(X_test_proc)[:, 1]) * 100).round(1)
risk_labels = pd.cut(risk_scores, bins=[0, 25, 50, 75, 100],
                     labels=["✅ Low Risk", "🟡 Moderate Risk",
                             "🟠 High Risk", "🔴 Critical Risk"])

output_df = pd.DataFrame({
    "Risk Score (0-100)": risk_scores[:20],
    "Risk Level":         risk_labels[:20],
    "Actual Outcome":     y_test.values[:20],
    "Predicted":          rf_preds[:20],
}).reset_index(drop=True)

output_df["Actual Outcome"] = output_df["Actual Outcome"].map(
    {0: "❌ Failed", 1: "✅ Passed"}
)
output_df["Predicted"] = output_df["Predicted"].map(
    {0: "⚠️ At-Risk", 1: "✅ On-Track"}
)
print(output_df.to_string(index=True))

# ── Step 6: Save evaluate_model.py as a script ───────────────────────────────
# Save a standalone version (no Colab needed) for the code repository
evaluate_script = '''"""
evaluate_model.py — Standalone Evaluation Script
Run: python evaluate_model.py
Requires: model.pkl, hprime_preprocessor.pkl, hprime_clean.csv
"""
import pandas as pd, numpy as np, joblib
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("hprime_clean.csv")
model = joblib.load("model.pkl")
preprocessor = joblib.load("hprime_preprocessor.pkl")

columns_to_drop = [
    "candidate_id","assessment_date","assessment_number","assessor_name",
    "assessor_title","competency_areas","focus_area","case_summary",
    "feedback","meets_standard","score_record_keeping","score_history_examination",
    "score_management_plan","score_clinical_judgement","score_history",
    "score_physical_exam","score_communication","score_clinical_judgement.1",
    "score_professionalism","score_organisation","score_overall_care",
    "avg_score","score_std","any_score_1"
]
y = df["meets_standard"]
X = df.drop(columns=columns_to_drop)
numeric_cols = X.select_dtypes(include=["float64","int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
X["days_since_last"] = X["days_since_last"].fillna(0)
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_proc = preprocessor.transform(X_test)
preds = model.predict(X_test_proc)
print(classification_report(y_test, preds, target_names=["At-Risk","On-Track"]))
print(f"AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test_proc)[:,0]):.3f}")
'''
with open("evaluate_model.py", "w") as f:
    f.write(evaluate_script)
print("\n  Saved: evaluate_model.py")

# ── Step 7: Flask API (app.py) ────────────────────────────────────────────────
# The API allows the hPrime system to send a trainee's stats and receive
# a risk score back in real time — like a doctor consulting a specialist
# by phone and getting an instant second opinion.

api_code = '''"""
app.py — At-Risk Trainee Detection API
Exposes the trained Random Forest model as a REST API endpoint.
Imzans/hPrime can call this to get a risk score for any trainee.

Run locally: python app.py
Then send a POST request to http://localhost:5000/predict
"""
from flask import Flask, request, jsonify
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load model and preprocessor on startup
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
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "At-Risk Trainee Detector v1.0"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a trainee assessment record and returns a risk score.

    Example request body (JSON):
    {
        "patient_age_years": 45,
        "patient_is_female": 1,
        "is_emergency": 0,
        "days_since_last": 7,
        "rolling_avg_3": 2.4,
        "rolling_fail_rate_3": 0.33,
        "cumulative_fail_count": 1,
        "discussion_time_mins": 15,
        "prep_time_mins": 10,
        "feedback_length": 18,
        "num_competencies": 2,
        "form_type": "Mini-CEX",
        "assessor_seniority": "senior",
        "setting": "inpatient"
    }
    """
    try:
        data = request.get_json()

        # Build a single-row DataFrame matching the training schema
        row = {col: data.get(col, 0) for col in NUMERIC_COLS}
        row.update({col: data.get(col, "other") for col in CATEGORICAL_COLS})
        df = pd.DataFrame([row])

        # Preprocess and predict
        X_processed = preprocessor.transform(df)
        risk_prob   = float(model.predict_proba(X_processed)[0][0])  # prob of failing
        risk_score  = round(risk_prob * 100, 1)
        prediction  = "at_risk" if risk_prob > 0.5 else "on_track"

        # Risk tier
        if risk_score >= 75:   tier = "critical"
        elif risk_score >= 50: tier = "high"
        elif risk_score >= 25: tier = "moderate"
        else:                  tier = "low"

        return jsonify({
            "risk_score":    risk_score,
            "risk_tier":     tier,
            "prediction":    prediction,
            "confidence":    round(max(risk_prob, 1 - risk_prob) * 100, 1),
            "recommendation": {
                "critical": "Immediate supervisor review required.",
                "high":     "Schedule check-in within the next assessment cycle.",
                "moderate": "Monitor closely in upcoming assessments.",
                "low":      "Trainee is progressing well. Continue standard monitoring."
            }[tier]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Accepts a list of trainee records and returns risk scores for all of them.
    Useful for generating the hPrime dashboard risk report.
    """
    try:
        records = request.get_json()
        results = []
        for rec in records:
            row = {col: rec.get(col, 0) for col in NUMERIC_COLS}
            row.update({col: rec.get(col, "other") for col in CATEGORICAL_COLS})
            df = pd.DataFrame([row])
            X_processed = preprocessor.transform(df)
            risk_prob   = float(model.predict_proba(X_processed)[0][0])
            results.append({
                "candidate_id": rec.get("candidate_id", "unknown"),
                "risk_score":   round(risk_prob * 100, 1),
                "risk_tier":    "critical" if risk_prob >= 0.75 else
                                "high"     if risk_prob >= 0.50 else
                                "moderate" if risk_prob >= 0.25 else "low"
            })
        # Sort by risk score descending so highest risk trainees appear first
        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return jsonify({"total": len(results), "predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("Starting hPrime At-Risk Trainee Detection API...")
    print("Endpoints:")
    print("  GET  /health         — check if API is running")
    print("  POST /predict        — single trainee risk score")
    print("  POST /batch_predict  — risk scores for multiple trainees")
    app.run(debug=True, port=5000)
'''
with open("app.py", "w") as f:
    f.write(api_code)
print("  Saved: app.py")

# ── Step 8: Test the API locally inside Colab ─────────────────────────────────
# We can test the prediction logic directly without running Flask
print("\n── Step 8: API Logic Test (simulating a real request) ──")
sample_trainee = {
    "patient_age_years":    38,
    "patient_is_female":     0,
    "is_emergency":          1,
    "days_since_last":      14,
    "rolling_avg_3":        2.1,
    "rolling_fail_rate_3":  0.67,
    "cumulative_fail_count": 2,
    "discussion_time_mins": 15,
    "prep_time_mins":       10,
    "feedback_length":      12,
    "num_competencies":      2,
    "form_type":            "Mini-CEX",
    "assessor_seniority":   "senior",
    "setting":              "emergency"
}

sample_df   = pd.DataFrame([sample_trainee])
sample_proc = preprocessor.transform(sample_df)
risk_prob   = float(rf_model.predict_proba(sample_proc)[0][0])
risk_score  = round(risk_prob * 100, 1)
tier        = ("critical" if risk_score >= 75 else
               "high"     if risk_score >= 50 else
               "moderate" if risk_score >= 25 else "low")

print(f"\n  Sample Trainee Input:")
for k, v in sample_trainee.items():
    print(f"    {k:<30}: {v}")
print(f"\n  API Response:")
print(f"    risk_score  : {risk_score}")
print(f"    risk_tier   : {tier}")
print(f"    prediction  : {'at_risk' if risk_prob > 0.5 else 'on_track'}")

print("\n" + "=" * 60)
print("COMPONENT 3 COMPLETE")
print("\nFiles ready for your GitHub repo:")
print("  ✅ data_preprocessing.py   (Component 1 — teammate)")
print("  ✅ model_training.py        (Component 2)")
print("  ✅ evaluate_model.py        (Component 3)")
print("  ✅ app.py                   (Component 3 — API)")
print("  ✅ model.pkl                (trained model)")
print("  ✅ hprime_preprocessor.pkl  (data pipeline)")
print("  ✅ evaluation_plots.png     (for report & presentation)")
print("=" * 60)
