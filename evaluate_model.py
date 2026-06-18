"""
evaluate_model.py — Standalone Model Evaluation Script
CSCI323 Modern Artificial Intelligence — Spring 2026
University of Wollongong in Dubai

Run: python evaluate_model.py
Requires: model.pkl, hprime_preprocessor.pkl, hprime_clean.csv
"""
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, recall_score,
    precision_score, accuracy_score, roc_curve, auc
)

print("Loading model and preprocessor...")
model        = joblib.load("model.pkl")
preprocessor = joblib.load("hprime_preprocessor.pkl")

df = pd.read_csv("hprime_clean.csv")
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
numeric_cols     = X.select_dtypes(include=["float64","int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
X["days_since_last"] = X["days_since_last"].fillna(0)
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_proc = preprocessor.transform(X_test)
preds       = model.predict(X_test_proc)
fail_proba  = model.predict_proba(X_test_proc)[:, 0]
fpr, tpr, _ = roc_curve(y_test, fail_proba, pos_label=0)

print("\n" + "="*55)
print("MODEL EVALUATION REPORT")
print("At-Risk Junior Doctor Detection — hPrime Dataset")
print("="*55)
print(classification_report(y_test, preds,
      target_names=["At-Risk (Fail)","On-Track (Pass)"]))
print(f"AUC-ROC : {auc(fpr, tpr):.3f}")
print(f"Recall  : {recall_score(y_test, preds, pos_label=0):.3f}")
print(f"F1-Score: {f1_score(y_test, preds, pos_label=0, zero_division=0):.3f}")
