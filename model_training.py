# =============================================================================
# CODE COMPONENT 2: The Machine Learning Core Developer
# CSCI323 Modern Artificial Intelligence — Spring 2026
# University of Wollongong in Dubai
#
# PURPOSE: Takes the cleaned & preprocessed data from Component 1 and builds,
#          tunes, and saves the best predictive model for at-risk trainee detection.
#
# HOW TO RUN: Paste this into a new Colab cell and run it directly after
#             Component 1. All variables from Component 1 are reused.
# =============================================================================

# ── Step 1: Install SMOTE (run once per Colab session) ───────────────────────
# SMOTE = Synthetic Minority Over-sampling Technique
# Fixes the class imbalance: 1,102 passes vs only 54 fails in training data
import subprocess
subprocess.run(["pip", "install", "imbalanced-learn", "-q"])

# ── Step 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, recall_score)

print("=" * 60)
print("COMPONENT 2: Model Training")
print("=" * 60)

# ── Step 3: Verify Component 1 output is available ───────────────────────────
# These variables were created in Component 1
# If you're running fresh, re-run Component 1 first
try:
    print(f"\nUsing data from Component 1:")
    print(f"  X_train shape : {X_train_processed.shape}")
    print(f"  X_test shape  : {X_test_processed.shape}")
    print(f"  Train labels  : {dict(y_train.value_counts())} (0=fail, 1=pass)")
    print(f"  Test labels   : {dict(y_test.value_counts())} (0=fail, 1=pass)")
except NameError:
    raise RuntimeError("Component 1 variables not found. Please run Component 1 first.")

# ── Step 4: Apply SMOTE to balance training data ──────────────────────────────
# SMOTE synthetically generates new minority-class examples (fails) by
# interpolating between existing fail cases — like a teacher creating
# practice exam questions based on the types of errors students already made.
#
# We only apply SMOTE to TRAINING data. Test data stays untouched so our
# evaluation reflects real-world class distribution.

print("\n── Step 4: Applying SMOTE ──")
print(f"  Before SMOTE → Pass: {(y_train==1).sum()}, Fail: {(y_train==0).sum()}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

print(f"  After SMOTE  → Pass: {(y_train_balanced==1).sum()}, Fail: {(y_train_balanced==0).sum()}")
print(f"  New training shape: {X_train_balanced.shape}")

# ── Step 5: Baseline Model — Logistic Regression ─────────────────────────────
# Logistic Regression is the "ruler" baseline. Before building a complex model,
# we need to know what a simple model can already achieve. If our Random Forest
# is only 1% better, the complexity isn't worth it.

print("\n── Step 5: Training Baseline (Logistic Regression) ──")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'   # extra safety on top of SMOTE
)
lr_model.fit(X_train_balanced, y_train_balanced)

lr_preds = lr_model.predict(X_test_processed)
lr_recall = recall_score(y_test, lr_preds, pos_label=0)
lr_f1     = f1_score(y_test, lr_preds, pos_label=0)
lr_auc    = roc_auc_score(y_test, lr_model.predict_proba(X_test_processed)[:, 0])

print(f"  Logistic Regression Results (failing class = 0):")
print(f"    Recall    : {lr_recall:.3f}  ← most important metric (catching at-risk trainees)")
print(f"    F1 Score  : {lr_f1:.3f}")
print(f"    AUC-ROC   : {lr_auc:.3f}")

# ── Step 6: Main Model — Random Forest with GridSearchCV ─────────────────────
# Random Forest is like asking 100 different doctors to independently assess
# a patient, then taking a majority vote. It's robust, handles mixed feature
# types well, and gives us feature importances to explain predictions.
#
# GridSearchCV tries every combination of hyperparameters systematically —
# like a chef trying every combination of ingredient quantities to find the
# best recipe.

print("\n── Step 6: Training Main Model (Random Forest + GridSearchCV) ──")
print("  This may take 1-2 minutes...")

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators':      [100, 200, 300],    # how many trees in the forest
    'max_depth':         [5, 10, 15, None],  # how deep each tree grows
    'min_samples_split': [2, 5, 10],         # min samples needed to split a node
    'class_weight':      ['balanced']        # automatically handle class imbalance
}

rf_base = RandomForestClassifier(random_state=42)

# StratifiedKFold ensures each fold has proportional class representation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator  = rf_base,
    param_grid = param_grid,
    cv         = cv,
    scoring    = 'recall',    # optimise for catching failing trainees
    n_jobs     = -1,          # use all CPU cores
    verbose    = 1
)

grid_search.fit(X_train_balanced, y_train_balanced)

best_rf = grid_search.best_estimator_
print(f"\n  Best Parameters found: {grid_search.best_params_}")

# Evaluate best RF on test set
rf_preds  = best_rf.predict(X_test_processed)
rf_proba  = best_rf.predict_proba(X_test_processed)[:, 0]  # probability of failing
rf_recall = recall_score(y_test, rf_preds, pos_label=0)
rf_f1     = f1_score(y_test, rf_preds, pos_label=0)
rf_auc    = roc_auc_score(y_test, rf_proba)

print(f"\n  Random Forest Results (failing class = 0):")
print(f"    Recall    : {rf_recall:.3f}")
print(f"    F1 Score  : {rf_f1:.3f}")
print(f"    AUC-ROC   : {rf_auc:.3f}")

# Full classification report
print(f"\n  Full Classification Report:")
print(classification_report(y_test, rf_preds, target_names=["At-Risk (Fail)", "On-Track (Pass)"]))

# ── Step 7: Cross-Validation on balanced training data ───────────────────────
# Cross-validation checks if the model's performance is consistent or just
# got lucky with a particular train/test split — like retaking a test 5 times
# to confirm you genuinely understand the material.

print("── Step 7: Cross-Validation (5-Fold) ──")
cv_scores = cross_val_score(
    best_rf, X_train_balanced, y_train_balanced,
    cv=cv, scoring='recall', n_jobs=-1
)
print(f"  CV Recall Scores: {[round(s,3) for s in cv_scores]}")
print(f"  Mean CV Recall  : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ── Step 8: Feature Importances ───────────────────────────────────────────────
# Which features does the model rely on most? This tells us what actually
# predicts at-risk trainees — critical for the report's findings section.

print("\n── Step 8: Feature Importances ──")
feature_names = (
    numeric_cols +
    list(preprocessor.named_transformers_['cat']
         .get_feature_names_out(categorical_cols))
)
importances = pd.Series(
    best_rf.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("  Top 10 Most Important Features:")
for feat, imp in importances.head(10).items():
    bar = "█" * int(imp * 100)
    print(f"    {feat:<35} {imp:.4f}  {bar}")

# ── Step 9: Compare Baseline vs Main Model ────────────────────────────────────
print("\n── Step 9: Model Comparison Summary ──")
print(f"  {'Metric':<15} {'Logistic Regression':>22} {'Random Forest':>18}")
print(f"  {'─'*55}")
print(f"  {'Recall (Fail)':<15} {lr_recall:>22.3f} {rf_recall:>18.3f}")
print(f"  {'F1 (Fail)':<15} {lr_f1:>22.3f} {rf_f1:>18.3f}")
print(f"  {'AUC-ROC':<15} {lr_auc:>22.3f} {rf_auc:>18.3f}")

winner = "Random Forest" if rf_recall >= lr_recall else "Logistic Regression"
print(f"\n  Selected Model: {winner} (higher recall on failing class)")

# ── Step 10: Save model artifacts ─────────────────────────────────────────────
best_model = best_rf if rf_recall >= lr_recall else lr_model

joblib.dump(best_model,   'model.pkl')
joblib.dump(best_rf,      'random_forest_model.pkl')
joblib.dump(lr_model,     'logistic_regression_model.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

# Also save the processed test data for Component 3
joblib.dump(X_test_processed, 'X_test_processed.pkl')
joblib.dump(y_test,           'y_test.pkl')
joblib.dump(rf_proba,         'rf_proba.pkl')
joblib.dump(lr_model.predict_proba(X_test_processed)[:, 0], 'lr_proba.pkl')

print("\n── Step 10: Saved Model Files ──")
print("  model.pkl                  ← best overall model")
print("  random_forest_model.pkl    ← random forest (main)")
print("  logistic_regression_model.pkl ← baseline")
print("  feature_names.pkl          ← feature list for API")
print("  X_test_processed.pkl       ← test features for Component 3")
print("  y_test.pkl                 ← test labels for Component 3")
print("  rf_proba.pkl               ← RF probabilities for ROC curve")
print("  lr_proba.pkl               ← LR probabilities for ROC curve")

print("\n" + "=" * 60)
print("COMPONENT 2 COMPLETE")
print("Run Component 3 next for evaluation plots and API.")
print("=" * 60)
