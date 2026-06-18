# At-Risk Junior Doctor Detection: hPrime

CSCI323 Modern Artificial Intelligence, Spring 2026
University of Wollongong in Dubai

## Overview
Machine learning pipeline that predicts which junior doctors are at risk 
of falling below assessment standards, using historical Mini-CEX and 
CBD assessment data from hPrime (Imzans).

## Files
- `preprocess.py` - Raw hPrime export parsing pipeline
- `eda.py` - Exploratory data analysis (7 figures)
- `model_training.py` - Component 2: Model training (Logistic Regression + Random Forest, SMOTE, GridSearchCV)
- `evaluate_and_api.py` - Component 3: Evaluation script (Colab version)
- `evaluate_model.py` - Component 3: Standalone evaluation script
- `app.py` - Component 3: Flask API for real-time risk scoring
- `hprime_clean.csv` - Cleaned dataset (1,445 records, 120 candidates)
- `CSCI323_Project.ipynb` - Full Colab notebook (all 3 components, run end-to-end)
- `model.pkl` - Final selected model
- `random_forest_model.pkl` - Trained Random Forest
- `logistic_regression_model.pkl` - Trained baseline model
- `hprime_preprocessor.pkl` - Fitted ColumnTransformer (scaling + encoding)
- `feature_names.pkl` - Feature names for API integration
- `y_test.pkl`, `rf_proba.pkl` - Saved test labels and prediction probabilities
- `evaluation_plots.png` - Confusion matrices, ROC curve, PR curve, feature importances

## How to Run
1. Run `preprocess.py` and `model_training.py` in sequence
2. Run `evaluate_and_api.py` for evaluation plots and API generation
3. To start the API: `python app.py`

## Results
- Random Forest Recall (At-Risk class): 0.875
- AUC-ROC: 0.985
