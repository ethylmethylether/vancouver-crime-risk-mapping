# -*- coding: utf-8 -*-
"""
Vancouver Crime GIS Project
Machine Learning Model Training

Author: Uzair

Description:
    This script trains a Random Forest classifier using the ML-ready
    Vancouver crime grid dataset.

    The model predicts crime risk class for each grid cell using spatial,
    temporal, and neighbourhood-based features.

Inputs:
    - ml_data.csv

Outputs:
    - crime_model.pkl
    - feature_columns.pkl
"""

# ============================================================
# Libraries
# ============================================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# File Paths
# ============================================================

OUTPUT_DIR = Path("D:/Programming/MyPythonProjects/GIS/Project4/outputs")

DATA_PATH = OUTPUT_DIR / "ml_data.csv"
MODEL_PATH = OUTPUT_DIR / "crime_model.pkl"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.pkl"



# ============================================================
# Load ML Dataset
# ============================================================

ml_data = pd.read_csv(DATA_PATH)


# ============================================================
# Prepare Features and Target
# ============================================================

# Target column:
# 0 = Low risk
# 1 = Medium risk
# 2 = High risk
y = ml_data["risk"]

# Feature columns:
# Remove columns that should not be used as model inputs.
X = ml_data.drop(
    columns=[
        "crime_count",
        "risk",
        "grid_id"
    ],
    errors="ignore"
)

# Ensure all feature columns are numeric.
X = X.select_dtypes(include=["number", "bool"])

# Fill any remaining missing values.
X = X.fillna(0)


# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# Train Random Forest Model
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# ============================================================
# Model Prediction
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# Model Evaluation
# ============================================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

accuracy = model.score(X_test, y_test)
print("Accuracy:", round(accuracy*100, 3))


# ============================================================
# Save Model and Feature Columns
# ============================================================

joblib.dump(model, MODEL_PATH)
joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)

print("\nModel saved successfully!")
print("Feature columns saved successfully!")
