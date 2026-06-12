# -*- coding: utf-8 -*-
"""
Vancouver Crime GIS Project
Crime Risk Prediction Mapping

Author: Uzair

Description:
    This script loads the trained crime risk model, applies predictions
    to the ML dataset, attaches predicted risk classes back to the grid,
    and visualizes the predicted crime risk map for Vancouver.

    It also plots the top 15 most important model features.

Inputs:
    - ml_data.csv
    - grid.geojson
    - crime_model.pkl
    - feature_columns.pkl

Outputs:
    - Predicted crime risk map
    - Top 15 feature importances chart
"""

# ============================================================
# Libraries
# ============================================================

from pathlib import Path

import contextily as ctx
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


# ============================================================
# File Paths
# ============================================================

OUTPUT_DIR = Path("D:/Programming/MyPythonProjects/GIS/Project4/outputs")

ML_DATA_PATH = OUTPUT_DIR / "ml_data.csv"
GRID_PATH = OUTPUT_DIR / "grid.geojson"
MODEL_PATH = OUTPUT_DIR / "crime_model.pkl"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.pkl"

PREDICTED_RISK_MAP_PATH = OUTPUT_DIR / "predicted_crime_risk_map.png"

# ============================================================
# Load Data
# ============================================================

ml_data = pd.read_csv(ML_DATA_PATH)
grid = gpd.read_file(GRID_PATH)


# ============================================================
# Load Model and Feature Schema
# ============================================================

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_COLUMNS_PATH)


# ============================================================
# Prepare Prediction Features
# ============================================================

# Rebuild model input using the saved feature schema.
# Any missing columns are filled with 0 to match training structure.
X = ml_data.reindex(columns=features, fill_value=0)


# ============================================================
# Predict Risk Classes
# ============================================================

ml_data["predicted_risk"] = model.predict(X)


# ============================================================
# Attach Predictions to Grid
# ============================================================

grid["predicted_risk"] = ml_data["predicted_risk"]


# ============================================================
# Convert to Web Mercator for Basemap
# ============================================================

grid_web = grid.to_crs(epsg=3857)


# ============================================================
# Define Categorical Risk Colors
# ============================================================

# 0 = Low Risk
# 1 = Medium Risk
# 2 = High Risk
cmap = ListedColormap(["green", "yellow", "red"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)


# ============================================================
# Plot Predicted Risk Map
# ============================================================

fig, ax = plt.subplots(figsize=(18, 10))

grid_web.plot(
    column="predicted_risk",
    cmap=cmap,
    norm=norm,
    ax=ax,
    alpha=0.65,
    edgecolor="black",
    linewidth=0.2
)

ctx.add_basemap(
    ax,
    source=ctx.providers.OpenStreetMap.Mapnik
)

legend_items = [
    Patch(facecolor="green", edgecolor="black", label="Low Risk"),
    Patch(facecolor="yellow", edgecolor="black", label="Medium Risk"),
    Patch(facecolor="red", edgecolor="black", label="High Risk")
]

ax.legend(
    handles=legend_items,
    title="Predicted Risk",
    loc="lower left"
)

ax.set_title(
    "Vancouver Crime Risk Prediction Map",
    fontsize=16,
    fontweight="bold"
)

ax.set_axis_off()
plt.tight_layout()

fig.savefig(
    PREDICTED_RISK_MAP_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Diagnostics
# ============================================================

print("\nPredicted Risk Class Distribution:")
print(ml_data["predicted_risk"].value_counts().sort_index())

print("\nPrediction mapping completed successfully!")