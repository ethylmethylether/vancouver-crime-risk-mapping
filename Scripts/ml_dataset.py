# -*- coding: utf-8 -*-
"""
Vancouver Crime GIS Project
ML Dataset Preparation

Author: Uzair

Description:
    This script prepares a machine learning dataset from the processed
    Vancouver crime grid.

    It loads the crime grid and crime-to-grid spatial join output,
    creates spatial and temporal features, one-hot encodes neighbourhoods,
    creates risk classes, and saves the final ML-ready CSV file.

Inputs:
    - grid.geojson
    - crime_grid.geojson

Outputs:
    - ml_data.csv
    - feature_columns.csv
"""

# ============================================================
# Libraries
# ============================================================

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import contextily as ctx


# ============================================================
# File Paths and Project Settings
# ============================================================

OUTPUT_DIR = Path("D:/Programming/MyPythonProjects/GIS/Project4/outputs")

GRID_PATH = OUTPUT_DIR / "grid.geojson"
CRIME_GRID_PATH = OUTPUT_DIR / "crime_grid.geojson"

ML_DATA_PATH = OUTPUT_DIR / "ml_data.csv"
FEATURE_COLUMNS_PATH = OUTPUT_DIR / "feature_columns.csv"

NEIGHBOUHOODS_PATH = OUTPUT_DIR / "neighbourhoods.geojson"

PLOT1 = OUTPUT_DIR / "night_crime_ratio_map.png"
PLOT2 = OUTPUT_DIR / "Average_pattern_risk.png"

# ============================================================
# Load Data
# ============================================================

grid = gpd.read_file(GRID_PATH)
crime_grid = gpd.read_file(CRIME_GRID_PATH)
neighbourhoods = gpd.read_file(NEIGHBOUHOODS_PATH)

# ============================================================
# Prepare Base ML Dataset
# ============================================================

grid = grid.reset_index(drop=True)
ml_data = grid.copy()

# Keep neighbourhood name in a separate column before one-hot encoding.
ml_data["neighbourhoods"] = ml_data["name"]


# ============================================================
# Spatial Feature Engineering
# ============================================================

# Approximate city center reference point in EPSG:26910.
city_center_x = 491000
city_center_y = 5459000

# Use grid cell centroids to create spatial features.
ml_data["centroid"] = ml_data.geometry.centroid

ml_data["x"] = ml_data["centroid"].x
ml_data["y"] = ml_data["centroid"].y

ml_data["distance_to_center"] = np.sqrt(
    (ml_data["x"] - city_center_x) ** 2 +
    (ml_data["y"] - city_center_y) ** 2
)


# ============================================================
# Temporal Feature Engineering
# ============================================================

# Remove crime records that were not assigned to a grid cell.
crime_grid = crime_grid.dropna(subset=["grid_id"]).copy()
crime_grid["grid_id"] = crime_grid["grid_id"].astype(int)

# Make sure HOUR is numeric.
crime_grid["HOUR"] = pd.to_numeric(crime_grid["HOUR"], errors="coerce")

# Average crime hour per grid cell.
avg_hour = crime_grid.groupby("grid_id")["HOUR"].mean()

# Night crime ratio:
# 1 = crime happened between 8 PM and 5 AM.
crime_grid["night_crime"] = (
    (crime_grid["HOUR"] >= 20) |
    (crime_grid["HOUR"] <= 5)
).astype(int)

night_ratio = crime_grid.groupby("grid_id")["night_crime"].mean()

# Create date column for weekend calculation.
crime_grid["date"] = pd.to_datetime(
    crime_grid[["YEAR", "MONTH", "DAY"]]
    .rename(columns={
        "YEAR": "year",
        "MONTH": "month",
        "DAY": "day"
    }),
    errors="coerce"
)

# Weekend ratio:
# 1 = Saturday or Sunday.
crime_grid["weekend"] = (
    crime_grid["date"].dt.dayofweek >= 5
).astype(int)

weekend_ratio = crime_grid.groupby("grid_id")["weekend"].mean()

# Combine temporal features into one table.
temporal_features = pd.DataFrame({
    "avg_hour": avg_hour,
    "night_ratio": night_ratio,
    "weekend_ratio": weekend_ratio
}).reset_index()


# ============================================================
# Merge Temporal Features with ML Dataset
# ============================================================

ml_data = ml_data.merge(
    temporal_features,
    on="grid_id",
    how="left"
)

# Fill empty temporal values for grid cells with no crimes.
ml_data[["avg_hour", "night_ratio", "weekend_ratio"]] = (
    ml_data[["avg_hour", "night_ratio", "weekend_ratio"]]
    .fillna(0)
)


# ============================================================
# One-Hot Encode Neighbourhoods
# ============================================================

ml_data = pd.get_dummies(
    ml_data,
    columns=["neighbourhoods"],
    drop_first=True
)


# ============================================================
# Create Risk Classes
# ============================================================

low_cutoff = ml_data["crime_count"].quantile(0.60)
high_cutoff = ml_data["crime_count"].quantile(0.75)

ml_data["risk"] = np.where(
    ml_data["crime_count"] <= low_cutoff,
    0,
    np.where(
        ml_data["crime_count"] <= high_cutoff,
        1,
        2
    )
)

print("Low/Medium cutoff:", low_cutoff)
print("Medium/High cutoff:", high_cutoff)

print("\nRisk class distribution:")
print(ml_data["risk"].value_counts().sort_index())


# ============================================================
# Plotting Using Calculated Data
# ============================================================

# -------------------- Night Crime Ratio Map --------------------
grid_plot = grid.copy()

grid_plot["night_ratio"] = ml_data["night_ratio"].values
grid_plot_web = grid_plot.to_crs(epsg=3857)
neighbourhoods_web = neighbourhoods.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(18, 10))

grid_plot_web.plot(
    column="night_ratio",
    cmap="Purples",
    legend=True,
    ax=ax,
    edgecolor="gray",
    linewidth=0.15,
    alpha=0.75,
    vmin=0,
    vmax=1,
    legend_kwds={
        "label": "Night Crime Ratio",
        "shrink": 0.75
    }
)

neighbourhoods_web.plot(
    ax=ax,
    color="none",
    edgecolor="black",
    linewidth=0.8
)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

ax.set_title("Night-Time Crime Ratio by 250m Grid Cell, Vancouver", fontsize=18, fontweight="bold")
ax.set_axis_off()
plt.tight_layout()

'''fig.savefig(
    PLOT1,
    dpi=600,
    bbox_inches="tight"
)'''

plt.show()

# -------------------- Average Temporal Features by Risk Class --------------------
risk_summary = ml_data.groupby("risk")[
    ["night_ratio", "weekend_ratio", "avg_hour"]
].mean()

risk_summary.index = ["Low Risk", "Medium Risk", "High Risk"]

print(risk_summary)

#fig, ax = plt.subplots(figsize=(10, 6))
risk_summary[["night_ratio", "weekend_ratio"]].plot(
    kind="bar",
)

plt.title("Average Night and Weekend Crime Ratios by Risk Class")
plt.ylabel("Average Ratio")
plt.xlabel("Risk Class")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

'''fig.savefig(
    PLOT2,
    dpi=600,
    bbox_inches="tight"
)'''

plt.show()

# ============================================================
# Clean Columns Before Saving
# ============================================================

ml_data = ml_data.drop(
    columns=[
        "geometry",
        "centroid",
        "geo_point_2d",
        "name"
    ],
    errors="ignore"
)


# ============================================================
# Save ML Dataset
# ============================================================

ml_data.to_csv(ML_DATA_PATH, index=False)


# ============================================================
# Save Feature Columns
# ============================================================

feature_columns = (
    ml_data
    .select_dtypes(include=[np.number])
    .columns
    .drop(["crime_count", "risk", "grid_id"])
)

pd.Series(feature_columns).to_csv(
    FEATURE_COLUMNS_PATH,
    index=False
)


# ============================================================
# Final Confirmation
# ============================================================

print("\nML dataset saved successfully!")
print("Feature column schema saved successfully!")