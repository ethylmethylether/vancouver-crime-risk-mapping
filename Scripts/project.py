# -*- coding: utf-8 -*-
"""
Vancouver Crime GIS Project

Author: Uzair
Student Email: m.khalid@student.fdu.edu
Institution: Fairleigh Dickinson University, Vancouver Campus
Program: Bachelor of Information Technology

Project Type:
    GIS / Spatial Crime Analysis / Portfolio Project

Description:
    This script analyzes Vancouver crime incident locations.
    It converts crime records into spatial points, creates a 250m grid,
    counts crimes per grid cell, assigns neighbourhood names, visualizes
    crime density, and saves GIS-ready output files.

Data Sources:
    - City of Vancouver Crime Dataset
    - City of Vancouver Local Area Boundary GeoJSON

Coordinate Reference Systems:
    - Analysis CRS: EPSG:26910  UTM Zone 10N
    - Web Map CRS: EPSG:3857   Web Mercator

Outputs:
    - crime_points.geojson
    - neighbourhoods.geojson
    - crime_grid.geojson
    - grid.geojson
    - crime_point_distribution.png
    - crime_density_heatmap.png
"""

# ============================================================
# Libraries
# ============================================================

from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np

from shapely.geometry import Point, box


# ============================================================
# File Paths and Project Settings
# ============================================================

CRIME_PATH = Path("D:/Programming/MyPythonProjects/GIS/Project4/Data/crime.csv")

NEIGHBOURHOODS_PATH = Path(
    "D:/Programming/MyPythonProjects/GIS/Project4/Data/local-area-boundary.geojson"
)

OUTPUT_DIR = Path("D:/Programming/MyPythonProjects/GIS/Project4/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_CRS = "EPSG:26910"
WEB_CRS = "EPSG:3857"

CELL_SIZE = 250  # metres

CRIME_POINT_PLOT_PATH = OUTPUT_DIR / "crime_point_distribution.png"
CRIME_HEATMAP_PLOT_PATH = OUTPUT_DIR / "crime_density_heatmap.png"


# ============================================================
# Load Data
# ============================================================

crime = pd.read_csv(CRIME_PATH)
neighbourhoods = gpd.read_file(NEIGHBOURHOODS_PATH)


# ============================================================
# Clean Crime Data
# ============================================================

crime = crime.dropna(subset=["X", "Y"])
crime = crime[(crime["X"] != 0) & (crime["Y"] != 0)]


# ============================================================
# Convert Crime Data to GeoDataFrame
# ============================================================

geometry = [Point(xy) for xy in zip(crime["X"], crime["Y"])]

crime_gdf = gpd.GeoDataFrame(
    crime,
    geometry=geometry,
    crs=ANALYSIS_CRS
)


# ============================================================
# CRS Alignment
# ============================================================

neighbourhoods = neighbourhoods.to_crs(ANALYSIS_CRS)


# ============================================================
# Plot 1: Observed Crime Point Distribution
# ============================================================

crime_web = crime_gdf.to_crs(WEB_CRS)
neighbourhoods_web = neighbourhoods.to_crs(WEB_CRS)

fig, ax = plt.subplots(figsize=(14, 10))

neighbourhoods_web.plot(
    ax=ax,
    color="none",
    edgecolor="black",
    linewidth=0.8,
    alpha=0.9,
    zorder=2
)

crime_web.plot(
    ax=ax,
    markersize=0.6,
    color="red",
    alpha=0.30,
    zorder=3
)

ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.Positron,
    zorder=1
)

ax.set_title(
    "Observed Crime Point Distribution in Vancouver",
    fontsize=16,
    fontweight="bold"
)

ax.set_axis_off()
plt.tight_layout()

fig.savefig(
    CRIME_POINT_PLOT_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Create Spatial Grid
# ============================================================

minx, miny, maxx, maxy = crime_gdf.total_bounds

x_coords = np.arange(minx, maxx, CELL_SIZE)
y_coords = np.arange(miny, maxy, CELL_SIZE)

grid_cells = [
    box(x, y, x + CELL_SIZE, y + CELL_SIZE)
    for x in x_coords
    for y in y_coords
]

grid = gpd.GeoDataFrame(
    {"geometry": grid_cells},
    crs=crime_gdf.crs
)


# ============================================================
# Clip Grid to Vancouver Boundary
# ============================================================

# Dissolve neighbourhoods first so the grid is clipped to Vancouver
# as one city boundary instead of being split by each neighbourhood.
vancouver_boundary = neighbourhoods.dissolve().reset_index(drop=True)

grid = gpd.overlay(
    grid,
    vancouver_boundary[["geometry"]],
    how="intersection"
)

# Create clean unique grid IDs.
grid = grid.reset_index(drop=True)
grid["grid_id"] = grid.index


# ============================================================
# Spatial Join: Crimes to Grid
# ============================================================

crime_grid = gpd.sjoin(
    crime_gdf,
    grid[["grid_id", "geometry"]],
    how="left",
    predicate="within"
)

missing_grid = crime_grid["grid_id"].isna().sum()
print("Crimes not assigned to any grid:", missing_grid)


# ============================================================
# Count Crimes Per Grid Cell
# ============================================================

grid_counts = crime_grid.groupby("grid_id").size()

grid["crime_count"] = grid["grid_id"].map(grid_counts).fillna(0)


# ============================================================
# Assign One Neighbourhood Per Grid Cell
# ============================================================

# representative_point() is used instead of centroid because it is
# guaranteed to fall inside the polygon.
grid_points = grid.copy()
grid_points["geometry"] = grid_points.geometry.representative_point()

grid_neigh = gpd.sjoin(
    grid_points[["grid_id", "geometry"]],
    neighbourhoods[["name", "geometry"]],
    how="left",
    predicate="within"
)

# Keep one neighbourhood name per grid cell.
grid_neigh = grid_neigh[["grid_id", "name"]].drop_duplicates("grid_id")

grid = grid.merge(
    grid_neigh,
    on="grid_id",
    how="left"
)

grid["name"] = grid["name"].fillna("Unknown")


# ============================================================
# Plot 2: Crime Density Heatmap
# ============================================================

grid_web = grid.to_crs(WEB_CRS)
crime_web = crime_gdf.to_crs(WEB_CRS)
neighbourhoods_web = neighbourhoods.to_crs(WEB_CRS)

# Cap colour scale at the 95th percentile so one extreme hotspot
# does not dominate the whole map.
vmax = grid_web["crime_count"].quantile(0.95)

fig, ax = plt.subplots(figsize=(18, 10))

grid_web.plot(
    column="crime_count",
    cmap="Reds",
    legend=True,
    ax=ax,
    edgecolor="gray",
    linewidth=0.15,
    alpha=0.70,
    vmin=0,
    vmax=vmax,
    zorder=2,
    legend_kwds={
        "label": f"Crime Count per {CELL_SIZE}m Grid Cell",
        "shrink": 0.75
    }
)

neighbourhoods_web.plot(
    ax=ax,
    color="none",
    edgecolor="black",
    linewidth=0.8,
    alpha=0.8,
    zorder=3
)

crime_web.plot(
    ax=ax,
    markersize=0.2,
    color="black",
    alpha=0.25,
    zorder=4
)

ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.Positron,
    zorder=1
)

ax.set_title(
    f"Observed Crime Density by {CELL_SIZE}m Grid Cell, Vancouver",
    fontsize=18,
    fontweight="bold"
)

ax.set_axis_off()
plt.tight_layout()

fig.savefig(
    CRIME_HEATMAP_PLOT_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Save Outputs
# ============================================================

# Remove spatial join index column before saving, if it exists.
crime_grid = crime_grid.drop(columns=["index_right"], errors="ignore")

crime_gdf.to_file(OUTPUT_DIR / "crime_points.geojson", driver="GeoJSON")
neighbourhoods.to_file(OUTPUT_DIR / "neighbourhoods.geojson", driver="GeoJSON")
crime_grid.to_file(OUTPUT_DIR / "crime_grid.geojson", driver="GeoJSON")
grid.to_file(OUTPUT_DIR / "grid.geojson", driver="GeoJSON")


# ============================================================
# Diagnostics
# ============================================================

print("\nCrime Count Summary:")
print(grid["crime_count"].describe())

print("\nTop 10 Crime Grid Cells:")
print(
    grid[["grid_id", "crime_count", "name"]]
    .sort_values("crime_count", ascending=False)
    .head(10)
)

print("\nFiles saved successfully!")
