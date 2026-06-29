# ==========================================
# Yield Prediction Model
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error
)
# ==========================================
# Load Dataset
# ==========================================

FILE_NAME = "reactor_data.xlsx"

df = pd.read_excel(FILE_NAME)

print("\nDataset Loaded Successfully!")
print("-" * 50)
print("Number of Rows    :", len(df))
print("Number of Columns :", len(df.columns))

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe(include="all"))
# ==========================================
# Define Features and Target
# ==========================================

FEATURES = [
    "Temperature (°C)",
    "Pressure (bar)",
    "Concentration (mol/L)",
    "Catalyst (wt%)",
    "Residence Time (min)",
    "Reactor Type"
]

TARGET = "Yield (%)"

# Keep only required columns
df = df[FEATURES + [TARGET]].copy()

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

print("\nData after cleaning:")
print(df.head())

print("\nNumber of records after cleaning:", len(df))

# ==========================================
# Separate Inputs (X) and Output (y)
# ==========================================

X = df[FEATURES]
y = df[TARGET]

print("\nInput Shape :", X.shape)
print("Output Shape:", y.shape)
# ==========================================
# Data Preprocessing
# ==========================================

numeric_features = [
    "Temperature (°C)",
    "Pressure (bar)",
    "Concentration (mol/L)",
    "Catalyst (wt%)",
    "Residence Time (min)"
]

categorical_features = [
    "Reactor Type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

print("\nPreprocessor created successfully!")

# ==========================================
# Models to Compare
# ==========================================

models = {

    "Ridge": Ridge(alpha=5),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "Hist Gradient Boosting":
        HistGradientBoostingRegressor(
            random_state=42
        )
}

print("\nModels loaded successfully!")
# ==========================================
# Cross Validation
# ==========================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Cross-validation ready!")

# ==========================================
# Model Comparison
# ==========================================

best_model = None
best_name = None
best_r2 = -999

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    preds = cross_val_predict(
        pipe,
        X,
        y,
        cv=kf
    )

    r2 = r2_score(y, preds)

    mae = mean_absolute_error(
        y,
        preds
    )

    print(
        f"{name:25s} "
        f"R² = {r2:.3f}   "
        f"MAE = {mae:.2f}"
    )

    if r2 > best_r2:
        best_r2 = r2
        best_model = pipe
        best_name = name

print("\nBest Model :", best_name)
print("Best R²    :", round(best_r2, 3))
# ==========================================
# Save Clean Dataset
# ==========================================

df.to_csv("yield_clean.csv", index=False)

print("\nClean dataset saved as yield_clean.csv")

# ==========================================
# Train Best Model on Entire Dataset
# ==========================================

best_model.fit(X, y)

print("\nBest model trained on complete dataset.")
# ==========================================
# Save Model
# ==========================================

joblib.dump(
    {
        "model": best_model,
        "features": FEATURES,
        "target": TARGET,
        "best_model": best_name,
        "cv_r2": float(best_r2),
        "cv_mae": float(
            mean_absolute_error(
                y,
                cross_val_predict(best_model, X, y, cv=kf)
            )
        ),
        "y_mean": float(y.mean()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "feat_ranges": {
            col: (
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].median())
            )
            for col in numeric_features
        },
        "categories": {
            "Reactor Type": sorted(df["Reactor Type"].unique())
        }
    },
    "yield_model.joblib"
)

print("\nModel saved as yield_model.joblib")
