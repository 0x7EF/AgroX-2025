"""
Train a RandomForest model on genus_processed.csv to predict Hybridization Propensityr with an 80/20 split.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = Path("genus_processed.csv")

def main():
    # 1. Load already-processed data (no extra cleaning)
    df = pd.read_csv(DATA_PATH)
    print(f"Data shape (rows, cols): {df.shape}")

    # 2. Split features/target
    y = df["HybProp"].values
    X = df.drop(columns=["HybProp", "Hyb_Ratio"])

    # Keep Genus for reporting only
    genus_names = df["Genus"].values

    numeric_cols = [c for c in X.columns if c not in ["Genus", "Family", "Order"]]
    cat_cols = ["Family", "Order"]

    # 3. Train/test split (80/20)
    X_train, X_test, y_train, y_test, genus_train, genus_test = train_test_split(
        X, y, genus_names, test_size=0.2, random_state=42
    )

    # 4. Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # 5. RandomForest model
    rf = RandomForestRegressor(
        n_estimators=760,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )

    rf_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", rf),
    ])

    # Train and evaluate on 20% test set
    rf_pipe.fit(X_train, y_train)
    y_pred = rf_pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"RandomForest: R2={r2:.4f}, RMSE={rmse:.4f}")

    joblib.dump(rf_pipe, "genus_hybprop_model.pkl")


if __name__ == "__main__":
    main()