"""
Extract R², RMSE, MAE from trained model
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load model and data
model = joblib.load("genus_hybprop_model.pkl")
df = pd.read_csv("genus_processed.csv")

print("Calculating metrics...")
print("=" * 40)

# Predict on full dataset
y_true = df["HybProp"].values
X = df.drop(columns=["HybProp", "Hyb_Ratio"])
y_pred = model.predict(X)

# Compute metrics
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"R² Score:   {r2:.4f}")
print(f"RMSE:       {rmse:.4f}")
print(f"MAE:        {mae:.4f}")
print(f"Range:     [{y_true.min():.2f}, {y_true.max():.2f}]")
