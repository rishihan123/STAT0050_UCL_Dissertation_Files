from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pathlib import Path


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
print("Script folder:", BASE_DIR)


def file_path(filename: str) -> Path:
    return BASE_DIR / filename


print("Files in script folder:")
for f in BASE_DIR.iterdir():
    print(" -", f.name)


# Load the merged dataset containing both PageRank scores and FotMob ratings
df = pd.read_csv(file_path("merged_fotmob_modelA_2023.csv"), encoding="utf-8")


# Define which statistics matter for each position group
# Each list contains the per-90 features we think best capture that role's contribution
feature_sets = {
    "GK": ["saves_per90"],
    "DF": ["tackles_per90", "interceptions_per90", "blocks_per90", "passes_per90"],
    "MF": ["passes_per90", "assists_per90", "big_chances_created_per90",
           "tackles_per90", "interceptions_per90"],
    "FW": ["goals_per90", "sot_per90", "assists_per90", "big_chances_created_per90"],
}


weights_by_pos = {}
model_fits = {}


# Run a separate linear regression for each position
# This tells us how much each statistic predicts the external FotMob rating
for pos, feats in feature_sets.items():
    df_pos = df[df["Category"] == pos].copy()

    # Only keep players with enough minutes so small sample sizes don't distort the weights
    # Goalkeepers are given a more lenient threshold since there are fewer of them
    if "minutes_played" in df_pos.columns:
        if pos == "GK":
            df_pos = df_pos[df_pos["minutes_played"] >= 300]   # lenient for goalkeepers
        else:
            df_pos = df_pos[df_pos["minutes_played"] >= 600]   # stricter for outfield

    # Drop rows missing rating or any of the features
    df_pos = df_pos.dropna(subset=["fotmob_rating"] + feats)

    if df_pos.empty:
        print(f"No data for {pos}")
        continue

    X = df_pos[feats].values
    y = df_pos["fotmob_rating"].values

    # Standardise the features so coefficients are on the same scale
    # and can be fairly compared across different statistics
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Allow intercept
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X_std, y)

    coefs = pd.Series(reg.coef_, index=feats)

    # Use only the positive coefficients to derive feature weights
    # Negative coefficients would imply a stat hurts a player's rating, which we ignore here
    coefs_pos = coefs.clip(lower=0)
    if coefs_pos.sum() > 0:
        weights = coefs_pos / coefs_pos.sum()
    else:
        # Fallback: equal weights if nothing is positive
        weights = pd.Series([1 / len(feats)] * len(feats), index=feats)

    weights_by_pos[pos] = weights

    r2 = reg.score(X_std, y)
    model_fits[pos] = {
        "r2": r2,
        "n": len(df_pos),
        "intercept": reg.intercept_,
    }

    print(f"\n=== {pos} regression ===")
    print("Samples used:", len(df_pos))
    print("R^2:", r2)
    print("Intercept:", reg.intercept_)
    print("Raw coefficients:")
    print(coefs)
    print("Normalised positive weights:")
    print(weights.round(3))


# Collect the learned weights into a single DataFrame so they can be
# passed into the PageRank model as position-specific feature importance scores
weights_rows = []
for pos, w in weights_by_pos.items():
    for feat, val in w.items():
        weights_rows.append({
            "Category": pos,
            "Feature": feat,
            "Weight": val,
        })

weights_df = pd.DataFrame(weights_rows)

# Save weights to csv 
out_path = file_path("modelB_feature_weights_from_regression.csv")
weights_df.to_csv(out_path, index=False, encoding="utf-8")
print("\nSaved weights to:", out_path)
print(weights_df)


# Save a summary of each position's regression fit so we can report
# how well external ratings are explained by our chosen statistics
fits_rows = []
for pos, meta in model_fits.items():
    fits_rows.append({
        "Category": pos,
        "R2": meta["r2"],
        "n": meta["n"],
        "intercept": meta["intercept"],
    })

fits_df = pd.DataFrame(fits_rows)
fits_out_path = file_path("modelB_regression_fit_summary.csv")
fits_df.to_csv(fits_out_path, index=False, encoding="utf-8")
print("\nSaved regression fit summary to:", fits_out_path)
print(fits_df)