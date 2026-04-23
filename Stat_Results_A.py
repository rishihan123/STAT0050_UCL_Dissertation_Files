from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau


BASE_DIR = Path(__file__).resolve().parent
print("Script folder:", BASE_DIR)


def file_path(name: str) -> Path:
    return BASE_DIR / name

# File name is changed for each model to produce the same results for each one
# Load the merged dataset containing both Model C PageRank scores and FotMob ratings
df = pd.read_csv(file_path("merged_fotmob_modelC_2023.csv"), encoding="utf-8")


# Define which statistics are relevant for each position
# These are used to determine which players we are evaluating per role
feature_sets = {
    "GK": ["saves_per90"],
    "DF": ["tackles_per90", "interceptions_per90", "blocks_per90", "passes_per90"],
    "MF": ["passes_per90", "assists_per90", "big_chances_created_per90",
           "tackles_per90", "interceptions_per90"],
    "FW": ["goals_per90", "sot_per90", "assists_per90", "big_chances_created_per90"],
}


rows = []


# Compute rank correlation and prediction error metrics for each position separately
# so we can see where Model C agrees most and least with the external benchmark
for pos in feature_sets.keys():
    df_pos = df[df["Category"] == pos].copy()

    # Drop rows where either score is missing as we cannot compare those
    df_pos = df_pos.dropna(subset=["Score", "fotmob_rating"])

    if df_pos.empty:
        print(f"No data for {pos}, skipping.")
        continue

    # Convert raw scores to ranks so correlation measures are order-based
    # rather than sensitive to the scale of the scores themselves
    df_pos["rank_modelC"] = df_pos["Score"].rank(ascending=False, method="average")
    df_pos["rank_fotmob"] = df_pos["fotmob_rating"].rank(ascending=False, method="average")

    # Spearman and Kendall tau are calculated to measure rank agreement in different ways
    rho_s, p_s = spearmanr(df_pos["rank_modelC"], df_pos["rank_fotmob"])
    tau_k, p_k = kendalltau(df_pos["rank_modelC"], df_pos["rank_fotmob"])

    # Fit a linear trend from PageRank score to FotMob rating and measure
    # how far off the predictions are in rating units (RMSE and MAE)
    b, a = np.polyfit(df_pos["Score"].values, df_pos["fotmob_rating"].values, 1)
    y_hat = a + b * df_pos["Score"].values
    mse = np.mean((y_hat - df_pos["fotmob_rating"].values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_hat - df_pos["fotmob_rating"].values))

    rows.append({
        "Category": pos,
        "n": len(df_pos),
        "Spearman_rho": rho_s,
        "Spearman_p": p_s,
        "Kendall_tau": tau_k,
        "Kendall_p": p_k,
        "RMSE": rmse,
        "MAE": mae,
        "a_intercept": a,
        "b_slope": b,
    })


# Save all metrics to disk so they can be reported directly in the results section
resB = pd.DataFrame(rows)
out_path = file_path("SpearmanC.csv")
resB.to_csv(out_path, index=False, encoding="utf-8")
print("Saved Model C metrics to:", out_path)
print(resB)