from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
print("Script folder:", BASE_DIR)


def file_path(name: str) -> Path:
    return BASE_DIR / name

# Name of file is changed for each model to produce the same figures for each one
# Load the merged dataset containing both Model A PageRank scores and FotMob ratings
df = pd.read_csv(file_path("merged_fotmob_modelA_2023.csv"), encoding="utf-8")


positions = ["GK", "DF", "MF", "FW"]


# Draw a separate scatter plot for each position so we can assess
# how well PageRank scores correlate with external ratings by role
for pos in positions:
    df_pos = df[df["Category"] == pos].copy()

    # Drop rows where either score is missing as we cannot plot or fit those
    df_pos = df_pos.dropna(subset=["Score", "fotmob_rating"])

    if df_pos.empty:
        print(f"No data for {pos}, skipping.")
        continue

    x = df_pos["Score"].values
    y = df_pos["fotmob_rating"].values

    # Need at least 2 points to fit a line
    if len(df_pos) < 2:
        print(f"Not enough data for {pos}, skipping.")
        continue

    # Fit a simple linear trend line so we can visually assess the relationship
    b, a = np.polyfit(x, y, 1)

    # Sort x so the trend line draws cleanly from left to right
    x_line = np.sort(x)
    y_line = a + b * x_line

    # Plot each player as a dot and overlay the trend line
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.65, color="steelblue", edgecolor="white", s=90)
    plt.plot(x_line, y_line, color="crimson", linewidth=3, label="Linear fit")

    plt.xlabel(f"Model C PageRank score")
    plt.ylabel("FotMob rating")
    plt.title(f"Model C vs FotMob rating ({pos})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to disk next to this script so it can be dropped straight into the report
    out_path = file_path(f"modelC_vs_fotmob_{pos}_scatter_fit.jpg")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")