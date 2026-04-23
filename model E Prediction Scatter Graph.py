import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def file_path(name: str) -> Path:
    return BASE_DIR / name


# Load the predicted vs actual table produced by the Model E prediction step
tbl = pd.read_csv(file_path("modelE_2024_table_pred_vs_actual.csv"),
                  encoding="utf-8")


# Plot each team as a dot with actual points on the x-axis and predicted on the y-axis
# Teams close to the diagonal line are predicted most accurately
plt.figure(figsize=(6, 6))
plt.scatter(tbl["act_points"], tbl["pred_points"],
            color="steelblue", edgecolor="white", s=80, alpha=0.8)

# Draw the y = x reference line so perfect predictions would sit exactly on it
max_pts = max(tbl["act_points"].max(), tbl["pred_points"].max()) + 2
plt.plot([0, max_pts], [0, max_pts], color="red", linestyle="--", label="y = x")

# Label each dot with the team name so we can identify which clubs are over or under predicted
for _, row in tbl.iterrows():
    plt.text(row["act_points"] + 0.2, row["pred_points"] + 0.2,
             row["team"], fontsize=8)

plt.xlabel("Actual 2024 points")
plt.ylabel("Predicted points (Model E)")
plt.title("Model E: Predicted vs actual 2024 points")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save to disk next to this script so it can be dropped straight into the report
out_scatter = file_path("modelE_points_scatter_2024.jpg")
plt.savefig(out_scatter, dpi=300)
plt.close()
print("Saved scatter plot:", out_scatter)