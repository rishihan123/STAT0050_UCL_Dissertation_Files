import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
print("Script folder:", BASE_DIR)


def file_path(name: str) -> Path:
    return BASE_DIR / name


# Load the feature weights learned from the Model B regression
weights = pd.read_csv(file_path("modelB_feature_weights_from_regression.csv"))


# Filter down to forwards only and sort so the largest bar appears at the top
fw = weights[weights["Category"] == "FW"].copy() #This is changed to MF, GK or DF for the other charts
fw = fw.sort_values("Weight", ascending=True)


# Draw a horizontal bar chart so long feature names stay readable
plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(fw["Feature"], fw["Weight"], color="#01696f")
ax.set_xlabel("Weight")
ax.set_title("Model B feature weights – Forwards")


plt.tight_layout()


# Save to csv next to this script so it can be dropped straight into the report
out_path = file_path("modelB_FW_weights_bar.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)


print("Saved FW bar chart to:", out_path)