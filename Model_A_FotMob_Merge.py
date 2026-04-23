import pandas as pd
from pathlib import Path
from unidecode import unidecode  


BASE_DIR = Path(__file__).resolve().parent
print("Script folder:", BASE_DIR)


def file_path(filename: str) -> Path:
    return BASE_DIR / filename


print("Files in script folder:")
for f in BASE_DIR.iterdir():
    print(" -", f.name)


# Normalise a player or team name so accents, capitalisation and spacing
# do not prevent two records from matching across datasets
def make_key(s) -> str:
    if pd.isna(s):
        return ""
    # remove accents, lower, strip spaces
    return unidecode(str(s)).strip().lower()


# Load the FotMob external ratings which will serve as our benchmark
fot = pd.read_csv(file_path("kaggle_fotmob_2023_master.csv"), encoding="utf-8")

# Build normalised keys so fuzzy name differences do not break the merge
fot["Player_key"] = fot["Player"].apply(make_key)
fot["Team_key"]   = fot["Team"].apply(make_key)


# File name is changed for each model to produce a merged file for each one with the same structure 
# Load the Model A PageRank scores
pr = pd.read_csv(file_path("modelA_pagerank_2023.csv"), encoding="utf-8")

# Build the same normalised keys on the PageRank side
pr["Player_key"] = pr["Player"].apply(make_key)
pr["Team_key"]   = pr["team"].apply(make_key)


# Merge the two datasets on both player name and team so we only match
# records where the same player appears at the same club in both sources
merged = fot.merge(
    pr[["Player_key", "Team_key", "Category", "Score"]],
    on=["Player_key", "Team_key"],
    how="inner"
)


print("Merged rows:", len(merged))

# Quick check of unique players and goalkeepers
print("Unique players in merged:", merged["Player"].nunique())
print("GKs in merged (by Category == 'GK'):", merged[merged["Category"] == "GK"]["Player"].nunique())


# Save the merged file to disk so the regression step can load it directly
# without needing to repeat the merging process
out_path = file_path("merged_fotmob_modelA_2023.csv")
merged.to_csv(out_path, index=False, encoding="utf-8")
print("Saved merged file to:", out_path)
print(merged.head())