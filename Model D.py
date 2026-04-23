import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def file_path(name: str) -> Path:
    return BASE_DIR / name


# Load Model C output files for multiple seasons (include 2024 as this is a pure ranking)
# We stack all seasons so we can aggregate each player's career performance across years
season_files = [
    ("modelC_pagerank_2024.csv", 2024),
    ("modelC_pagerank_2023.csv", 2023),
    ("modelC_pagerank_2022.csv", 2022),
    ("modelC_pagerank_2021.csv", 2021),
    ("modelC_pagerank_2020.csv", 2020),
    ("modelC_pagerank_2019.csv", 2019),
    ("modelC_pagerank_2018.csv", 2018),
    ("modelC_pagerank_2017.csv", 2017),
    ("modelC_pagerank_2016.csv", 2016),
    ("modelC_pagerank_2015.csv", 2015),
]


# Read each season file and tag it with its year before stacking
dfs = []
for fname, season in season_files:
    fpath = file_path(fname)
    df_season = pd.read_csv(fpath, encoding="utf-8")
    df_season["season"] = season
    dfs.append(df_season)


df_all = pd.concat(dfs, ignore_index=True)


# Use player name as ID since there is no player_id column in the Model C outputs
id_col = "Player"


# Compute total score and number of seasons played per player and position
# Total score rewards longevity while seasons_played lets us contextualise it
total_scores = (
    df_all
    .groupby([id_col, "Category"], as_index=False)
    .agg(
        total_score=("Score", "sum"),
        seasons_played=("season", "nunique")
    )
)


# Compute average score separately so we can also assess peak season performance
avg_scores = (
    df_all
    .groupby([id_col, "Category"], as_index=False)
    .agg(
        avg_score=("Score", "mean")
    )
)


agg_df = total_scores.merge(avg_scores, on=[id_col, "Category"], how="left")


# Attach the most recent team for each player so the output shows where they last played
latest_team = (
    df_all.sort_values("season")
          .drop_duplicates(subset=[id_col, "Category"], keep="last")
          [[id_col, "Category", "team"]]
)
agg_df = agg_df.merge(latest_team, on=[id_col, "Category"], how="left")


# Save the full combined file so it can be inspected 
all_out = file_path("modelD_aggregation_candidates_by_position.csv")
agg_df.to_csv(all_out, index=False, encoding="utf-8")
print("Saved combined file:", all_out)


# Save a separate ranked CSV for each position and print the top 10
# Sorting by total_score first then avg_score breaks ties in favour of consistent players
for cat in ["GK", "DF", "MF", "FW"]:
    pos_df = (
        agg_df[agg_df["Category"] == cat]
        .sort_values(["total_score", "avg_score"], ascending=False)
    )

    # Save to disk next to this script so rankings can be dropped straight into the report
    pos_out = file_path(f"modelD_{cat}_rankings.csv")
    pos_df.to_csv(pos_out, index=False, encoding="utf-8")
    print(f"Saved {cat} rankings:", pos_out)

    # Print out the rankings of each position
    print(f"\nTop 10 {cat} by total_score:")
    print(
        pos_df.head(10)[[id_col, "team", "Category",
                         "total_score", "avg_score", "seasons_played"]]
    )