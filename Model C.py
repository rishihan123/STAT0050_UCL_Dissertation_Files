import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import networkx as nx
from pathlib import Path
import os


# Database Connection
engine = create_engine("mysql+mysqlconnector://root:RoomierCanine24!@localhost/footballdata")

print("Current working directory:", os.getcwd())
BASE_DIR = Path(__file__).resolve().parent
print("Script folder:", BASE_DIR)


# Function to simplify positions into categories
def get_position_group(pos):
    """
    Footballers have many specific roles (CB, LW, DM).
    This function simplifies them into 4 big buckets: GK, DF, MF, FW.
    """
    if not pos:
        return None
    pos = str(pos).upper()
    if "G" in pos:
        return "GK"
    if "D" in pos:
        return "DF"
    if "M" in pos:
        return "MF"
    if "F" in pos:
        return "FW"
    return None


# Data-driven feature weights inherited from the Model B regression
# These determine how much each statistic contributes to a player's influence score
WEIGHTS_B = {
    "GK": {
        "saves": 1.0,
    },
    "DF": {
        "tackles_total": 0.191058,
        "interceptions": 0.279464,
        "passes_total": 0.529478,
    },
    "MF": {
        "passes_total": 0.431431,
        "assists": 0.225221,
        "key_passes": 0.280058,   # big chances created
        "interceptions": 0.063290,
    },
    "FW": {
        "goals": 0.395906,
        "assists": 0.286039,
        "key_passes": 0.318055,   # big chances created
    },
}

# Model C adds discipline penalties and clean sheet bonuses on top of the Model B metric
# These penalise players who commit fouls or receive cards and reward defensive solidity
YELLOW_W      = 0.03
RED_W         = 0.10
FOUL_W        = 0.005
PEN_COMM_W    = 0.10
CS_BONUS_GK   = 0.08   # clean sheet bonus for GK
CS_BONUS_DF   = 0.04   # clean sheet bonus for DF


def run_model_c_pagerank(season_year):
    # Grab the data from our SQL, including discipline and clean sheet columns
    # that were not present in Models A and B
    query = f"""
    SELECT
        season, game_id, team, player_name, position, minutes_played, 
        goals, assists, shots_on_target,
        passes_total, key_passes,
        tackles_total, blocks, interceptions,
        fouls_committed, cards_yellow, cards_red,
        penalty_committed,
        saves, goals_conceded
    FROM player_data_2
    WHERE season = {season_year}
    """
    df = pd.read_sql(query, engine)

    # If there is no data for this season, return early
    if df.empty:
        return pd.DataFrame()

    # We only care about people who actually played (minutes > 0)
    df = df[df["minutes_played"] > 0].copy().fillna(0)
    df["pos_group"] = df["position"].apply(get_position_group)

    # Run PageRank for each position graph
    categories = ["GK", "DF", "MF", "FW"]
    category_results = []

    for cat in categories:
        # Create a new empty graph for this position
        G = nx.DiGraph()
        cat_df = df[df["pos_group"] == cat].copy()
        if cat_df.empty:
            continue

        w = WEIGHTS_B[cat]

        for game_id, game_group in cat_df.groupby("game_id"):
            game_group = game_group.copy()

            # Need at least 2 players to form any connections
            if len(game_group) < 2:
                continue

            # First compute the Model B positive metric for each position,
            # which captures the constructive side of each player's contribution
            if cat == "GK":
                # Keeper score = their share of the team's total saves
                total_saves = game_group["saves"].sum()
                share_saves = game_group["saves"] / total_saves if total_saves > 0 else 0
                game_group["metric_pos"] = w["saves"] * share_saves

            elif cat == "DF":
                # Defender score = regression-weighted mix of tackles, interceptions and passes
                total_tkl = game_group["tackles_total"].sum()
                total_int = game_group["interceptions"].sum()
                total_pass = game_group["passes_total"].sum()

                tkl_share = game_group["tackles_total"] / total_tkl if total_tkl > 0 else 0
                int_share = game_group["interceptions"] / total_int if total_int > 0 else 0
                pass_share = game_group["passes_total"] / total_pass if total_pass > 0 else 0

                game_group["metric_pos"] = (
                    w["tackles_total"] * tkl_share +
                    w["interceptions"] * int_share +
                    w["passes_total"] * pass_share
                )

            elif cat == "MF":
                # Midfielder score = regression-weighted mix of passes, assists,
                # key passes and interceptions
                total_passes = game_group["passes_total"].sum()
                total_ast = game_group["assists"].sum()
                total_key = game_group["key_passes"].sum()
                total_int = game_group["interceptions"].sum()

                pass_share = game_group["passes_total"] / total_passes if total_passes > 0 else 0
                ast_share = game_group["assists"] / total_ast if total_ast > 0 else 0
                key_share = game_group["key_passes"] / total_key if total_key > 0 else 0
                int_share = game_group["interceptions"] / total_int if total_int > 0 else 0

                game_group["metric_pos"] = (
                    w["passes_total"] * pass_share +
                    w["assists"] * ast_share +
                    w["key_passes"] * key_share +
                    w["interceptions"] * int_share
                )

            elif cat == "FW":
                # Forward score = regression-weighted mix of goals, assists and key passes
                total_goals = game_group["goals"].sum()
                total_ast = game_group["assists"].sum()
                total_key = game_group["key_passes"].sum()

                goal_share = game_group["goals"] / total_goals if total_goals > 0 else 0
                ast_share = game_group["assists"] / total_ast if total_ast > 0 else 0
                key_share = game_group["key_passes"] / total_key if total_key > 0 else 0

                game_group["metric_pos"] = (
                    w["goals"] * goal_share +
                    w["assists"] * ast_share +
                    w["key_passes"] * key_share
                )

            # Start Model C metric from the Model B baseline,
            # then apply bonuses and penalties on top
            game_group["metric_val"] = game_group["metric_pos"].copy()

            # Reward keepers and defenders who kept a clean sheet in this match,
            # since goals conceded is a team-level stat stored per player row
            if "goals_conceded" in game_group.columns:
                is_clean_sheet = game_group["goals_conceded"] == 0
                if cat == "GK":
                    game_group.loc[is_clean_sheet, "metric_val"] += CS_BONUS_GK
                elif cat == "DF":
                    game_group.loc[is_clean_sheet, "metric_val"] += CS_BONUS_DF

            # Penalise players for disciplinary actions since these hurt the team
            if "cards_yellow" in game_group.columns:
                game_group["metric_val"] -= YELLOW_W * game_group["cards_yellow"]
            if "cards_red" in game_group.columns:
                game_group["metric_val"] -= RED_W * game_group["cards_red"]
            if "fouls_committed" in game_group.columns:
                game_group["metric_val"] -= FOUL_W * game_group["fouls_committed"]
            if "penalty_committed" in game_group.columns:
                game_group["metric_val"] -= PEN_COMM_W * game_group["penalty_committed"]

            # Scale each player's influence by how long they actually played,
            # so a substitute who played 10 minutes has less impact than a starter
            game_group["metric_val"] *= (game_group["minutes_played"] / 90.0)
            game_group["metric_val"] = game_group["metric_val"].clip(lower=0)

            # Players vote for each other proportionally to each receiver's performance
            players = game_group[["player_name", "team", "metric_val"]].to_dict("records")

            for p1 in players:
                sender = p1["player_name"]
                targets = [p2 for p2 in players if p2["player_name"] != sender]
                if not targets:
                    continue

                target_strength_sum = sum(p2["metric_val"] for p2 in targets)

                # If everyone scored zero, distribute votes equally so no one is isolated
                if target_strength_sum == 0:
                    uniform_weight = 1.0 / len(targets)
                    for p2 in targets:
                        receiver = p2["player_name"]
                        if G.has_edge(sender, receiver):
                            G[sender][receiver]["weight"] += uniform_weight
                        else:
                            G.add_edge(sender, receiver, weight=uniform_weight)
                else:
                    # Otherwise, stronger performers attract more of the sender's vote
                    for p2 in targets:
                        receiver = p2["player_name"]
                        normalized_weight = p2["metric_val"] / target_strength_sum
                        if G.has_edge(sender, receiver):
                            G[sender][receiver]["weight"] += normalized_weight
                        else:
                            G.add_edge(sender, receiver, weight=normalized_weight)

        # Use the PageRank maths to settle the final scores
        if len(G) > 0:
            scores = nx.pagerank(G, alpha=0.85, weight="weight")
            res_df = pd.DataFrame(list(scores.items()), columns=["Player", "Score"])
            res_df["Category"] = cat

            # Clean up and add the team names back so we can see who plays where
            latest_team = cat_df.drop_duplicates("player_name", keep="last")[["player_name", "team"]]
            res_df = res_df.merge(
                latest_team, left_on="Player", right_on="player_name", how="left"
            )
            res_df = res_df.drop(columns=["player_name"])
            category_results.append(res_df)

    if category_results:
        return pd.concat(category_results, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    target_season = 2023
    results = run_model_c_pagerank(target_season)

    # Save CSV next to this script
    out_path = BASE_DIR / f"modelC_pagerank_{target_season}.csv"
    results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved Model C PageRank results to: {out_path}")

    # Print out the rankings of each position
    for cat in ["GK", "DF", "MF", "FW"]:
        print(f"\n--- Top 10 {cat}s (Model C) ---")
        top = results[results["Category"] == cat].sort_values(
            "Score", ascending=False
        ).head(10)
        print(top[["Player", "team", "Score"]])