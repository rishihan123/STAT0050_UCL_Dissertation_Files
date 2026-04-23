import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path


# Database Connection
ENGINE_URL = "mysql+mysqlconnector://root:RoomierCanine24!@localhost/footballdata"
BASE_DIR = Path(__file__).resolve().parent


# Model C season files used to build recency-weighted player strengths
# We stop at 2023 so the 2024 season can be used as a prediction season
MODEL_E_SEASONS = [
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

# Controls how quickly older seasons are discounted relative to recent ones
# Higher values make the model more focused on recent form
LAMBDA_DECAY = 0.25

# Scaling parameter for the logistic win probability function
# Controls how quickly probabilities move away from 50% as the strength gap grows
LOGIT_K = 4.0


def file_path(name: str) -> Path:
    return BASE_DIR / name


# Mapping from short database team names to full names
# This ensures the same club is not counted as two different teams across sources
TEAM_NAME_MAP = {
    "Manchester United": "Manchester United",
    "Tottenham": "Tottenham Hotspur",
    "Bournemouth": "Bournemouth",
    "Aston Villa": "Aston Villa",
    "Everton": "Everton",
    "Crystal Palace": "Crystal Palace",
    "Chelsea": "Chelsea",
    "Newcastle": "Newcastle United",
    "Southampton": "Southampton",
    "Arsenal": "Arsenal",
    "West Ham": "West Ham United",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Brighton": "Brighton and Hove Albion",
    "Fulham": "Fulham",
    "Wolves": "Wolverhampton Wanderers",
    "Brentford": "Brentford",
    "Nottingham Forest": "Nottingham Forest",
    "Ipswich": "Ipswich Town",
    "Leicester": "Leicester City",
}

# Only keep rows belonging to the 20 clubs in the 2024/25 season
VALID_2024_25_TEAMS = set(TEAM_NAME_MAP.keys())


def normalise_team_name(team):
    # Replace short or variant names with their canonical full name
    if pd.isna(team):
        return team
    team = str(team).strip()
    return TEAM_NAME_MAP.get(team, team)


# Build recency-weighted player strengths
def build_player_strengths():
    # Load all historical Model C seasons and tag each row with its year
    dfs = []
    for fname, season in MODEL_E_SEASONS:
        fpath = file_path(fname)
        df_season = pd.read_csv(fpath, encoding="utf-8")
        df_season["season"] = season
        if "team" in df_season.columns:
            df_season["team"] = df_season["team"].apply(normalise_team_name)
        dfs.append(df_season)

    df_all = pd.concat(dfs, ignore_index=True)

    id_col = "Player"

    # Assign exponentially increasing weights so recent seasons count more
    # than older ones when computing a player's career strength score
    min_season = df_all["season"].min()
    df_all["w_recency"] = np.exp(LAMBDA_DECAY * (df_all["season"] - min_season))

    # Collapse to one row per player per season, summing their score
    player_season = (
        df_all
        .groupby([id_col, "team", "season"], as_index=False)
        .agg(
            season_score=("Score", "sum"),
            w_recency=("w_recency", "mean"),
        )
    )

    # Compute a single recency-weighted average strength score for each player
    # across all seasons they appeared in, then record their most recent team
    player_strength_E = (
        player_season
        .groupby(id_col, as_index=False)
        .apply(lambda g: pd.Series({
            "strength_E": (g["season_score"] * g["w_recency"]).sum() / g["w_recency"].sum(),
            "last_team": g.sort_values("season")["team"].iloc[-1],
            "last_season": g["season"].max(),
        }))
        .reset_index(drop=True)
    )

    # Save to disk so team strength building can load it without rerunning
    out_path = file_path("modelE_player_strengths_2015_2023.csv")
    player_strength_E.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved player strengths to:", out_path)

    return player_strength_E


# Build 2024 team strengths from player_data_2 
def build_team_strengths_2024(player_strength_E, engine):
    # Fetch the 2024 squad data so we know which players appeared for each club
    squads_2024 = pd.read_sql(
        """
        SELECT DISTINCT
            season,
            team,
            player_name AS Player,
            minutes_played
        FROM player_data_2
        WHERE season = 2024
        """,
        engine,
    )

    # Normalise team names and filter to only the 20 valid 2024/25 clubs
    squads_2024["team_raw"] = squads_2024["team"]
    squads_2024["team"] = squads_2024["team"].apply(normalise_team_name)
    squads_2024 = squads_2024[squads_2024["team_raw"].isin(VALID_2024_25_TEAMS)].copy()

    # Attach each player's career strength score; players with no historical data get 0
    squads_2024 = squads_2024.merge(
        player_strength_E[["Player", "strength_E"]],
        on="Player",
        how="left",
    )

    squads_2024["strength_E"] = squads_2024["strength_E"].fillna(0.0)
    squads_2024["minutes_played"] = squads_2024["minutes_played"].fillna(0.0)
    squads_2024["min_weight"] = squads_2024["minutes_played"].clip(lower=0.0)

    # Compute team strength as a minutes-weighted average of its players' strengths
    # Players who featured more heavily have a larger influence on the team score
    def team_strength(group):
        w_sum = group["min_weight"].sum()
        if w_sum <= 0:
            return 0.0
        return (group["strength_E"] * group["min_weight"]).sum() / w_sum

    team_strength_2024 = (
        squads_2024
        .groupby("team", as_index=False)
        .apply(lambda g: pd.Series({"team_strength_E": team_strength(g)}))
        .reset_index(drop=True)
    )

    # Save team strengths to disk so the prediction step can load them directly
    out_path = file_path("modelE_team_strengths_2024.csv")
    team_strength_2024.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved team strengths for 2024 to:", out_path)

    return team_strength_2024


#  Derive 2024 fixtures from player_data_2 
def build_matches_2024_from_player_data(engine):
    # Fetch raw match-level data so we can reconstruct home and away teams
    # along with the goals each side scored in every game
    df = pd.read_sql(
        """
        SELECT
            season,
            game_id,
            team,
            goals,
            player_id
        FROM player_data_2
        WHERE season = 2024
        ORDER BY game_id, team, player_id
        """,
        engine,
    )

    # Filter to valid clubs and normalise team names
    df["team_raw"] = df["team"]
    df = df[df["team_raw"].isin(VALID_2024_25_TEAMS)].copy()
    df["team"] = df["team"].apply(normalise_team_name)

    # Identify the home team for each game as the first team appearing in the data
    first_team = (
        df.groupby("game_id")["team"]
          .first()
          .reset_index()
          .rename(columns={"team": "home_team"})
    )

    df = df.merge(first_team, on="game_id", how="left")
    df["is_home"] = df["team"] == df["home_team"]

    # Aggregate goals separately for home and away sides then join into one fixture row
    home_goals = (
        df[df["is_home"]]
        .groupby(["season", "game_id", "team"], as_index=False)["goals"]
        .sum()
        .rename(columns={"team": "home_team", "goals": "home_goals"})
    )

    away_goals = (
        df[~df["is_home"]]
        .groupby(["season", "game_id", "team"], as_index=False)["goals"]
        .sum()
        .rename(columns={"team": "away_team", "goals": "away_goals"})
    )

    matches = home_goals.merge(away_goals, on=["season", "game_id"], how="inner")

    # Save fixture list to disk so it can be reused without re-querying the database
    out_path = file_path("matches_2024_from_player_data.csv")
    matches.to_csv(out_path, index=False, encoding="utf-8")
    print("Saved inferred 2024 fixtures to:", out_path)

    return matches


# Hard-coded actual 2024 league table 
def load_actual_table_2024():
    # Store the real final standings so we can compare them against our predictions
    data = [
        ("Liverpool", 84),
        ("Arsenal", 74),
        ("Manchester City", 71),
        ("Chelsea", 69),
        ("Newcastle United", 66),
        ("Aston Villa", 66),
        ("Nottingham Forest", 65),
        ("Brighton and Hove Albion", 61),
        ("Bournemouth", 56),
        ("Brentford", 56),
        ("Fulham", 54),
        ("Crystal Palace", 53),
        ("Everton", 48),
        ("West Ham United", 43),
        ("Manchester United", 42),
        ("Wolverhampton Wanderers", 42),
        ("Tottenham Hotspur", 38),
        ("Leicester City", 25),
        ("Ipswich Town", 22),
        ("Southampton", 12),
    ]
    return pd.DataFrame(data, columns=["team", "act_points"])


# Predict results and compare
def predict_2024_matches(team_strength_2024, matches):
    fixtures = matches.copy()

    # Attach home and away team strengths to each fixture
    fixtures = fixtures.merge(
        team_strength_2024.rename(
            columns={"team": "home_team", "team_strength_E": "home_strength"}
        ),
        on="home_team",
        how="left",
    ).merge(
        team_strength_2024.rename(
            columns={"team": "away_team", "team_strength_E": "away_strength"}
        ),
        on="away_team",
        how="left",
    )

    fixtures[["home_strength", "away_strength"]] = fixtures[
        ["home_strength", "away_strength"]
    ].fillna(0.0)

    # Compute the strength difference and pass it through the logistic function
    # to produce a home win probability in [0, 1]
    diff = fixtures["home_strength"] - fixtures["away_strength"]
    fixtures["p_home"] = 1.0 / (1.0 + np.exp(-LOGIT_K * diff))
    fixtures["p_away"] = 1.0 - fixtures["p_home"]

    # Allocate a small portion of remaining probability to draws
    # clipped to keep the draw probability within a sensible range
    fixtures["p_draw"] = np.clip(
        1.0 - (fixtures["p_home"] + fixtures["p_away"]) * 0.8, 0.05, 0.9
    )

    # Predict the most likely outcome for each match by picking the highest probability
    def pred(row):
        probs = {"H": row["p_home"], "D": row["p_draw"], "A": row["p_away"]}
        return max(probs, key=probs.get)

    fixtures["pred_result"] = fixtures.apply(pred, axis=1)

    # Convert predicted outcomes into points for home and away teams
    fixtures["home_points_pred"] = fixtures["pred_result"].map(
        {"H": 3, "D": 1, "A": 0}
    )
    fixtures["away_points_pred"] = fixtures["pred_result"].map(
        {"H": 0, "D": 1, "A": 3}
    )

    # Sum predicted points across all fixtures to build a predicted league table
    home_pred = fixtures.groupby("home_team", as_index=False)["home_points_pred"].sum()
    away_pred = fixtures.groupby("away_team", as_index=False)["away_points_pred"].sum()

    tbl_pred = home_pred.rename(columns={"home_team": "team"}).merge(
        away_pred.rename(columns={"away_team": "team"}),
        on="team",
        how="outer",
    ).fillna(0.0)

    tbl_pred["pred_points"] = (
        tbl_pred["home_points_pred"] + tbl_pred["away_points_pred"]
    )
    tbl_pred = tbl_pred[["team", "pred_points"]]

    # use actual table as master list so all 20 teams appear
    tbl_act = load_actual_table_2024()
    table_compare = tbl_act.merge(tbl_pred, on="team", how="left").fillna(0.0)

    # Assign predicted and actual ranks so we can measure ranking accuracy directly
    table_compare["pred_rank"] = table_compare["pred_points"].rank(
        ascending=False, method="min"
    )
    table_compare["act_rank"] = table_compare["act_points"].rank(
        ascending=False, method="min"
    )

    # Compute summary metrics to report in the results section
    rank_accuracy = (table_compare["pred_rank"] == table_compare["act_rank"]).mean()
    mae_points = np.mean(np.abs(table_compare["pred_points"] - table_compare["act_points"]))
    rmse_points = np.sqrt(np.mean((table_compare["pred_points"] - table_compare["act_points"]) ** 2))

    print(f"\nModel E league-table exact-rank accuracy 2024: {rank_accuracy:.3f}")
    print(f"Model E points MAE 2024: {mae_points:.3f}")
    print(f"Model E points RMSE 2024: {rmse_points:.3f}")

    table_compare = table_compare.sort_values("pred_points", ascending=False)

    # Save the full comparison table to disk so it can be dropped into the report
    out_tbl = file_path("modelE_2024_table_pred_vs_actual.csv")
    table_compare.to_csv(out_tbl, index=False, encoding="utf-8")
    print("Saved predicted vs actual 2024 table to:", out_tbl)

    return rank_accuracy, mae_points, rmse_points, table_compare


if __name__ == "__main__":
    engine = create_engine(ENGINE_URL)

    print("Building Model E player strengths (2015–2023)...")
    player_strength_E = build_player_strengths()

    print("\nBuilding Model E team strengths for 2024...")
    team_strength_2024 = build_team_strengths_2024(player_strength_E, engine)

    print("\nDeriving 2024 fixtures from player_data_2...")
    matches_2024 = build_matches_2024_from_player_data(engine)

    print("\nPredicting 2024 results and comparing league table...")
    rank_acc, mae_pts, rmse_pts, table_compare = predict_2024_matches(
        team_strength_2024, matches_2024
    )

    print("\nFull predicted vs actual table:")
    print(table_compare.sort_values("pred_rank"))

    # Print out the top 10 teams by predicted and actual points for a quick check
    print("\nTop 10 teams by predicted points:")
    print(
        table_compare.sort_values("pred_points", ascending=False)
                     .head(10)
    )

    print("\nTop 10 teams by actual points:")
    print(
        table_compare.sort_values("act_points", ascending=False)
                     .head(10)
    )