import soccerdata as sd
import pandas as pd
import time
import mysql.connector
from sqlalchemy import create_engine
import os

# THIS WASN'T USED FOR THE MAIN PART OF THE PROJECT

# Spoof a real browser user agent so FBref does not block the scraper requests
os.environ['SOCCERDATA_USER_AGENT'] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


# Set up the SQL database connection
sqldb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="RoomierCanine24!",
    database="footballdata"
)

cursor = sqldb.cursor()

# Create the Player_Data table if it does not already exist
# This stores one row per player per match with all FBref summary statistics
cursor.execute("""
CREATE TABLE  IF NOT EXISTS Player_Data (
    -- Identification & Meta Data
    league VARCHAR(100),
    season VARCHAR(20),
    game VARCHAR(255),
    team VARCHAR(100),
    player VARCHAR(255),
    player_id VARCHAR(100),
    jersey_number INT,
    nation VARCHAR(10),
    pos VARCHAR(10),
    age VARCHAR(10),
    min INT,

    -- Performance Stats
    Performance_Gls INT,
    Performance_Ast INT,
    Performance_PK INT,
    Performance_PKatt INT,
    Performance_Sh INT,
    Performance_SoT INT,
    Performance_CrdY INT,
    Performance_CrdR INT,
    Performance_Touches INT,
    Performance_Tkl INT,
    Performance_Int INT,
    Performance_Blocks INT,

    -- Expected Metrics
    Expected_xG DECIMAL(5,2),
    Expected_npxG DECIMAL(5,2),
    Expected_xAG DECIMAL(5,2),

    -- Shot Creation & Passing
    SCA_SCA INT,
    SCA_GCA INT,
    Passes_Cmp INT,
    Passes_Att INT,
    `Passes_Cmp%` DECIMAL(5,2),
    Passes_PrgP INT,

    -- Progression & Take-ons
    Carries_Carries INT,
    Carries_PrgC INT,
    `Take-Ons_Att` INT,
    `Take-Ons_Succ` INT,

    -- IDs
    game_id VARCHAR(100)
);
""")

engine = create_engine(
    "mysql+mysqlconnector://root:RoomierCanine24!@localhost/footballdata"
)

# Initialise the FBref reader for the Premier League seasons we want to scrape
fbref = sd.FBref(leagues=["ENG-Premier League"], seasons=["2022","2023","2024"])


print("Fetching season schedule...")
games = fbref.read_schedule()
print(f"Successfully retrieved schedule for {len(games)} entries.")


# Only keep matches that have a match report link so we know player data is available
REPORT_COLUMN_KEY = "match_report"
played_games = games[games[REPORT_COLUMN_KEY].notna()]
print(f"Filtered down to {len(played_games)} played matches with data available.")

# If there are no matches to process, there is nothing to scrape
if played_games.empty:
    print("\n--- Scraping Halted ---")
    exit()


all_player_stats = []
match_counter = 0

for i, row in played_games.iterrows():
    match_report_url = row[REPORT_COLUMN_KEY]
    match_id_str = match_report_url.split("/")[-2]

    # Check if the game is already in the DB
    # If count > 0, we skip the entire iteration so we never double-insert a match
    cursor.execute("SELECT count(*) FROM Player_Data WHERE game_id = %s", (match_id_str,))
    if cursor.fetchone()[0] > 0:
        print(f"Skipping match {match_id_str} - Already fully recorded.")
        continue

    print(f"Processing new match: {row['home_team']} vs {row['away_team']}...")

    try:
        # Fetch full data
        # The summary stat type returns all columns we need in a single request
        player_match_df = fbref.read_player_match_stats(match_id=match_id_str, stat_type='summary')
        player_match_df = player_match_df.reset_index()

        # Prepare Columns
        # Flatten MultiIndex to "Category_Stat" format so column names are SQL-safe
        player_match_df.columns = ["_".join(col).strip("_") for col in player_match_df.columns]

        # Tag every row with the match ID 
        player_match_df["game_id"] = match_id_str

        # Rename the pass completion percentage column to match the SQL schema definition
        if "Passes_CmpP" in player_match_df.columns:
            player_match_df = player_match_df.rename(columns={"Passes_CmpP": "Passes_Cmp%"})

        # Upload using index=False ensures we don't upload the Pandas index as a column
        player_match_df.to_sql("Player_Data", con=engine, if_exists="append", index=False)

        print(f"-> Success: Added {len(player_match_df)} player rows for {match_id_str}")

        # Sleep between requests so FBref does not throttle or block the scraper
        time.sleep(5)

    except Exception as e:
        print(f"-> ERROR on match {match_id_str}: {e}")