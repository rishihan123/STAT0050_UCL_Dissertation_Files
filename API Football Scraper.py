import http.client
import json
import mysql.connector
import time
from tqdm import tqdm # Progress bar library


# Configuration for the API, database connection, and which seasons to scrape
API_KEY = "4c1f4ea5e7400154878b3ce4c9519c02"
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "RoomierCanine24!",
    "database": "footballdata"
}
LEAGUE_ID = 39
SEASONS = list(range(2015, 2025))


def call_api(endpoint):
    # Send a GET request to the API-Football endpoint and return the parsed JSON response
    conn = http.client.HTTPSConnection("v3.football.api-sports.io")
    headers = {'x-apisports-key': API_KEY}
    conn.request("GET", endpoint, headers=headers)
    res = conn.getresponse()
    return json.loads(res.read().decode("utf-8"))


def fixture_exists(cursor, fixture_id):
    # Check if we already have any player rows for this match so we can skip re-scraping it
    cursor.execute("SELECT 1 FROM player_data_2 WHERE game_id = %s LIMIT 1", (fixture_id,))
    return cursor.fetchone() is not None


def run_bulk_scrape():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Create the player_data_2 table if it does not exist yet
    # All per-90 statistics will be stored as raw per-match counts here
    create_table_query = """
    CREATE TABLE IF NOT EXISTS player_data_2 (
        id INT AUTO_INCREMENT PRIMARY KEY,
        season INT,
        game_id INT,
        team VARCHAR(255),
        player_name VARCHAR(255),
        player_id INT,
        position VARCHAR(10),
        minutes_played INT,
        goals INT DEFAULT 0,
        assists INT DEFAULT 0,
        shots_total INT DEFAULT 0,
        shots_on_target INT DEFAULT 0,
        passes_total INT DEFAULT 0,
        passes_accuracy VARCHAR(10),
        key_passes INT DEFAULT 0,
        tackles_total INT DEFAULT 0,
        blocks INT DEFAULT 0,
        interceptions INT DEFAULT 0,
        duels_total INT DEFAULT 0,
        duels_won INT DEFAULT 0,
        dribbles_attempts INT DEFAULT 0,
        dribbles_success INT DEFAULT 0,
        dribbles_past INT DEFAULT 0,
        fouls_drawn INT DEFAULT 0,
        fouls_committed INT DEFAULT 0,
        cards_yellow INT DEFAULT 0,
        cards_red INT DEFAULT 0,
        penalty_won INT DEFAULT 0,
        penalty_committed INT DEFAULT 0,
        penalty_scored INT DEFAULT 0,
        penalty_missed INT DEFAULT 0,
        penalty_saved INT DEFAULT 0,
        saves INT DEFAULT 0,
        goals_conceded INT DEFAULT 0,
        INDEX (game_id),
        INDEX (player_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cursor.execute(create_table_query)

    total_matches_processed = 0

    for season in SEASONS:
        # Fetch the full list of completed fixtures for this season
        print(f"\nGathering Fixture List for Season {season}...")
        fixtures = call_api(f"/fixtures?league={LEAGUE_ID}&season={season}&status=FT")

        if not fixtures.get('response'):
            print(f"Skipping {season}: No data.")
            continue

        all_fixtures = fixtures['response']

        # Filter out matches we already have to make the progress bar accurate
        # so restarting the script does not waste API quota on matches already in the database
        fixtures_to_process = []
        for f in all_fixtures:
            if not fixture_exists(cursor, f['fixture']['id']):
                fixtures_to_process.append(f)

        skip_count = len(all_fixtures) - len(fixtures_to_process)
        if skip_count > 0:
            print(f"⏩ Already have {skip_count} matches in DB. Scraping remaining {len(fixtures_to_process)}.")

        # Progress Bar for the current season
        for fix in tqdm(fixtures_to_process, desc=f"Season {season}", unit="match"):
            fix_id = fix['fixture']['id']

            # Fetch player-level statistics for this specific match
            player_data = call_api(f"/fixtures/players?fixture={fix_id}")
            if not player_data.get('response'):
                continue

            # Loop over both teams and every player who appeared in the match
            for team_entry in player_data['response']:
                team_name = team_entry['team']['name']
                for p_entry in team_entry['players']:
                    p = p_entry['player']
                    s = p_entry['statistics'][0]

                    # Insert one row per player per match, filling in 0 for missing values
                    insert_query = """
                    INSERT INTO player_data_2 (
                        season, game_id, team, player_name, player_id, position, minutes_played,
                        goals, assists, shots_total, shots_on_target, 
                        passes_total, passes_accuracy, key_passes,
                        tackles_total, blocks, interceptions,
                        duels_total, duels_won,
                        dribbles_attempts, dribbles_success, dribbles_past,
                        fouls_drawn, fouls_committed,
                        cards_yellow, cards_red,
                        penalty_won, penalty_committed, penalty_scored, penalty_missed, penalty_saved,
                        saves, goals_conceded
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    data_tuple = (
                        season, fix_id, team_name, p['name'], p['id'], s['games']['position'], s['games']['minutes'] or 0,
                        s['goals']['total'] or 0, s['goals']['assists'] or 0, s['shots']['total'] or 0, s['shots']['on'] or 0,
                        s['passes']['total'] or 0, s['passes']['accuracy'] or "0%", s['passes']['key'] or 0,
                        s['tackles']['total'] or 0, s['tackles']['blocks'] or 0, s['tackles']['interceptions'] or 0,
                        s['duels']['total'] or 0, s['duels']['won'] or 0,
                        s['dribbles']['attempts'] or 0, s['dribbles']['success'] or 0, s['dribbles']['past'] or 0,
                        s['fouls']['drawn'] or 0, s['fouls']['committed'] or 0,
                        s['cards']['yellow'] or 0, s['cards']['red'] or 0,
                        s['penalty']['won'] or 0, s['penalty']['commited'] or 0, s['penalty']['scored'] or 0,
                        s['penalty']['missed'] or 0, s['penalty']['saved'] or 0,
                        s['goals'].get('saves') or 0, s['goals'].get('conceded') or 0
                    )
                    cursor.execute(insert_query, data_tuple)

            # Commit after each match and sleep briefly to stay within the API rate limit
            conn.commit()
            total_matches_processed += 1
            time.sleep(0.4)

    cursor.close()
    conn.close()
    print(f"\n Added {total_matches_processed} new matches to 'player_data_2'.")


if __name__ == "__main__":
    run_bulk_scrape()