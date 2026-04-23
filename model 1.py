import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

# Database Connection
engine = create_engine("mysql+mysqlconnector://root:RoomierCanine24!@localhost/footballdata")

def run_winner_weighted_pagerank():
    # Fetch relevant columns
    query = "SELECT game_id, team, player, Passes_Cmp, min, Performance_Gls FROM Player_Data WHERE season = 2324"
    df = pd.read_sql(query, engine)
    
    # Data Cleaning
    df = df[df['min'].notnull()]
    df['Passes_Cmp'] = pd.to_numeric(df['Passes_Cmp'], errors='coerce').fillna(0)
    df['Performance_Gls'] = pd.to_numeric(df['Performance_Gls'], errors='coerce').fillna(0)

    # Determine Match Winners
    # Calculate goals per team per match
    team_goals = df.groupby(['game_id', 'team'])['Performance_Gls'].sum().reset_index()
    
    # Create a dictionary to store the winner of each game_id
    game_winners = {}
    for gid, group in team_goals.groupby('game_id'):
        if len(group) == 2:
            t1_name, t1_goals = group.iloc[0]['team'], group.iloc[0]['Performance_Gls']
            t2_name, t2_goals = group.iloc[1]['team'], group.iloc[1]['Performance_Gls']
            
            if t1_goals > t2_goals:
                game_winners[gid] = t1_name
            elif t2_goals > t1_goals:
                game_winners[gid] = t2_name
            else:
                game_winners[gid] = "Draw"

    # Build the Graph
    G = nx.DiGraph()

    for game_id, game_group in df.groupby('game_id'):
        match_total_passes = game_group['Passes_Cmp'].sum()
        if match_total_passes == 0:
            continue
            
        winner_team = game_winners.get(game_id)
        players = game_group.to_dict('records')
        
        for p1 in players:
            # Fraction of total match passes
            share = p1['Passes_Cmp'] / match_total_passes
            
            # Apply 20% Winning Bonus
            if winner_team and p1['team'] == winner_team:
                share *= 1.20 
            
            if share == 0:
                continue
                
            for p2 in players:
                if p1['player'] != p2['player']:
                    # p1 votes for p2 with their calculated share
                    if G.has_edge(p2['player'], p1['player']):
                        G[p2['player']][p1['player']]['weight'] += share
                    else:
                        G.add_edge(p2['player'], p1['player'], weight=share)

    # Run PageRank
    pagerank_scores = nx.pagerank(G, weight='weight', alpha=0.85)

    # Format Results
    results = pd.DataFrame(list(pagerank_scores.items()), columns=['Player', 'Score'])
    
    # Attach team name for display (from their most recent match)
    latest_teams = df.drop_duplicates('player', keep='last')[['player', 'team']]
    results = results.merge(latest_teams, left_on='Player', right_on='player').drop(columns=['player'])
    
    return results.sort_values(by='Score', ascending=False)

if __name__ == "__main__":
    print("Calculating PageRank with Fractional Passing & Winning Bonus...")
    rankings = run_winner_weighted_pagerank()
    print("\n--- Top 20 Ranked Players ---")
    print(rankings.head(20).to_string(index=False))
