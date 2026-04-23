import pandas as pd
from sqlalchemy import create_engine
import networkx as nx

engine = create_engine("mysql+mysqlconnector://root:RoomierCanine24!@localhost/footballdata")

def get_position_group(pos):
    if not pos: return None
    pos = pos.upper()
    if 'GK' in pos: return 'GK'
    if any(df_pos in pos for df_pos in ['DF', 'CB', 'LB', 'RB', 'WB']): return 'DF'
    if any(mf_pos in pos for mf_pos in ['MF', 'DM', 'CM', 'AM', 'LM', 'RM']): return 'MF'
    if any(fw_pos in pos for fw_pos in ['FW', 'ST', 'CF', 'LW', 'RW']): return 'FW'
    return None

def run_advanced_pos_pagerank():
    query = "SELECT game_id, team, player, pos, min, Passes_Cmp, Performance_SoT, Performance_Tkl, Performance_Gls FROM Player_Data WHERE season = 2324"
    df = pd.read_sql(query, engine)
    df = df[df['min'].notnull()]
    df['pos_group'] = df['pos'].apply(get_position_group)

# Database Connection
engine = create_engine("mysql+mysqlconnector://root:RoomierCanine24!@localhost/footballdata")

# Function to simplify positions into categories
def get_position_group(pos):
    """
    Footballers have many specific roles (CB, LW, DM). 
    This function simplifies them into 4 big buckets: GK, DF, MF, FW.
    """
    if not pos: return None
    pos = pos.upper()
    if 'GK' in pos: return 'GK'
    if any(df_pos in pos for df_pos in ['DF', 'CB', 'LB', 'RB', 'WB']): return 'DF'
    if any(mf_pos in pos for mf_pos in ['MF', 'DM', 'CM', 'AM', 'LM', 'RM']): return 'MF'
    if any(fw_pos in pos for fw_pos in ['FW', 'ST', 'CF', 'LW', 'RW']): return 'FW'
    return None

def run_advanced_pos_pagerank():
    # Grab the data from our SQL 
    query = "SELECT game_id, team, player, pos, min, Passes_Cmp, Performance_SoT, Performance_Tkl, Performance_Gls FROM Player_Data"
    df = pd.read_sql(query, engine)
    
    # We only care about people who actually played (minutes not null)
    df = df[df['min'].notnull()]
    df['pos_group'] = df['pos'].apply(get_position_group)

    # Calculate how busy the Goalkeepers were
    # Since our data doesn't explicitly list saves we find them by looking at
    # how many shots the opposing team took compared to how many goals they scored.
    match_team_stats = df.groupby(['game_id', 'team']).agg({
        'Performance_SoT': 'sum',
        'Performance_Gls': 'sum'
    }).reset_index()

    def get_gk_stats(row):
        # Look at the other team in the same match to see what the keeper faced
        match_stats = match_team_stats[match_team_stats['game_id'] == row['game_id']]
        opp_stats = match_stats[match_stats['team'] != row['team']]
        
        if opp_stats.empty: return 0, 0
        
        sot_faced = opp_stats['Performance_SoT'].values[0]
        goals_conceded = opp_stats['Performance_Gls'].values[0]
        
        # Logic: If they faced 5 shots and 1 went in, they must have made 4 saves
        saves = max(0, sot_faced - goals_conceded)
        return saves, sot_faced

    # Identify who won each match
    # Winners get a boost in our rankings
    game_winners = {}
    for gid, group in match_team_stats.groupby('game_id'):
        if len(group) == 2:
            t1, t2 = group.iloc[0], group.iloc[1]
            if t1['Performance_Gls'] > t2['Performance_Gls']: game_winners[gid] = t1['team']
            elif t2['Performance_Gls'] > t1['Performance_Gls']: game_winners[gid] = t2['team']

    # Run PageRank for each position graph
    categories = ['GK', 'DF', 'MF', 'FW']
    category_results = []

    for cat in categories:
        # Create a new empty graph for this position
        G = nx.DiGraph()
        cat_df = df[df['pos_group'] == cat].copy()

        for game_id, game_group in cat_df.groupby('game_id'):
            winner_team = game_winners.get(game_id)
            
            # Decide the score for this game based on the player's role
            if cat == 'GK':
                for i, gk in game_group.iterrows():
                    saves, sot_faced = get_gk_stats(gk)
                    # Keeper score = Save Percentage
                    game_group.at[i, 'metric_val'] = saves / sot_faced if sot_faced > 0 else 0
            elif cat == 'DF':
                # Defender score = Their share of the team's total tackles
                total = game_group['Performance_Tkl'].sum()
                game_group['metric_val'] = game_group['Performance_Tkl'] / total if total > 0 else 0
            elif cat == 'MF':
                # Midfielder score = Their share of the team's total passes
                total = game_group['Passes_Cmp'].sum()
                game_group['metric_val'] = game_group['Passes_Cmp'] / total if total > 0 else 0
            elif cat == 'FW':
                # Forward score = Their share of the team's shots on target
                total = game_group['Performance_SoT'].sum()
                game_group['metric_val'] = game_group['Performance_SoT'] / total if total > 0 else 0
            
            #  Players vote for each other
            players = game_group.to_dict('records')
            for p1 in players:
                # share is how much influence this player has to give away
                share = p1['metric_val']
                # winners boost
                if winner_team and p1['team'] == winner_team: share *= 1.2
                
                for p2 in players:
                    if p1['player'] != p2['player']:
                        # Create a connection: p1 votes for p2 based on their performance
                        if G.has_edge(p2['player'], p1['player']):
                            G[p2['player']][p1['player']]['weight'] += share
                        else:
                            G.add_edge(p2['player'], p1['player'], weight=share)

        # Use the PageRank maths to settle the final scores
        if len(G) > 0:
            scores = nx.pagerank(G, weight='weight', alpha=0.85)
            res_df = pd.DataFrame(list(scores.items()), columns=['Player', 'Score'])
            res_df['Category'] = cat
            
            # Clean up and add the team names back so we can see who plays where
            latest_team = cat_df.drop_duplicates('player', keep='last')[['player', 'team']]
            category_results.append(res_df.merge(latest_team, left_on='Player', right_on='player').drop(columns=['player']))

    return pd.concat(category_results)

if __name__ == "__main__":
    print("Running the Footballer PageRank...")
    results = run_advanced_pos_pagerank()
    
    # Print out the rankings of each position
    for cat in ['GK', 'DF', 'MF', 'FW']:
        print(f"\n--- Top 10 {cat}s ---")
        print(results[results['Category'] == cat].sort_values('Score', ascending=False).head(10))