import pandas as pd
from collections import deque
import networkx as nx

# Load the datasets
schedule_path = "C:/Users/benny/OneDrive/Desktop/NFL Game Predictor/nfl-2024-UTC.csv"
weather_path = "C:/Users/benny/OneDrive/Desktop/NFL Game Predictor/games_weather.csv"
stadium_path = "C:/Users/benny/OneDrive/Desktop/NFL Game Predictor/stadium_coordinates.csv"

real_schedule = pd.read_csv(schedule_path)
real_weather = pd.read_csv(weather_path)
stadium_data = pd.read_csv(stadium_path)

# Mapping of full team names to abbreviations
full_name_to_abbr = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

# Step 1: Process schedule data
real_schedule.rename(
    columns={"Round Number": "Week", "Home Team": "HomeTeam", "Away Team": "VisitorTeam", "Date": "GameDate"},
    inplace=True,
)
real_schedule['GameDate'] = pd.to_datetime(real_schedule['GameDate'], format='%d/%m/%Y %H:%M')
real_schedule['HomeTeamAbbr'] = real_schedule['HomeTeam'].map(full_name_to_abbr)
real_schedule['VisitorTeamAbbr'] = real_schedule['VisitorTeam'].map(full_name_to_abbr)
real_schedule = real_schedule.dropna(subset=['HomeTeamAbbr', 'VisitorTeamAbbr'])

# Step 2: Process weather data
real_weather['Year'] = real_weather['game_id'].astype(str).str[:4]
real_weather['Date'] = real_weather['TimeMeasure'].str.extract(r'(\d{1,2}/\d{1,2})')
real_weather['Date'] = pd.to_datetime(real_weather['Date'] + '/' + real_weather['Year'], format='%m/%d/%Y')

# Merge schedule with weather data
real_schedule['GameDateOnly'] = real_schedule['GameDate'].dt.date
real_weather['DateOnly'] = real_weather['Date'].dt.date
merged_schedule_weather = pd.merge(
    real_schedule, real_weather, left_on='GameDateOnly', right_on='DateOnly', how='left'
)

# Step 3: Handle missing weather data
merged_schedule_weather['Temperature'] = merged_schedule_weather['Temperature'].fillna(60)  # Default temperature
merged_schedule_weather['Humidity'] = merged_schedule_weather['Humidity'].fillna(50)       # Default humidity
merged_schedule_weather['EstimatedCondition'] = merged_schedule_weather['EstimatedCondition'].fillna('Clear')

# Define prediction logic
def predict_winner(row):
    home_score = 80  # Base score for Home Team
    away_score = 80  # Base score for Visitor Team

    # Adjust scores based on weather
    if row['EstimatedCondition'] in ['Rain', 'Snow']:
        home_score *= 0.9
        away_score *= 0.9
    return row['HomeTeam'] if home_score > away_score else row['VisitorTeam']

# Apply prediction logic
merged_schedule_weather['PredictedWinner'] = merged_schedule_weather.apply(predict_winner, axis=1)

# Implement a Queue for scheduling games
game_queue = deque(merged_schedule_weather[['HomeTeam', 'VisitorTeam']].values.tolist())

# Process games using a queue
while game_queue:
    game = game_queue.popleft()
    print(f"Processing game: {game[0]} vs {game[1]}")

# Implement a Tree to represent a playoff bracket
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Example: Creating a playoff tree
root = TreeNode("Season")
week1 = TreeNode("Week 1")
week2 = TreeNode("Week 2")
root.left = week1
root.right = week2

# Implement a Graph for team relationships
team_graph = nx.Graph()

# Add teams and edges (rivalries or matchups)
for _, row in merged_schedule_weather.iterrows():
    team_graph.add_edge(row['HomeTeamAbbr'], row['VisitorTeamAbbr'])

# Display graph nodes and edges
print("Graph Nodes:", team_graph.nodes())
print("Graph Edges:", team_graph.edges())

# Group predictions by week
weekly_predictions = merged_schedule_weather.groupby('Week')[
    ['HomeTeam', 'VisitorTeam', 'PredictedWinner', 'Temperature', 'EstimatedCondition']
].apply(lambda x: x.reset_index(drop=True))

# Display weekly predictions
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
print(weekly_predictions)
