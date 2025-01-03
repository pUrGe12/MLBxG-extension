"""import csv

player_code_mapping = {}						# We'll pass this directly to the LLM
file_path = r"razzball.csv"

with open(file_path, mode='r', encoding='utf-8') as fp:
	csv_reader = csv.DictReader(fp)
	for row in csv_reader:
		player_code_mapping[row['Name'].lower()] = row['MLBAMID']

# print(player_code_mapping)

output = "player: Gary SanCheZ"

if 'player' in output.split(':')[0].strip().lower():
			# This means the user needs the player's information
	player_name_normalised = output.split(':')[1].strip().lower() 			# since we have added all lowercased names to the dictionary, we'll first lower the input name (even if its only partial)

	matched_entry = next((								# using next because we want to find the first match and that's exactly what this does
						(name, code) for name, code in player_code_mapping.items() if player_name_normalised in name.lower()
						))			# Still lowering the name just in case.

	if matched_entry:
		matched_name, matched_code = matched_entry 			# This will be a tuple and hence we must unpack this
		print(matched_code)
	else:
		print('no such player!')"""


# Testing the parsing of the API webpages.

import requests
# from bs4 import BeautifulSoup
# import json

year = 2024
game_type = "R"
team_code = 119
player_code = 660271

url_1 = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType={game_type}" # Getting the schedule
url_2 = f"https://statsapi.mlb.com/api/v1/teams/{team_code}/roster?season={year}"
url_3 = f"https://statsapi.mlb.com/api/v1/teams/{team_code}"
url_4 = f"https://statsapi.mlb.com/api/v1/people/{player_code}"

webpage = requests.get(url_1).text

# from typing import Dict, List
# from dataclasses import dataclass
# from enum import Enum

# class PlayerStatus(Enum):
#     ACTIVE = "A"
#     FORTY_MAN = "40M"
#     MINOR_LEAGUE = "MIN"
#     INJURED_60_DAY = "D60"
#     RELEASED = "RL"
#     TRADED = "TR"
#     CLAIMED = "CL"

# @dataclass
# class Position:
#     code: str
#     name: str
#     type: str
#     abbreviation: str

# @dataclass
# class Player:
#     id: int
#     full_name: str
#     jersey_number: str
#     position: Position
#     status: str
#     status_description: str

# def parse_roster_data(json_data: str) -> List[Player]:
#     """
#     Parse MLB roster JSON data and return a list of Player objects.
    
#     Args:
#         json_data (str): JSON string containing MLB roster data
        
#     Returns:
#         List[Player]: List of parsed Player objects
#     """
#     # Parse JSON data
#     data = json.loads(json_data)
    
#     # Extract roster information
#     roster_data = data.get('roster', [])
    
#     # Process each player
#     players = []
#     for player_data in roster_data:
#         # Extract person info
#         person = player_data['person']
        
#         # Create Position object
#         position_data = player_data['position']
#         position = Position(
#             code=position_data['code'],
#             name=position_data['name'],
#             type=position_data['type'],
#             abbreviation=position_data['abbreviation']
#         )
        
#         # Create Player object
#         player = Player(
#             id=person['id'],
#             full_name=person['fullName'],
#             jersey_number=player_data['jerseyNumber'],
#             position=position,
#             status=player_data['status']['code'],
#             status_description=player_data['status']['description']
#         )
        
#         players.append(player)
    
#     return players

# def get_roster_analysis(players: List[Player]) -> Dict:
#     """
#     Analyze roster data and return various statistics.
    
#     Args:
#         players (List[Player]): List of Player objects
        
#     Returns:
#         Dict: Dictionary containing roster analysis
#     """
#     analysis = {
#         'total_players': len(players),
#         'active_players': len([p for p in players if p.status == PlayerStatus.ACTIVE.value]),
#         'positions': {},
#         'status_breakdown': {},
#     }
    
#     # Count players by position
#     for player in players:
#         pos_type = player.position.type
#         analysis['positions'][pos_type] = analysis['positions'].get(pos_type, 0) + 1
        
#         status = player.status
#         analysis['status_breakdown'][status] = analysis['status_breakdown'].get(status, 0) + 1
    
#     return analysis

# # Example usage
# if __name__ == "__main__":
# # Example JSON string (would be replaced with actual data)
# 	output = ""
# 	json_str = webpage # Your JSON data here

# 	# Parse roster
# 	roster = parse_roster_data(json_str)

# 	# Get analysis
# 	analysis = get_roster_analysis(roster)

# 	# Print some example outputs
# 	print(f"Total players: {analysis['total_players']}")
# 	output += f"Total players: {analysis['total_players']} \n"

# 	print(f"Active players: {analysis['active_players']}")
# 	output += f"Active players: {analysis['active_players']} \n\n"

# 	print("\nPosition breakdown:")
# 	output += f"Active players:\n"

# 	for pos, count in analysis['positions'].items():
# 		print(f"{pos}: {count}")
# 		output += f"{pos}: {count} \n"

# 	print("\nStatus breakdown:")
# 	output += "Status breakdown:"

# 	for status, count in analysis['status_breakdown'].items():
# 		print(f"{status}: {count}")
# 		output += f"{status}: {count} \n"

# 	# Example of filtering players
# 	active_pitchers = [
# 	p for p in roster 
# 	if p.status == PlayerStatus.ACTIVE.value and p.position.type == "Pitcher"
# 	]
# 	print("\nActive pitchers:")
# 	output += "Active pitchers:"

# 	for pitcher in active_pitchers:
# 		print(f"{pitcher.full_name} (#{pitcher.jersey_number})")
# 		output += f"{pitcher.full_name} (#{pitcher.jersey_number}) \n"

# 	print(output)


import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

@dataclass
class TeamRecord:
    wins: int
    losses: int
    pct: str

@dataclass
class TeamInfo:
    id: int
    name: str
    link: str

@dataclass
class TeamGameInfo:
    team: TeamInfo
    is_winner: bool
    league_record: TeamRecord
    split_squad: bool
    series_number: int

@dataclass
class Venue:
    id: int
    name: str
    link: str

@dataclass
class GameStatus:
    abstract_state: str
    coded_state: str
    detailed_state: str
    status_code: str
    abstract_code: str

@dataclass
class Game:
    game_pk: int
    game_guid: str
    game_type: str
    season: str
    game_date: datetime
    status: GameStatus
    home_team: TeamGameInfo
    away_team: TeamGameInfo
    venue: Venue
    day_night: str
    description: Optional[str]
    scheduled_innings: int
    series_description: str
    series_game_number: int

output = ""

def add_to_output(text: str):
    """Helper function to add text to the output variable"""
    global output
    output += text + "\n"

def parse_team_info(data: Dict) -> TeamGameInfo:
    """Parse team information from game data."""
    add_to_output("\nParsing team info:")
    add_to_output(f"Team Name: {data['team']['name']}")
    add_to_output(f"Team Record: {data['leagueRecord']['wins']}-{data['leagueRecord']['losses']} ({data['leagueRecord']['pct']})")
    add_to_output(f"Is Winner: {data.get('isWinner', False)}")
    add_to_output(f"Split Squad: {data['splitSquad']}")
    add_to_output(f"Series Number: {data['seriesNumber']}")
    
    team_data = data['team']
    record = TeamRecord(
        wins=data['leagueRecord']['wins'],
        losses=data['leagueRecord']['losses'],
        pct=data['leagueRecord']['pct']
    )
    team = TeamInfo(
        id=team_data['id'],
        name=team_data['name'],
        link=team_data['link']
    )
    return TeamGameInfo(
        team=team,
        is_winner=data.get('isWinner', False),
        league_record=record,
        split_squad=data['splitSquad'],
        series_number=data['seriesNumber']
    )

def parse_game(game_data: Dict) -> Game:
    """Parse individual game data."""
    add_to_output("\nParsing game:")
    add_to_output(f"Game ID: {game_data['gamePk']}")
    add_to_output(f"Game Date: {game_data['gameDate']}")
    add_to_output(f"Status: {game_data['status']['detailedState']}")
    add_to_output(f"Venue: {game_data['venue']['name']}")
    add_to_output(f"Game Type: {game_data['gameType']}")
    add_to_output(f"Series Description: {game_data['seriesDescription']}")
    add_to_output(f"Game Number in Series: {game_data['seriesGameNumber']}")
    
    status = GameStatus(
        abstract_state=game_data['status']['abstractGameState'],
        coded_state=game_data['status']['codedGameState'],
        detailed_state=game_data['status']['detailedState'],
        status_code=game_data['status']['statusCode'],
        abstract_code=game_data['status']['abstractGameCode']
    )
    
    venue = Venue(
        id=game_data['venue']['id'],
        name=game_data['venue']['name'],
        link=game_data['venue']['link']
    )
    
    return Game(
        game_pk=game_data['gamePk'],
        game_guid=game_data['gameGuid'],
        game_type=game_data['gameType'],
        season=game_data['season'],
        game_date=datetime.strptime(game_data['gameDate'], "%Y-%m-%dT%H:%M:%SZ"),
        status=status,
        home_team=parse_team_info(game_data['teams']['home']),
        away_team=parse_team_info(game_data['teams']['away']),
        venue=venue,
        day_night=game_data['dayNight'],
        description=game_data.get('description'),
        scheduled_innings=game_data['scheduledInnings'],
        series_description=game_data['seriesDescription'],
        series_game_number=game_data['seriesGameNumber']
    )

def parse_schedule(json_data: str) -> Dict[str, List[Game]]:
    """Parse MLB schedule JSON data and return games organized by date."""
    add_to_output("\nParsing schedule data:")
    data = json.loads(json_data)
    schedule = {}
    
    for date_data in data['dates']:
        date = date_data['date']
        add_to_output(f"\nProcessing date: {date}")
        add_to_output(f"Number of games on this date: {len(date_data['games'])}")
        games = [parse_game(game) for game in date_data['games']]
        schedule[date] = games
    
    return schedule

def get_schedule_analysis(schedule: Dict[str, List[Game]]) -> Dict:
    """Analyze schedule data including matchups and results."""
    add_to_output("\nAnalyzing schedule:")
    
    analysis = {
        'total_games': 0,
        'games_by_date': {},
        'matchups_by_date': defaultdict(list),
        'team_records': defaultdict(lambda: {'wins': 0, 'losses': 0, 'games': 0}),
        'home_team_win_pct': 0,
        'results_summary': []
    }
    
    total_completed_games = 0
    home_team_wins = 0
    
    analysis['total_games'] = sum(len(games) for games in schedule.values())
    add_to_output(f"\nTotal games in schedule: {analysis['total_games']}")
    
    # Process each date and its games
    for date, games in schedule.items():
        add_to_output(f"\nAnalyzing games for {date}:")
        analysis['games_by_date'][date] = len(games)
        add_to_output(f"Games scheduled: {len(games)}")
        
        for game in games:
            home_team = game.home_team.team.name
            away_team = game.away_team.team.name
            add_to_output(f"\nMatchup: {away_team} @ {home_team}")
            add_to_output(f"Status: {game.status.detailed_state}")
            add_to_output(f"Venue: {game.venue.name}")
            add_to_output(f"Time: {game.game_date.strftime('%I:%M %p')}")
            
            # Track matchup with score
            matchup_info = {
                'away_team': away_team,
                'home_team': home_team,
                'status': game.status.detailed_state,
                'venue': game.venue.name,
                'time': game.game_date.strftime('%I:%M %p'),
                'description': game.description
            }
            
            analysis['matchups_by_date'][date].append(matchup_info)
            
            # Update team records
            for team_info in [game.home_team, game.away_team]:
                team_name = team_info.team.name
                analysis['team_records'][team_name]['games'] += 1
                add_to_output(f"\nUpdating record for {team_name}:")
                add_to_output(f"Games played: {analysis['team_records'][team_name]['games']}")
                
                if team_info.is_winner:
                    analysis['team_records'][team_name]['wins'] += 1
                    add_to_output(f"Win recorded - New wins: {analysis['team_records'][team_name]['wins']}")
                elif game.status.abstract_state == "Final":
                    analysis['team_records'][team_name]['losses'] += 1
                    add_to_output(f"Loss recorded - New losses: {analysis['team_records'][team_name]['losses']}")
            
            # Track home team advantage
            if game.status.abstract_state == "Final":
                total_completed_games += 1
                add_to_output(f"\nCompleted game - Total completed games: {total_completed_games}")
                
                if game.home_team.is_winner:
                    home_team_wins += 1
                    add_to_output(f"Home team win recorded - Total home team wins: {home_team_wins}")
    
    if total_completed_games > 0:
        analysis['home_team_win_pct'] = round(home_team_wins / total_completed_games, 3)
        add_to_output(f"\nFinal home team win percentage: {analysis['home_team_win_pct']}")
    
    return analysis

def print_schedule_summary(analysis: Dict):
    """Print a formatted summary of the schedule analysis."""
    add_to_output("\n=== SCHEDULE SUMMARY ===")
    add_to_output(f"Total games scheduled: {analysis['total_games']}\n")
    
    add_to_output("Games by Date:")
    for date, matchups in analysis['matchups_by_date'].items():
        add_to_output(f"\n=== {date} ({len(matchups)} games) ===")
        for matchup in matchups:
            if matchup['status'] == "Final":
                add_to_output(f"{matchup['away_team']} @ "
                          f"{matchup['home_team']} "
                          f"(Final)")
            else:
                add_to_output(f"{matchup['away_team']} @ {matchup['home_team']} "
                          f"({matchup['time']})")
            if matchup['description']:
                add_to_output(f"  Note: {matchup['description']}")
    
    add_to_output("\nTeam Records:")
    for team, record in analysis['team_records'].items():
        if record['games'] > 0:
            win_pct = round(record['wins'] / record['games'], 3)
            add_to_output(f"{team}: {record['wins']}-{record['losses']} ({win_pct})")
    
    add_to_output(f"\nHome Team Win Percentage: {analysis['home_team_win_pct']}")

# Example usage
if __name__ == "__main__":
    # Reset output variable
    output = ""
    
    add_to_output("\n=== Starting MLB Schedule Analysis ===")
    schedule = parse_schedule(webpage)  # Your JSON data here
    analysis = get_schedule_analysis(schedule)
    print_schedule_summary(analysis)
    
    # At this point, the 'output' variable contains all the logged information
    print(output)  # Or use the output variable as needed