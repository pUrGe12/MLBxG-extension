# Imports for scraping
import requests
from bs4 import BeautifulSoup
import re

# Imports for url_1, schedules 
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

# Imports for url_4, players 
from dataclasses import dataclass
from typing import Optional, List, Dict
import json
from datetime import datetime

# Imports for url_2, roster
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

# Import for player code mapping
import csv

# Imports for .env file
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Imports for Gemini
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#																					Creating dictionary mappings
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}

# This is the code used to create the dictionaries

# mapping_ = r"https://github.com/jasonlttl/gameday-api-docs/blob/master/team-information.md"
# website = requests.get(mapping_, headers=headers).text
# soup = BeautifulSoup(website, "html.parser")

# for i in soup.find_all('tr'):
# 	print(i.text)

raw = """

108
ana
ana
LAA
LA Angels
Los Angeles Angels
Angels


109
ari
ari
ARI
Arizona
Arizona Diamondbacks
D-backs


110
bal
bal
BAL
Baltimore
Baltimore Orioles
Orioles


111
bos
bos
BOS
Boston
Boston Red Sox
Red Sox


112
chn
chc
CHC
Chi Cubs
Chicago Cubs
Cubs


113
cin
cin
CIN
Cincinnati
Cincinnati Reds
Reds


114
cle
cle
CLE
Cleveland
Cleveland Indians
Indians


115
col
col
COL
Colorado
Colorado Rockies
Rockies


116
det
det
DET
Detroit
Detroit Tigers
Tigers


117
hou
hou
HOU
Houston
Houston Astros
Astros


118
kca
kc
KC
Kansas City
Kansas City Royals
Royals


119
lan
la
LAD
LA Dodgers
Los Angeles Dodgers
Dodgers


120
was
was
WSH
Washington
Washington Nationals
Nationals


121
nyn
nym
NYM
NY Mets
New York Mets
Mets


133
oak
oak
OAK
Oakland
Oakland Athletics
Athletics


134
pit
pit
PIT
Pittsburgh
Pittsburgh Pirates
Pirates


135
sdn
sd
SD
San Diego
San Diego Padres
Padres


136
sea
sea
SEA
Seattle
Seattle Mariners
Mariners


137
sfn
sf
SF
San Francisco
San Francisco Giants
Giants


138
sln
stl
STL
St. Louis
St. Louis Cardinals
Cardinals


139
tba
tb
TB
Tampa Bay
Tampa Bay Rays
Rays


140
tex
tex
TEX
Texas
Texas Rangers
Rangers


141
tor
tor
TOR
Toronto
Toronto Blue Jays
Blue Jays


142
min
min
MIN
Minnesota
Minnesota Twins
Twins


143
phi
phi
PHI
Philadelphia
Philadelphia Phillies
Phillies


144
atl
atl
ATL
Atlanta
Atlanta Braves
Braves


145
cha
cws
CWS
Chi White Sox
Chicago White Sox
White Sox


146
mia
mia
MIA
Miami
Miami Marlins
Marlins


147
nya
nyy
NYY
NY Yankees
New York Yankees
Yankees


158
mil
mil
MIL
Milwaukee
Milwaukee Brewers
Brewers


159
aas
al
AL
AL All-Stars
American League All-Stars
AL All-Stars


160
nas
nl
NL
NL All-Stars
National League All-Stars
NL All-Stars

"""

# Creating team code mapping
team_code_mapping = {} 							# We'll pass this directly to the LLM 

splitted_raw = raw.split('\n')

try:
	for ind, val in enumerate(splitted_raw):
		if val == "" and splitted_raw[ind+1] == "":
			team_code_mapping[splitted_raw[ind+7].lower()] = int(splitted_raw[ind+2])				# Ensuring that we add the lowercased team names so that its easier to check in the future
except IndexError:
	pass

player_code_mapping = {}						# We'll pass this directly to the LLM
file_path = r"API_querying/razzball.csv"

with open(file_path, mode='r', encoding='utf-8') as fp:
	csv_reader = csv.DictReader(fp)
	for row in csv_reader:
		player_code_mapping[row['Name'].lower()] = row['MLBAMID']

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#																				The Extraction model
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

global figure_out_prompt, pretty_print_prompt
figure_out_prompt = '''
You are duxMLB, an agent designed to answer baseball questions. You focus specifically on providing information on baseball teams and players. 

You will be given the user's prompt. Your job is to extract one of the three things from that.

1. The player's name
2. The team's name
3. If the user is asking for the "schedule"

If both the team's name and the player's name is present then give the player's name

For example, 
Prompt: "Is ottavino playing this season?"
output: player: ottavino

Prompt: "How has Red Sox performed this season?"
output:	team: Red Sox

Prompt: "Is Tommy Pham playing in Oakland Athletics?"
output:	player: Tommy Pham

Prompt: "So, whats MLB gonna be like now? Whats the schedule?"
output: schedule

Ensure that you output in exactly in this format. Do not include any brackets of any kind or anything else in the output.
'''

pretty_print_prompt = """
You are duxMLB. You have been provided some raw information on baseball stats of either a team or a player. Your task is to beautify it and make it look cleaner to present to a user.

You must output the facts exactly as provided, only ensure that they follow grammatical rules and are easy to read by a user.
"""

def figure_out_code(team_code_mapping, player_code_mapping, user_prompt):
	'''
		Takes the user's input and determines the code. We can't directly use the dictionaries because the names given to us is not exact.
	'''

	assert type(team_code_mapping) == dict and type(player_code_mapping) == dict, 'entered parameters are not of type dict'

	load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

	API_KEY = str(os.getenv("API_KEY")).strip()

	# print(API_KEY)

	chrome_extension_id = str(os.getenv("chrome_extension_id")).strip()

	genai.configure(api_key=API_KEY)

	model = genai.GenerativeModel('gemini-pro')
	chat = model.start_chat(history=[])

	print('model initialised')

	prompt = figure_out_prompt + f"""
		This is the user's prompt: {user_prompt} \n
						"""
	try:
		output = ''
		response = chat.send_message(prompt, stream=False, safety_settings={
			HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
			HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
		})

		# Sometimes the model acts up...
		if not response:
			raise ValueError("No response received")

		for chunk in response:
			if chunk.text:
				output += str(chunk.text)

		print(output)

		try:
			if 'player' in output.split(':')[0].strip().lower():
				# This means the user needs the player's information

				# print("inside the player if")

				player_name_normalised = output.split(':')[1].strip().lower() 			# since we have added all lowercased names to the dictionary, we'll first lower the input name (even if its only partial)

				matched_entry = next((								# using next because we want to find the first match and that's exactly what this does
									(name, code) for name, code in player_code_mapping.items() if player_name_normalised in name.lower()
									))			# Still lowering the name just in case.

				if matched_entry:
					matched_name, matched_code = matched_entry 			# This will be a tuple and hence we must unpack this
					print(matched_code)

					return ('player', matched_code)

				else:
					print('No such player!')					 # we'll have to handle this differently by displaying something like can't really find out about this team etc.

					return 'Try again, you have got the wrong player name'

			elif 'team' in output.split(':')[0].strip().lower():
				# This means user wants the team's information, so we need the team code

				print('inside team if statement')

				team_name_normalised = output.split(':')[1].strip().lower()					# Getting the lowercase name of the team

				matched_entry = next((
					(name, code) for name, code in team_code_mapping.items() if team_name_normalised in name.lower()			# search for any matching entries in the dictionary
					))				

				if matched_entry:
					matched_name, matched_code = matched_entry			# Unpacking the tuple
					print(matched_code)

					return ('team', matched_code)

				else:
					print('No such team!')

					return 'Try again, you have got the wrong team name'

			elif "schedule" in output.strip().lower():
				return ('schedule', 'schedule')							# Just ensuring that we know the schedule is being asked of, since we need to unpack the tuple later on, gotta do this.

			else:
				print("something's seriously wrong")
		
		except Exception as e:
			print(e)

	except Exception as e:
		print(f"Error generating: {e}")
		return 'Try again, there was an error in generating the response'


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#																			Helper classes and functions for parsing of player data
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Position_:
    code: str
    name: str
    type: str
    abbreviation: str

@dataclass
class BatPitchSide:
    code: str
    description: str

@dataclass
class Player_:
    id: int
    full_name: str
    first_name: str
    last_name: str
    primary_number: Optional[str]
    birth_date: datetime
    current_age: int
    birth_city: str
    birth_state_province: Optional[str]
    birth_country: str
    height: str
    weight: int
    active: bool
    primary_position: Position_
    use_name: str
    use_last_name: str
    middle_name: Optional[str]
    boxscore_name: str
    nick_name: Optional[str]
    gender: str
    is_player: bool
    is_verified: bool
    draft_year: Optional[int]
    pronunciation: Optional[str]
    last_played_date: Optional[str]
    mlb_debut_date: Optional[str]
    bat_side: BatPitchSide
    pitch_hand: BatPitchSide
    name_first_last: str
    name_slug: str
    first_last_name: str
    last_first_name: str
    last_init_name: str
    init_last_name: str
    full_fml_name: str
    full_lfm_name: str
    strike_zone_top: float
    strike_zone_bottom: float

def parse_player_data(json_data: str) -> List[Player_]:
    """
    Parse MLB player JSON data and return a list of Player objects.
    """
    data = json.loads(json_data)
    players = []
    
    for player_data in data['people']:
        # Parse position
        position = Position(
            code=player_data['primaryPosition']['code'],
            name=player_data['primaryPosition']['name'],
            type=player_data['primaryPosition']['type'],
            abbreviation=player_data['primaryPosition']['abbreviation']
        )
        
        # Parse bat and pitch sides
        bat_side = BatPitchSide(
            code=player_data['batSide']['code'],
            description=player_data['batSide']['description']
        )
        pitch_hand = BatPitchSide(
            code=player_data['pitchHand']['code'],
            description=player_data['pitchHand']['description']
        )
        
        # Parse birth date
        birth_date = datetime.strptime(player_data['birthDate'], '%Y-%m-%d')
        
        # Create Player object
        player = Player_(
            id=player_data['id'],
            full_name=player_data['fullName'],
            first_name=player_data['firstName'],
            last_name=player_data['lastName'],
            primary_number=player_data.get('primaryNumber'),
            birth_date=birth_date,
            current_age=player_data['currentAge'],
            birth_city=player_data['birthCity'],
            birth_state_province=player_data.get('birthStateProvince'),
            birth_country=player_data['birthCountry'],
            height=player_data['height'],
            weight=player_data['weight'],
            active=player_data['active'],
            primary_position=position,
            use_name=player_data['useName'],
            use_last_name=player_data['useLastName'],
            middle_name=player_data.get('middleName'),
            boxscore_name=player_data['boxscoreName'],
            nick_name=player_data.get('nickName'),
            gender=player_data['gender'],
            is_player=player_data['isPlayer'],
            is_verified=player_data['isVerified'],
            draft_year=player_data.get('draftYear'),
            pronunciation=player_data.get('pronunciation'),
            last_played_date=player_data.get('lastPlayedDate'),
            mlb_debut_date=player_data.get('mlbDebutDate'),
            bat_side=bat_side,
            pitch_hand=pitch_hand,
            name_first_last=player_data['nameFirstLast'],
            name_slug=player_data['nameSlug'],
            first_last_name=player_data['firstLastName'],
            last_first_name=player_data['lastFirstName'],
            last_init_name=player_data['lastInitName'],
            init_last_name=player_data['initLastName'],
            full_fml_name=player_data['fullFMLName'],
            full_lfm_name=player_data['fullLFMName'],
            strike_zone_top=player_data['strikeZoneTop'],
            strike_zone_bottom=player_data['strikeZoneBottom']
        )
        players.append(player)
    
    return players


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 																			Helper classes and functions for helping with parsing of url_2
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class PlayerStatus(Enum):
	ACTIVE = "A"
	FORTY_MAN = "40M"
	MINOR_LEAGUE = "MIN"
	INJURED_60_DAY = "D60"
	RELEASED = "RL"
	TRADED = "TR"
	CLAIMED = "CL"

@dataclass
class Position:
	code: str
	name: str
	type: str
	abbreviation: str

@dataclass
class Player:
	id: int
	full_name: str
	jersey_number: str
	position: Position
	status: str
	status_description: str

def parse_roster_data(json_data: str) -> List[Player]:
	"""
	Parse MLB roster JSON data and return a list of Player objects.

	Args:
	json_data (str): JSON string containing MLB roster data

	Returns:
	List[Player]: List of parsed Player objects
	"""

	data = json.loads(json_data)

	roster_data = data.get('roster', [])

	# Process each player
	players = []
	try:
		for player_data in roster_data:
			# Extract person info
			person = player_data['person']

			# Create Position object
			position_data = player_data['position']
			position = Position(
				code=position_data['code'],
				name=position_data['name'],
				type=position_data['type'],
				abbreviation=position_data['abbreviation']
				)

			# Create Player object
			player = Player(
				id=person['id'],
				full_name=person['fullName'],
				jersey_number=player_data['jerseyNumber'],
				position=position,
				status=player_data['status']['code'],
				status_description=player_data['status']['description']
			)

			players.append(player)
	    
		return players

	except Exception as e:
		return f"Found error: {e}"

def get_roster_analysis(players: List[Player]) -> Dict:
    """
    Analyze roster data and return various statistics.
    
    Args:
        players (List[Player]): List of Player objects
        
    Returns:
        Dict: Dictionary containing roster analysis
    """
    analysis = {
        'total_players': len(players),
        'active_players': len([p for p in players if p.status == PlayerStatus.ACTIVE.value]),
        'positions': {},
        'status_breakdown': {},
    }
    
    # Count players by position
    for player in players:
        pos_type = player.position.type
        analysis['positions'][pos_type] = analysis['positions'].get(pos_type, 0) + 1
        
        status = player.status
        analysis['status_breakdown'][status] = analysis['status_breakdown'].get(status, 0) + 1
    
    return analysis

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 																	Helper classes and functions for helping with parsing of url_1 (schedule)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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

global output_
output_ = ""

def add_to_output(text: str):
    """
    Helper function to add text to the output variable. 
    Have defined output as the global variable cause will be using it directly in the parsing code as well
    """
    global output_
    output_ += text + "\n"

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#																						Pretty printing LLM
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def pretty_print(raw_information):
	'''
	This is a pretty_printer which takes in the raw information obtained via the API endpoints and prints it in a easy to ready manner
	'''

	load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

	API_KEY = str(os.getenv("API_KEY")).strip()

	# print(API_KEY)

	chrome_extension_id = str(os.getenv("chrome_extension_id")).strip()

	genai.configure(api_key=API_KEY)

	model_ = genai.GenerativeModel('gemini-pro')
	_chat_ = model_.start_chat(history=[])

	prompt = pretty_print_prompt + f"""
			This is the raw information: {raw_information} \n
					"""

	try:
		output = ''
		response = _chat_.send_message(prompt, stream=False, safety_settings={
			HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
			HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
		})

		if not response:
			raise ValueError('No response received')

		for chunk in response:
			if chunk.text:
				output += str(chunk.text)

		return output

	except Exception as e:
		print(f"Something went wrong: {e}\n")
		return "Try again"


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#																				API calling and parsing here
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def call_API(name_code_tuple, year=2024, game_type='R'):
	'''
	This is the main calling API. It takes the player or the team's code, and retrieves information regarding it.
	
	I have entered the default values for the year and game_type because I am sure the user will not be entering the year and all
	'''

	assert type(name_code_tuple) == tuple, 'Maybe you made a mistake in the name? Check your spellings please! I am doing this all manually, an AI can get tired!'
	# print(name_code_tuple)

	# After assert we don't have to use try and accept
	type_, code = name_code_tuple

	if type_ == code: 			# That is both are schedule, we're gonna assume 2024 for now, later we can add the yaer as well
		url_1 = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType={game_type}"
		webpage = requests.get(url_1, headers=headers).text

		schedule = parse_schedule(webpage)
		analysis = get_schedule_analysis(schedule)
		print_schedule_summary(analysis)

		# At this point, the 'output' variable contains all the logged information

		# We don't have to pass this to the LLM, we'll get the relevant stuff here, cause it won't be able to handle this much data
		# For now returning output, but we should parse this more carefully. 
		return output_		

	if 'player' in type_:	 		# No need to go lower and all here, I have myself defined this

		# This is the relevant url for the player's data
		url_4 = f"https://statsapi.mlb.com/api/v1/people/{code}"
		webpage = requests.get(url_4, headers=headers).text

		# Now parse this
		players = parse_player_data(webpage)

		_output_ = """"""  										# ensuring this is empty
		for player in players:
			print(f"\nPlayer Information:")
			_output_ += "\nPlayer Information:"

			print(f"Name: {player.full_name}")
			_output_ += f"\nName: {player.full_name}"

			print(f"Position: {player.primary_position.name}")
			_output_ +=  f"\nPosition: {player.primary_position.name}"

			print(f"Birth Date: {player.birth_date.strftime('%B %d, %Y')}")
			_output_ += f"\nBirth Date: {player.birth_date.strftime('%B %d, %Y')}"

			print(f"From: {player.birth_city}, {player.birth_state_province}, {player.birth_country}")
			_output_ += f"\nFrom: {player.birth_city}, {player.birth_state_province}, {player.birth_country}"

			print(f"Height/Weight: {player.height}, {player.weight} lbs")
			_output_ += f"\nHeight/Weight: {player.height}, {player.weight} lbs"

			print(f"Bats: {player.bat_side.description}")
			_output_ += f"\nBats: {player.bat_side.description}"

			print(f"Throws: {player.pitch_hand.description}")
			_output_ += f"\nThrows: {player.pitch_hand.description}"

			print(f"Draft Year: {player.draft_year}")
			_output_ += f"\nDraft Year: {player.draft_year}"

			print(f"MLB Debut: {player.mlb_debut_date}")
			_output_ += f"\nMLB Debut: {player.mlb_debut_date}"

			print(f"Last Played: {player.last_played_date}")
			_output_ += f"\nLast Played: {player.last_played_date}"

			print(f"Active: {player.active}")
			_output_ += f"\nActive: {player.active}"

			print(f"Nick_name: {player.nick_name}")
			_output_ += f"\nNick_name: {player.nick_name}"

			print(f"Name_slug: {player.name_slug}")
			_output_ += f"\nName_slug: {player.name_slug}"

			print(f"Bat_side: {player.bat_side}")
			_output_ += f"\nBat_side: {player.bat_side}"

			print(f"Top strike zone: {player.strike_zone_top}")
			_output_ += f"\nTop strike zone: {player.strike_zone_top}"

			print(f"Pitch hand: {player.pitch_hand}")
			_output_ += f"\nPitch hand: {player.pitch_hand}"

			print(f"Gender: {player.gender}")
			_output_ += f"\nGender: {player.gender}"

			print(f"Primary position: {player.primary_position}")
			_output_ += f"\nPrimary position: {player.primary_position}"

			print(f"Primary number: {player.primary_number}")
			_output_ += f"\nPrimary number: {player.primary_number}"

			print(f"Current age: {player.current_age}")
			_output_ += f"\nCurrent age: {player.current_age}"

		return _output_

	elif 'team' in type_:
		output = "" 				# we'll write everything we print here
		url_2 = f"https://statsapi.mlb.com/api/v1/teams/{code}/roster?season={year}"

		json_str = requests.get(url_2, headers=headers).text

		roster = parse_roster_data(json_str)

		# Get analysis
		analysis = get_roster_analysis(roster)

		# Print some example outputs
		# print(f"Total players: {analysis['total_players']}")
		output += f"Total players: {analysis['total_players']} \n"

		# print(f"Active players: {analysis['active_players']}")
		output += f"Active players: {analysis['active_players']} \n"

		# print("\nPosition breakdown:")
		output += f"Position breakdown:\n"

		for pos, count in analysis['positions'].items():
			# print(f"{pos}: {count}")
			output += f"{pos}: {count} \n"

		# print("\nStatus breakdown:")
		output += "\nStatus breakdown:"

		for status, count in analysis['status_breakdown'].items():
			# print(f"{status}: {count}")
			output += f"{status}: {count} \n"

		# Example of filtering players
		active_pitchers = [
			p for p in roster 
			if p.status == PlayerStatus.ACTIVE.value and p.position.type == "Pitcher"
			]
		# print("\nActive pitchers:")
		output += "\nActive pitchers:"

		for pitcher in active_pitchers:
			# print(f"{pitcher.full_name} (#{pitcher.jersey_number})")
			output += f"{pitcher.full_name} (#{pitcher.jersey_number}) \n"

		return output

	else:
		output = "Nah, shoudn't come here at all. Just for safety"

		# Since we have added all the information in the output (exactly as it is being printed) we are good to go
		return output