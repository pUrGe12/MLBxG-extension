import requests
from bs4 import BeautifulSoup
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
import re

headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}

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
			print(splitted_raw[ind+2])
			print(splitted_raw[ind+7])
			team_code_mapping[splitted_raw[ind+7].lower()] = int(splitted_raw[ind+2])				# Ensuring that we add the lowercased team names so that its easier to check in the future
except IndexError:
	print('all done')

print(team_code_mapping)

# Player code mapping
import csv

player_code_mapping = {}						# We'll pass this directly to the LLM
file_path = r"razzball.csv"

with open(file_path, mode='r', encoding='utf-8') as fp:
	csv_reader = csv.DictReader(fp)
	for row in csv_reader:
		player_code_mapping[row['Name'].lower()] = row['MLBAMID']

print(player_code_mapping)


import os
import sys

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()
chrome_extension_id = str(os.getenv("chrome_extension_id")).strip()

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

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
output: "player": "ottavino"

Prompt: "How has Red Sox performed this season?"
output:	"team": "Red Sox"

Prompt: "Is Tommy Pham playing in Oakland Athletics?"
output:	"player": "Tommy Pham"

Prompt: "So, whats MLB gonna be like now? Whats the schedule?"
output: "schedule"

Ensure that you output in exactly in this format. \n
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

		if 'player' in output.split(':')[0].strip().lower():
			# This means the user needs the player's information

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

			team_name_normalised = output.split(':')[1].strip().lower()

			matched_entry = next((
				(name, code) for name, code in team_code_mapping.items() if team_name_normalised in name.lower()
				))

			if matched_entry:
				matched_name, matched_code = matched_entry			# Unpacking the tuple
				print(matched_code)

				return ('team', matched_code)

			else:
				print('No such team!')

				return 'Try again, you have got the wrong team name'

		elif "schedule" in output.strip().lower():
			return ('schedule', 'schedule')

		else:
			print("something's seriously wrong")

	except Exception as e:
		print(f"Error generating GPT response in model_json: {e}")
		return 'Try again, there was an error in generating the response'


def call_API(name_code_tuple, year=2024, game_type='R'):
	'''
	This is the main calling API. It takes the player or the team's code, and retrieves information regarding it.
	
	I have entered the default values for the year and game_type because I am sure the user will not be entering the year and all
	'''

	assert type(name_code_tuple) == tuple, 'You are trying to feed me bad data! Try again'

	type, code = name_code_tuple

	if type == code: 			# That is both are schedule, we're gonna assume 2024 for now, later we can add the yaer as well
		url_1 = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType={game_type}"
		requests.get(url_1).text

		# Now parse this 

		output = 'parsed content here'

	if 'player' in type:	 		# No need to go lower and all here, I have myself defined this

		# This is the relevant url for the player's data
		url_4 = f"https://statsapi.mlb.com/api/v1/people/{code}"
		requests.get(url_4).text

		# Now parse this

		output = 'parsed content here'

	elif 'team' in type:
		url_2 = f"https://statsapi.mlb.com/api/v1/teams/{team_code}/roster?season={year}"
		url_3 = f"https://statsapi.mlb.com/api/v1/teams/{team_code}"

		webpage_2 = requests.get(url_2).text
		webpage_3 = requests.get(url_3).text

		# Now parse these

		output = 'parsed content here'

	else:
		output = "Nah, shoudn't come here at all. Just for safety"

	return output


def pretty_print(raw_information):
	'''
	This is a pretty_printer which takes in the raw information obtained via the API endpoints and prints it in a easy to ready manner
	'''

	prompt = pretty_print_prompt + f"""
This is the raw information: {raw_information} \n
					"""

	try:
		output = ''
		response = chat.send_message(prompt, stream=False, safety_settings={
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



user_prompt = input('enter stuff: ')
print(figure_out_code(team_code_mapping, player_code_mapping, user_prompt))

year = 2024
game_type = "R"
team_code = 119
player_code = 660271

url_1 = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType={game_type}" # Getting the schedule
url_2 = f"https://statsapi.mlb.com/api/v1/teams/{team_code}/roster?season={year}"
url_3 = f"https://statsapi.mlb.com/api/v1/teams/{team_code}"
url_4 = f"https://statsapi.mlb.com/api/v1/people/{player_code}"
