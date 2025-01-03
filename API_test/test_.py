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