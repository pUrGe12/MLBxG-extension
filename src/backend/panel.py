from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import sys

import requests             # Making post requests to the getBuffer method

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # adding the root directory to path
from prompts import genPrompt, buffer_needed_prompt

from dotenv import load_dotenv
from pathlib import Path

# Imports for API querying 
from API_querying.query import call_API, pretty_print, figure_out_code
from API_querying.query import team_code_mapping, player_code_mapping

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()
chrome_extension_id = str(os.getenv("chrome_extension_id")).strip()

genai.configure(api_key=API_KEY)

model_ = genai.GenerativeModel('gemini-pro')
chat_ = model_.start_chat(history=[])

model__ = genai.GenerativeModel('gemini-pro')
chat__ = model__.start_chat(history=[])

def check_buffer_needed(user_prompt):
    prompt = str(buffer_needed_prompt[0]) + f"""
                                This is the user's prompt: {user_prompt}
    """

    try:
        output = ''
        response = chat_.send_message(prompt, stream=False, safety_settings={
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

        if 'yes' in output.lower().strip():
            return True

        else:
            return False

    except Exception as e:
        print(f"Error generating response: {e}")
        return 'Try again'


def is_it_gen_stuff(user_prompt):
    prompt = str(genPrompt[0]) + f"""
                        This is the user's prompt: {user_prompt}
        """
    
    try:
        output = ''
        response = chat__.send_message(prompt, stream=False, safety_settings={
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

        if 'yes' in output.lower().strip():
            return True

        else:
            return False

    except Exception as e:
        print(f"Error generating GPT response in model_json: {e}")
        return 'Try again'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.json.get('input')  # Receive input from the panel
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    '''
    Here is how the processing takes place
    1. Figure out if the buffer is required. 
    2. If it is not required, then answer general baseball stats. This is what panel_response is
    3. If it is required, then process the video and answer.

    We'll focus on figuring out 2 statcast datas from the videos, those are the pitcher's speed, and the batsman's hit velocities
    '''
    
    boolean = check_buffer_needed(user_input)
    
    if boolean == False:             # That is no buffer needed
        # print('buffer not needed!')
        
        '''
        Here, we may have to call the APIs
        The idea here is to parse the input, determine if its general baseball stuff or particular to a player, or a team or schedule, in which case we'll call the API querying function.
        '''

        if is_it_gen_stuff(user_input):
            print('Requires APIs')
            name_code_tuple = figure_out_code(team_code_mapping, player_code_mapping, user_input)
            output = call_API(name_code_tuple)

            # processed_output = output
            # print(output)

            processed_output = pretty_print(output)
        
        else:
            processed_output = "hello always!"          # We'll add a normal response generator here

        return jsonify({"response": processed_output})


    else:            # that is buffer is required
        print('buffer is needed!')

        # Make a post request to the extension with the action getBuffer. (I am worried about the syntax here)
        response = requests.post(
            f"http://127.0.0.1:5000/extensions/{chrome_extension_id}/getBuffer",
            json={'action': "getBuffer"}
        )
        if response.status_code == 200:
            video_data = response.content
            with open("received_video.webm", "wb") as f:
                f.write(video_data)

            # now we gotta process this
            return jsonify({"response": "Buffer received and saved"}), 200
        else:
            return jsonify({"error": "Failed to get buffer"}), 500



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
