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

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()
chrome_extension_id = str(os.getenv("chrome_extension_id")).strip()

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def check_buffer_needed(user_prompt):
    prompt = str(buffer_needed_prompt[0]) + f"""
                                This is the user's prompt: {user_prompt}
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

        if 'yes' in output.lower().strip():
            return True

        else:
            return False

    except Exception as e:
        print(f"Error generating GPT response in model_json: {e}")
        return 'Try again'


def panel_response(user_prompt):
    prompt = str(genPrompt[0]) + f"""
                                This is the user's prompt: {user_prompt}
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

        return output

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
        print('buffer not needed!')
        
        # What if we use a custom model here pre-trained in sports and specially trained in baseball?
        processed_output = panel_response(user_input)                                           
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
