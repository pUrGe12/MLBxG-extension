from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # adding the root directory to path
from prompts import genPrompt

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def GPT_response(user_prompt):
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

    # Process the user input (dummy response for now)
    processed_output = GPT_response(user_input)

    return jsonify({"response": processed_output})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
