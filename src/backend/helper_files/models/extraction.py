from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

import re
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # adding the root directory to path
from backend.prompts import extractionPrompt

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

def parse_extractor_dict(data):
    pattern = r'(\w+),\s([\d\.]+)'
    matches = re.findall(pattern, data[0])
    return {key: float(value) for key, value in matches}

def parse_additional_params(data):
    pattern = r'(\w+),\s"([^"]+)"'
    matches = re.findall(pattern, data[0])
    return {key: value for key, value in matches}

def parse_incomplete_text(data):
    pattern = r'\^\^\^incomplete(.*?)\^\^\^'
    
    match = re.search(pattern, data[0], re.DOTALL)

    return match.group(1) if match else None

def extractor(user_prompt):
    prompt = str(extractionPrompt[0]) + f"""
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

        if output.split('\n')[0].strip().startswith('&') or '&' in output.strip():
            extractor_dictionary = re.findall('&&&dict.*&&&', output, re.DOTALL)
            additional_params = re.findall('@@@addparam.*@@@', output, re.DOTALL)

            final_extractor_dict = parse_extractor_dict(extractor_dictionary)
            final_additional_params = parse_additional_params(additional_params)
            
            return (final_extractor_dict, final_additional_params)
        else:

            incomplete_text = re.findall('\^\^\^incomplete.*\^\^\^', output, re.DOTALL)
            
            print(incomplete_text)
            
            return parse_incomplete_text(incomplete_text)

    except Exception as e:
        print(f"Error generating GPT response in model_json: {e}")
        return output