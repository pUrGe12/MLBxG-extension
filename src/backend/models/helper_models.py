from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))       # adding the root directory to path
from prompts import genPrompt, buffer_needed_prompt, talk_normally, check_statcast_prompt

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()

genai.configure(api_key=API_KEY)

model_ = genai.GenerativeModel('gemini-pro')
chat_ = model_.start_chat(history=[])

model__ = genai.GenerativeModel('gemini-pro')
chat__ = model__.start_chat(history=[])

model___ = genai.GenerativeModel('gemini-pro')
chat___ = model___.start_chat(history=[])

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


def gen_talk(user_prompt):
    prompt = str(talk_normally[0]) + f"""
                                This is the user's prompt: {user_prompt}
    """

    try:
        output = ''
        response = chat___.send_message(prompt, stream=False, safety_settings={
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
        print(f"Error generating response: {e}")
        return 'Try again'

def check_statcast(user_prompt):
    prompt = str(check_statcast_prompt[0]) + f"""
                                This is the user's prompt: {user_prompt}
    """

    try:
        output = ''
        response = chat___.send_message(prompt, stream=False, safety_settings={
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
        print(f"Error generating response: {e}")
        return 'Try again'