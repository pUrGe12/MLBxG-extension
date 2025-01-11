"""
This is the backend file for the options page. This is supposed to be the MLB future predictor. The way this is done is as follows

Step 1 --> Extract the user's stats based on their prompt and give it numerical values. 
Step 2 --> Create a query of those stats and send it to pinecone.
Step 3 --> Access the top players from pinecone who match the user's performance.
Step 4 --> Based on additional information (if provided by the user) further single out 2 players from the top few.
Step 5 --> Wrap it around Gemini's response and push it to the user as a statistical estimate of how well their future in MLB may turn out to be based on their current performance.
"""

import numpy as np
import pandas as pd

from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # adding the root directory to path
from prompts import statPrompt

# print(statPrompt[0])
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent.parent / '.env')

API_KEY = str(os.getenv("API_KEY")).strip()
pinecone_api_key = str(os.getenv("pinecone_api_key")).strip()

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# ---------------------------------------------------------------------------
# Extraction model calling and implementation
# ---------------------------------------------------------------------------
from models.extraction import extractor

# ---------------------------------------------------------------------------
# This is the code for loading the trained models, and querying the vectorDB
# ---------------------------------------------------------------------------

from tensorflow.keras.models import load_model, Model
import joblib
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

def load_models():
    '''
        The encoder and the scaler have been trained through the notebook present in the prediction directory
        The datasets used to train these are present inside the datasets directory. These are exactly what were provided by the organisers.
    
        input: None
        return: instances of the encoder and the scaler model.
    '''

    encoder = load_model('../prediction/models/encoder_model.h5')
    scaler = joblib.load('../prediction/models/scaler.joblib')
    
    return encoder, scaler


global encoder, scaler
encoder, scaler = load_models()             # Load the encoder and the scaler using the above defined function


def process_new_hit(new_hit_data, encoder, scaler):
    """
        This function takes in the players stats and generates embeddings using the trained models
    """
    
    assert type(new_hit_data) == dict, 'The hit data must be a dictionary'
    features = np.array([[
        new_hit_data['ExitVelocity'],
        new_hit_data['HitDistance'],
        new_hit_data['LaunchAngle']
    ]])
    
    embedding = encoder.predict(scaler.transform(features))
    return embedding[0]

def find_similar_hits(embedding, index_name="baseball-hits", top_k=5):
    """Find similar hits in the database"""

    pinecone = Pinecone(api_key=pinecone_api_key)
    
    index = pinecone.Index(index_name)
    results = index.query(
        vector=embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return results

def store_similar_hits(results):
    """Store the details of similar hits in a structured format."""

    matches = []
    for idx, match in enumerate(results.matches, 1):
        match_details = {
            "SimilarityScore": round(match.score, 3),
            "Title": match.metadata.get("title", "Unknown"),
            "ExitVelocity": f"{float(match.metadata.get('exit_velocity', 0.0)):.1f} mph",
            "HitDistance": f"{float(match.metadata.get('hit_distance', 0.0)):.1f} feet",
            "LaunchAngle": f"{float(match.metadata.get('launch_angle', 0.0)):.1f} degrees",
        }
        matches.append(match_details)
    
    return matches

# ---------------------------------------------------------------------------
# Model to wrap things and output final answer
# ---------------------------------------------------------------------------

def GPT_response(top_similar_hits, additional_params, user_input):
    prompt = str(statPrompt[0]) + f"""
                                These are the top similar homeruns in MLB: {top_similar_hits}
                                These are the additional user stats: {additional_params}

                                This is the user's input: {user_input}
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
        print(f"Error generating response: {e}")
        return 'Try again'
