from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import sys

from moviepy.editor import VideoFileClip

import requests             # Making post requests to the getBuffer method

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # adding the root directory to path
# This should be done before any relative imports, adding the "backend" directory as root

# Imports for API querying 
from API_querying.query import call_API, pretty_print, figure_out_code
from API_querying.query import team_code_mapping, player_code_mapping

# Imports for helpers
from models.helper_models import check_buffer_needed, is_it_gen_stuff
from models.helper_models import gen_talk

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

from flask import Flask, request, jsonify
from flask_cors import CORS

from prompts import statPrompt

# from dotenv import load_dotenv                # Don't need to do this if we've specified .env contents in render
from pathlib import Path

# load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / '.env')

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

    encoder = load_model('/opt/render/project/src/src/prediction/models/encoder_model.h5')
    scaler = joblib.load('/opt/render/project/src/src/prediction/models/scaler.joblib')
    
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

def GPT_response(top_similar_hits, additional_params):
    prompt = str(statPrompt[0]) + f"""
                                These are the top similar homeruns in MLB: {top_similar_hits}
                                These are the additional user stats: {additional_params}
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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                              Code for generating baseball speed
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from ultralytics import YOLO

import requests
import os
from tqdm import tqdm
import zipfile
import io
from typing import Optional, Union, List, Tuple, Dict
import shutil

import cv2
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

class LoadTools:
    """
    Class dedicated to downloading / loading models and datasets from either the BallDataLab API or specified text files.
    
    Attributes:
        session (requests.Session): Session object for making requests.
        chunk_size (int): Size of chunks to use when downloading files.
        BDL_MODEL_API (str): Base URL for the BallDataLab model API.
        BDL_DATASET_API (str): Base URL for the BallDataLab dataset API.
    
    Methods:
        load_model(model_alias: str, model_type: str = 'YOLO', use_bdl_api: Optional[bool] = True) -> str:
            Loads a given baseball computer vision model into the repository.
        load_dataset(dataset_alias: str, use_bdl_api: Optional[bool] = True) -> str:
            Loads a zipped dataset and extracts it to a folder.
        _download_files(url: str, dest: Union[str, os.PathLike], is_dataset: bool = False) -> None:
            Protected method to handle model and dataset downloads.
        _get_url(alias: str, txt_path: str, use_bdl_api: bool, api_endpoint: str) -> str:
            Protected method to obtain the download URL from the BDL API or a text file.
    """

    def __init__(self):
        self.session = requests.Session()
        self.chunk_size = 1024
        self.BDL_MODEL_API = "https://balldatalab.com/api/models/"
        self.BDL_DATASET_API = "https://balldatalab.com/api/datasets/"
        self.yolo_model_aliases = {
            'phc_detector': 'models/YOLO/pitcher_hitter_catcher_detector/model_weights/pitcher_hitter_catcher_detector_v4.txt',
            'bat_tracking': 'models/YOLO/bat_tracking/model_weights/bat_tracking.txt',
            'ball_tracking': './ball_tracking/ball_tracking.txt',
            'glove_tracking': 'models/YOLO/glove_tracking/model_weights/glove_tracking.txt',
            'ball_trackingv4': './ball_tracking/ball_trackingv4.txt'
        }
        self.florence_model_aliases = {              # No need (maybe)
            'ball_tracking': 'models/FLORENCE2/ball_tracking/model_weights/florence_ball_tracking.txt',
            'florence_ball_tracking': 'models/FLORENCE2/ball_tracking/model_weights/florence_ball_tracking.txt'
        }
        self.dataset_aliases = {                    # Don't have to worry about this
            'okd_nokd': 'datasets/yolo/OKD_NOKD.txt',
            'baseball_rubber_home_glove': 'datasets/yolo/baseball_rubber_home_glove.txt',
            'baseball_rubber_home': 'datasets/yolo/baseball_rubber_home.txt',
            'broadcast_10k_frames': 'datasets/raw_photos/broadcast_10k_frames.txt',
            'broadcast_15k_frames': 'datasets/raw_photos/broadcast_15k_frames.txt',
            'baseball_rubber_home_COCO': 'datasets/COCO/baseball_rubber_home_COCO.txt',
            'baseball_rubber_home_glove_COCO': 'datasets/COCO/baseball_rubber_home_glove_COCO.txt',
            'baseball': 'datasets/yolo/baseball.txt'
        }

    def _download_files(self, url: str, dest: Union[str, os.PathLike], is_folder: bool = False, is_labeled: bool = False) -> None:
        response = self.session.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(dest)}")
            
            if is_folder: 
                content = io.BytesIO()
                for data in response.iter_content(chunk_size=self.chunk_size):
                    size = content.write(data)
                    progress_bar.update(size)
                
                progress_bar.close()

                with zipfile.ZipFile(content) as zip_ref:
                    for file in zip_ref.namelist():
                        if not file.startswith('__MACOSX') and not file.startswith('._'):
                            if is_labeled:
                                zip_ref.extract(file, dest)
                            else:
                                if '/' in file:
                                    filename = file.split('/')[-1]
                                    if filename:
                                        with zip_ref.open(file) as source, open(os.path.join(dest, filename), 'wb') as target:
                                            shutil.copyfileobj(source, target)
                                else:
                                    zip_ref.extract(file, dest)
                
                if not is_labeled:
                    for root, dirs, files in os.walk(dest, topdown=False):
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                
                print(f"Dataset downloaded and extracted to {dest}")
            else:
                with open(dest, 'wb') as file:
                    for data in response.iter_content(chunk_size=self.chunk_size):
                        size = file.write(data)
                        progress_bar.update(size)
                
                progress_bar.close()
                print(f"Model downloaded to {dest}")
        else:
            print(f"Download failed. STATUS: {response.status_code}")

    def _get_url(self, alias: str, txt_path: str, use_bdl_api: bool, api_endpoint: str) -> str:
        if use_bdl_api:
            return f"{api_endpoint}{alias}"
        else:
            with open(txt_path, 'r') as file:
                return file.read().strip()

    def load_model(self, model_alias: str, model_type: str = 'YOLO', use_bdl_api: Optional[bool] = True, model_txt_path: Optional[str] = None) -> str:
        '''
        Loads a given baseball computer vision model into the repository.

        Args:
            model_alias (str): Alias of the model to load.
            model_type (str): The type of the model to utilize. Defaults to YOLO.
            use_bdl_api (Optional[bool]): Whether to use the BallDataLab API.
            model_txt_path (Optional[str]): Path to .txt file containing download link to model weights. 
                                            Only used if use_bdl_api is specified as False.

        Returns:
            model_weights_path (str):  Path to where the model weights are saved within the repo.
        '''
        if model_type == 'YOLO':
            model_txt_path = self.yolo_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'FLORENCE2':
            model_txt_path = self.florence_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        if not model_txt_path:
            raise ValueError(f"Invalid alias: {model_alias}")

        base_dir = os.path.dirname(model_txt_path)
        base_name = os.path.splitext(os.path.basename(model_txt_path))[0]

        if model_type == 'YOLO':
            model_weights_path = f"{base_dir}/{base_name}.pt"
        else:
            model_weights_path = f"{base_dir}/{base_name}"
            os.makedirs(model_weights_path, exist_ok=True)

        if os.path.exists(model_weights_path):
            print(f"Model found at {model_weights_path}")
            return model_weights_path

        url = self._get_url(model_alias, model_txt_path, use_bdl_api, self.BDL_MODEL_API)
        self._download_files(url, model_weights_path, is_folder=(model_type=='FLORENCE2'))
        
        return model_weights_path

    def load_dataset(self, dataset_alias: str, use_bdl_api: Optional[bool] = True, file_txt_path: Optional[str] = None) -> str:
        '''
        Loads a zipped dataset and extracts it to a folder.

        Args:
            dataset_alias (str): Alias of the dataset to load that corresponds to a dataset folder to download
            use_bdl_api (Optional[bool]): Whether to use the BallDataLab API. Defaults to True.
            file_txt_path (Optional[str]): Path to .txt file containing download link to zip file containing dataset. 
                                           Only used if use_bdl_api is specified as False.

        Returns:
            dir_name (str): Path to the folder containing the dataset.
        '''
        txt_path = self.dataset_aliases.get(dataset_alias) if use_bdl_api else file_txt_path
        if not txt_path:
            raise ValueError(f"Invalid alias or missing path: {dataset_alias}")

        base = os.path.splitext(os.path.basename(txt_path))[0]
        dir_name = "unlabeled_" + base if 'raw_photos' in base or 'frames' in base or 'frames' in dataset_alias else base

        if os.path.exists(dir_name):
            print(f"Dataset found at {dir_name}")
            return dir_name

        url = self._get_url(dataset_alias, txt_path, use_bdl_api, self.BDL_DATASET_API)
        os.makedirs(dir_name, exist_ok=True)
        self._download_files(url, dir_name, is_folder=True)

        return dir_name

@dataclass
class BallDetection:
    frame_number: int
    timestamp: float  # in seconds
    box_coords: np.ndarray  # XYXY coordinates
    confidence: float
    class_value: float
    track_id: int = None

class BaseballTracker:
    def __init__(
        self, 
        model: YOLO,
        min_confidence: float = 0.5,
        max_displacement: float = 100,  # max pixels between frames
        min_sequence_length: int = 7,
        pitch_distance_range: Tuple[float, float] = (50, 70)  # feet
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.max_displacement = max_displacement
        self.min_sequence_length = min_sequence_length
        self.pitch_distance_range = pitch_distance_range
        
    def _get_box_center(self, box_coords: np.ndarray) -> Tuple[float, float]:
        """Calculate center coordinates from XYXY box coordinates."""
        x1, y1, x2, y2 = box_coords[0]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def process_video(self, video_path: str) -> Dict:
        """Process video and return ball tracking data and speed estimates."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Store all detections
        all_detections: List[BallDetection] = []
        
        # Process each frame
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            timestamp = frame_number / fps
            
            # Get detections for current frame
            results = self.model.predict(frame, verbose=False)
            print(f"detecting for frame {frame_number}")
            # Process each detection
            for r in results:
                for box in r.boxes.cpu().numpy():
                    confidence = float(box.conf)
                    if confidence >= self.min_confidence:
                        detection = BallDetection(
                            frame_number=frame_number,
                            timestamp=timestamp,
                            box_coords=box.xyxy,
                            confidence=confidence,
                            class_value=float(box.cls),
                            track_id=int(box.id) if box.id is not None else None
                        )
                        all_detections.append(detection)
            
            frame_number += 1
            
        cap.release()
        
        # Find continuous sequences
        sequences = self._find_continuous_sequences(all_detections)
        
        # Calculate speeds for valid sequences
        speed_estimates = self._calculate_speeds(sequences, fps)
        
        return {
            "total_frames": frame_count,
            "fps": fps,
            "sequences": sequences,
            "speed_estimates": speed_estimates,
            "all_detections": all_detections
        }
    
    def _find_continuous_sequences( # We must pass this a mp4 video!
        self, 
        detections: List[BallDetection]
    ) -> List[List[BallDetection]]:
        """Find continuous sequences of ball detections."""
        sequences = []
        current_sequence = []
        
        for detection in detections:
            if not current_sequence:
                current_sequence.append(detection)
                continue
                
            last_detection = current_sequence[-1]
            
            # Get centers for displacement calculation
            current_center = self._get_box_center(detection.box_coords)
            last_center = self._get_box_center(last_detection.box_coords)
            
            # Calculate displacement
            dx = current_center[0] - last_center[0]
            dy = current_center[1] - last_center[1]
            displacement = np.sqrt(dx*dx + dy*dy)
            
            # Check if frames are consecutive and displacement is reasonable
            if (detection.frame_number == last_detection.frame_number + 1 and 
                displacement <= self.max_displacement):
                current_sequence.append(detection)
            else:
                if len(current_sequence) >= self.min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = [detection]
        
        # Add last sequence if valid
        if len(current_sequence) >= self.min_sequence_length:
            sequences.append(current_sequence)
            
        return sequences
    
    def _calculate_speeds(
        self, 
        sequences: List[List[BallDetection]], 
        fps: float
    ) -> List[Dict]:
        """Calculate speed estimates for each valid sequence."""
        speed_estimates = []
        
        for sequence in sequences:
            # Calculate time duration
            time_duration = sequence[-1].timestamp - sequence[0].timestamp
            
            # Calculate pixel displacement using centers
            start_center = self._get_box_center(sequence[0].box_coords)
            end_center = self._get_box_center(sequence[-1].box_coords)
            
            pixel_displacement = np.sqrt(
                (end_center[0] - start_center[0])**2 + 
                (end_center[1] - start_center[1])**2
            )
            
            # Estimate speeds using pitch distance range
            min_speed = (self.pitch_distance_range[0] / time_duration)  # ft/s
            max_speed = (self.pitch_distance_range[1] / time_duration)  # ft/s
            
            speed_estimates.append({
                "sequence_length": len(sequence),
                "time_duration": time_duration,
                "pixel_displacement": pixel_displacement,
                "min_speed_ft_per_sec": min_speed,
                "max_speed_ft_per_sec": max_speed,
                "min_speed_mph": min_speed * 0.681818,  # convert ft/s to mph
                "max_speed_mph": max_speed * 0.681818,  # convert ft/s to mph
                "start_frame": sequence[0].frame_number,
                "end_frame": sequence[-1].frame_number,
                "average_confidence": np.mean([d.confidence for d in sequence])
            })
            
        return speed_estimates


def convert_webm_to_mp4(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264")
    clip.close()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                                       API endpoints
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Upload configs for classics-video
UPLOAD_FOLDER = 'uploads/videos'                                    # To save the uploaded videos
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

            processed_output = pretty_print(output)             # not doing this if the output is too big as in the case of the schedule
        
        else:
            processed_output = gen_talk(user_input)          # We'll add a normal response generator here

        return jsonify({"response": processed_output})


    else:            # that is buffer is required
        print('buffer is needed!')

        # Make a post request to the extension with the action getBuffer. (I am worried about the syntax here)
        response = requests.post(
            f"http://127.0.0.1:5000/extensions/{chrome_extension_id}/getBuffer",                    # Hope we're actually getting the buffer
            json={'action': "getBuffer"}
        )
        if response.status_code == 200:
            video_data = response.content
            with open("/opt/render/project/src/src/backend/input_files/received_video.webm", "wb") as fp:               # Saving this in the input files directory (hopefully this works)
                fp.write(video_data)

            # Convert webm to mp4 and save it
            convert_webm_to_mp4("/opt/render/project/src/src/backend/input_files/received_video.webm", "/opt/render/project/src/src/backend/input_files/converted_input.mp4")
            
            VideoPath = "/opt/render/project/src/src/backend/input_files/converted_input.mp4"


            # -------------------------------------------------------------------------------------------
            #               Now we need a logic to see if they're asking for speed or what
            # -------------------------------------------------------------------------------------------

            what_is_needed = check_statcast(user_input)

            if 'baseballspeed' in what_is_needed.strip().lower():
                
                # now we gotta process this, assuming we get the data as a mp4 data
                load_tools = LoadTools()
                model_weights = load_tools.load_model(model_alias='ball_trackingv4')
                model = YOLO(model_weights)

                tracker = BaseballTracker(
                model=model,
                min_confidence=0.3,         # 0.3 confidence works good enough, gives realistic predictions
                max_displacement=100,       # adjust based on your video resolution
                min_sequence_length=7,
                pitch_distance_range=(50, 70)  # feet
                )
            
                # Process video
                results = tracker.process_video(VideoPath)                                  # This should probably work
                
                # Printing and saving results
                output = """"""

                print(f"\nProcessed {results['total_frames']} frames at {results['fps']} FPS")
                print(f"Found {len(results['sequences'])} valid ball sequences")
                
                output += f"\nProcessed {results['total_frames']} frames at {results['fps']} FPS" + f"\n Found {len(results['sequences'])} valid ball sequences"

                for i, speed_est in enumerate(results['speed_estimates'], 1):
                    print(f"\nSequence {i}:")
                    print(f"Frames: {speed_est['start_frame']} to {speed_est['end_frame']}")
                    print(f"Duration: {speed_est['time_duration']:.2f} seconds")
                    print(f"Average confidence: {speed_est['average_confidence']:.3f}")
                    print(f"Estimated speed: {speed_est['min_speed_mph']:.1f} to "
                          f"{speed_est['max_speed_mph']:.1f} mph")

                    output += f"""
        \nSequence {i}:
        Frames: {speed_est['start_frame']} to {speed_est['end_frame']}
        Duration: {speed_est['time_duration']:.2f} seconds
        Average confidence: {speed_est['average_confidence']:.3f}
        Estimated speed: {speed_est['min_speed_mph']:.1f}""" + f""" to {speed_est['max_speed_mph']:.1f} mph
                    """

                return jsonify({"response": output}), 200
            
            elif 'exitvelocity' in what_is_needed.strip().lower():
                # This now requires tracking the bat.
                pass

            else:
                return jsonify({"error": "Failed to get buffer"}), 500




@app.route('/user-stat/', methods=['POST'])
def user_stat():
    user_input = request.json.get('input')  # Receive input from the panel

    print(user_input)

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        print('in the try block')
        extractor_dictionary, additional_params = extractor(user_input)

        print("unpacked the tuple!")

        '''  
            extractor output is a dictionary that describes the following of the user
            1. ExitVelocity
            2. HitDistance
            3. LaunchAngle

            We parse the "AdditionalParams" seperately. The idea is to send the extractor_dictionary to the pineconeDB and get top similar hits

        '''

        embedding = process_new_hit(extractor_dictionary, encoder, scaler)                           
        
        print("embeddings generated")

        # Find the top 5 matches to the user's stats being entered above.
        found_similar_hits = find_similar_hits(embedding, top_k = 5)
        top_similar_hits = store_similar_hits(found_similar_hits)               # Storing that to send it to the model for prettifying

        print(top_similar_hits)

        processed_output = GPT_response(top_similar_hits, additional_params.get('AdditionalParams'))                    # Both additional params and extractor dict are dictionaries.

        print(processed_output)

        return jsonify({
            "response": processed_output
            })

    except Exception as e:
        incomplete_text = extractor(user_input)

        print(incomplete_text)
        print('here in exception', e)

        '''
            Enter this block of code when unpacking the tuple leads to an exception
            This will only happen when there is only one element being returned by the extractor, that is the incomplete text. 

            We will show the incomplete text directly to the user. 
        '''

        return jsonify({
            "response": incomplete_text
            })


# -----------------------------------------------------------X Classics endpoints X--------------------------------------------------------------

from werkzeug.utils import secure_filename                              # Adding this here cause not always requireds

@app.route('/classics-video', methods=['POST'])
def classics_video_processing():
    '''
    Video processing essentially means applying the yolo model and computing the statcast data.

    We additionally wanna make sure that the entered file is mp4 itself! Not to have a php file being executed here and my laptop becoming compromised!
    '''
    try:
        # Check if video file is in request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file in request'}), 400
        
        video_file = request.files['video']                                                 # This is a mp4 video file
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Secure the filename and save the file
        filename = secure_filename(video_file.filename)
        
        # Ensure upload directory exists and saving it
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(file_path)
        
        # Process the video
        load_tools = LoadTools()
        model_weights = load_tools.load_model(model_alias='ball_trackingv4')
        model = YOLO(model_weights)

        tracker = BaseballTracker(
        model=model,
        min_confidence=0.3,         # 0.3 confidence works good enough, gives realistic predictions
        max_displacement=100,       # adjust based on your video resolution
        min_sequence_length=7,
        pitch_distance_range=(50, 70)  # feet
        )
    
        # Process video
        results = tracker.process_video(SOURCE_VIDEO_PATH)
        
        # Printing and saving results
        output = """"""

        print(f"\nProcessed {results['total_frames']} frames at {results['fps']} FPS")
        print(f"Found {len(results['sequences'])} valid ball sequences")
        
        output += f"\nProcessed {results['total_frames']} frames at {results['fps']} FPS" + f"\n Found {len(results['sequences'])} valid ball sequences"

        for i, speed_est in enumerate(results['speed_estimates'], 1):
            print(f"\nSequence {i}:")
            print(f"Frames: {speed_est['start_frame']} to {speed_est['end_frame']}")
            print(f"Duration: {speed_est['time_duration']:.2f} seconds")
            print(f"Average confidence: {speed_est['average_confidence']:.3f}")
            print(f"Estimated speed: {speed_est['min_speed_mph']:.1f} to "
                  f"{speed_est['max_speed_mph']:.1f} mph")

            output += f"""
\nSequence {i}:
Frames: {speed_est['start_frame']} to {speed_est['end_frame']}
Duration: {speed_est['time_duration']:.2f} seconds
Average confidence: {speed_est['average_confidence']:.3f}
Estimated speed: {speed_est['min_speed_mph']:.1f}""" + f""" to {speed_est['max_speed_mph']:.1f} mph"""

        return jsonify({                                                            # Also need to return uploaded message
            'message': 'Video analysed successfully, now running plan',
            'filename': filename,
            'output': output                                                                # Handle this properly in the frontend
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classics-text', methods=['POST'])
def classics_text_processing():
    '''
    Here we will have to first scrape the video! Then apply the same concept as above
    '''
    pass

# Trying to host this on render

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render will provide PORT env variable
    app.run(host="0.0.0.0", port=port)