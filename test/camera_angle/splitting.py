'''
okay so, top left corner is most likely to be the origin. To the right, and to the bottom are positive axes. '''

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
    def __init__(self):
        self.session = requests.Session()
        self.chunk_size = 1024
        self.BDL_MODEL_API = "https://balldatalab.com/api/models/"
        self.BDL_DATASET_API = "https://balldatalab.com/api/datasets/"
        self.yolo_model_aliases = {
            'phc_detector': './ball_tracking/pitcher_hitter_catcher_detector_v4.txt',
            'bat_tracking': './ball_tracking/bat_tracking.txt',
            'ball_tracking': './ball_tracking/ball_tracking.txt',
            'glove_tracking': './ball_tracking/glove_tracking.txt',
            'ball_trackingv4': './ball_tracking/ball_trackingv4.txt'
        }

    def _download_files(self, url: str, dest: Union[str, os.PathLike], is_folder: bool = False, is_labeled: bool = False) -> None:
        response = self.session.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(dest)}")    
        
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
        model_txt_path = self.yolo_model_aliases.get(model_alias) if use_bdl_api else model_txt_path    
        if not model_txt_path:
            raise ValueError(f"Invalid alias: {model_alias}")

        base_dir = os.path.dirname(model_txt_path)
        base_name = os.path.splitext(os.path.basename(model_txt_path))[0]

        if model_type == 'YOLO':
            model_weights_path = f"{base_dir}/{base_name}.pt"

        if os.path.exists(model_weights_path):
            print(f"Model found at {model_weights_path}")
            return model_weights_path

        url = self._get_url(model_alias, model_txt_path, use_bdl_api, self.BDL_MODEL_API)
        self._download_files(url, model_weights_path, is_folder=(model_type=='FLORENCE2'))
        
        return model_weights_path


load_tools = LoadTools()
model_weights1 = load_tools.load_model(model_alias='phc_detector')
model = YOLO(model_weights1)

SOURCE_VIDEO_PATH = "./input/baseball_2.mp4"

# Dictionary to store positions for each player type across frames
player_positions = defaultdict(list)

# Define class IDs
PITCHER_CLASS_ID = 1
CATCHER_CLASS_ID = 2

# Run prediction on the source video
results = model.predict(source=SOURCE_VIDEO_PATH, save=True)  # Set save=False if you want to draw custom boxes

# Process each frame
for frame_idx, r in enumerate(results):
    frame_boxes = {
        'pitcher': None,
        'catcher': None
    }
    
    # Process detections in current frame
    for box in r.boxes:
        class_id = int(box.cls[0])
        box_data = box.xywh.numpy()[0]  # Convert to numpy, get first (and only) box
        center = (box_data[0], box_data[1])  # x, y coordinates
        
        if class_id == PITCHER_CLASS_ID:
            frame_boxes['pitcher'] = center
        elif class_id == CATCHER_CLASS_ID:
            frame_boxes['catcher'] = center
    
    # Store positions for this frame
    for player_type, center in frame_boxes.items():
        if center is not None:
            player_positions[player_type].append(center)
            player_positions[player_type].append(center)
            
# Calculate average positions
average_positions = {}
for player_type, positions in player_positions.items():
    if positions:  # Check if we have any detections for this player
        positions_array = np.array(positions)
        average_position = np.mean(positions_array, axis=0)
        average_positions[player_type] = average_position

# Print results
for player_type, avg_pos in average_positions.items():
    print(f"Average {player_type} position: x={avg_pos[0]:.2f}, y={avg_pos[1]:.2f}")
    
# Optional: Print detection statistics
for player_type, positions in player_positions.items():
    detection_rate = len(positions) / len(results) * 100
    print(f"{player_type.capitalize()} detected in {detection_rate:.1f}% of frames")


# So, this is working good, we're getting believeable data about their positions
