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
            'bat_tracking': '../big_weights/bat_tracking.txt',
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

        model_weights_path = f"{base_dir}/{base_name}.pt"

        if os.path.exists(model_weights_path):
            print(f"Model found at {model_weights_path}")
            return model_weights_path

        url = self._get_url(model_alias, model_txt_path, use_bdl_api, self.BDL_MODEL_API)
        self._download_files(url, model_weights_path, is_folder=False)
        
        return model_weights_path


load_tools = LoadTools()
model_weights = load_tools.load_model(model_alias='bat_tracking')
model = YOLO(model_weights)

SOURCE_VIDEO_PATH = "baseball_3.mp4"

results = model.predict(source=SOURCE_VIDEO_PATH, save=True)