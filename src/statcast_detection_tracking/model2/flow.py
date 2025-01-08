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
    
    def _find_continuous_sequences(
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

# Example usage
if __name__ == "__main__":
    SOURCE_VIDEO_PATH = "./input/baseball.mp4"
    
    # Load your model
    load_tools = LoadTools()
    model_weights = load_tools.load_model(model_alias='ball_trackingv4')
    model = YOLO(model_weights)
    
    # Initialize tracker with your model
    tracker = BaseballTracker(
        model=model,
        min_confidence=0.3,         # 0.3 confidence works good enough, gives realistic predictions
        max_displacement=100,       # adjust based on your video resolution
        min_sequence_length=7,
        pitch_distance_range=(50, 70)  # feet
    )
    
    # Process video
    results = tracker.process_video(SOURCE_VIDEO_PATH)
    
    # Print results
    print(f"\nProcessed {results['total_frames']} frames at {results['fps']} FPS")
    print(f"Found {len(results['sequences'])} valid ball sequences")
    
    for i, speed_est in enumerate(results['speed_estimates'], 1):
        print(f"\nSequence {i}:")
        print(f"Frames: {speed_est['start_frame']} to {speed_est['end_frame']}")
        print(f"Duration: {speed_est['time_duration']:.2f} seconds")
        print(f"Average confidence: {speed_est['average_confidence']:.3f}")
        print(f"Estimated speed: {speed_est['min_speed_mph']:.1f} to "
              f"{speed_est['max_speed_mph']:.1f} mph")