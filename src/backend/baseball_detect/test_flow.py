"""

We will have to make some calibrations

Assuming the pitch is exactly 60 feet long, and the average pitch speed is around 97mph (around 145ft/s). 
For a video having an average fps of 30 fps (which is what seems to be the case).

Time taken to cover the pitch by the ball on average is 60/145 = 0.413 seconds. This means a total number of frames being 30*0.413 = 12 to 13.

So, if you're detecting the baseball in around 12 to 13 frames then you're good to go. This means the threshold should be around 10 minimum.

-------------------------------------------------------------------------------------------------------------------------------------------------------

Added logic:

- If there is a baseball being detected in frame x, then not in x+1, then again in x+2, then we can consider the position of the ball in the x+1th frame 
as the average of the two

- This means we can add that in our valid sequence! (thus getting longer and more accurate results.)


This we'll do using interpolated fields

-------------------------------------------------------------------------------------------------------------------------------------------------------

Added logic:

Taking camera angle into account!

"""


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
            'phc_detector': './baseball_detect/ball_tracking/pitcher_hitter_catcher_detector_v4.txt',
            'bat_tracking': './baseball_detect/bat_tracking/bat_tracking.txt',
            'ball_tracking': './baseball_detect/ball_tracking/ball_tracking.txt',
            'glove_tracking': 'models/YOLO/glove_tracking/model_weights/glove_tracking.txt',
            'ball_trackingv4': './baseball_detect/ball_tracking/ball_trackingv4.txt'
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
    interpolated: bool = False  # New field to mark interpolated detections

class BaseballTracker:
    def __init__(
        self, 
        model: YOLO,
        min_confidence: float = 0.5,
        max_displacement: float = 100,  # max pixels between frames
        min_sequence_length: int = 7,
        pitch_distance_range: Tuple[float, float] = (50, 70),  # feet
        max_interpolation_gap: int = 2  # maximum frames to interpolate
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.max_displacement = max_displacement
        self.min_sequence_length = min_sequence_length
        self.pitch_distance_range = pitch_distance_range
        self.max_interpolation_gap = max_interpolation_gap
        
    def _get_box_center(self, box_coords: np.ndarray) -> Tuple[float, float]:
        """Calculate center coordinates from XYXY box coordinates."""
        x1, y1, x2, y2 = box_coords[0]
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _interpolate_detection(
        self, 
        prev_detection: BallDetection, 
        next_detection: BallDetection, 
        frame_number: int,
        fps: float
    ) -> BallDetection:
        """Create an interpolated detection between two valid detections."""
        # Calculate the fraction of the way between prev and next detection
        total_frames = next_detection.frame_number - prev_detection.frame_number
        frames_from_prev = frame_number - prev_detection.frame_number
        fraction = frames_from_prev / total_frames
        
        # Interpolate box coordinates
        prev_coords = prev_detection.box_coords[0]
        next_coords = next_detection.box_coords[0]
        interpolated_coords = prev_coords + (next_coords - prev_coords) * fraction
        
        # Create interpolated detection
        return BallDetection(
            frame_number=frame_number,
            timestamp=frame_number / fps,
            box_coords=np.array([interpolated_coords]),
            confidence=(prev_detection.confidence + next_detection.confidence) / 2,
            class_value=prev_detection.class_value,
            track_id=prev_detection.track_id,
            interpolated=True
        )
    
    def _organize_detections_by_frame(
        self, 
        detections: List[BallDetection]
    ) -> Dict[int, BallDetection]:
        """Organize detections into a dictionary keyed by frame number."""
        return {d.frame_number: d for d in detections}
    
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
        
        # Organize detections by frame number
        detections_by_frame = self._organize_detections_by_frame(all_detections)
        
        # Find and interpolate gaps
        interpolated_detections = self._interpolate_gaps(detections_by_frame, frame_count, fps)
        
        # Combine original and interpolated detections
        all_detections = sorted(
            interpolated_detections.values(),
            key=lambda x: x.frame_number
        )
        
        # Find continuous sequences
        sequences = self._find_continuous_sequences(all_detections)
        
        return {
            "total_frames": frame_count,
            "fps": fps,
            "sequences": sequences,
            "all_detections": all_detections
        }
    
    def _interpolate_gaps(              # This is for taking the average and all
        self, 
        detections_by_frame: Dict[int, BallDetection],
        frame_count: int,
        fps: float
    ) -> Dict[int, BallDetection]:
        """Interpolate gaps in detection sequences."""
        interpolated_detections = detections_by_frame.copy()
        
        # Sort frame numbers
        frame_numbers = sorted(detections_by_frame.keys())
        
        if not frame_numbers:
            return interpolated_detections
            
        # Iterate through frames to find gaps
        for i in range(len(frame_numbers) - 1):
            current_frame = frame_numbers[i]
            next_frame = frame_numbers[i + 1]
            frame_gap = next_frame - current_frame
            
            # If there's a gap that's within our interpolation limit
            if 1 < frame_gap <= self.max_interpolation_gap:
                prev_detection = detections_by_frame[current_frame]
                next_detection = detections_by_frame[next_frame]
                
                # Check if the displacement between detections is reasonable
                prev_center = self._get_box_center(prev_detection.box_coords)
                next_center = self._get_box_center(next_detection.box_coords)
                displacement = np.sqrt(
                    (next_center[0] - prev_center[0])**2 + 
                    (next_center[1] - prev_center[1])**2
                )
                
                # Only interpolate if the total displacement is reasonable
                if displacement <= self.max_displacement * frame_gap:
                    # Interpolate for each missing frame
                    for missing_frame in range(current_frame + 1, next_frame):
                        interpolated_detection = self._interpolate_detection(
                            prev_detection,
                            next_detection,
                            missing_frame,
                            fps
                        )
                        interpolated_detections[missing_frame] = interpolated_detection
        
        return interpolated_detections
    
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
    
    def _calculate_speeds(              # used scaling factor here, needs to be found using the catcher and the pitcher
        self, 
        sequences: List[List[BallDetection]], 
        fps: float, 
        scale_factor: float  # Conversion factor: pixels to feet
    ) -> List[Dict]:
        """Calculate speed estimates for each valid sequence, accounting for parabolic motion."""
        speed_estimates = []

        for sequence in sequences:
            # Calculate time duration of the sequence
            time_duration = sequence[-1].timestamp - sequence[0].timestamp

            # Initialize total horizontal and vertical displacement
            total_horizontal_displacement = 0.0
            total_vertical_displacement = 0.0

            # Loop through consecutive detections to calculate displacements
            for i in range(len(sequence) - 1):
                print('looping here!')
                current_center = self._get_box_center(sequence[i].box_coords)
                next_center = self._get_box_center(sequence[i + 1].box_coords)

                # Calculate pixel displacements
                dx = next_center[0] - current_center[0]  # Horizontal displacement
                dy = next_center[1] - current_center[1]  # Vertical displacement

                # Convert to real-world units using scale_factor
                total_horizontal_displacement += abs(dx) * scale_factor
                total_vertical_displacement += abs(dy) * scale_factor
                print('used scale factor')

            # Calculate observed pixel displacement in real-world distance
            observed_displacement = np.sqrt(
                total_horizontal_displacement**2 + total_vertical_displacement**2
            )

            # Calculate minimum and maximum speeds using pitch distance range
            min_speed = self.pitch_distance_range[0] / time_duration  # ft/s
            max_speed = self.pitch_distance_range[1] / time_duration  # ft/s

            # Convert to mph
            min_speed_mph = min_speed * 0.681818  # Convert ft/s to mph
            max_speed_mph = max_speed * 0.681818  # Convert ft/s to mph

            # Count interpolated frames
            interpolated_frames = sum(1 for d in sequence if d.interpolated)

            # Append results
            speed_estimates.append({
                "sequence_length": len(sequence),
                "interpolated_frames": interpolated_frames,
                "time_duration": time_duration,
                "horizontal_displacement_ft": total_horizontal_displacement,
                "vertical_displacement_ft": total_vertical_displacement,
                "observed_displacement_ft": observed_displacement,
                "min_speed_ft_per_sec": min_speed,
                "max_speed_ft_per_sec": max_speed,
                "min_speed_mph": min_speed_mph,
                "max_speed_mph": max_speed_mph,
                "start_frame": sequence[0].frame_number,
                "end_frame": sequence[-1].frame_number,
                "average_confidence": np.mean([d.confidence for d in sequence])
            })

        return speed_estimates


def calculate_pitcher_and_catcher(SOURCE_VIDEO_PATH):
    print('here')
    load_tools = LoadTools()
    model_weights2 = load_tools.load_model(model_alias="phc_detector")
    model2 = YOLO(model_weights2)

    player_positions = defaultdict(list)
    player_positions_normalized = defaultdict(list)

    # These are the class IDs for detection as defined by the model itself. 1-pitcher, 2-catcher. Not that it matters, cause we reference it by name. 
    # Found these using another testing code.
    PITCHER_CLASS_ID = 1
    CATCHER_CLASS_ID = 2

    # Run prediction on the source video and calculate positions
    results = model2.predict(source=SOURCE_VIDEO_PATH, save=False)

    # Process each frame
    for frame_idx, r in enumerate(results):
        frame_boxes = {
            'pitcher': None,
            'catcher': None
        }
        
        # Get frame dimensions
        if hasattr(r, 'orig_img'):
            frame_height, frame_width = r.orig_img.shape[:2]
        else:
            print('using these!')
            # If we can't get original image dimensions, you'll need to specify them. This is the default size for many screens. optionally we can use tkinter or something and find it dynamically
            frame_width = 1920
            frame_height = 1080
        
        # Process detections in current frame
        for box in r.boxes:
            class_id = int(box.cls[0])
            box_data = box.xywh.numpy()[0]  # Convert to numpy, get first (and only) box
            
            # Store both absolute and normalized coordinates
            center_absolute = (box_data[0], box_data[1])  # x, y coordinates
            center_normalized = (box_data[0] / frame_width, box_data[1] / frame_height)  # normalized coordinates
            
            if class_id == PITCHER_CLASS_ID:
                frame_boxes['pitcher'] = (center_absolute, center_normalized)
            elif class_id == CATCHER_CLASS_ID:
                frame_boxes['catcher'] = (center_absolute, center_normalized)
        
        for player_type, centers in frame_boxes.items():
            if centers is not None:
                center_absolute, center_normalized = centers
                player_positions[player_type].append(center_absolute)
                player_positions_normalized[player_type].append(center_normalized)

    average_positions = {}
    average_positions_normalized = {}

    for player_type, positions in player_positions.items():
        if positions:
            # Calculate absolute average
            positions_array = np.array(positions)
            average_positions[player_type] = np.mean(positions_array, axis=0)
            
            # Calculate normalized average
            positions_array_norm = np.array(player_positions_normalized[player_type])
            average_positions_normalized[player_type] = np.mean(positions_array_norm, axis=0)

    # print("\nAbsolute Coordinates (in pixels):")
    # for player_type, avg_pos in average_positions.items():
    #     print(f"Average {player_type} position: x={avg_pos[0]:.2f}, y={avg_pos[1]:.2f}")

    # x2, y2 is the pitcher and x1, y1 is the catcher
    x1 = float(average_positions.get('catcher')[0])
    y1 = float(average_positions.get('catcher')[1])

    x2 = float(average_positions.get('pitcher')[0])
    y2 = float(average_positions.get('pitcher')[1])

    print('coordinates found')
    return (x1, y1, x2, y2)
