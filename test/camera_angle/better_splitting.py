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
import tkinter as tk

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

@dataclass
class BallDetection:
    frame_number: int
    timestamp: float  # in seconds
    box_coords: np.ndarray  # XYXY coordinates
    confidence: float
    class_value: float
    track_id: Optional[int] = None
    interpolated: bool = False

class BaseballTracker:
    def __init__(
        self, 
        model: YOLO,
        min_confidence: float = 0.2,
        max_displacement: float = 100,
        min_sequence_length: int = 10,
        pitch_distance_range: Tuple[float, float] = (55, 65)
    ):
        self.model = model
        self.min_confidence = min_confidence
        self.max_displacement = max_displacement
        self.min_sequence_length = min_sequence_length
        self.pitch_distance_range = pitch_distance_range
        self.max_interpolation_gap = 2

    def process_video(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_detections: List[BallDetection] = []
        frame_number = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            timestamp = frame_number / fps
            results = self.model.predict(frame, verbose=False)
            print(f"detecting for frame {frame_number}")
            
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
        
        detections_by_frame = self._organize_detections_by_frame(all_detections)
        interpolated_detections = self._interpolate_gaps(detections_by_frame, frame_count, fps)
        all_detections = sorted(interpolated_detections.values(), key=lambda x: x.frame_number)
        sequences = self._find_continuous_sequences(all_detections)
        speed_estimates = self._calculate_speeds(sequences, fps)
        
        return {
            "total_frames": frame_count,
            "fps": fps,
            "sequences": sequences,
            "speed_estimates": speed_estimates,
            "all_detections": all_detections
        }

    def _get_box_center(self, box_coords: np.ndarray) -> Tuple[float, float]:
        x1, y1, x2, y2 = box_coords[0]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _interpolate_detection(self, prev_detection: BallDetection, next_detection: BallDetection, 
                             frame_number: int, fps: float) -> BallDetection:
        total_frames = next_detection.frame_number - prev_detection.frame_number
        frames_from_prev = frame_number - prev_detection.frame_number
        fraction = frames_from_prev / total_frames
        
        prev_coords = prev_detection.box_coords[0]
        next_coords = next_detection.box_coords[0]
        interpolated_coords = prev_coords + (next_coords - prev_coords) * fraction
        
        return BallDetection(
            frame_number=frame_number,
            timestamp=frame_number / fps,
            box_coords=np.array([interpolated_coords]),
            confidence=(prev_detection.confidence + next_detection.confidence) / 2,
            class_value=prev_detection.class_value,
            track_id=prev_detection.track_id,
            interpolated=True
        )

    def _organize_detections_by_frame(self, detections: List[BallDetection]) -> Dict[int, BallDetection]:
        return {d.frame_number: d for d in detections}

    def _interpolate_gaps(self, detections_by_frame: Dict[int, BallDetection],
                         frame_count: int, fps: float) -> Dict[int, BallDetection]:
        interpolated_detections = detections_by_frame.copy()
        frame_numbers = sorted(detections_by_frame.keys())
        
        if not frame_numbers:
            return interpolated_detections
            
        for i in range(len(frame_numbers) - 1):
            current_frame = frame_numbers[i]
            next_frame = frame_numbers[i + 1]
            frame_gap = next_frame - current_frame
            
            if 1 < frame_gap <= self.max_interpolation_gap:
                prev_detection = detections_by_frame[current_frame]
                next_detection = detections_by_frame[next_frame]
                
                prev_center = self._get_box_center(prev_detection.box_coords)
                next_center = self._get_box_center(next_detection.box_coords)
                displacement = np.sqrt(
                    (next_center[0] - prev_center[0])**2 + 
                    (next_center[1] - prev_center[1])**2
                )
                
                if displacement <= self.max_displacement * frame_gap:
                    for missing_frame in range(current_frame + 1, next_frame):
                        interpolated_detection = self._interpolate_detection(
                            prev_detection,
                            next_detection,
                            missing_frame,
                            fps
                        )
                        interpolated_detections[missing_frame] = interpolated_detection
        
        return interpolated_detections

    def _find_continuous_sequences(self, detections: List[BallDetection]) -> List[List[BallDetection]]:
        sequences = []
        current_sequence = []
        
        for detection in detections:
            if not current_sequence:
                current_sequence.append(detection)
                continue
                
            last_detection = current_sequence[-1]
            current_center = self._get_box_center(detection.box_coords)
            last_center = self._get_box_center(last_detection.box_coords)
            
            displacement = np.sqrt(
                (current_center[0] - last_center[0])**2 + 
                (current_center[1] - last_center[1])**2
            )
            
            if (detection.frame_number == last_detection.frame_number + 1 and 
                displacement <= self.max_displacement):
                current_sequence.append(detection)
            else:
                if len(current_sequence) >= self.min_sequence_length:
                    sequences.append(current_sequence)
                current_sequence = [detection]
        
        if len(current_sequence) >= self.min_sequence_length:
            sequences.append(current_sequence)
            
        return sequences

    def _calculate_speeds(self, sequences: List[List[BallDetection]], fps: float) -> List[Dict]:
        speed_estimates = []
        
        for sequence in sequences:
            time_duration = sequence[-1].timestamp - sequence[0].timestamp
            start_center = self._get_box_center(sequence[0].box_coords)
            end_center = self._get_box_center(sequence[-1].box_coords)
            
            pixel_displacement = np.sqrt(
                (end_center[0] - start_center[0])**2 + 
                (end_center[1] - start_center[1])**2
            )
            
            min_speed = (self.pitch_distance_range[0] / time_duration)
            max_speed = (self.pitch_distance_range[1] / time_duration)
            
            interpolated_frames = sum(1 for d in sequence if d.interpolated)
            
            speed_estimates.append({
                "sequence_length": len(sequence),
                "interpolated_frames": interpolated_frames,
                "time_duration": time_duration,
                "pixel_displacement": pixel_displacement,
                "min_speed_ft_per_sec": min_speed,
                "max_speed_ft_per_sec": max_speed,
                "min_speed_mph": min_speed * 0.681818,
                "max_speed_mph": max_speed * 0.681818,
                "start_frame": sequence[0].frame_number,
                "end_frame": sequence[-1].frame_number,
                "average_confidence": np.mean([d.confidence for d in sequence])
            })
            
        return speed_estimates


class PlayerTracker:
    """Tracks player positions (pitcher and catcher) in video."""
    PITCHER_CLASS_ID = 1
    CATCHER_CLASS_ID = 2

    def __init__(self, model: YOLO):
        self.model = model

    def _detect_players_in_video(self, video_path: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Process video and detect players in each frame.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing lists of positions for each player type
        """
        # Initialize position storage
        positions = {
            'pitcher': {'absolute': [], 'normalized': []},
            'catcher': {'absolute': [], 'normalized': []}
        }
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Run predictions on video
        results = self.model.predict(source=video_path, save=False)
        
        # Process each frame's results
        for frame_idx, result in enumerate(results):
            frame_detections = {
                'pitcher': None,
                'catcher': None
            }
            
            # Process each detection in the frame
            for box in result.boxes:
                class_id = int(box.cls[0])
                box_data = box.xywh.numpy()[0]  # Convert to numpy, get first box
                
                # Calculate center coordinates
                center_absolute = (box_data[0], box_data[1])
                center_normalized = (
                    box_data[0] / frame_width,
                    box_data[1] / frame_height
                )
                
                # Store positions based on player type
                if class_id == self.PITCHER_CLASS_ID:
                    frame_detections['pitcher'] = (center_absolute, center_normalized)
                elif class_id == self.CATCHER_CLASS_ID:
                    frame_detections['catcher'] = (center_absolute, center_normalized)
            
            # Store valid detections
            for player_type, detection in frame_detections.items():
                if detection:
                    absolute_pos, normalized_pos = detection
                    positions[player_type]['absolute'].append(absolute_pos)
                    positions[player_type]['normalized'].append(normalized_pos)
        
        return positions

    def _calculate_average_positions(
        self, 
        positions: Dict[str, Dict[str, List[Tuple[float, float]]]]
    ) -> Dict[str, PlayerPosition]:
        """
        Calculate average positions for each player from all detections.
        
        Args:
            positions: Dictionary containing lists of positions for each player
            
        Returns:
            Dictionary containing average PlayerPosition for each player type
        """
        average_positions = {}
        
        for player_type, coords in positions.items():
            if coords['absolute'] and coords['normalized']:
                # Calculate absolute average
                abs_positions = np.array(coords['absolute'])
                avg_absolute = tuple(np.mean(abs_positions, axis=0))
                
                # Calculate normalized average
                norm_positions = np.array(coords['normalized'])
                avg_normalized = tuple(np.mean(norm_positions, axis=0))
                
                # Create PlayerPosition object
                average_positions[player_type] = PlayerPosition(
                    absolute_coords=avg_absolute,
                    normalized_coords=avg_normalized
                )
        
        return average_positions

    def track_players(self, video_path: str) -> Dict[str, PlayerPosition]:
        """
        Track players throughout the video and return their average positions.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing average positions for each player type
        """
        # Detect players in all frames
        positions = self._detect_players_in_video(video_path)
        
        # Calculate and return average positions
        average_positions = self._calculate_average_positions(positions)
        
        # Print detection statistics
        self._print_detection_stats(positions, video_path)
        
        return average_positions

    def _print_detection_stats(
        self,
        positions: Dict[str, Dict[str, List[Tuple[float, float]]]],
        video_path: str
    ) -> None:
        """Print statistics about player detections."""
        # Get total frame count
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print("\nDetection Statistics:")
        for player_type, coords in positions.items():
            detection_count = len(coords['absolute'])
            detection_rate = (detection_count / total_frames) * 100
            print(f"{player_type.capitalize()} detected in {detection_rate:.1f}% of frames "
                  f"({detection_count}/{total_frames})")

class SpeedCalculator:
    """Calculates real pitch speed using player positions and apparent speed."""
    def __init__(self):
        # Get screen dimensions
        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.half_width = float(self.screen_width // 2)
        self.half_height = float(self.screen_height // 2)
        print(f"Screen width: {self.screen_width}")
        print(f"Screen height: {self.screen_height}")
        root.destroy()

    def calculate_real_speed(
        self,
        average_positions: Dict[str, np.ndarray],
        apparent_speeds: Tuple[float, float]  # (min_speed, max_speed)
    ) -> Tuple[float, float]:
        """
        Calculate real pitch speed using geometry and player positions.
        
        Args:
            average_positions: Dictionary containing average positions of players
            apparent_speeds: Tuple of (min_speed, max_speed) in mph
            
        Returns:
            Tuple of (min_real_speed, max_real_speed) in mph
        """
        # Extract pitcher and catcher positions
        x2 = float(f"{average_positions['pitcher'][0]:.2f}")  # pitcher x
        y2 = float(f"{average_positions['pitcher'][1]:.2f}")  # pitcher y
        x1 = float(f"{average_positions['catcher'][0]:.2f}")  # catcher x
        y1 = float(f"{average_positions['catcher'][1]:.2f}")  # catcher y

        # Calculate y_ (intersection point)
        y_ = ((y2 - y1) / (x2 - x1)) * (self.half_width - x1) + y1

        # Calculate angle based on geometry
        if y_ < 0:
            Delta_y = abs(y_) + y2  # add the pitcher's coordinates
            Delta_x = abs(x2 - x1)
            alpha = np.arctan(Delta_x / Delta_y)
        elif y_ > 0:
            Delta_y = abs(y_ - y2)  # This should always be positive regardless of abs()
            Delta_x = abs(x2 - x1)
            alpha = np.arctan(Delta_x / Delta_y)
        else:
            print("Very rare case: y_ = 0")
            return apparent_speeds  # Return apparent speeds if we can't calculate

        # Calculate real speeds
        v_real_max = apparent_speeds[1] * (1 / np.sin(alpha))
        v_real_min = apparent_speeds[0] * (1 / np.sin(alpha))

        return v_real_min, v_real_max

def main():
    SOURCE_VIDEO_PATH = "./input/baseball_3.mp4"

    # Initialize tools and load models
    load_tools = LoadTools()
    
    # Load and initialize ball tracking model
    model_weights1 = load_tools.load_model(model_alias='ball_trackingv4')
    ball_model = YOLO(model_weights1)
    ball_tracker = BaseballTracker(
        model=ball_model,
        min_confidence=0.2,
        max_displacement=100,
        min_sequence_length=10,
        pitch_distance_range=(55, 65)
    )

    # Process video for ball tracking
    ball_results = ball_tracker.process_video(SOURCE_VIDEO_PATH)
    
    # Extract apparent speeds from ball tracking
    if ball_results['speed_estimates']:
        estimated_speed_min = ball_results['speed_estimates'][0]['min_speed_mph']
        estimated_speed_max = ball_results['speed_estimates'][0]['max_speed_mph']
    else:
        print("No valid ball sequences found")
        return

    # Load and initialize player tracking model
    model_weights2 = load_tools.load_model(model_alias="phc_detector")
    player_model = YOLO(model_weights2)
    player_tracker = PlayerTracker(player_model)

    # Track players
    average_positions, average_positions_normalized = player_tracker.track_players(SOURCE_VIDEO_PATH)

    # Calculate real speeds
    speed_calculator = SpeedCalculator()
    v_real_min, v_real_max = speed_calculator.calculate_real_speed(
        average_positions,
        (estimated_speed_min, estimated_speed_max)
    )

    # Print results
    print("\nSpeed Analysis Results:")
    print(f"Apparent Speed Range: {estimated_speed_min:.1f} - {estimated_speed_max:.1f} mph")
    print(f"Calculated Real Speed Range: {v_real_min:.1f} - {v_real_max:.1f} mph")

if __name__ == "__main__":
    main()