from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import cv2
import numpy as np
from time import time

def calcualte_pixel_to_feet(video_path):
    ''' 
    using the number of pixels taken for a pitch length, calculate the pixel_to_feet calib
    '''
    pass

def calculate_speed(positions, frame_rate, pixel_to_feet):
    """
    Calculate speed given positions, frame rate and pixel to feet conversion
    Returns speed in mph
    """
    if len(positions) < 2:
        return 0
    
    distances = []
    for i in range(1, len(positions)):
        # Calculate Euclidean distance between consecutive points
        dist = np.sqrt(
            (positions[i][0] - positions[i-1][0])**2 +
            (positions[i][1] - positions[i-1][1])**2
        )
        distances.append(dist * pixel_to_feet)  # Convert to feet
    
    avg_distance = np.mean(distances)
    speed_fps = avg_distance * frame_rate
    
    # Convert to mph
    speed_mph = speed_fps * 0.681818  # Convert fps to mph
    
    return speed_mph

class BallTracker:
    def __init__(self, weights_path, conf_threshold=0.5, pixel_to_feet=0.1):
        """
        Initialize ball tracker
        weights_path: path to YOLOv5 weights (best.pt)
        conf_threshold: confidence threshold for detection
        pixel_to_feet: conversion factor from pixels to feet
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.conf = conf_threshold
        
        # Initialize DeepSORT - no model path needed
        self.tracker = DeepSort(
            max_age=70,
            n_init=3,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=100,
            override_track_class=None
        )
        
        self.pixel_to_feet = pixel_to_feet
        
    def process_video(self, video_path, output_path=None):
        """
        Process video file and track ball
        Returns list of speeds for each tracked ball
        """
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        
        track_history = {}  # Dictionary to store tracking history
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Run YOLOv5 detection
            results = self.model(frame)
            detections = results.xyxy[0].cpu().numpy()
            
            # Format detections for DeepSORT
            detections_for_deepsort = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                ltrb = [x1, y1, x2, y2]
                detections_for_deepsort.append((ltrb, conf, frame[int(y1):int(y2), int(x1):int(x2)]))
            
            # Update tracker
            tracks = self.tracker.update_tracks(detections_for_deepsort, frame=frame)
            
            # Process each tracked object
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                # Calculate center point of bounding box
                center_x = (ltrb[0] + ltrb[2]) / 2
                center_y = (ltrb[1] + ltrb[3]) / 2
                
                # Store position history
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((center_x, center_y))
                
                # Draw bounding box and ID
                if output_path:
                    cv2.rectangle(frame, 
                                (int(ltrb[0]), int(ltrb[1])), 
                                (int(ltrb[2]), int(ltrb[3])), 
                                (0, 255, 0), 2)
                    
                    # Calculate and display speed if we have enough positions
                    if len(track_history[track_id]) > 5:
                        speed = calculate_speed(
                            track_history[track_id][-5:],
                            frame_rate,
                            self.pixel_to_feet
                        )
                        cv2.putText(frame, 
                                  f"ID: {track_id} Speed: {speed:.1f} mph",
                                  (int(ltrb[0]), int(ltrb[1] - 10)),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5,
                                  (0, 255, 0),
                                  2)
            
            if output_path:
                out.write(frame)
        
        cap.release()
        if output_path:
            out.release()
        
        # Calculate final speeds for each track
        speeds = {}
        for track_id, positions in track_history.items():
            if len(positions) > 5:
                speed = calculate_speed(positions, frame_rate, self.pixel_to_feet)
                speeds[track_id] = speed
        
        return speeds

# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = BallTracker(
        weights_path='baseball_detection4/weights/best.pt',
        conf_threshold=0.5,
        pixel_to_feet=0.1  # This value needs to be calibrated for your setup
    )
    
    # Process video
    speeds = tracker.process_video(
        video_path='../baseball.mp4',
        output_path='output.mp4'
    )
    
    # Print results
    for track_id, speed in speeds.items():
        print(f"Ball {track_id}: {speed:.1f} mph")