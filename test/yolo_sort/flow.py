# Making bounding box for a single image

# import torch
# import cv2

# class BaseballDetector:
#     def __init__(self, weights_path, ball_class=0, conf_threshold=0.5):
#         """
#         Initialize the baseball detector.
#         weights_path: Path to YOLOv5 weights (e.g., best.pt).
#         ball_class: Class index for the baseball in the YOLO model.
#         conf_threshold: Confidence threshold for detection.
#         """
#         self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
#         self.model.conf = conf_threshold
#         self.ball_class = ball_class

#     def detect_baseball(self, image_path, output_path=None):
#         """
#         Detect baseball in a still image.
#         image_path: Path to the input image.
#         output_path: Path to save the output image with detections (optional).
#         """
#         # Load image
#         image = cv2.imread(image_path)
#         if image is None:
#             raise FileNotFoundError(f"Image not found: {image_path}")
        
#         # Run YOLO detection
#         results = self.model(image)                       # Since the model was trained with images of size 416x416, but if I use that then it messes up!
#         detections = results.xyxy[0].cpu().numpy()
        
#         # Filter detections for the baseball class
#         baseball_detections = [
#             det for det in detections if int(det[5]) == self.ball_class
#         ]
        
#         # Draw bounding boxes around detected baseballs
#         for det in baseball_detections:
#             x1, y1, x2, y2, conf, cls = det
#             cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(
#                 image,
#                 f"Baseball {conf:.2f}",
#                 (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2,
#             )
        
#         # Save the output image if an output path is provided
#         if output_path:
#             cv2.imwrite(output_path, image)
        
#         return baseball_detections

# # Example usage
# if __name__ == "__main__":
#     # Initialize detector
#     print('starting')
#     detector = BaseballDetector(
#         weights_path='./baseball_detection4/weights/best.pt',
#         ball_class=0,  # Set the class index for the baseball
#         conf_threshold=0.10                                 # For some reason, we're getting detections with 0.12 confidence threshold. (probably cause the images are so different)
#     )
    
#     # Detect baseball in an image
#     detections = detector.detect_baseball(
#         image_path='./baseball_img.png',
#         output_path='output_image.png'  # Save annotated image
#     )
    
#     # Print detection results
#     for i, det in enumerate(detections, 1):
#         print(f"Detection {i}: Bounding Box: {det[:4]}, Confidence: {det[4]:.2f}")

# Making bounding box for the video frames

'''
Working:

Take the code for a single image
Apply it to each frame
Stitch them together
'''

# import torch
# import cv2
# import os

# class BaseballDetector:
#     def __init__(self, weights_path, ball_class=0, conf_threshold=0.5):
#         """
#         Initialize the baseball detector.
#         weights_path: Path to YOLOv5 weights (e.g., best.pt).
#         ball_class: Class index for the baseball in the YOLO model.
#         conf_threshold: Confidence threshold for detection.
#         """
#         self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
#         self.model.conf = conf_threshold
#         self.ball_class = ball_class

#     def detect_baseball_in_frame(self, frame):
#         """
#         Detect baseball in a single frame.
#         frame: A single frame from the video.
#         Returns the annotated frame and detections.
#         """
#         results = self.model(frame)
#         detections = results.xyxy[0].cpu().numpy()
        
#         # Filter detections for the baseball class
#         baseball_detections = [
#             det for det in detections if int(det[5]) == self.ball_class
#         ]
        
#         # Draw bounding boxes around detected baseballs
#         for det in baseball_detections:
#             x1, y1, x2, y2, conf, cls = det
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 f"Baseball {conf:.2f}",
#                 (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2,
#             )
        
#         return frame, baseball_detections

#     def process_video(self, video_path, output_path):
#         """
#         Process a video to detect baseballs in each frame.
#         video_path: Path to the input video.
#         output_path: Path to save the output video.
#         """
#         # Open video file
#         cap = cv2.VideoCapture(video_path)
#         frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
#         # Initialize video writer
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        
#         # Process each frame
#         frame_count = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
#             print(f"Processing frame {frame_count}...")
            
#             # Detect baseballs in the current frame
#             annotated_frame, _ = self.detect_baseball_in_frame(frame)
            
#             # Write the annotated frame to the output video
#             out.write(annotated_frame)
        
#         # Release resources
#         cap.release()
#         out.release()
#         print(f"Video processing completed. Output saved at {output_path}")

# # Example usage
# if __name__ == "__main__":
#     # Initialize detector
#     detector = BaseballDetector(
#         weights_path='./baseball_detection4/weights/best.pt',
#         ball_class=0,  # Set the class index for the baseball
#         conf_threshold=0.10                         # Forced to keep this small because the model shows a very low probability
#     )
    
#     # Process video
#     detector.process_video(
#         video_path='../baseball.mp4',
#         output_path='output_video.mp4'
#     )

# Tracking

import torch
import cv2
import os
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class BaseballDetector:
    def __init__(self, weights_path, ball_class=0, conf_threshold=0.5):
        """
        Initialize the baseball detector.
        weights_path: Path to YOLOv5 weights (e.g., best.pt).
        ball_class: Class index for the baseball in the YOLO model.
        conf_threshold: Confidence threshold for detection.
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.conf = conf_threshold
        self.ball_class = ball_class
        self.tracker = DeepSort(max_age=30, n_init=3)  # Initialize DeepSORT tracker

    def detect_and_track(self, frame):
        """
        Detect and track the baseball in a single frame.
        frame: A single frame from the video.
        Returns the annotated frame, detections, and tracked objects.
        """
        # Detect baseball in the frame
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        # Filter detections for the baseball class
        baseball_detections = [
            det for det in detections if int(det[5]) == self.ball_class
        ]
        
        # Format detections for DeepSORT
        bbox_xywh = []
        confs = []
        for det in baseball_detections:
            x1, y1, x2, y2, conf, cls = det
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = x1 + bbox_width / 2
            center_y = y1 + bbox_height / 2
            bbox_xywh.append([center_x, center_y, bbox_width, bbox_height])
            confs.append(conf)
        
        # Validate bbox_xywh and confs
        if not bbox_xywh:
            return frame, {}
        
        # Debugging print statements
        print("bbox_xywh:", bbox_xywh)
        print("confs:", confs)
        
        try:
            # Track objects
            tracks = self.tracker.update_tracks(bbox_xywh, confs, frame=frame)
        except Exception as e:
            print(f"Error in tracker.update_tracks: {e}")
            print("bbox_xywh:", bbox_xywh)
            print("confs:", confs)
            raise
        
        # Draw bounding boxes and calculate velocities
        velocities = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()
            velocity = track.speed  # Velocity calculated by DeepSORT
            
            # Annotate frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id} | Speed: {velocity:.2f} px/frame",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            velocities[track_id] = velocity
        
        return frame, velocities


    def process_video(self, video_path, output_path):
        """
        Process a video to detect, track, and calculate velocities of baseballs in each frame.
        video_path: Path to the input video.
        output_path: Path to save the output video.
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))
        
        # Process each frame
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}...")
            
            # Detect and track baseballs in the current frame
            annotated_frame, velocities = self.detect_and_track(frame)
            
            # Debug: Print velocities for each tracked object
            for track_id, velocity in velocities.items():
                print(f"Frame {frame_count}: Track ID {track_id} -> Velocity: {velocity:.2f} px/frame")
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)
        
        # Release resources
        cap.release()
        out.release()
        print(f"Video processing completed. Output saved at {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = BaseballDetector(
        weights_path='./baseball_detection4/weights/best.pt',
        ball_class=0,  # Set the class index for the baseball
        conf_threshold=0.10
    )
    
    # Process video
    detector.process_video(
        video_path='../baseball.mp4',
        output_path='output_video_with_velocity.mp4'
    )


# okay some issue

# Using cache found in /home/purge/.cache/torch/hub/ultralytics_yolov5_master
# YOLOv5 🚀 2025-1-2 Python-3.11.11 torch-2.3.1 CPU

# Fusing layers... 
# Model summary: 157 layers, 7015519 parameters, 0 gradients, 15.8 GFLOPs
# Adding AutoShape... 
# Processing frame 1...
# bbox_xywh: [[405.29469299316406, 243.2591094970703, 31.322662, 26.462555]]
# confs: [0.10053124]
# Error in tracker.update_tracks: object of type 'numpy.float64' has no len()
# bbox_xywh: [[405.29469299316406, 243.2591094970703, 31.322662, 26.462555]]
# confs: [0.10053124]
# Traceback (most recent call last):
#   File "/home/purge/Desktop/MLBxG-extension/test/yolo_sort/flow.py", line 671, in <module>
#     detector.process_video(
#   File "/home/purge/Desktop/MLBxG-extension/test/yolo_sort/flow.py", line 647, in process_video
#     annotated_frame, velocities = self.detect_and_track(frame)
#                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/purge/Desktop/MLBxG-extension/test/yolo_sort/flow.py", line 587, in detect_and_track
#     tracks = self.tracker.update_tracks(bbox_xywh, confs, frame=frame)
#              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/purge/miniconda3/lib/python3.11/site-packages/deep_sort_realtime/deepsort_tracker.py", line 195, in update_tracks
#     assert len(raw_detections[0][0])==4
#            ^^^^^^^^^^^^^^^^^^^^^^^^^
# TypeError: object of type 'numpy.float64' has no len()

# Figure this out