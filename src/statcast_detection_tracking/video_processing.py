'''
Working:

- Take the code for a single image (along with the tweaks)
- Apply it to each frame
- Stitch them together
'''

import torch
import cv2
import numpy as np

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

    def stylize_frame(self, frame):
        """
        Apply artistic stylization to a frame similar to the reference example.
        Includes contrast enhancement, sharpening, and color adjustments.
        """
        # Convert frame to float32
        img_float = frame.astype(np.float32) / 255.0
        
        # Increase contrast using gamma correction
        gamma = 0.9
        img_contrast = np.power(img_float, gamma)

        # Apply unsharp masking for sharpening
        blur = cv2.GaussianBlur(img_contrast, (0, 0), 30)
        img_sharp = cv2.addWeighted(img_contrast, 3, blur, -1, 0)
        
        # Enhance color saturation
        img_hsv = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2HSV)
        img_hsv[:,:,1] = img_hsv[:,:,1] * 2  # Increase saturation
        img_enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        # Add slight vignette effect
        rows, cols = img_enhanced.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/4)
        kernel_y = cv2.getGaussianKernel(rows, rows/4)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        # Apply vignette
        for i in range(3):
            img_enhanced[:,:,i] = img_enhanced[:,:,i] * mask
        
        # Ensure values are in valid range
        img_enhanced = np.clip(img_enhanced, 0, 1)
        
        # Convert back to uint8
        result = (img_enhanced * 255).astype(np.uint8)
        
        return result

    def detect_baseball_in_frame(self, frame):
        """
        Detect baseball in a single frame.
        frame: A single frame from the video.
        Returns the annotated frame and detections.
        """
        # First, stylize the frame
        stylized_frame = self.stylize_frame(frame)
        
        # Run detection on stylized frame
        results = self.model(stylized_frame)
        detections = results.xyxy[0].cpu().numpy()
        
        # Filter detections for the baseball class
        baseball_detections = [
            det for det in detections if int(det[5]) == self.ball_class
        ]
        
        # Draw bounding boxes around detected baseballs on the original frame
        for det in baseball_detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Baseball {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        
        return frame, baseball_detections

    def process_video(self, video_path, output_path):
        """
        Process a video to detect baseballs in each frame.
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
            
            # Detect baseballs in the current frame
            annotated_frame, detections = self.detect_baseball_in_frame(frame)
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)
            
            # Print detection results for this frame
            for i, det in enumerate(detections, 1):
                print(f"Frame {frame_count} - Detection {i}: Confidence: {det[4]:.2f}")
        
        # Release resources
        cap.release()
        out.release()
        print(f"Video processing completed. Output saved at {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = BaseballDetector(
        weights_path='./baseball_detection/weights/best.pt',
        ball_class=0,  # Set the class index for the baseball               # This model is capable of detecting the bat and some other things as well
        conf_threshold=0.1  # Using the same threshold as in your image example
    )
    
    # Process video
    detector.process_video(
        video_path='../../baseball.mp4',
        output_path='output_video_stylized.mp4'
    )



# Tracking

# import torch
# import cv2
# import os
# import numpy as np
# from deep_sort_realtime.deepsort_tracker import DeepSort

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
#         self.tracker = DeepSort(max_age=30, n_init=3)  # Initialize DeepSORT tracker

#     def detect_and_track(self, frame):
#         """
#         Detect and track the baseball in a single frame.
#         frame: A single frame from the video.
#         Returns the annotated frame, detections, and tracked objects.
#         """
#         # Detect baseball in the frame
#         results = self.model(frame)
#         detections = results.xyxy[0].cpu().numpy()
        
#         # Filter detections for the baseball class
#         baseball_detections = [
#             det for det in detections if int(det[5]) == self.ball_class
#         ]
        
#         # Format detections for DeepSORT
#         bbox_xywh = []
#         confs = []
#         for det in baseball_detections:
#             x1, y1, x2, y2, conf, cls = det
#             bbox_width = x2 - x1
#             bbox_height = y2 - y1
#             center_x = x1 + bbox_width / 2
#             center_y = y1 + bbox_height / 2
#             bbox_xywh.append([center_x, center_y, bbox_width, bbox_height])
#             confs.append(conf)
        
#         # Validate bbox_xywh and confs
#         if not bbox_xywh:
#             return frame, {}
        
#         # Debugging print statements
#         print("bbox_xywh:", bbox_xywh)
#         print("confs:", confs)
        
#         try:
#             # Track objects
#             tracks = self.tracker.update_tracks(bbox_xywh, confs, frame=frame)
#         except Exception as e:
#             print(f"Error in tracker.update_tracks: {e}")
#             print("bbox_xywh:", bbox_xywh)
#             print("confs:", confs)
#             raise
        
#         # Draw bounding boxes and calculate velocities
#         velocities = {}
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
            
#             track_id = track.track_id
#             x1, y1, x2, y2 = track.to_tlbr()
#             velocity = track.speed  # Velocity calculated by DeepSORT
            
#             # Annotate frame
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(
#                 frame,
#                 f"ID {track_id} | Speed: {velocity:.2f} px/frame",
#                 (int(x1), int(y1) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (0, 255, 0),
#                 2,
#             )
#             velocities[track_id] = velocity
        
#         return frame, velocities


#     def process_video(self, video_path, output_path):
#         """
#         Process a video to detect, track, and calculate velocities of baseballs in each frame.
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
            
#             # Detect and track baseballs in the current frame
#             annotated_frame, velocities = self.detect_and_track(frame)
            
#             # Debug: Print velocities for each tracked object
#             for track_id, velocity in velocities.items():
#                 print(f"Frame {frame_count}: Track ID {track_id} -> Velocity: {velocity:.2f} px/frame")
            
#             # Write the annotated frame to the output video
#             out.write(annotated_frame)
        
#         # Release resources
#         cap.release()
#         out.release()
#         print(f"Video processing completed. Output saved at {output_path}")

# # Example usage
# if __name__ == "__main__":
#     # Initialize detector
#     print('starting')
#     detector = BaseballDetector(
#         weights_path='./baseball_detection4/weights/best.pt',
#         ball_class=0,  # Set the class index for the baseball
#         conf_threshold=0.10
#     )
    
#     # Process video
#     detector.process_video(
#         video_path='../baseball.mp4',
#         output_path='output_video_with_velocity.mp4'
#     )


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