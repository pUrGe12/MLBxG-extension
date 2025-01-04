'''
The goal here is to tweak the image in a certain way such that it matches the training dataset. Then see if the confindence threshold for baseball detection increases or not.
'''

from PIL import Image
import piexif
from pathlib import Path

def preprocess_image(image_path, output_path=None):
    """
    Preprocess an image to match the training dataset preprocessing:
    1. Auto-orient using EXIF data and strip EXIF
    2. Resize to 640x640 with stretching
    
    Args:
        image_path (str): Path to input image
        output_path (str, optional): Path to save processed image. If None, overwrites original.
    
    Returns:
        PIL.Image: Processed image object
    """
    # Open the image
    img = Image.open(image_path)
    
    # Auto-orient based on EXIF data
    try:
        # Check if image has EXIF data
        exif = img._getexif()
        if exif:
            # Get the orientation tag (274)
            orientation = exif.get(274)
            if orientation is not None:
                # Rotate/flip based on EXIF orientation
                exif_orientation_map = {
                    2: (Image.FLIP_LEFT_RIGHT,),
                    3: (Image.ROTATE_180,),
                    4: (Image.FLIP_TOP_BOTTOM,),
                    5: (Image.ROTATE_270, Image.FLIP_LEFT_RIGHT),
                    6: (Image.ROTATE_270,),
                    7: (Image.ROTATE_90, Image.FLIP_LEFT_RIGHT),
                    8: (Image.ROTATE_90,)
                }
                if orientation in exif_orientation_map:
                    for operation in exif_orientation_map[orientation]:
                        img = img.transpose(operation)
    except:
        # If EXIF processing fails, continue with original image
        pass
    
    # Strip all EXIF data
    img_without_exif = Image.new(img.mode, img.size)
    img_without_exif.putdata(list(img.getdata()))
    
    # Resize to 640x640 with stretching
    img_resized = img_without_exif.resize((640, 640), Image.Resampling.LANCZOS)
    
    # Save if output path provided
    if output_path:
        img_resized.save(output_path)
    else:
        img_resized.save(image_path)
    
    return img_resized

input_path = r'./input/baseball_img.png'
output_path = r'./input/baseball_img_tweaked.png'

preprocess_image(input_path, output_path)
print(f"Tweaked image saved to {output_path}")

import torch
import cv2

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

    def detect_baseball(self, image_path, output_path=None):
        """
        Detect baseball in a still image.
        image_path: Path to the input image.
        output_path: Path to save the output image with detections (optional).
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Run YOLO detection
        results = self.model(image)                       # Since the model was trained with images of size 416x416, but if I use that then it messes up!
        detections = results.xyxy[0].cpu().numpy()
        
        # Filter detections for the baseball class
        baseball_detections = [
            det for det in detections if int(det[5]) == self.ball_class
        ]
        
        # Draw bounding boxes around detected baseballs
        for det in baseball_detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Baseball {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        
        # Save the output image if an output path is provided
        if output_path:
            cv2.imwrite(output_path, image)
        
        return baseball_detections

# Example usage
if __name__ == "__main__":
    # Initialize detector
    i = 3
    print('starting detection')
    detector = BaseballDetector(
        weights_path='./baseball_detection/weights/best.pt',
        ball_class=0,  # Set the class index for the baseball
        conf_threshold=0.065                                 # For some reason, we're getting detections with 0.12 confidence threshold. (probably cause the images are so different --> FIX THIS)
    )
    
    # Detect baseball in an image
    detections = detector.detect_baseball(
        image_path='./input/baseball_img_tweaked.png',							
        output_path=f'./output/baseball_image_detected_1.png'  # Save annotated image
    )
    
    # Print detection results
    for i, det in enumerate(detections, 1):
        print(f"Detection {i}: Bounding Box: {det[:4]}, Confidence: {det[4]:.2f}")