'''
The goal here is to tweak the image in a certain way such that it matches the training dataset. Then see if the confindence threshold for baseball detection increases or not.
'''

from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

input_path = "./input/baseball_img_2.png"  # Path to the pitcher+batsman image
output_path = "./baseball_img_tweaked.png"  # Path to save the tweaked image


def stylize_image(input_path, output_path):
    """
    Apply artistic stylization to an image similar to the reference example.
    Includes contrast enhancement, sharpening, and color adjustments.
    """
    img = cv2.imread(input_path)
    
    img_float = img.astype(np.float32) / 255.0
    
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
    
    # Save the processed image
    cv2.imwrite(output_path, result)


stylize_image(input_path, output_path)
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
        weights_path='../baseball_detection4/weights/best.pt',
        ball_class=0,  # Set the class index for the baseball
        conf_threshold=0.065                                 # For some reason, we're getting detections with 0.12 confidence threshold. (probably cause the images are so different --> FIX THIS)
    )
    
    # Detect baseball in an image
    detections = detector.detect_baseball(
        image_path='./baseball_img_tweaked.png',							# using the tweaked image for detection
        output_path=f'./output/output_image{i}.png'  # Save annotated image
    )
    
    # Print detection results
    for i, det in enumerate(detections, 1):
        print(f"Detection {i}: Bounding Box: {det[:4]}, Confidence: {det[4]:.2f}")