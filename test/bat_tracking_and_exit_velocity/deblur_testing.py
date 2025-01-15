import cv2
import numpy as np
from skimage.restoration import richardson_lucy

# Function to create a motion blur PSF (Point Spread Function)
def motion_blur_psf(length, angle):
    psf = np.zeros((length, length))
    center = length // 2
    for i in range(length):
        offset = int(np.tan(np.radians(angle)) * (i - center))
        if 0 <= center + offset < length:
            psf[i, center + offset] = 1
    return psf / psf.sum()

# Function to deblur an image using Richardson-Lucy deconvolution
def deblur_frame(frame, psf, iterations=30):
    # Normalize frame for deconvolution
    frame = frame / 255.0
    # Perform Richardson-Lucy deconvolution
    deblurred = np.zeros_like(frame)
    for c in range(frame.shape[2]):  # Apply for each channel
        deblurred[..., c] = richardson_lucy(frame[..., c], psf, iterations=iterations)
    # Scale back to [0, 255]
    return np.clip(deblurred * 255, 0, 255).astype(np.uint8)

# Main function to process the video
def process_video(input_video, output_video, psf_length, psf_angle, iterations=30):
    # Open the video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Generate the PSF
    psf = motion_blur_psf(psf_length, psf_angle)
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Deblur the frame
        deblurred_frame = deblur_frame(frame, psf, iterations)
        
        # Write the processed frame to the output video
        out.write(deblurred_frame)
        cv2.imshow('Deblurred Frame', deblurred_frame)
        
        # Break on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_video}")

# Example usage
input_video_path = 'baseball_3.mp4'
output_video_path = 'deblurred_video.mp4'
psf_length = 15  # Adjust this based on the blur extent
psf_angle = 0    # Adjust this based on the blur direction (in degrees)
process_video(input_video_path, output_video_path, psf_length, psf_angle)
