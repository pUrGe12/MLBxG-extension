import os
import cv2

def processVideoForStatcastData(video_complete, output_path):
    video_complete = os.path.expanduser(video_complete)
    cap = cv2.VideoCapture(video_complete)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:

            # Filtering small areas to eliminate noise and highlight the ball (ensuring the ball doesn't get filtered out)
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            # Calculate perimeter and circularity and using that to zero in on the ball
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:			# can't divide by zero
                continue
            circularity = (4 * 3.14159 * area) / (perimeter * perimeter)

            if circularity < 0.7:  # Ignore non-circular objects
                continue

            # Draw bounding rectangle around the detected object. This is not working at all.
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        out.write(fgmask)

    cap.release()
    out.release()
    print("Processing complete. Video saved at:", output_path)

# Example Usage
processVideoForStatcastData("~/Downloads/baseball.mp4", "output_processed.mp4")
