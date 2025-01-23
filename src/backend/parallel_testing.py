import time 
t1 = time.time()
import multiprocessing
import numpy as np
import cv2
from ultralytics import YOLO
from baseball_detect.test_flow import LoadTools, BaseballTracker, calculate_pitcher_and_catcher

def process_baseball_detection(video_path, result_queue):
    """
    Perform baseball detection and put results in a queue.
    
    Args:
        video_path (str): Path to the video file
        result_queue (multiprocessing.Queue): Queue to store detection results
    """
    load_tools = LoadTools()
    model_weights = load_tools.load_model(model_alias='ball_trackingv4')
    model = YOLO(model_weights)
    
    tracker = BaseballTracker(
        model=model,
        min_confidence=0.3,
        max_displacement=100,
        min_sequence_length=7,
        pitch_distance_range=(60, 61)
    )
    
    # Perform detection without scale factor
    results = tracker.process_video(video_path)
    result_queue.put(('detection', results))

def process_pitcher_catcher(video_path, result_queue):
    """
    Calculate pitcher and catcher coordinates and put results in a queue.
    
    Args:
        video_path (str): Path to the video file
        result_queue (multiprocessing.Queue): Queue to store coordinate results
    """
    coordinates = calculate_pitcher_and_catcher(video_path)
    x1, y1, x2, y2 = coordinates
    
    # Calculate scale factor
    scale_factor = float(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 60.5)
    beta = float(np.arctan(2*(x1-x2)/(y2-y1)))
    
    result_queue.put(('coordinates', {
        'coordinates': coordinates, 
        'scale_factor': scale_factor, 
        'beta': beta
    }))

def calculate_speed_ball_parallel(video_path):
    """
    Main function to parallelize baseball detection and metadata calculation.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        str: Formatted output of processing results
    """
    # Create a queue for inter-process communication
    result_queue = multiprocessing.Queue()
    
    # Create processes
    detection_process = multiprocessing.Process(
        target=process_baseball_detection, 
        args=(video_path, result_queue)
    )
    coordinate_process = multiprocessing.Process(
        target=process_pitcher_catcher, 
        args=(video_path, result_queue)
    )
    
    # Start both processes
    detection_process.start()
    coordinate_process.start()
    
    # Collect results
    results = {}
    for _ in range(2):
        key, value = result_queue.get()
        results[key] = value
    
    # Wait for processes to complete
    detection_process.join()
    coordinate_process.join()
    
    # Process final results
    load_tools = LoadTools()
    model_weights = load_tools.load_model(model_alias='ball_trackingv4')
    model = YOLO(model_weights)
    
    tracker = BaseballTracker(
        model=model,
        min_confidence=0.3,
        max_displacement=100,
        min_sequence_length=7,
        pitch_distance_range=(60, 61)
    )

    # Calculate speed estimates
    speed_estimates = tracker._calculate_speeds(
        results['detection']['sequences'], 
        results['detection']['fps'], 
        results['coordinates']['scale_factor']
    )
    
    # Merge results
    final_results = results['detection']
    final_results['speed_estimates'] = speed_estimates
    final_results['scale_factor'] = results['coordinates']['scale_factor']
    final_results['pitcher_catcher_coordinates'] = results['coordinates']['coordinates']
    beta = results['coordinates']['beta']

    # Format output (similar to your original implementation)
    output = f"\nProcessed {final_results['total_frames']} frames at {final_results['fps']} FPS"
    output += f"\nFound {len(final_results['sequences'])} valid ball sequences"
    output += f"\nScale Factor: {final_results.get('scale_factor', 'N/A')}"
    
    for i, speed_est in enumerate(final_results['speed_estimates'], 1):
        output += f"""\n\nSequence {i}:
        Frames: {speed_est['start_frame']} to {speed_est['end_frame']}
        Duration: {speed_est['time_duration']:.2f} seconds
        Average confidence: {speed_est['average_confidence']:.3f}
        """
        estimated_speed_min = float(speed_est['min_speed_mph'])
        estimated_speed_max = float(speed_est['max_speed_mph'])

        if beta*180/3.1415926 > 10:
            v_real_max = estimated_speed_max * 1/np.sin(beta)
            v_real_min = estimated_speed_min * 1/np.sin(beta)
            output += f"\nEstimated speed: {v_real_max:.1f}" + f" to {v_real_min:.1f} mph"
        else:
            output += f"\nEstimated speed: {estimated_speed_min:.1f}" + f" to {estimated_speed_max:.1f} mph \n"
            print(f'Estimated speed range: {estimated_speed_max, estimated_speed_min}')

        output += f"This was within the time frame: {speed_est['start_frame'] * 1/final_results['fps']} to {speed_est['end_frame'] * 1/final_results['fps']}"

    return output

# Example usage
print(calculate_speed_ball_parallel("./uploads/videos/baseball_3.mp4"))
print(f"time taken: {time.time() - t1}")

# This time is 442 seconds approx! This is too much which means either parallelisation isn't happening quite like I expected, or something's wrong
# The normal sequencial execution is taking 165 seconds only... which is 3 times faster!