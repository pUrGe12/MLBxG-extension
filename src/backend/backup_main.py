from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import sys

# For converting to mp4
from moviepy.editor import VideoFileClip

# imports for scraping and downloading youtube videos for old classical matches
from yt_dlp import YoutubeDL

# Selenium imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# Making post requests to the getBuffer method
import requests             

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # adding the root directory to path
# This should be done before any relative imports, adding the "backend" directory as root

# Imports for API querying 
from API_querying.query import call_API, pretty_print, figure_out_code
from API_querying.query import team_code_mapping, player_code_mapping

# Imports for helpers
from models.helper_models import check_buffer_needed, is_it_gen_stuff
from models.helper_models import gen_talk

import numpy as np
import pandas as pd

from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

# --------------------------------------------------------------------------
# Code for generating baseball speed
# --------------------------------------------------------------------------

from ultralytics import YOLO

from baseball_detect.flow import LoadTools, BallDetection, BaseballTracker          # This will handle everything
from baseball_detect.flow import calculate_pitcher_and_catcher
# ---------------------------------------------------------------------------
# Extraction model calling and implementation
# ---------------------------------------------------------------------------
from models.extraction import extractor

# ---------------------------------------------------------------------------
# This is the code for loading the trained models, and querying the vectorDB
# ---------------------------------------------------------------------------

from helper_files.options import GPT_response, load_models, process_new_hit, find_similar_hits, store_similar_hits

global encoder, scaler
encoder, scaler = load_models()             # Load the encoder and the scaler using the above defined function

# ---------------------------------------------------------------------------
# Import for bat detection and velocity calculation
# ---------------------------------------------------------------------------
from helper_files.tracking_bats import DeblurProcessor, BatTracker
from helper_files.tracking_bats import BatDetection

# ---------------------------------------------------------------------------
# Some helping functions
# ---------------------------------------------------------------------------

def convert_webm_to_mp4(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264")
    clip.close()

# For getting the video's link using selenium
# Note that in the future, this is the most likely part that will get fucked

# MAKE SURE THAT YOU CHECK THESE VALUES BEFORE SUBMISSION, that is, the values of the CSS selectors
def get_url(user_query):

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    chrome_options.add_argument("--window-size=1920x1080")  # Set window size for proper element rendering

    driver = webdriver.Chrome(options=chrome_options)

    driver.get(f'https://www.youtube.com/results?search_query={user_query}')

    # WHY CANT I CLICK HERE? This is for filtering based on 4-20 minutes. Commenting this for now
    # filter_for_four_to_twenty_minutes = driver.find_element(By.CSS_SELECTOR, "yt-chip-cloud-chip-renderer.style-scope:nth-child(9) > div:nth-child(2)").click()

    time.sleep(1)           # let the page load

    # This will always find the first link of the video (it won't pick up the sponsor messages so chill)
    link = driver.find_element(By.XPATH, "/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[1]/div[1]/div/div[1]/div/h3/a").get_attribute('href')

    # link has currently the & part which is something we don't care about
    link = str(link.split('&')[0])

    return link

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                           This is the function that can calculate the ball speed for us
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_speed_ball(video_path, min_confidence=0.5, max_displacement=100, min_sequence_length=7, pitch_distance_range=(55,65)):
    # First load the phc detector and find coordinates for the pitcher and the catcher
    SOURCE_VIDEO_PATH = video_path
    
    # print('here')
    # print(SOURCE_VIDEO_PATH)

    coordinates = calculate_pitcher_and_catcher(SOURCE_VIDEO_PATH)                     # Returns coordinates as (x1, y1, x2, y2)
    # x2, y2 is the pitcher and x1, y1 is the catcher
    
    x1, y1, x2, y2 = coordinates

    scale_factor = float(np.sqrt((x2-x1)**2 + (y2-y1)**2)/(60.5))              # scale factor is pixel distance between those niggas divided by actual distance between the niggas
    print(f"scale_factor: {scale_factor}")

    beta = float(np.arctan(2*(x1-x2)/(y2-y1)))                              # The math is explained in the readme docs, but we aren't really using it yet
    print(f"sin(beta) = {np.sin(beta)}")

    load_tools = LoadTools()
    model_weights = load_tools.load_model(model_alias='ball_trackingv4')
    model = YOLO(model_weights)

    tracker = BaseballTracker(
    model=model,
    min_confidence=0.3,         # 0.3 confidence works good enough, gives realistic predictions
    max_displacement=100,       # adjust based on your video resolution
    min_sequence_length=7,
    pitch_distance_range=(60, 61)  # feet
    )

    print('processing video')
    # Process video
    results = tracker.process_video(video_path, scale_factor) 

    output = """"""

    output += f"\nProcessed {results['total_frames']} frames at {results['fps']} FPS" + f"\n Found {len(results['sequences'])} valid ball sequences"

    for i, speed_est in enumerate(results['speed_estimates'], 1):
        # Run the beta calculations for each frame
        output += f"""
            \nSequence {i}:
            Frames: {speed_est['start_frame']} to {speed_est['end_frame']}
            Duration: {speed_est['time_duration']:.2f} seconds
            Average confidence: {speed_est['average_confidence']:.3f}\n
        """

        estimated_speed_min = float(speed_est['min_speed_mph'])
        estimated_speed_max = float(speed_est['max_speed_mph'])

        if beta*180/3.1415926 > 10:                               # if the angle is less than 10 degree, then the calculations become absurd!
            v_real_max = estimated_speed_max * 1/np.sin(beta)                    # This necessarily implies that v_real is more than v_app which is true.
            v_real_min = estimated_speed_min * 1/np.sin(beta)
            output += f"\nEsttimated speed: {v_real_max:.1f}" + f" to {v_real_min:.1f} mph"

        else:
            output += f"\nEstimated speed: {estimated_speed_min:.1f}" + f" to {estimated_speed_max:.1f} mph \n"
            print(f'Estimated speed range: {estimated_speed_max, estimated_speed_min}')

        output += f"This was within the time frame: {speed_est['start_frame'] * 1/results['fps']} to {speed_est['end_frame'] * 1/results['fps']}"

    return output

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
#                                           This is the function that can calculate the ball speed for us
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_speed_bat(video_path, min_confidence = 0.5, deblur_iterations = 30):

    SOURCE_VIDEO_PATH = video_path

    load_tools = LoadTools()
    model_weights = load_tools.load_model(model_alias='bat_tracking')
    model = YOLO(model_weights)

    tracker = BatTracker(
        model = model,
        min_confidence = 0.2,                       # 0.2 also works good enough
        deblur_iterations=30
    )

    results = tracker.process_video(SOURCE_VIDEO_PATH)

    return results

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                                       API endpoints
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Upload configs for classics-video
UPLOAD_FOLDER = 'uploads/videos'                                    # To save the uploaded videos
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/check-buffer', methods=['POST'])
def check_buffer():
    """Endpoint that receives buffer but only saves it if we're expecting it"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    # Simply return 404 if we're not expecting buffer
    # This tells frontend to keep trying
    if not getattr(app, 'waiting_for_buffer', False):
        return jsonify({'message': 'Buffer not needed yet'}), 404
    
    # If we are waiting for buffer, save it
    video_file = request.files['video']
    with open("./input_files/received_video.webm", "wb") as fp:
        video_file.save(fp)
    
    # Signal that we've received the buffer
    app.waiting_for_buffer = False
    app.buffer_received = True
    
    return jsonify({'message': 'Buffer received'}), 200


@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.json.get('input')  # Receive input from the panel
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    '''
    Here is how the processing takes place
    1. Figure out if the buffer is required. 
    2. If it is not required, then answer general baseball stats. This is what panel_response is
    3. If it is required, then process the video and answer.

    We'll focus on figuring out 2 statcast datas from the videos, those are the pitcher's speed, and the batsman's hit velocities
    
    How do we ensure that this works?
    '''
    
    boolean = check_buffer_needed(user_input)
    
    if boolean == False:             # That is no buffer needed
        # print('buffer not needed!')
        
        '''
        Here, we may have to call the APIs
        The idea here is to parse the input, determine if its general baseball stuff or particular to a player, or a team or schedule, in which case we'll call the API querying function.
        '''

        if is_it_gen_stuff(user_input):
            print('Requires APIs')

            name_code_tuple = figure_out_code(team_code_mapping, player_code_mapping, user_input)

            output = call_API(name_code_tuple)

            # processed_output = output
            # print(output)

            if 'schedule' not in name_code_tuple[0]:
                processed_output = pretty_print(output, user_input)             # not doing this if the output is too big as in the case of the schedule
            else:
                processed_output = output

        else:
            processed_output = gen_talk(user_input)          # We'll add a normal response generator here

        return jsonify({"response": processed_output})

    else:                                                            # that is buffer is required
        print('buffer is needed!')

        app.waiting_for_buffer = True
        app.buffer_received = False

        print('initialised flags')
        import time
        max_wait = 5  # Maximum seconds to wait
        start_time = time.time()

        while not getattr(app, 'buffer_received', False):
            print('inside while loop')
            time.sleep(0.1)
            if time.time() - start_time > max_wait:
                return jsonify({"error": "Timeout waiting for video buffer"}), 500

        convert_webm_to_mp4("./input_files/received_video.webm", "./input_files/converted_input.mp4")
        print('converted')
        video_path = "./input_files/converted_input.mp4"
        what_is_needed = check_statcast(user_input)

        # -------------------------------------------------------------------------------------------
        #               Now we need a logic to see if they're asking for speed or what
        # -------------------------------------------------------------------------------------------

        if 'baseballspeed' in what_is_needed.strip().lower():
            output = calculate_speed_ball(video_path)
            return jsonify({"response": output}), 200
        
        elif 'exitvelocity' in what_is_needed.strip().lower():
            output = calculate_speed_bat(video_path)
            return jsonify({"response": output}), 200
        else:
            return jsonify({"error": "Unknown analysis type requested"}), 500


@app.route('/user-stat/', methods=['POST'])
def user_stat():
    user_input = request.json.get('input')  # Receive input from the panel

    print(user_input)

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        print('in the try block')
        extractor_dictionary, additional_params = extractor(user_input)

        print("unpacked the tuple!")

        '''  
            extractor output is a dictionary that describes the following of the user
            1. ExitVelocity
            2. HitDistance
            3. LaunchAngle

            We parse the "AdditionalParams" seperately. The idea is to send the extractor_dictionary to the pineconeDB and get top similar hits

        '''

        embedding = process_new_hit(extractor_dictionary, encoder, scaler)                           
        
        print("embeddings generated")

        # Find the top 5 matches to the user's stats being entered above.
        found_similar_hits = find_similar_hits(embedding, top_k = 5)
        top_similar_hits = store_similar_hits(found_similar_hits)               # Storing that to send it to the model for prettifying

        print(top_similar_hits)

        processed_output = GPT_response(top_similar_hits, additional_params.get('AdditionalParams'), user_input)                    # Both additional params and extractor dict are dictionaries.

        print(processed_output)

        return jsonify({
            "response": processed_output
            })

    except Exception as e:
        incomplete_text = extractor(user_input)

        print(incomplete_text)
        print('here in exception', e)

        '''
            Enter this block of code when unpacking the tuple leads to an exception
            This will only happen when there is only one element being returned by the extractor, that is the incomplete text. 

            We will show the incomplete text directly to the user. 
        '''

        return jsonify({
            "response": incomplete_text
            })


# -----------------------------------------------------------X Classics endpoints X--------------------------------------------------------------

from werkzeug.utils import secure_filename                              # Adding this here cause not always requireds

@app.route('/classics-video/', methods=['POST'])
def classics_video_processing():
    '''
    This is where the user is suppossed to enter the video as a mp4 file.

    We additionally wanna make sure that the entered file is mp4 itself! Not to have a php file being executed here and my laptop becoming compromised!
    '''
    try:
        # Check if video file is in request
        if 'video' not in request.files:
            return jsonify({'error': 'No video file in request'}), 400
        
        video_file = request.files['video']                                                 # This is a mp4 video file
        
        if video_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(video_file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Secure the filename and save the file
        filename = secure_filename(video_file.filename)
        
        print(filename)

        # Ensure upload directory exists and saving it. We'll keep the name as above so that its easy to process
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(file_path)
        
        SOURCE_VIDEO_PATH = f'/home/purge/Desktop/MLBxG-extension/src/backend/uploads/videos/{filename}'
        output = calculate_speed_ball(SOURCE_VIDEO_PATH)

        return jsonify({                                                            # Also need to return uploaded message
            'message': 'Video analysed successfully, now running plan',
            'filename': filename,
            'output': output                                                                # Handle this properly in the frontend
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classics-text/', methods=['POST'])
def classics_text_processing():
    '''
    Here we will have to first scrape the video! Then apply the same concept as above
    
    With this we have downloaded the video, now we can start processing.
    Now, this video is likely going to be very long! (may even be hours long). So how do we parse this and generate meaningful output?

    1. Ensure that the video we're downloading is less than 20 minutes (so, its more like highlights rather than the entire match)
    2. If the video length is more than 20 minutes, ask the user to be generous otherwise it'll take way too much time!
    3. So, we're detecting valid sequences of continous ball movements. In highlights, this may refer to either a pitch, or a hit and travel by the ball, or a player's throw
    4. We need to see how we differentiate that. Also, we can use timestamps of the video itself to let the know the ball speeds for the sequence within that timestamp!

    So basically, the user get's an output like "within the 5th to 6th second, the ball was travelling at a speed of 97mph" and that one second was the pitch. This way. 

    Now, we can do similar things for the batsman's hit. This will be easier because the logic is then to have both the bat and the ball in the same frame, with distance between them reducing, 
    and the bat swinging. Just code these conditions and we're done.
    '''

    user_input = request.json.get('input')  # Receive input from the panel
    print(user_input)

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # assuming user_input is a properly formatted input that we can directly use to search youtube (later we'll add LLM)

    custom_url = get_url(user_input)                                                    # based on user's input search youtube and select the first link
    path = '/home/purge/Desktop/MLBxG-extension/src/backend/downloads_classical'                # note that we can later change all these to match the path of the user

    ydl_opts = {
        'outtmpl': os.path.join(path, '%(title)s.%(ext)s'),
    }

    with YoutubeDL(ydl_opts) as ydl:
        # Getting the filename first
        video_info = ydl.extract_info(custom_url, download=False)
        filename = ydl.prepare_filename(video_info)

        ydl.download([custom_url])

    
    SOURCE_VIDEO_PATH = filename

    output = calculate_speed_ball(SOURCE_VIDEO_PATH)

    return jsonify({
        'message': f"This is what we found: \n{output}"
        })


@app.route('/load-yolo-model/', methods=['POST'])
def loading_yolo_models():
    load_tools = LoadTools()
    model_weights1 = load_tools.load_model(model_alias='ball_trackingv4')
    model1 = YOLO(model_weights1)

    model_weights2 = load_tools.load_model(model_alias='phc_detector')
    model2 = YOLO(model_weights2)

    model_weights3 = load_tools.load_model(model_alias='bat_tracking')
    model3 = YOLO(model_weights3)

    return jsonify({
        'message': "loaded deeplearning model"
        })

if __name__ == "__main__":
    app.waiting_for_buffer = False
    app.buffer_received = False
    app.run(host="127.0.0.1", port=5000, debug=True)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))  # Render will provide PORT env variable
#     app.run(host="0.0.0.0", port=port)