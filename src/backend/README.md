# Backend

This is where the processing happens. The important files of consideration are:

1. The [main file](./main.py)
2. [API querying files](./API_querying)
3. The helper models [part 1](./models)
4. The helper files [part 2](./helper_files)

Additionally we have the [uploads file](./uploads/videos) which is where the user's inputted videos (as will be explained later) go.

The directory also have some test files (which I should've performed in the test directory itself but got carried away). Those are for parallelly detecting baseballs and catcher+pitcher but it was so much slower in my laptop!

## Functionality

The backend performs the following functions

### Panel

1. LLM intergration that answers basic baseball questions
2. API query pipeline for the LLM to answer questions based on player's and team's information as well as the schedule for MLB
3. Generate statcast data for a live video being watched on YouTube (this is a bit iffy)

For example:

The user can ask questions like "how fast was that ball?" and the LLM will be able to answer that! 

### Options

1. Using pinecone to store home-run stats for top MLB players and comparing the user's stats with them to let them know how good or bad they are (through an LLM again)
2. There is a guide that explains the techincal workings of the extension (much like this page itself)
3. User's can enter a classical MLB match directly (in mp4 format) or enter the name of the match (and we'll scrape for ourselves) to generate it's statcast data 
(as data recorders were not there)

## How we do it?

The complete techincal explanations are here:

- The `background.js` script is always active and it detects for when the user enters YouTube. This is when the extension becomes active. Then once the user clicks on a video, a buffer is initialised which stores individual frames at 30 FPS. This is a rolling buffer which always holds 5 seconds worth of video feed in `webp` format.

### Panel

1. Flask endpoint with LLM processing the message. We first determine if the user needs the buffer or not.
2. If the user needs the buffer, we send it over using the extension's ID and some internal mechs (Not sure if this works actually!) and then start processing it.

We are assuming that in these 5 seconds will the pitch that the user is talking about. Very drastic thing to assume but lets roll with it for now.

3. Using the video feed, we save that locally, run the speed prediction model in that and tell the user the output.

4. We also reference this [database](https://github.com/MajorLeagueBaseball/google-cloud-mlb-hackathon) provided by the organisers. These APIs and their parsing code is integrated in the panel, and hence the user can get up-to-date information about players, schedules or teams.

The way these APIs are implemented is:

- The LLM parses the user's input and determines if APIs are required or not (this is pure prompt engineering) and returns a structured output if yes.
- Then we use the relevant dictionaries (that I created) that maps the user's input names to the code used in MLB.
- Then we query the API endpoint, recieve the information and parse it using another custom set of codes. This is finally provided to a `pretty printing` LLM which takes the user's input and this data and extrapolates the right answer.

#### Speed prediction

- Here, we're using a custom YOLOv8 model that I found over [here](https://github.com/dylandru/BaseballCV). This has been trained on data specific to MLB and is especially accurate in determing baseballs when it is thrown in a pitch.

- We take this model, locate the baseball at individual frames of the image. Taking the ball's position to be the center of the bounding box

1. Figure out how many pixels the ball travels in between frames. In case of anomalous detections (which are inevitable) I am taking averages for now, later we can look at interpolations and splines. A continous sequence of these detections (minimum of 10 frames) constitues a `valid sequence`.
2. Get a `pixel_to_feet_ratio` which depends on the `pitch length` and the `camera angles`. The pitch length is more of less fixed, we only need to care about the camera angle.


There are two angles, the azimuthal and radial. Radial is comparatively easier to calculate, as it can be done using the relative positions of the batter/catcher and the pitcher during the pitch and trig functions.

The azimuthal angle is hard. We know for a fact that when the azimuth angle is 90, we are essentially viewing it from the very top (as if hovering right above the playing field) and when its 0, we're basically behind the pitcher and can't see the ball moving. 

I am guessing sin(l_azimuth) comes into picture. Not entirely sure.

Anyway, figure this out!

3. After this calibration, we can figure out the total number of pixels moved by the ball (until it was caught by the catcher, or hit by the batter) and hence the total feet moved.
4. Since we know the FPS and the total number of frames for the balls detection (in a `valid sequence`) we know the total time the ball was travelling for and hence the average speed can be calculated.

Points of doubt:

1. Finding `pixel_to_feet_ratio` accurately
2. Calculating the angles

### The options page

The options page contains 3 things as of now
1. MLB future predictor
2. Guide page
3. Classics video statcast generator

#### MLB future predictor (add a video uploader as well!! This will take the video and compare swings?! I guess?)

This is another flask endpoint which is again connected to an LLM to parse the user's input and extract the relevant information out of it

1. The user is required to enter their own statcast data for their homeruns, such as their hit distance, exit-velocity (how fast the ball went after hitting it), and launch angle. 

This is mostly for experienced players or players who are being trained in a proper enviornment. We don't expect everyday players to have such information. For them we have something else!

- This data is added to a `pinecone` database where I have already collected embeddings of MLB stars with their statcast data.

- With this I've generated a similarity score which tells the user how they compare to the stars in the world. The user can also enter `additional information` and we expect the LLM to weigh these in appropriately. 

- Finally the output is rendered in a markdown format and a report is presented to the user.

(Not yet implemented)

2. The novice users can enter a `mp4` video file that contains them practising their swings or pitches.

How do we compare? Will have to learn...

#### Guide page

This is essentially the contents of the guide page. Its there to just tell the users (who don't read READMEs often) about how the extension does what it does and hence any prediction made by the extension is based entirely on the mathematics behind these implementations

#### Classics statcast

This is the part where we ask the user to either enter a name of a classics match or a mp4 video of it. I have done this because I don't expect every person to have a video of match lying around somewhere, so they can just enter the name itself and we'll scrape.

1. If the user enters a name, I have used `selenium` to search that up on `YouTube` and get the link for a video that is within 4-20 mins long. Then using `youtube-dl` I download that video and save it in the downloads folder (not the default one).

2. Then applying the same processing code as before I calculate the speeds of the pitches and hits of the bats.

Here the crucial part is displaying the output. In the intial implementation I have calculated all the valid sequences (as defined above) and output the speeds for each of those sequences. 

This is done along with the timestamps and time-intervals for when these values are calculated from. Its a bit tedious but the user can reference the video during the given time-intervals and watch the ball being throwing (for example).

(not yet implemented)
In the second edition I am planning on annotating the video feed itself and displaying the entire video as an output. This will be much easier for the user to comprehend!

3. If the user enters a video itself, then we process it in a similar fashion and output results also in a similar fashion (this is easier for me as I don't have to scrape anymore)

